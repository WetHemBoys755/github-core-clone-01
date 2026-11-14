// The HTTP server: one shared Store, four surfaces.
//
// Routing (all on one process, one ephemeral port — never 8080):
//   GET  /healthz                                  the env's boot probe
//   GET  /api/v3[/]                                the API root
//   GET|POST /<owner>/<repo>.git/info/refs         git smart-HTTP advertisement
//   POST /<owner>/<repo>.git/git-upload-pack       git fetch (v2 command= / v0 wants)
//   POST /<owner>/<repo>.git/git-receive-pack      git push
//   *    /api/v3/*                                 REST v3 (rest.ts)
//   POST /api/graphql                              GraphQL v4 (graphql.ts)
//   POST /mcp                                      MCP (mcp.ts)
//
// The viewer is the login the environment's token belongs to (the scenario's
// fixed viewer, `adaiversen`): GET /api/v3/user and GraphQL viewer{login}
// both report it, and repositories are created under it. A repo name under
// any other login is 404, never 403. The git endpoints answer from repo
// existence alone (a missing repo is 404 on both services, and the server
// stays healthy through malformed requests); REST, GraphQL and MCP require
// a non-empty Bearer token and answer 401 without one.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { Store } from './store.ts';
import { readBody } from './util.ts';
import { decodePktLines, handleFetchV0, handleFetchV2, handleLsRefs, handleReceivePack, v0Advertisement, v2Advertisement } from './gitproto.ts';
import { handleRest, type RestRequest } from './rest.ts';
import { executeGraphQL } from './graphql.ts';
import { handleMcp } from './mcp.ts';

/** The login the environment's token belongs to (the scenario's viewer). */
const VIEWER_LOGIN = 'adaiversen';

const JSON_TYPE = 'application/json; charset=utf-8';

/** Answer one request with a JSON body. */
function sendJson(res: ServerResponse, status: number, body: unknown, headers: Readonly<Record<string, string>> = {}): void {
  const text = `${JSON.stringify(body)}\n`;
  res.writeHead(status, { 'content-type': JSON_TYPE, 'content-length': Buffer.byteLength(text), ...headers });
  res.end(text);
}

/** Answer one request with raw bytes and a content type. */
function sendBytes(res: ServerResponse, status: number, body: Buffer, contentType: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': body.length });
  res.end(body);
}

/** The path + query of a request URL (never a throw from a request). */
function urlOf(req: IncomingMessage): { path: string; query: URLSearchParams } {
  try {
    const u = new URL(req.url ?? '/', 'http://entry');
    return { path: u.pathname, query: u.searchParams };
  } catch {
    return { path: '/', query: new URLSearchParams() };
  }
}

/** Lower-cased request headers as a plain record. */
function headersOf(req: IncomingMessage): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(req.headers)) {
    if (typeof value === 'string') out[key.toLowerCase()] = value;
    else if (Array.isArray(value)) out[key.toLowerCase()] = value.join(', ');
  }
  return out;
}

/** A git path `<owner>/<repo>.git/...` split into its parts, or null. */
function gitPath(path: string): { owner: string; repo: string; rest: string[] } | null {
  const parts = path.split('/').filter((s) => s.length > 0);
  if (parts.length < 2) return null;
  const repoPart = parts[1]!;
  if (!repoPart.endsWith('.git')) return null;
  return {
    owner: decodeURIComponent(parts[0]!),
    repo: decodeURIComponent(repoPart.slice(0, -4)),
    rest: parts.slice(2),
  };
}

/** Build the server. It binds nothing until the caller listens; the entry listens on an ephemeral port. */
export function createApp(): Server {
  const store = new Store(VIEWER_LOGIN);

  return createServer((req: IncomingMessage, res: ServerResponse) => {
    void (async () => {
      const { path, query } = urlOf(req);
      const method = req.method ?? 'GET';
      const headers = headersOf(req);

      try {
        // The env's boot check calls this twice: 200 {"ok":true}.
        if (method === 'GET' && path === '/healthz') {
          sendJson(res, 200, { ok: true });
          return;
        }

        // (The GitHub API root GET /api/v3 is answered by the REST layer
        // below, which also stamps the x-github-request-id every reply
        // carries.)

        // GraphQL v4: POST /api/graphql, JSON {query, variables, operationName}
        // (or an array of them for a batch). No token -> 401; bad JSON -> 400.
        if (method === 'POST' && path === '/api/graphql') {
          const auth = headers.authorization ?? '';
          const token = auth.startsWith('Bearer ') ? auth.slice('Bearer '.length).trim() : '';
          if (token === '') {
            sendJson(res, 401, { errors: [{ message: 'Requires authentication' }] });
            return;
          }
          const raw = await readBody(req);
          let parsed: unknown;
          try {
            parsed = JSON.parse(raw.toString('utf8'));
          } catch {
            sendJson(res, 400, { errors: [{ message: 'Problems parsing JSON' }] });
            return;
          }
          const requests = Array.isArray(parsed) ? parsed : [parsed];
          const answers = requests.map((item) => {
            if (typeof item !== 'object' || item === null || typeof (item as Record<string, unknown>).query !== 'string') {
              return { data: null, errors: [{ message: 'Invalid GraphQL request: query is required' }] };
            }
            const q = item as { query: string; variables?: Record<string, unknown>; operationName?: string };
            return executeGraphQL(store, { query: q.query, variables: q.variables, operationName: q.operationName });
          });
          sendJson(res, 200, Array.isArray(parsed) ? answers : answers[0]);
          return;
        }

        // MCP: POST /mcp (the handler owns auth, parse errors and JSON-RPC).
        if (method === 'POST' && path === '/mcp') {
          const raw = await readBody(req);
          const answer = handleMcp(store, { method, headers, bodyText: raw.toString('utf8') });
          if (typeof answer.body === 'string') {
            res.writeHead(answer.status, { 'content-type': 'application/json', ...answer.headers });
            res.end(answer.body);
          } else {
            sendJson(res, answer.status, answer.body, answer.headers);
          }
          return;
        }

        // Git smart HTTP: /<owner>/<repo>.git/info/refs, /git-upload-pack, /git-receive-pack.
        const git = gitPath(path);
        if (git !== null) {
          const repo = git.owner === store.owner ? store.getRepo(git.owner, git.repo) : undefined;
          if (repo === undefined) {
            sendJson(res, 404, { message: 'Not Found' });
            return;
          }
          if (method === 'GET' && git.rest.length === 2 && git.rest[0] === 'info' && git.rest[1] === 'refs') {
            const service = query.get('service');
            if (service !== 'git-upload-pack' && service !== 'git-receive-pack') {
              sendJson(res, 400, { message: 'service is required' });
              return;
            }
            // A client that asks for protocol v2 (Git-Protocol: version=2) gets
            // the v2 capability advertisement; anything else gets the v0 ref
            // advertisement. Answering v0 to a v2 request is a protocol error
            // on the client's side, so the header decides.
            const wantsV2 = service === 'git-upload-pack' && (headers['git-protocol'] ?? '') === 'version=2';
            const body = wantsV2 ? v2Advertisement() : v0Advertisement(repo, service);
            sendBytes(res, 200, body, `application/x-${service}-advertisement`);
            return;
          }
          if (method === 'POST' && git.rest.length === 1 && git.rest[0] === 'git-upload-pack') {
            const body = await readBody(req);
            // v2 requests start with a `command=` pkt-line; v0 starts with `want`.
            const { packets } = decodePktLines(body);
            const first = packets.find((p) => p.kind === 'data');
            const firstLine = first ? first.payload.toString('utf8').trim() : '';
            let out: Buffer;
            if (firstLine.startsWith('command=ls-refs')) out = handleLsRefs(repo, body);
            else if (firstLine.startsWith('command=fetch')) out = handleFetchV2(repo, body);
            else out = handleFetchV0(repo, body);
            sendBytes(res, 200, out, 'application/x-git-upload-pack-result');
            return;
          }
          if (method === 'POST' && git.rest.length === 1 && git.rest[0] === 'git-receive-pack') {
            const body = await readBody(req);
            const out = handleReceivePack(repo, body);
            sendBytes(res, 200, out, 'application/x-git-receive-pack-result');
            return;
          }
          sendJson(res, 404, { message: 'Not Found' });
          return;
        }

        // REST v3: everything under /api/v3.
        if (path === '/api/v3' || path.startsWith('/api/v3/')) {
          // A body can ride on POST, PUT, PATCH — and on DELETE (the
          // contents delete carries {message, sha}).
          const needsBody = method === 'POST' || method === 'PUT' || method === 'PATCH' || method === 'DELETE';
          let body: unknown = null;
          let bodyParseError = false;
          if (needsBody) {
            const raw = await readBody(req);
            if (raw.length > 0) {
              try {
                body = JSON.parse(raw.toString('utf8'));
              } catch {
                bodyParseError = true;
              }
            }
          }
          const restReq: RestRequest = {
            method,
            path,
            query,
            headers,
            body,
            bodyParseError,
            host: headers.host ?? 'entry',
          };
          const answer = handleRest(store, restReq);
          if (answer.body === null) {
            res.writeHead(answer.status, { ...answer.headers });
            res.end();
          } else {
            sendJson(res, answer.status, answer.body, answer.headers);
          }
          return;
        }

        // Everything else, every method: GitHub's 404 body.
        sendJson(res, 404, { message: 'Not Found' });
      } catch (err) {
        // A request must never take the server down (the suite probes
        // malformed bodies and missing repos, then keeps working).
        try {
          sendJson(res, 500, { message: err instanceof Error ? err.message : 'Internal Server Error' });
        } catch {
          // The response is already gone; nothing left to do.
        }
      }
    })();
  });
}
