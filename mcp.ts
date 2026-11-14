// Model Context Protocol endpoint (POST /mcp), stdlib only.
//
// The endpoint speaks the two protocol eras the suite's client accepts
// (see the kit's program/env/src/mcp-client.ts):
//
//   modern  2026-07-28: `server/discover` returns identity, capabilities and
//           `supportedVersions` in one request; no handshake, no session.
//   legacy  2025-11-25 and earlier: `initialize` + `notifications/initialized`,
//           an optional `Mcp-Session-Id` (accepted and echoed when sent).
//
// JSON-RPC 2.0 over HTTP: every answer carries the request's id, a result or
// an error, and Content-Type application/json. A body that is not JSON gets
// HTTP 400 with JSON-RPC -32700; an unknown method gets -32601; an unknown
// tool gets -32602; a tool call that fails (bad repo, missing arguments,
// merge conflict) answers isError:true in the result, never HTTP 500.
// A tools/call with no Authorization header answers HTTP 401.
//
// The six required tools (github-mcp-server's names) all work on the same
// store the REST and GraphQL surfaces use, so a repository or issue created
// over MCP is the same object over REST, with stable ids. A few extra tools
// (list_repos, get_issue, create_comment, list_pull_requests) round out the
// surface; every tool has a description and an object inputSchema, and the
// `required` list of each tool names only properties the suite can fill.

import { globalId } from './util.ts';
import { tick, type Issue, type PullRequest, type Repo, type Store } from './store.ts';
import { mergePullRequest } from './rest.ts';
import { refTip } from './treeops.ts';

export interface McpRequest {
  method: string;
  headers: Record<string, string>;
  /** The raw request body text (a body that is not JSON is a -32700). */
  bodyText: string;
}

export interface McpResponse {
  status: number;
  headers: Record<string, string>;
  /** JSON-serializable body, or a raw string when already serialized. */
  body: unknown;
}

const SERVER_NAME = 'github-core-clone-forge';
const SERVER_VERSION = '1.0.0';
const SUPPORTED_VERSIONS = ['2026-07-28', '2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05'];
const PREFERRED_VERSION = '2026-07-28';

const PARSE_ERROR = -32700;
const INVALID_REQUEST = -32600;
const METHOD_NOT_FOUND = -32601;
const INVALID_PARAMS = -32602;

function jsonResponse(status: number, body: unknown): McpResponse {
  return { status, headers: { 'content-type': 'application/json' }, body };
}

function rpcError(id: unknown, code: number, message: string, data?: unknown): McpResponse {
  const error: Record<string, unknown> = { code, message };
  if (data !== undefined) error.data = data;
  return jsonResponse(200, { jsonrpc: '2.0', id: id ?? null, error });
}

function rpcResult(id: unknown, result: unknown): McpResponse {
  return jsonResponse(200, { jsonrpc: '2.0', id: id ?? null, result });
}

/** Whether the request carries a non-empty Bearer token (the viewer). */
function hasAuth(headers: Record<string, string>): boolean {
  const auth = headers.authorization ?? headers.Authorization ?? '';
  return /^Bearer\s+\S+$/.test(auth);
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function asString(v: unknown): string | null {
  return typeof v === 'string' ? v : null;
}

function asNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isInteger(v) ? v : null;
}

/* ------------------------------------------------------------------ */
/* Tool definitions                                                    */
/* ------------------------------------------------------------------ */

interface ToolDef {
  name: string;
  description: string;
  inputSchema: {
    type: 'object';
    properties: Record<string, { type: string; description: string }>;
    required: string[];
    additionalProperties: boolean;
  };
}

function tool(
  name: string,
  description: string,
  properties: Record<string, { type: string; description: string }>,
  required: string[],
): ToolDef {
  return { name, description, inputSchema: { type: 'object', properties, required, additionalProperties: false } };
}

const TOOLS: ToolDef[] = [
  tool(
    'create_repo',
    'Create a repository owned by the authenticated user.',
    {
      name: { type: 'string', description: 'The repository name.' },
      owner: { type: 'string', description: 'The owner login (defaults to the authenticated user).' },
      description: { type: 'string', description: 'A short description of the repository.' },
      private: { type: 'boolean', description: 'Whether the repository is private.' },
    },
    ['name'],
  ),
  tool(
    'get_repo',
    'Read a repository by owner and name.',
    {
      owner: { type: 'string', description: 'The repository owner login.' },
      repo: { type: 'string', description: 'The repository name.' },
    },
    ['owner', 'repo'],
  ),
  tool(
    'create_issue',
    'Create an issue in a repository.',
    {
      owner: { type: 'string', description: 'The repository owner login.' },
      repo: { type: 'string', description: 'The repository name.' },
      title: { type: 'string', description: 'The issue title.' },
      body: { type: 'string', description: 'The issue body.' },
    },
    ['owner', 'repo', 'title'],
  ),
  tool(
    'list_issues',
    'List the issues of a repository, optionally filtered by state.',
    {
      owner: { type: 'string', description: 'The repository owner login.' },
      repo: { type: 'string', description: 'The repository name.' },
      state: { type: 'string', description: 'open, closed, or all (default open).' },
    },
    ['owner', 'repo'],
  ),
  tool(
    'create_pull_request',
    'Open a pull request from a head branch into a base branch.',
    {
      owner: { type: 'string', description: 'The repository owner login.' },
      repo: { type: 'string', description: 'The repository name.' },
      title: { type: 'string', description: 'The pull request title.' },
      body: { type: 'string', description: 'The pull request body.' },
      head: { type: 'string', description: 'The head branch name.' },
      base: { type: 'string', description: 'The base branch name.' },
    },
    ['owner', 'repo', 'title', 'head', 'base'],
  ),
  tool(
    'merge_pull_request',
    'Merge an open pull request onto its base branch.',
    {
      owner: { type: 'string', description: 'The repository owner login.' },
      repo: { type: 'string', description: 'The repository name.' },
      pull_number: { type: 'integer', description: 'The pull request number.' },
    },
    ['owner', 'repo', 'pull_number'],
  ),
  tool(
    'list_repos',
    'List the repositories owned by the authenticated user.',
    {
      owner: { type: 'string', description: 'The owner login (defaults to the authenticated user).' },
    },
    [],
  ),
  tool(
    'get_issue',
    'Read one issue by owner, repository and number.',
    {
      owner: { type: 'string', description: 'The repository owner login.' },
      repo: { type: 'string', description: 'The repository name.' },
      issue_number: { type: 'integer', description: 'The issue number.' },
    },
    ['owner', 'repo', 'issue_number'],
  ),
  tool(
    'create_comment',
    'Add a comment to an issue.',
    {
      owner: { type: 'string', description: 'The repository owner login.' },
      repo: { type: 'string', description: 'The repository name.' },
      issue_number: { type: 'integer', description: 'The issue number.' },
      body: { type: 'string', description: 'The comment body.' },
    },
    ['owner', 'repo', 'issue_number', 'body'],
  ),
  tool(
    'list_pull_requests',
    'List the pull requests of a repository, optionally filtered by state.',
    {
      owner: { type: 'string', description: 'The repository owner login.' },
      repo: { type: 'string', description: 'The repository name.' },
      state: { type: 'string', description: 'open, closed, or all (default open).' },
    },
    ['owner', 'repo'],
  ),
];

/* ------------------------------------------------------------------ */
/* Shared JSON views (same shapes as the REST surface)                 */
/* ------------------------------------------------------------------ */

function repoJson(store: Store, repo: Repo): Record<string, unknown> {
  return {
    id: repo.id,
    node_id: globalId('Repository', repo.id),
    name: repo.name,
    full_name: `${repo.owner}/${repo.name}`,
    owner: { login: repo.owner, id: 1000, type: 'User' },
    private: repo.private,
    description: repo.description,
    default_branch: repo.defaultBranch,
    created_at: repo.createdAt,
    updated_at: repo.updatedAt,
    html_url: `/api/v3/repos/${repo.owner}/${repo.name}`,
    url: `/api/v3/repos/${repo.owner}/${repo.name}`,
  };
}

function issueJson(store: Store, repo: Repo, issue: Issue): Record<string, unknown> {
  return {
    id: issue.id,
    node_id: globalId('Issue', issue.id),
    number: issue.number,
    title: issue.title,
    body: issue.body,
    state: issue.state,
    user: { login: issue.user, id: 1000, type: 'User' },
    labels: issue.labels.map((name) => ({ name })),
    created_at: issue.createdAt,
    updated_at: issue.updatedAt,
    closed_at: issue.closedAt,
    html_url: `/api/v3/repos/${repo.owner}/${repo.name}/issues/${issue.number}`,
    url: `/api/v3/repos/${repo.owner}/${repo.name}/issues/${issue.number}`,
  };
}

function prJson(store: Store, repo: Repo, pr: PullRequest): Record<string, unknown> {
  return {
    id: pr.id,
    node_id: globalId('PullRequest', pr.id),
    number: pr.number,
    pull_number: pr.number,
    title: pr.title,
    body: pr.body,
    state: pr.state,
    merged: pr.merged,
    merge_commit_sha: pr.mergeCommitSha,
    head: { ref: pr.headRef, sha: pr.headSha },
    base: { ref: pr.baseRef, sha: pr.baseSha },
    user: { login: store.owner, id: 1000, type: 'User' },
    created_at: pr.createdAt,
    updated_at: pr.updatedAt,
    closed_at: pr.closedAt,
    merged_at: pr.mergedAt,
    html_url: `/api/v3/repos/${repo.owner}/${repo.name}/pulls/${pr.number}`,
    url: `/api/v3/repos/${repo.owner}/${repo.name}/pulls/${pr.number}`,
  };
}

/* ------------------------------------------------------------------ */
/* Tool implementations                                                */
/* ------------------------------------------------------------------ */

/** A tool result: `{content, structuredContent, isError}`. */
interface ToolResult {
  content: Array<{ type: 'text'; text: string }>;
  structuredContent: unknown;
  isError: boolean;
}

function okResult(data: unknown): ToolResult {
  return { content: [{ type: 'text', text: JSON.stringify(data) }], structuredContent: data, isError: false };
}

function errResult(message: string): ToolResult {
  return { content: [{ type: 'text', text: message }], structuredContent: { error: message }, isError: true };
}

/** The repo a tool call names, or an error result when it does not exist. */
function lookupRepo(store: Store, args: Record<string, unknown>): { repo: Repo } | { error: ToolResult } {
  const owner = asString(args.owner);
  const repoName = asString(args.repo);
  if (owner === null || repoName === null) return { error: errResult('owner and repo are required') };
  if (owner !== store.owner) return { error: errResult(`Repository not found: ${owner}/${repoName}`) };
  const repo = store.getRepo(owner, repoName);
  if (!repo) return { error: errResult(`Repository not found: ${owner}/${repoName}`) };
  return { repo };
}

function isValidRepoName(name: string): boolean {
  return name.length > 0 && name.length <= 100 && /^[A-Za-z0-9._-]+$/.test(name) && !name.includes('..');
}

function callTool(store: Store, name: string, args: Record<string, unknown>): ToolResult {
  switch (name) {
    case 'create_repo': {
      const repoName = asString(args.name);
      if (repoName === null || repoName === '') return errResult('name is required');
      if (!isValidRepoName(repoName)) return errResult(`Invalid repository name: ${repoName}`);
      if (store.getRepo(store.owner, repoName)) return errResult(`Repository already exists: ${store.owner}/${repoName}`);
      const repo = store.createRepo(repoName, {
        description: asString(args.description) ?? '',
        private: args.private === true,
      });
      return okResult(repoJson(store, repo));
    }
    case 'get_repo': {
      const found = lookupRepo(store, args);
      if ('error' in found) return found.error;
      return okResult(repoJson(store, found.repo));
    }
    case 'create_issue': {
      const found = lookupRepo(store, args);
      if ('error' in found) return found.error;
      const title = asString(args.title);
      if (title === null || title === '') return errResult('title is required');
      const issue = store.createIssue(found.repo, title, asString(args.body) ?? '', []);
      return okResult(issueJson(store, found.repo, issue));
    }
    case 'list_issues': {
      const found = lookupRepo(store, args);
      if ('error' in found) return found.error;
      const state = asString(args.state) ?? 'open';
      let issues = [...found.repo.issues];
      if (state !== 'all') issues = issues.filter((i) => i.state === state);
      issues.sort((a, b) => b.number - a.number);
      return okResult(issues.map((i) => issueJson(store, found.repo, i)));
    }
    case 'create_pull_request': {
      const found = lookupRepo(store, args);
      if ('error' in found) return found.error;
      const title = asString(args.title);
      const head = asString(args.head);
      const base = asString(args.base);
      if (title === null || title === '') return errResult('title is required');
      if (head === null || base === null) return errResult('head and base are required');
      const headSha = refTip(found.repo, `refs/heads/${head}`);
      const baseSha = refTip(found.repo, `refs/heads/${base}`);
      if (!headSha || !baseSha) return errResult(`head or base branch not found: ${head} -> ${base}`);
      const pr = store.createPull(found.repo, title, asString(args.body) ?? '', head, base, headSha, baseSha);
      return okResult(prJson(store, found.repo, pr));
    }
    case 'merge_pull_request': {
      const found = lookupRepo(store, args);
      if ('error' in found) return found.error;
      const number = asNumber(args.pull_number);
      if (number === null) return errResult('pull_number is required');
      const pr = found.repo.pulls.find((p) => p.number === number);
      if (!pr) return errResult(`Pull request not found: #${number}`);
      const result = mergePullRequest(store, found.repo, pr);
      if (!result.ok) return errResult(result.reason);
      return okResult({ ...prJson(store, found.repo, pr), merged: true, sha: result.sha });
    }
    case 'list_repos': {
      const owner = asString(args.owner);
      if (owner !== null && owner !== store.owner) return okResult([]);
      return okResult(store.repoList.map((r) => repoJson(store, r)));
    }
    case 'get_issue': {
      const found = lookupRepo(store, args);
      if ('error' in found) return found.error;
      const number = asNumber(args.issue_number);
      if (number === null) return errResult('issue_number is required');
      const issue = found.repo.issues.find((i) => i.number === number);
      if (!issue) return errResult(`Issue not found: #${number}`);
      return okResult(issueJson(store, found.repo, issue));
    }
    case 'create_comment': {
      const found = lookupRepo(store, args);
      if ('error' in found) return found.error;
      const number = asNumber(args.issue_number);
      const body = asString(args.body);
      if (number === null) return errResult('issue_number is required');
      if (body === null || body === '') return errResult('body is required');
      const issue = found.repo.issues.find((i) => i.number === number);
      if (!issue) return errResult(`Issue not found: #${number}`);
      const comment = {
        id: store.nextId(),
        body,
        user: store.owner,
        createdAt: tick(),
        updatedAt: tick(),
      };
      issue.comments.push(comment);
      issue.updatedAt = tick();
      return okResult({
        id: comment.id,
        node_id: globalId('IssueComment', comment.id),
        body: comment.body,
        user: { login: comment.user, id: 1000, type: 'User' },
        created_at: comment.createdAt,
        updated_at: comment.updatedAt,
      });
    }
    case 'list_pull_requests': {
      const found = lookupRepo(store, args);
      if ('error' in found) return found.error;
      const state = asString(args.state) ?? 'open';
      let pulls = [...found.repo.pulls];
      if (state !== 'all') pulls = pulls.filter((p) => p.state === state);
      pulls.sort((a, b) => b.number - a.number);
      return okResult(pulls.map((p) => prJson(store, found.repo, p)));
    }
    default:
      return errResult(`Unknown tool: ${name}`);
  }
}

/* ------------------------------------------------------------------ */
/* JSON-RPC dispatch                                                   */
/* ------------------------------------------------------------------ */

export function handleMcp(store: Store, req: McpRequest): McpResponse {
  let parsed: unknown;
  try {
    parsed = JSON.parse(req.bodyText);
  } catch {
    return jsonResponse(400, { jsonrpc: '2.0', id: null, error: { code: PARSE_ERROR, message: 'Parse error' } });
  }

  // A batch of requests: answer each, in order.
  if (Array.isArray(parsed)) {
    const answers: unknown[] = [];
    for (const item of parsed) {
      const r = dispatch(store, req, item);
      if (r !== null) answers.push(r.body);
    }
    return jsonResponse(200, answers);
  }
  const r = dispatch(store, req, parsed);
  return r ?? { status: 202, headers: {}, body: '' };
}

/** Dispatch one JSON-RPC message; null when it is a notification (no id). */
function dispatch(store: Store, req: McpRequest, message: unknown): McpResponse | null {
  if (!isRecord(message)) {
    return jsonResponse(200, { jsonrpc: '2.0', id: null, error: { code: INVALID_REQUEST, message: 'Invalid Request' } });
  }
  const id = message.id;
  const method = asString(message.method);
  const params = isRecord(message.params) ? message.params : {};
  const isNotification = id === undefined;

  if (method === null) {
    return isNotification ? null : rpcError(id, INVALID_REQUEST, 'Invalid Request: method is required');
  }

  // Notifications (no id) never get a body back.
  if (isNotification) return null;

  // Authentication: a non-empty Bearer token is the viewer. A tools/call
  // without one is HTTP 401; every other method needs one too.
  if (!hasAuth(req.headers)) {
    return jsonResponse(401, {
      jsonrpc: '2.0',
      id: id ?? null,
      error: { code: -32001, message: 'Requires authentication' },
    });
  }

  if (method === 'server/discover') {
    return rpcResult(id, {
      resultType: 'DiscoverResult',
      supportedVersions: SUPPORTED_VERSIONS,
      capabilities: { tools: {} },
      instructions: 'A GitHub-compatible git forge: smart-HTTP git, REST v3, GraphQL v4, and MCP tools.',
      serverInfo: { name: SERVER_NAME, version: SERVER_VERSION },
      _meta: {
        'io.modelcontextprotocol/serverInfo': { name: SERVER_NAME, version: SERVER_VERSION },
        'io.modelcontextprotocol/protocolVersion': PREFERRED_VERSION,
      },
    });
  }

  if (method === 'initialize') {
    const offered = asString(params.protocolVersion);
    const negotiated = offered !== null && SUPPORTED_VERSIONS.includes(offered) ? offered : PREFERRED_VERSION;
    return rpcResult(id, {
      protocolVersion: negotiated,
      capabilities: { tools: {} },
      serverInfo: { name: SERVER_NAME, version: SERVER_VERSION },
    });
  }

  if (method === 'ping') {
    return rpcResult(id, {});
  }

  if (method === 'tools/list') {
    return rpcResult(id, { tools: TOOLS });
  }

  if (method === 'tools/call') {
    const name = asString(params.name);
    if (name === null || name === '') {
      return rpcError(id, INVALID_PARAMS, 'tools/call requires a tool name');
    }
    const toolDef = TOOLS.find((t) => t.name === name);
    if (!toolDef) {
      return rpcError(id, INVALID_PARAMS, `Unknown tool: ${name}`);
    }
    const args = isRecord(params.arguments) ? params.arguments : {};
    const missing = toolDef.inputSchema.required.filter((key) => !(key in args) || args[key] === undefined || args[key] === null || args[key] === '');
    if (missing.length > 0) {
      return rpcResult(id, {
        content: [{ type: 'text', text: `Missing required argument(s): ${missing.join(', ')}` }],
        structuredContent: { error: `Missing required argument(s): ${missing.join(', ')}` },
        isError: true,
      });
    }
    const result = callTool(store, name, args);
    return rpcResult(id, result);
  }

  return rpcError(id, METHOD_NOT_FOUND, `Method not found: ${method}`);
}
