// GitHub REST v3 surface (/api/v3), stdlib only.
//
// The endpoint contract is the one the suite's checks require (see the
// kit's program/env/src/checks-rest.ts header): repos, contents (backed by
// real git objects, so a clone sees what the contents API committed),
// branches, commits, git refs, labels, issues + comments, and pull requests
// with a three-way squash merge. One bearer token is the viewer; a repo
// belongs to the login the token names, so the same name under another login
// is 404 (never 403), and a request with no token is 401. Every reply is
// application/json (never HTML), an unknown route is a JSON 404, a malformed
// JSON body is a 400, and every reply carries X-GitHub-Request-Id.

import { globalId, gitObjectId, isSha1, ZERO_OID } from './util.ts';
import { tick, type Issue, type Label, type PullRequest, type Repo, type Store } from './store.ts';
import { makeCommit, parseCommit, parseTree } from './gitobj.ts';
import {
  commitTree,
  deleteFile,
  fileOid,
  listTree,
  lookupPath,
  mergeTrees,
  readFile,
  refTip,
  setFile,
  storeBlob,
  storeCommit,
  storeTree,
} from './treeops.ts';

export interface RestRequest {
  method: string;
  path: string;
  query: URLSearchParams;
  headers: Record<string, string>;
  /** Parsed JSON body, or null when there is none. */
  body: unknown;
  /** True when a body was present but did not parse as JSON. */
  bodyParseError: boolean;
  /** The Host header, for building absolute pagination URLs. */
  host: string;
}

export interface RestResponse {
  status: number;
  headers: Record<string, string>;
  body: unknown;
}

let requestCounter = 0;
function nextRequestId(): string {
  requestCounter += 1;
  return `forge-${requestCounter.toString(36)}-${Date.now().toString(36)}`;
}

function json(status: number, body: unknown, extra: Record<string, string> = {}): RestResponse {
  return { status, headers: { 'x-github-request-id': nextRequestId(), ...extra }, body };
}

function err(status: number, message: string): RestResponse {
  return json(status, { message });
}

function noContent(): RestResponse {
  return { status: 204, headers: { 'x-github-request-id': nextRequestId() }, body: null };
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function asString(v: unknown, fallback = ''): string {
  return typeof v === 'string' ? v : fallback;
}

function asBool(v: unknown, fallback = false): boolean {
  return typeof v === 'boolean' ? v : fallback;
}

/** Parse a page/per_page query pair (GitHub defaults: page 1, per_page 30, max 100). */
function pagination(query: URLSearchParams): { page: number; perPage: number } {
  const page = Math.max(1, Number.parseInt(query.get('page') ?? '1', 10) || 1);
  let perPage = Number.parseInt(query.get('per_page') ?? '30', 10) || 30;
  if (perPage < 1) perPage = 1;
  if (perPage > 100) perPage = 100;
  return { page, perPage };
}

/** Slice a list into one page and build the Link header (next/prev/first/last). */
function paginate<T>(items: T[], query: URLSearchParams, basePath: string, host: string): { page: T[]; link: string } {
  const { page, perPage } = pagination(query);
  const totalPages = Math.max(1, Math.ceil(items.length / perPage));
  const start = (page - 1) * perPage;
  const pageItems = items.slice(start, start + perPage);
  const url = (p: number): string => {
    const q = new URLSearchParams(query);
    q.set('page', String(p));
    q.set('per_page', String(perPage));
    return `http://${host}${basePath}?${q.toString()}`;
  };
  const links: string[] = [];
  if (page < totalPages) links.push(`<${url(page + 1)}>; rel="next"`);
  if (page > 1) links.push(`<${url(page - 1)}>; rel="prev"`);
  links.push(`<${url(1)}>; rel="first"`);
  links.push(`<${url(totalPages)}>; rel="last"`);
  return { page: pageItems, link: links.join(', ') };
}

/* ------------------------------------------------------------------ */
/* JSON shapes (GitHub-shaped, with the fields the checks require)     */
/* ------------------------------------------------------------------ */

function userJson(store: Store, login: string): Record<string, unknown> {
  const id = login === store.owner ? 1000 : 1002;
  return {
    login,
    id,
    node_id: globalId('User', id),
    avatar_url: `https://avatars.githubusercontent.com/u/${id}?v=4`,
    gravatar_id: '',
    url: `/api/v3/users/${login}`,
    html_url: `https://github.com/${login}`,
    type: 'User',
    site_admin: false,
  };
}

function repoJson(store: Store, repo: Repo): Record<string, unknown> {
  return {
    id: repo.id,
    node_id: globalId('Repository', repo.id),
    name: repo.name,
    full_name: `${repo.owner}/${repo.name}`,
    owner: userJson(store, repo.owner),
    private: repo.private,
    html_url: `https://github.com/${repo.owner}/${repo.name}`,
    description: repo.description,
    fork: false,
    url: `/api/v3/repos/${repo.owner}/${repo.name}`,
    created_at: repo.createdAt,
    updated_at: repo.updatedAt,
    pushed_at: repo.updatedAt,
    homepage: null,
    size: 0,
    stargazers_count: 0,
    watchers_count: 0,
    language: null,
    has_issues: true,
    has_projects: true,
    has_downloads: true,
    has_wiki: true,
    has_pages: false,
    forks_count: 0,
    open_issues_count: repo.issues.filter((i) => i.state === 'open').length,
    license: null,
    allow_forking: true,
    is_template: false,
    topics: [],
    visibility: repo.private ? 'private' : 'public',
    default_branch: repo.defaultBranch,
    permissions: { admin: true, maintain: true, push: true, triage: true, pull: true },
  };
}

function labelJson(label: Label): Record<string, unknown> {
  return {
    id: label.id,
    node_id: globalId('Label', label.id),
    url: '',
    name: label.name,
    color: label.color,
    description: label.description,
    default: false,
  };
}

function issueJson(store: Store, repo: Repo, issue: Issue): Record<string, unknown> {
  const labels = issue.labels
    .map((name) => repo.labels.find((l) => l.name === name))
    .filter((l): l is Label => l !== undefined)
    .map(labelJson);
  return {
    id: issue.id,
    node_id: globalId('Issue', issue.id),
    url: `/api/v3/repos/${repo.owner}/${repo.name}/issues/${issue.number}`,
    repository_url: `/api/v3/repos/${repo.owner}/${repo.name}`,
    html_url: `https://github.com/${repo.owner}/${repo.name}/issues/${issue.number}`,
    number: issue.number,
    title: issue.title,
    body: issue.body,
    user: userJson(store, issue.user),
    labels,
    state: issue.state,
    locked: false,
    assignee: null,
    assignees: [],
    comments: issue.comments.length,
    created_at: issue.createdAt,
    updated_at: issue.updatedAt,
    closed_at: issue.closedAt,
    author_association: 'OWNER',
  };
}

function pullJson(store: Store, repo: Repo, pr: PullRequest): Record<string, unknown> {
  return {
    id: pr.id,
    node_id: globalId('PullRequest', pr.id),
    url: `/api/v3/repos/${repo.owner}/${repo.name}/pulls/${pr.number}`,
    html_url: `https://github.com/${repo.owner}/${repo.name}/pull/${pr.number}`,
    number: pr.number,
    title: pr.title,
    body: pr.body,
    state: pr.state,
    locked: false,
    user: userJson(store, store.owner),
    head: { label: `${store.owner}:${pr.headRef}`, ref: pr.headRef, sha: pr.headSha, user: userJson(store, store.owner), repo: repoJson(store, repo) },
    base: { label: `${store.owner}:${pr.baseRef}`, ref: pr.baseRef, sha: pr.baseSha, user: userJson(store, store.owner), repo: repoJson(store, repo) },
    merged: pr.merged,
    mergeable: pr.state === 'open' ? true : null,
    merged_at: pr.mergedAt,
    merge_commit_sha: pr.mergeCommitSha,
    created_at: pr.createdAt,
    updated_at: pr.updatedAt,
    closed_at: pr.closedAt,
    author_association: 'OWNER',
  };
}

function commitJson(store: Store, repo: Repo, sha: string): Record<string, unknown> | null {
  const obj = repo.objects.get(sha);
  if (!obj || obj.type !== 'commit') return null;
  const c = parseCommit(obj.data);
  const ident = (line: string): { name: string; email: string; date: string } => {
    const m = /^(.*) <(.*)> (\d+) ([+-]\d+)$/.exec(line);
    if (!m) return { name: line, email: '', date: '' };
    return { name: m[1], email: m[2], date: new Date(Number(m[3]) * 1000).toISOString() };
  };
  return {
    sha,
    node_id: globalId('Commit', sha),
    url: `/api/v3/repos/${repo.owner}/${repo.name}/commits/${sha}`,
    html_url: `https://github.com/${repo.owner}/${repo.name}/commit/${sha}`,
    commit: {
      message: c.message,
      author: ident(c.author),
      committer: ident(c.committer),
      tree: { sha: c.tree },
    },
    author: userJson(store, store.owner),
    committer: userJson(store, store.owner),
    parents: c.parents.map((p) => ({ sha: p })),
  };
}

/* ------------------------------------------------------------------ */
/* Git helpers for the contents API                                    */
/* ------------------------------------------------------------------ */

/** Resolve a ref query (?ref=) to a commit oid: branch, full ref, sha, HEAD. */
function resolveRef(repo: Repo, ref: string | null): string {
  if (!ref || ref === 'HEAD') return refTip(repo, `refs/heads/${repo.defaultBranch}`);
  if (repo.refs.has(ref)) return repo.refs.get(ref)!;
  const branch = `refs/heads/${ref}`;
  if (repo.refs.has(branch)) return repo.refs.get(branch)!;
  if (isSha1(ref) && repo.objects.has(ref)) return ref;
  return '';
}

/** The identity lines for contents/merge commits (the viewer, one deterministic timestamp). */
function identLines(store: Store): { author: string; committer: string } {
  const ts = Math.floor(Date.parse(tick()) / 1000);
  const line = `${store.owner} <${store.owner}@example.invalid> ${ts} +0000`;
  return { author: line, committer: line };
}

/**
 * Commit a file change on `branch` (creating the branch's first commit when
 * the repo is unborn). Returns the commit sha and the blob sha.
 */
function commitFileChange(
  store: Store,
  repo: Repo,
  branch: string,
  path: string,
  content: Buffer,
  message: string,
  isDelete: boolean,
): { commitSha: string; blobSha: string } {
  const ref = `refs/heads/${branch}`;
  const tip = refTip(repo, ref);
  const parentTree = tip ? commitTree(repo, tip) : '';
  const blobSha = isDelete ? '' : storeBlob(repo, content);
  const newTree = isDelete
    ? deleteFile(repo, parentTree, path) ?? storeTree(repo, [])
    : setFile(repo, parentTree === '' ? null : parentTree, path, blobSha);
  const ident = identLines(store);
  const commitSha = storeCommit(repo, {
    tree: newTree,
    parents: tip ? [tip] : [],
    message,
    author: ident.author,
    committer: ident.committer,
  });
  repo.refs.set(ref, commitSha);
  repo.updatedAt = tick();
  return { commitSha, blobSha };
}

/** A simple line diff (LCS) returning added/removed line counts. */
function diffLineCounts(a: string, b: string): { additions: number; deletions: number } {
  const aLines = a === '' ? [] : a.split('\n');
  const bLines = b === '' ? [] : b.split('\n');
  // Strip a trailing empty line produced by a final newline.
  if (aLines.length > 0 && aLines[aLines.length - 1] === '') aLines.pop();
  if (bLines.length > 0 && bLines[bLines.length - 1] === '') bLines.pop();
  const n = aLines.length;
  const m = bLines.length;
  if (n === 0) return { additions: m, deletions: 0 };
  if (m === 0) return { additions: 0, deletions: n };
  // LCS lengths, two rows.
  let prev = new Array<number>(m + 1).fill(0);
  let cur = new Array<number>(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      cur[j] = aLines[i - 1] === bLines[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
    }
    [prev, cur] = [cur, prev];
  }
  const lcs = prev[m];
  return { additions: m - lcs, deletions: n - lcs };
}

/** Merge a pull request (squash-style single-parent commit onto the base ref).
 *  Returns the merge commit sha, or an error reason when it cannot merge.
 *  Shared by the REST, GraphQL and MCP surfaces so all three merge identically. */
export function mergePullRequest(
  store: Store,
  repo: Repo,
  pr: PullRequest,
): { ok: true; sha: string } | { ok: false; reason: string } {
  if (pr.state !== 'open') return { ok: false, reason: 'Pull request is already closed' };
  const baseTip = refTip(repo, `refs/heads/${pr.baseRef}`);
  const headTip = refTip(repo, `refs/heads/${pr.headRef}`);
  if (!baseTip || !headTip) return { ok: false, reason: 'Not Found' };
  const base = mergeBase(repo, baseTip, headTip);
  if (!base) return { ok: false, reason: 'No merge base found' };
  const { tree, conflicts } = mergeTrees(repo, commitTree(repo, base), commitTree(repo, baseTip), commitTree(repo, headTip));
  if (conflicts.length > 0) return { ok: false, reason: `Merge conflict in ${conflicts.join(', ')}` };
  const message = `Merge pull request #${pr.number} from ${store.owner}/${pr.headRef}`;
  const ident = identLines(store);
  const mergeSha = storeCommit(repo, {
    tree,
    parents: [baseTip],
    message,
    author: ident.author,
    committer: ident.committer,
  });
  repo.refs.set(`refs/heads/${pr.baseRef}`, mergeSha);
  repo.updatedAt = tick();
  pr.state = 'closed';
  pr.merged = true;
  pr.mergeCommitSha = mergeSha;
  pr.mergedAt = tick();
  pr.closedAt = pr.mergedAt;
  pr.updatedAt = tick();
  return { ok: true, sha: mergeSha };
}

/** The merge base of two commits: the first common ancestor (BFS from both). */
function mergeBase(repo: Repo, a: string, b: string): string {
  if (a === b) return a;
  const seenA = new Map<string, number>();
  const queueA: Array<{ oid: string; d: number }> = [{ oid: a, d: 0 }];
  seenA.set(a, 0);
  while (queueA.length > 0) {
    const { oid, d } = queueA.shift()!;
    const obj = repo.objects.get(oid);
    if (!obj || obj.type !== 'commit') continue;
    for (const p of parseCommit(obj.data).parents) {
      if (!seenA.has(p)) {
        seenA.set(p, d + 1);
        queueA.push({ oid: p, d: d + 1 });
      }
    }
  }
  const queueB: Array<{ oid: string; d: number }> = [{ oid: b, d: 0 }];
  const seenB = new Set<string>([b]);
  let best = '';
  let bestD = Number.POSITIVE_INFINITY;
  while (queueB.length > 0) {
    const { oid, d } = queueB.shift()!;
    if (seenA.has(oid) && seenA.get(oid)! + d < bestD) {
      best = oid;
      bestD = seenA.get(oid)! + d;
    }
    const obj = repo.objects.get(oid);
    if (!obj || obj.type !== 'commit') continue;
    for (const p of parseCommit(obj.data).parents) {
      if (!seenB.has(p)) {
        seenB.add(p);
        queueB.push({ oid: p, d: d + 1 });
      }
    }
  }
  return best;
}

/** All commits reachable from `tip`, newest first (BFS by generation). */
function listCommits(repo: Repo, tip: string): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  const queue = [tip];
  seen.add(tip);
  while (queue.length > 0) {
    const oid = queue.shift()!;
    const obj = repo.objects.get(oid);
    if (!obj || obj.type !== 'commit') continue;
    out.push(oid);
    for (const p of parseCommit(obj.data).parents) {
      if (!seen.has(p)) {
        seen.add(p);
        queue.push(p);
      }
    }
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Routing                                                             */
/* ------------------------------------------------------------------ */

export function handleRest(store: Store, req: RestRequest): RestResponse {
  const { method, path, query } = req;

  // Public roots (no token needed, as api.github.com answers them).
  if ((path === '/api/v3' || path === '/api/v3/') && method === 'GET') {
    return json(200, { current_user_url: '/api/v3/user', rate_limit_url: '/api/v3/rate_limit' }, { 'x-oauth-scopes': 'repo, read:org' });
  }
  if (path === '/api/v3/meta' && method === 'GET') {
    return json(200, { installed_version: '1.0.0', github_services_sha: '0000000000000000000000000000000000000000' });
  }

  // Everything else needs a bearer token.
  const auth = req.headers.authorization ?? '';
  const token = auth.startsWith('Bearer ') ? auth.slice('Bearer '.length).trim() : '';
  if (token === '') {
    return err(401, 'Requires authentication');
  }
  if (req.bodyParseError) {
    return err(400, 'Problems parsing JSON');
  }

  const segments = path.split('/').filter((s) => s.length > 0);
  // segments[0..1] = api, v3
  if (segments[0] !== 'api' || segments[1] !== 'v3') return err(404, 'Not Found');
  const rest = segments.slice(2);

  // GET /user
  if (rest.length === 1 && rest[0] === 'user' && method === 'GET') {
    return json(200, userJson(store, store.owner));
  }

  // POST /user/repos
  if (rest.length === 2 && rest[0] === 'user' && rest[1] === 'repos' && method === 'POST') {
    if (!isObject(req.body)) return err(422, 'Validation Failed: name is required');
    const name = asString(req.body.name);
    if (!name) return err(422, 'Validation Failed: name is required');
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$/.test(name)) return err(422, 'Validation Failed: name is invalid');
    if (store.getRepo(store.owner, name)) return err(422, 'Validation Failed: name already exists on this account');
    const repo = store.createRepo(name, {
      description: asString(req.body.description),
      private: asBool(req.body.private, false),
      defaultBranch: asString(req.body.default_branch, 'main'),
    });
    return json(201, repoJson(store, repo));
  }

  // GET /user/repos (paginated)
  if (rest.length === 2 && rest[0] === 'user' && rest[1] === 'repos' && method === 'GET') {
    const repos = store.reposOf(store.owner);
    const { page, link } = paginate(repos.map((r) => repoJson(store, r)), query, '/api/v3/user/repos', req.host);
    return json(200, page, { link });
  }

  // /repos/{owner}/{name}/...
  if (rest.length >= 3 && rest[0] === 'repos') {
    const owner = rest[1];
    const name = rest[2];
    const sub = rest.slice(3);
    // Ownership: only the viewer's repos are visible; any other owner is 404.
    if (owner !== store.owner) return err(404, 'Not Found');
    const repo = store.getRepo(owner, name);
    if (!repo && !(method === 'DELETE' && sub.length === 0)) {
      // DELETE of a missing repo is also 404 (the bench tolerates it).
      return err(404, 'Not Found');
    }
    if (!repo) return err(404, 'Not Found');

    // GET /repos/{o}/{r}
    if (sub.length === 0 && method === 'GET') return json(200, repoJson(store, repo));
    // PATCH /repos/{o}/{r}
    if (sub.length === 0 && method === 'PATCH') {
      if (isObject(req.body)) {
        if (typeof req.body.description === 'string') repo.description = req.body.description;
        if (typeof req.body.private === 'boolean') repo.private = req.body.private;
        if (typeof req.body.default_branch === 'string' && req.body.default_branch) repo.defaultBranch = req.body.default_branch;
        repo.updatedAt = tick();
      }
      return json(200, repoJson(store, repo));
    }
    // DELETE /repos/{o}/{r}
    if (sub.length === 0 && method === 'DELETE') {
      store.deleteRepo(owner, name);
      return noContent();
    }

    // GET /repos/{o}/{r}/branches
    if (sub.length === 1 && sub[0] === 'branches' && method === 'GET') {
      const branches = [...repo.refs.keys()]
        .filter((r) => r.startsWith('refs/heads/'))
        .map((r) => r.slice('refs/heads/'.length))
        .sort()
        .map((branch) => ({
          name: branch,
          commit: { sha: repo.refs.get(`refs/heads/${branch}`)!, url: `/api/v3/repos/${owner}/${name}/commits/${repo.refs.get(`refs/heads/${branch}`)!}` },
          protected: false,
        }));
      return json(200, branches);
    }

    // GET /repos/{o}/{r}/commits
    if (sub.length === 1 && sub[0] === 'commits' && method === 'GET') {
      const ref = query.get('sha');
      const tip = resolveRef(repo, ref);
      if (!tip) return err(404, 'Not Found');
      const commits = listCommits(repo, tip)
        .map((sha) => commitJson(store, repo, sha))
        .filter((c): c is Record<string, unknown> => c !== null);
      const { page } = paginate(commits, query, `/api/v3/repos/${owner}/${name}/commits`, req.host);
      return json(200, page);
    }

    // POST /repos/{o}/{r}/git/refs
    if (sub.length === 2 && sub[0] === 'git' && sub[1] === 'refs' && method === 'POST') {
      if (!isObject(req.body)) return err(422, 'Validation Failed');
      const ref = asString(req.body.ref);
      const sha = asString(req.body.sha);
      if (!ref.startsWith('refs/') || !isSha1(sha)) return err(422, 'Validation Failed: ref and sha are required');
      if (repo.refs.has(ref)) return err(422, 'Validation Failed: Reference already exists');
      const obj = repo.objects.get(sha);
      if (!obj || (ref.startsWith('refs/heads/') && obj.type !== 'commit')) return err(422, 'Validation Failed: Invalid SHA');
      repo.refs.set(ref, sha);
      return json(201, { ref, node_id: globalId('GitRef', sha), url: `/api/v3/repos/${owner}/${name}/git/refs/${encodeURIComponent(ref)}`, object: { sha, type: obj.type, url: '' } });
    }

    // GET /repos/{o}/{r}/git/refs/{ref...}
    if (sub.length >= 3 && sub[0] === 'git' && sub[1] === 'refs' && method === 'GET') {
      const ref = sub.slice(2).join('/');
      const oid = repo.refs.get(ref);
      if (!oid) return err(404, 'Not Found');
      const obj = repo.objects.get(oid);
      return json(200, { ref, node_id: globalId('GitRef', oid), url: '', object: { sha: oid, type: obj?.type ?? 'commit', url: '' } });
    }

    // /repos/{o}/{r}/contents/{path...}
    if (sub.length >= 2 && sub[0] === 'contents') {
      const filePath = sub.slice(1).map(decodeURIComponent).join('/');
      return handleContents(store, repo, filePath, req);
    }

    // /repos/{o}/{r}/labels
    if (sub.length === 1 && sub[0] === 'labels') {
      if (method === 'GET') return json(200, repo.labels.map(labelJson));
      if (method === 'POST') {
        if (!isObject(req.body) || !asString(req.body.name)) return err(422, 'Validation Failed: name is required');
        const name = asString(req.body.name);
        if (repo.labels.some((l) => l.name === name)) return err(422, 'Validation Failed: label already exists');
        const label = store.createLabel(repo, name, asString(req.body.color, 'ededed'), asString(req.body.description));
        return json(201, labelJson(label));
      }
    }

    // /repos/{o}/{r}/issues...
    if (sub.length >= 1 && sub[0] === 'issues') {
      return handleIssues(store, repo, sub.slice(1), req);
    }

    // /repos/{o}/{r}/pulls...
    if (sub.length >= 1 && sub[0] === 'pulls') {
      return handlePulls(store, repo, sub.slice(1), req);
    }
  }

  return err(404, 'Not Found');
}

/* ------------------------------------------------------------------ */
/* Contents                                                            */
/* ------------------------------------------------------------------ */

function handleContents(store: Store, repo: Repo, filePath: string, req: RestRequest): RestResponse {
  const { method, query } = req;
  const owner = repo.owner;
  const name = repo.name;

  if (method === 'GET') {
    const tip = resolveRef(repo, query.get('ref'));
    if (!tip) return err(404, 'Not Found');
    const tree = commitTree(repo, tip);
    if (!tree) return err(404, 'Not Found');
    const found = filePath === '' ? { type: 'tree' as const, oid: tree, mode: '40000' } : lookupPath(repo, tree, filePath);
    if (!found) return err(404, 'Not Found');
    if (found.type === 'tree') {
      const entries = listTree(repo, found.oid).map((e) => {
        const isDir = e.mode === '40000';
        const obj = repo.objects.get(e.oid);
        return {
          type: isDir ? 'dir' : 'file',
          size: isDir ? 0 : (obj?.data.length ?? 0),
          name: e.name,
          path: filePath === '' ? e.name : `${filePath}/${e.name}`,
          sha: e.oid,
          url: `/api/v3/repos/${owner}/${name}/contents/${filePath === '' ? e.name : `${filePath}/${e.name}`}`,
          html_url: '',
          git_url: '',
          download_url: isDir ? null : '',
        };
      });
      return json(200, entries);
    }
    if (found.type !== 'blob') return err(404, 'Not Found');
    const obj = repo.objects.get(found.oid)!;
    const baseName = filePath.split('/').pop() ?? filePath;
    return json(200, {
      type: 'file',
      encoding: 'base64',
      size: obj.data.length,
      name: baseName,
      path: filePath,
      content: obj.data.toString('base64'),
      sha: found.oid,
      url: `/api/v3/repos/${owner}/${name}/contents/${filePath}`,
      html_url: '',
      git_url: '',
      download_url: '',
    });
  }

  if (method === 'PUT') {
    if (!isObject(req.body)) return err(422, 'Validation Failed');
    const message = asString(req.body.message);
    const contentB64 = asString(req.body.content);
    const branch = asString(req.body.branch, repo.defaultBranch);
    const givenSha = asString(req.body.sha);
    if (!message) return err(422, 'Validation Failed: message is required');
    if (contentB64 === '') return err(422, 'Validation Failed: content is required');
    const ref = `refs/heads/${branch}`;
    // On an unborn repo the first PUT creates the branch; otherwise it must exist.
    if (repo.refs.size > 0 && !repo.refs.has(ref)) {
      return err(404, 'Not Found');
    }
    let content: Buffer;
    try {
      content = Buffer.from(contentB64, 'base64');
    } catch {
      return err(422, 'Validation Failed: content is not valid base64');
    }
    const tip = refTip(repo, ref);
    const isUpdate = tip !== '' && fileOid(repo, commitTree(repo, tip), filePath) !== null;
    if (isUpdate) {
      const currentSha = fileOid(repo, commitTree(repo, tip), filePath);
      if (givenSha && givenSha !== currentSha) return err(409, 'Conflict: the file has changed since the given sha');
      if (!givenSha) return err(422, 'Validation Failed: sha is required to update a file');
    } else if (givenSha) {
      return err(409, 'Conflict: sha was given for a file that does not exist');
    }
    const { commitSha, blobSha } = commitFileChange(store, repo, branch, filePath, content, message, false);
    const baseName = filePath.split('/').pop() ?? filePath;
    return json(isUpdate ? 200 : 201, {
      content: {
        name: baseName,
        path: filePath,
        sha: blobSha,
        size: content.length,
        url: `/api/v3/repos/${owner}/${name}/contents/${filePath}`,
        html_url: '',
        git_url: '',
        download_url: '',
        type: 'file',
        encoding: 'base64',
      },
      commit: { sha: commitSha, message, author: { name: store.owner }, committer: { name: store.owner } },
    });
  }

  if (method === 'DELETE') {
    if (!isObject(req.body)) return err(422, 'Validation Failed');
    const message = asString(req.body.message);
    const givenSha = asString(req.body.sha);
    const branch = asString(req.body.branch, repo.defaultBranch);
    const ref = `refs/heads/${branch}`;
    const tip = refTip(repo, ref);
    if (!tip) return err(404, 'Not Found');
    const tree = commitTree(repo, tip);
    const currentSha = fileOid(repo, tree, filePath);
    if (currentSha === null) return err(404, 'Not Found');
    if (givenSha && givenSha !== currentSha) return err(409, 'Conflict: the file has changed since the given sha');
    const { commitSha } = commitFileChange(store, repo, branch, filePath, Buffer.alloc(0), message || `Delete ${filePath}`, true);
    return json(200, { commit: { sha: commitSha, message: message || `Delete ${filePath}` } });
  }

  return err(404, 'Not Found');
}

/* ------------------------------------------------------------------ */
/* Issues                                                              */
/* ------------------------------------------------------------------ */

function handleIssues(store: Store, repo: Repo, sub: string[], req: RestRequest): RestResponse {
  const { method, query } = req;

  // POST /issues, GET /issues
  if (sub.length === 0) {
    if (method === 'POST') {
      if (!isObject(req.body)) return err(422, 'Validation Failed');
      const title = asString(req.body.title);
      if (!title) return err(422, 'Validation Failed: title is required');
      const body = asString(req.body.body);
      const labelNames: string[] = Array.isArray(req.body.labels) ? req.body.labels.filter((l): l is string => typeof l === 'string') : [];
      for (const name of labelNames) {
        if (!repo.labels.some((l) => l.name === name)) return err(422, `Validation Failed: label does not exist: ${name}`);
      }
      const issue = store.createIssue(repo, title, body, labelNames);
      return json(201, issueJson(store, repo, issue));
    }
    if (method === 'GET') {
      const state = query.get('state') ?? 'open';
      const sort = query.get('sort') ?? 'created';
      const direction = query.get('direction') ?? 'desc';
      let issues = [...repo.issues];
      if (state !== 'all') issues = issues.filter((i) => i.state === state);
      issues.sort((a, b) => {
        const ka = sort === 'updated' ? a.updatedAt : a.createdAt;
        const kb = sort === 'updated' ? b.updatedAt : b.createdAt;
        const cmp = ka < kb ? -1 : ka > kb ? 1 : a.number - b.number;
        return direction === 'asc' ? cmp : -cmp;
      });
      const { page, link } = paginate(issues.map((i) => issueJson(store, repo, i)), query, `/api/v3/repos/${repo.owner}/${repo.name}/issues`, req.host);
      return json(200, page, { link });
    }
    return err(404, 'Not Found');
  }

  // /issues/{n} and /issues/{n}/comments
  const number = Number.parseInt(sub[0], 10);
  if (!Number.isInteger(number)) return err(404, 'Not Found');
  const issue = repo.issues.find((i) => i.number === number);
  if (!issue) return err(404, 'Not Found');

  if (sub.length === 1) {
    if (method === 'GET') return json(200, issueJson(store, repo, issue));
    if (method === 'PATCH') {
      if (isObject(req.body)) {
        if (typeof req.body.title === 'string') issue.title = req.body.title;
        if (typeof req.body.body === 'string') issue.body = req.body.body;
        if (req.body.state === 'open' || req.body.state === 'closed') {
          issue.state = req.body.state;
          issue.closedAt = req.body.state === 'closed' ? tick() : null;
        }
        if (Array.isArray(req.body.labels)) {
          const names = req.body.labels.filter((l): l is string => typeof l === 'string');
          for (const n of names) {
            if (!repo.labels.some((l) => l.name === n)) return err(422, `Validation Failed: label does not exist: ${n}`);
          }
          issue.labels = names;
        }
        issue.updatedAt = tick();
      }
      return json(200, issueJson(store, repo, issue));
    }
    return err(404, 'Not Found');
  }

  if (sub.length === 2 && sub[1] === 'comments') {
    if (method === 'GET') {
      const comments = issue.comments.map((c) => ({
        id: c.id,
        node_id: globalId('IssueComment', c.id),
        url: `/api/v3/repos/${repo.owner}/${repo.name}/issues/${number}/comments/${c.id}`,
        body: c.body,
        user: userJson(store, c.user),
        created_at: c.createdAt,
        updated_at: c.updatedAt,
      }));
      return json(200, comments);
    }
    if (method === 'POST') {
      if (!isObject(req.body) || !asString(req.body.body)) return err(422, 'Validation Failed: body is required');
      const comment = {
        id: store.nextId(),
        body: asString(req.body.body),
        user: store.owner,
        createdAt: tick(),
        updatedAt: tick(),
      };
      issue.comments.push(comment);
      issue.updatedAt = tick();
      return json(201, {
        id: comment.id,
        node_id: globalId('IssueComment', comment.id),
        url: `/api/v3/repos/${repo.owner}/${repo.name}/issues/${number}/comments/${comment.id}`,
        body: comment.body,
        user: userJson(store, comment.user),
        created_at: comment.createdAt,
        updated_at: comment.updatedAt,
      });
    }
  }

  return err(404, 'Not Found');
}

/* ------------------------------------------------------------------ */
/* Pull requests                                                       */
/* ------------------------------------------------------------------ */

function handlePulls(store: Store, repo: Repo, sub: string[], req: RestRequest): RestResponse {
  const { method, query } = req;

  // POST /pulls, GET /pulls
  if (sub.length === 0) {
    if (method === 'POST') {
      if (!isObject(req.body)) return err(422, 'Validation Failed');
      const head = asString(req.body.head);
      const base = asString(req.body.base);
      const title = asString(req.body.title);
      if (!head || !base || !title) return err(422, 'Validation Failed: head, base and title are required');
      const headSha = refTip(repo, `refs/heads/${head}`);
      const baseSha = refTip(repo, `refs/heads/${base}`);
      if (!headSha || !baseSha) return err(422, 'Validation Failed: head or base branch not found');
      const pr = store.createPull(repo, title, asString(req.body.body), head, base, headSha, baseSha);
      return json(201, pullJson(store, repo, pr));
    }
    if (method === 'GET') {
      const state = query.get('state') ?? 'open';
      let pulls = [...repo.pulls];
      if (state !== 'all') pulls = pulls.filter((p) => p.state === state);
      pulls.sort((a, b) => b.number - a.number);
      const { page, link } = paginate(pulls.map((p) => pullJson(store, repo, p)), query, `/api/v3/repos/${repo.owner}/${repo.name}/pulls`, req.host);
      return json(200, page, { link });
    }
    return err(404, 'Not Found');
  }

  const number = Number.parseInt(sub[0], 10);
  if (!Number.isInteger(number)) return err(404, 'Not Found');
  const pr = repo.pulls.find((p) => p.number === number);
  if (!pr) return err(404, 'Not Found');

  // GET /pulls/{n}, PATCH /pulls/{n}
  if (sub.length === 1) {
    if (method === 'GET') return json(200, pullJson(store, repo, pr));
    if (method === 'PATCH') {
      if (isObject(req.body)) {
        if (typeof req.body.title === 'string') pr.title = req.body.title;
        if (typeof req.body.body === 'string') pr.body = req.body.body;
        if (req.body.state === 'open' || req.body.state === 'closed') {
          pr.state = req.body.state;
          pr.closedAt = req.body.state === 'closed' ? tick() : null;
        }
        pr.updatedAt = tick();
      }
      return json(200, pullJson(store, repo, pr));
    }
    return err(404, 'Not Found');
  }

  // GET /pulls/{n}/files
  if (sub.length === 2 && sub[1] === 'files' && method === 'GET') {
    const files = pullFiles(store, repo, pr);
    return json(200, files);
  }

  // PUT /pulls/{n}/merge
  if (sub.length === 2 && sub[1] === 'merge' && method === 'PUT') {
    const result = mergePullRequest(store, repo, pr);
    if (!result.ok) return err(result.reason === 'Not Found' ? 404 : 405, result.reason);
    return json(200, { sha: result.sha, merged: true, message: `Merge pull request #${pr.number} from ${store.owner}/${pr.headRef}` });
  }

  return err(404, 'Not Found');
}

/** The files a pull request changes (merge-base -> head), with line counts. */
function pullFiles(store: Store, repo: Repo, pr: PullRequest): Record<string, unknown>[] {
  const baseTip = refTip(repo, `refs/heads/${pr.baseRef}`);
  const headTip = refTip(repo, `refs/heads/${pr.headRef}`);
  if (!baseTip || !headTip) return [];
  const base = mergeBase(repo, baseTip, headTip);
  if (!base) return [];
  const flatten = (treeOid: string): Map<string, string> => {
    const out = new Map<string, string>();
    const walk = (oid: string, prefix: string): void => {
      const obj = repo.objects.get(oid);
      if (!obj || obj.type !== 'tree') return;
      for (const e of parseTree(obj.data)) {
        const p = prefix === '' ? e.name : `${prefix}/${e.name}`;
        if (e.mode === '40000') walk(e.oid, p);
        else out.set(p, e.oid);
      }
    };
    walk(treeOid, '');
    return out;
  };
  const baseFiles = flatten(commitTree(repo, base));
  const headFiles = flatten(commitTree(repo, headTip));
  const out: Record<string, unknown>[] = [];
  const paths = new Set([...baseFiles.keys(), ...headFiles.keys()]);
  for (const p of [...paths].sort()) {
    const b = baseFiles.get(p);
    const h = headFiles.get(p);
    if (b === h) continue;
    let status: string;
    let additions = 0;
    let deletions = 0;
    if (b === undefined) {
      status = 'added';
      const data = repo.objects.get(h!)?.data.toString('utf8') ?? '';
      additions = diffLineCounts('', data).additions;
    } else if (h === undefined) {
      status = 'removed';
      const data = repo.objects.get(b)?.data.toString('utf8') ?? '';
      deletions = diffLineCounts(data, '').deletions;
    } else {
      status = 'modified';
      const a = repo.objects.get(b)?.data.toString('utf8') ?? '';
      const c = repo.objects.get(h)?.data.toString('utf8') ?? '';
      const d = diffLineCounts(a, c);
      additions = d.additions;
      deletions = d.deletions;
    }
    out.push({
      sha: h ?? b,
      filename: p,
      status,
      additions,
      deletions,
      changes: additions + deletions,
      blob_url: '',
      raw_url: '',
      contents_url: '',
    });
  }
  return out;
}

/** Re-exported for the GraphQL/MCP layers. */
export { gitObjectId, ZERO_OID };
export { mergeBase, listCommits, resolveRef, commitTree };
