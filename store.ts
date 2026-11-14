// In-memory store for the forge: repositories, issues, pull requests, labels,
// comments, and the git object database + refs of every repository.
//
// Everything lives in memory (no disk, no outside service). Time is a
// deterministic counter, never the wall clock: every created/updated stamp
// comes from tick(), so a live run and its replay produce identical state.

export type ObjectType = 'commit' | 'tree' | 'blob' | 'tag';

export interface GitObject {
  type: ObjectType;
  data: Buffer;
  /** zlib-compressed "<type> <len>\0<data>", cached for fast pack building. */
  compressed: Buffer | null;
}

export interface Repo {
  id: number;
  name: string;
  owner: string;
  description: string;
  private: boolean;
  defaultBranch: string;
  /** git refs: full refname -> oid ('' for unborn). */
  refs: Map<string, string>;
  /** git objects by oid. */
  objects: Map<string, GitObject>;
  issues: Issue[];
  pulls: PullRequest[];
  labels: Label[];
  nextIssueNumber: number;
  nextPullNumber: number;
  createdAt: string;
  updatedAt: string;
}

export interface Issue {
  id: number;
  number: number;
  title: string;
  body: string;
  state: 'open' | 'closed';
  labels: string[];
  user: string;
  createdAt: string;
  updatedAt: string;
  closedAt: string | null;
  comments: Comment[];
}

export interface Comment {
  id: number;
  body: string;
  user: string;
  createdAt: string;
  updatedAt: string;
}

export interface Label {
  id: number;
  name: string;
  color: string;
  description: string;
}

export interface PullRequest {
  id: number;
  number: number;
  title: string;
  body: string;
  state: 'open' | 'closed';
  headRef: string;
  baseRef: string;
  headSha: string;
  baseSha: string;
  merged: boolean;
  mergeCommitSha: string | null;
  mergedAt: string | null;
  createdAt: string;
  updatedAt: string;
  closedAt: string | null;
}

export const ZERO_OID = '0000000000000000000000000000000000000000';

/** Deterministic clock: starts at a fixed epoch, +1s per tick. */
const EPOCH_MS = Date.UTC(2026, 10, 14, 0, 0, 0);
let tickCount = 0;

export function tick(): string {
  tickCount += 1;
  return new Date(EPOCH_MS + tickCount * 1000).toISOString();
}

export class Store {
  /** The login the bearer token names (the forge's one user). */
  readonly owner: string;
  /** Repos keyed by "owner/name". */
  readonly repos = new Map<string, Repo>();
  /** Repos in creation order (for GET /user/repos pagination). */
  readonly repoList: Repo[] = [];
  private nextRepoId = 1;
  private nextGlobalId = 1;

  constructor(owner: string) {
    this.owner = owner;
  }

  nextId(): number {
    const id = this.nextGlobalId;
    this.nextGlobalId += 1;
    return id;
  }

  repoKey(owner: string, name: string): string {
    return `${owner}/${name}`;
  }

  createRepo(name: string, opts: { description?: string; private?: boolean; defaultBranch?: string }): Repo {
    const now = tick();
    const repo: Repo = {
      id: this.nextRepoId++,
      name,
      owner: this.owner,
      description: opts.description ?? '',
      private: opts.private ?? false,
      defaultBranch: opts.defaultBranch ?? 'main',
      refs: new Map(),
      objects: new Map(),
      issues: [],
      pulls: [],
      labels: [],
      nextIssueNumber: 1,
      nextPullNumber: 1,
      createdAt: now,
      updatedAt: now,
    };
    this.repos.set(this.repoKey(this.owner, name), repo);
    this.repoList.push(repo);
    return repo;
  }

  getRepo(owner: string, name: string): Repo | undefined {
    return this.repos.get(this.repoKey(owner, name));
  }

  deleteRepo(owner: string, name: string): boolean {
    const key = this.repoKey(owner, name);
    const repo = this.repos.get(key);
    if (!repo) return false;
    this.repos.delete(key);
    const idx = this.repoList.indexOf(repo);
    if (idx >= 0) this.repoList.splice(idx, 1);
    return true;
  }

  /** All repos owned by `owner`, in creation order. */
  reposOf(owner: string): Repo[] {
    return this.repoList.filter((r) => r.owner === owner);
  }

  createIssue(repo: Repo, title: string, body: string, labels: string[]): Issue {
    const now = tick();
    const issue: Issue = {
      id: this.nextId(),
      number: repo.nextIssueNumber++,
      title,
      body,
      state: 'open',
      labels: [...labels],
      user: this.owner,
      createdAt: now,
      updatedAt: now,
      closedAt: null,
      comments: [],
    };
    repo.issues.push(issue);
    repo.updatedAt = now;
    return issue;
  }

  createPull(repo: Repo, title: string, body: string, headRef: string, baseRef: string, headSha: string, baseSha: string): PullRequest {
    const now = tick();
    const pr: PullRequest = {
      id: this.nextId(),
      number: repo.nextPullNumber++,
      title,
      body,
      state: 'open',
      headRef,
      baseRef,
      headSha,
      baseSha,
      merged: false,
      mergeCommitSha: null,
      mergedAt: null,
      createdAt: now,
      updatedAt: now,
      closedAt: null,
    };
    repo.pulls.push(pr);
    repo.updatedAt = now;
    return pr;
  }

  createLabel(repo: Repo, name: string, color: string, description: string): Label {
    const label: Label = { id: this.nextId(), name, color, description };
    repo.labels.push(label);
    return label;
  }

  /** Resolve a ref to its oid; '' when unborn, undefined when the ref is absent. */
  resolveRef(repo: Repo, ref: string): string | undefined {
    return repo.refs.get(ref);
  }

  /** Set a ref ('' deletes it when oid is the zero oid). */
  setRef(repo: Repo, ref: string, oid: string): void {
    if (oid === ZERO_OID) repo.refs.delete(ref);
    else repo.refs.set(ref, oid);
  }

  /** Store a git object; returns its oid. Caches nothing until asked. */
  putObject(repo: Repo, type: ObjectType, data: Buffer, oid: string): void {
    repo.objects.set(oid, { type, data, compressed: null });
  }

  getObject(repo: Repo, oid: string): GitObject | undefined {
    return repo.objects.get(oid);
  }
}
