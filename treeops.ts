// Tree manipulation on the store: read a path, set/delete a file, list a
// directory, and three-way merge trees. All operations build new immutable
// git objects (blob/tree/commit) in the store and return the new root tree
// oid; the caller's commit update moves the branch ref.

import { gitObjectId } from './util.ts';
import type { Repo } from './store.ts';
import { makeBlob, makeCommit, makeTree, parseCommit, parseTree, type TreeEntry } from './gitobj.ts';

export interface PathLookup {
  type: 'blob' | 'tree' | 'commit' | 'tag';
  oid: string;
  mode: string;
}

/** Walk `path` (slash-separated, no leading/trailing slash) from a tree oid. */
export function lookupPath(repo: Repo, treeOid: string, path: string): PathLookup | null {
  const segments = path.split('/').filter((s) => s.length > 0);
  if (segments.length === 0) return { type: 'tree', oid: treeOid, mode: '40000' };
  let currentOid = treeOid;
  for (let i = 0; i < segments.length; i++) {
    const obj = repo.objects.get(currentOid);
    if (!obj || obj.type !== 'tree') return null;
    const entry = parseTree(obj.data).find((e) => e.name === segments[i]);
    if (!entry) return null;
    if (i === segments.length - 1) {
      const target = repo.objects.get(entry.oid);
      if (!target) return null;
      return { type: target.type, oid: entry.oid, mode: entry.mode };
    }
    if (entry.mode !== '40000') return null;
    currentOid = entry.oid;
  }
  return null;
}

/** Store a tree object built from entries; returns its oid. */
export function storeTree(repo: Repo, entries: TreeEntry[]): string {
  const tree = makeTree(entries);
  repo.objects.set(tree.oid, { type: 'tree', data: tree.data, compressed: null });
  return tree.oid;
}

/** Store a blob; returns its oid. */
export function storeBlob(repo: Repo, data: Buffer): string {
  const blob = makeBlob(data);
  repo.objects.set(blob.oid, { type: 'blob', data: blob.data, compressed: null });
  return blob.oid;
}

/**
 * Set a file at `path` in the tree rooted at `rootTreeOid` (creating
 * intermediate directories). Returns the new root tree oid.
 */
export function setFile(repo: Repo, rootTreeOid: string | null, path: string, blobOid: string): string {
  const segments = path.split('/').filter((s) => s.length > 0);
  if (segments.length === 0) throw new Error('empty path');
  const setIn = (treeOid: string | null, depth: number): string => {
    const name = segments[depth];
    const entries: TreeEntry[] = treeOid ? parseTree(repo.objects.get(treeOid)!.data) : [];
    const isLast = depth === segments.length - 1;
    const idx = entries.findIndex((e) => e.name === name);
    if (isLast) {
      const entry: TreeEntry = { mode: '100644', name, oid: blobOid };
      if (idx >= 0) entries[idx] = entry;
      else entries.push(entry);
    } else {
      const childOid = idx >= 0 ? entries[idx].oid : null;
      const newChild = setIn(childOid, depth + 1);
      const entry: TreeEntry = { mode: '40000', name, oid: newChild };
      if (idx >= 0) entries[idx] = entry;
      else entries.push(entry);
    }
    return storeTree(repo, entries);
  };
  return setIn(rootTreeOid, 0);
}

/**
 * Delete the file at `path` from the tree rooted at `rootTreeOid`.
 * Returns the new root tree oid, or null when the path does not exist.
 * Empty directories left behind are pruned.
 */
export function deleteFile(repo: Repo, rootTreeOid: string, path: string): string | null {
  const segments = path.split('/').filter((s) => s.length > 0);
  if (segments.length === 0) return null;
  const delFrom = (treeOid: string, depth: number): string | null => {
    const obj = repo.objects.get(treeOid);
    if (!obj || obj.type !== 'tree') return null;
    const entries = parseTree(obj.data);
    const name = segments[depth];
    const idx = entries.findIndex((e) => e.name === name);
    if (idx < 0) return null;
    if (depth === segments.length - 1) {
      entries.splice(idx, 1);
    } else {
      if (entries[idx].mode !== '40000') return null;
      const newChild = delFrom(entries[idx].oid, depth + 1);
      if (newChild === null) return null;
      // Prune a directory that became empty (the root tree itself may stay empty).
      if (parseTree(repo.objects.get(newChild)!.data).length === 0) {
        entries.splice(idx, 1);
      } else {
        entries[idx] = { mode: '40000', name, oid: newChild };
      }
    }
    return storeTree(repo, entries);
  };
  return delFrom(rootTreeOid, 0);
}

/** List the entries of a tree (name, mode, oid, type). */
export function listTree(repo: Repo, treeOid: string): TreeEntry[] {
  const obj = repo.objects.get(treeOid);
  if (!obj || obj.type !== 'tree') return [];
  return parseTree(obj.data);
}

/** The blob bytes at `path` in the tree rooted at `rootTreeOid`, or null. */
export function readFile(repo: Repo, rootTreeOid: string, path: string): Buffer | null {
  const found = lookupPath(repo, rootTreeOid, path);
  if (!found || found.type !== 'blob') return null;
  const obj = repo.objects.get(found.oid);
  return obj ? obj.data : null;
}

/** The blob oid at `path` in the tree rooted at `rootTreeOid`, or null. */
export function fileOid(repo: Repo, rootTreeOid: string, path: string): string | null {
  const found = lookupPath(repo, rootTreeOid, path);
  return found && found.type === 'blob' ? found.oid : null;
}

export interface CommitInput {
  tree: string;
  parents: string[];
  message: string;
  author: string;
  committer: string;
}

/** Store a commit object; returns its oid. */
export function storeCommit(repo: Repo, input: CommitInput): string {
  const commit = makeCommit({
    tree: input.tree,
    parents: input.parents,
    author: input.author,
    committer: input.committer,
    message: input.message,
    extra: [],
  });
  repo.objects.set(commit.oid, { type: 'commit', data: commit.data, compressed: null });
  return commit.oid;
}

/** The commit oid at a ref ('' when the ref is unborn/absent). */
export function refTip(repo: Repo, ref: string): string {
  return repo.refs.get(ref) ?? '';
}

/** The tree oid of a commit, or '' when the commit is missing. */
export function commitTree(repo: Repo, commitOid: string): string {
  const obj = repo.objects.get(commitOid);
  if (!obj || obj.type !== 'commit') return '';
  return parseCommit(obj.data).tree;
}

/**
 * Three-way merge of trees. `baseTree` is the merge base, `oursTree` the
 * current branch tip's tree, `theirsTree` the head's tree. Returns the merged
 * root tree oid and the list of conflicted paths (a path conflicts when both
 * sides changed it from the base to different content).
 */
export function mergeTrees(
  repo: Repo,
  baseTree: string,
  oursTree: string,
  theirsTree: string,
): { tree: string; conflicts: string[] } {
  const conflicts: string[] = [];
  // Flatten each tree to path -> blob oid.
  const flatten = (treeOid: string, prefix: string, out: Map<string, string>): void => {
    const obj = repo.objects.get(treeOid);
    if (!obj || obj.type !== 'tree') return;
    for (const e of parseTree(obj.data)) {
      const p = prefix === '' ? e.name : `${prefix}/${e.name}`;
      if (e.mode === '40000') flatten(e.oid, p, out);
      else out.set(p, e.oid);
    }
  };
  const base = new Map<string, string>();
  const ours = new Map<string, string>();
  const theirs = new Map<string, string>();
  flatten(baseTree, '', base);
  flatten(oursTree, '', ours);
  flatten(theirsTree, '', theirs);
  const merged = new Map<string, string>(ours);
  const paths = new Set([...base.keys(), ...ours.keys(), ...theirs.keys()]);
  for (const p of paths) {
    const b = base.get(p);
    const o = ours.get(p);
    const t = theirs.get(p);
    if (o === t) continue;
    if (b === o) {
      // Only theirs changed it: take theirs (or delete when theirs removed it).
      if (t === undefined) merged.delete(p);
      else merged.set(p, t);
      continue;
    }
    if (b === t) continue; // Only ours changed it: keep ours.
    // Both changed it to different content: conflict.
    conflicts.push(p);
  }
  // Rebuild a tree from the merged path map.
  const build = (pathsMap: Map<string, string>): string => {
    const byDir = new Map<string, Map<string, { mode: string; oid: string }>>();
    const ensure = (dir: string): Map<string, { mode: string; oid: string }> => {
      let m = byDir.get(dir);
      if (!m) {
        m = new Map();
        byDir.set(dir, m);
      }
      return m;
    };
    for (const [p, oid] of pathsMap) {
      const segs = p.split('/');
      const file = segs[segs.length - 1];
      const dir = segs.slice(0, -1).join('/');
      ensure(dir).set(file, { mode: '100644', oid });
    }
    const buildDir = (dir: string): string => {
      const entries: TreeEntry[] = [];
      const direct = byDir.get(dir) ?? new Map();
      for (const [name, v] of direct) entries.push({ mode: v.mode, name, oid: v.oid });
      // Subdirectories: any dir whose parent is `dir`.
      const subdirs = new Set<string>();
      for (const d of byDir.keys()) {
        if (d === dir) continue;
        const parent = d.split('/').slice(0, -1).join('/');
        if (parent === dir) subdirs.add(d.split('/').pop()!);
      }
      for (const sub of subdirs) {
        const subPath = dir === '' ? sub : `${dir}/${sub}`;
        entries.push({ mode: '40000', name: sub, oid: buildDir(subPath) });
      }
      return storeTree(repo, entries);
    };
    return buildDir('');
  };
  return { tree: build(merged), conflicts: conflicts.sort() };
}

/** The git object id of a "<type> <len>\0<data>" payload (re-exported). */
export { gitObjectId };
