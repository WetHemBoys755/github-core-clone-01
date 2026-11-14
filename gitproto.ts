// Git smart-HTTP protocol (v2 + v0 upload-pack, v0 receive-pack), stdlib only.
//
// This implements exactly what the suite's git client speaks (see the kit's
// program/env/src/git-client.ts): pkt-line framing, the v2 capability
// advertisement, ls-refs, fetch (including shallow/deepen), and receive-pack
// with per-ref report-status. It is the server side of the protocol; the
// client drives it, so every response below mirrors the shapes the client
// parses (acknowledgments / shallow-info / wanted-refs / packfile sections,
// sideband band 1 for pack data and for the push report).

import { sha1, ZERO_OID } from './util.ts';
import type { ObjectType, Repo } from './store.ts';
import { buildPack, isAncestor, parseCommit, parsePack, parseTag, parseTree, reachable, verifyPackTrailer } from './gitobj.ts';

/* ------------------------------------------------------------------ */
/* pkt-line framing                                                    */
/* ------------------------------------------------------------------ */

export const PKT_FLUSH = Buffer.from('0000', 'ascii');
export const PKT_DELIM = Buffer.from('0001', 'ascii');

/** One pkt-line: 4 hex digits (total length incl. header) + payload. */
export function pktLine(data: string | Buffer): Buffer {
  const payload = typeof data === 'string' ? Buffer.from(data, 'utf8') : data;
  const len = payload.length + 4;
  const head = Buffer.alloc(4);
  head.write(len.toString(16).padStart(4, '0'), 0, 'ascii');
  return Buffer.concat([head, payload]);
}

export type PktKind = 'data' | 'flush' | 'delim' | 'response-end';
export interface Pkt {
  kind: PktKind;
  payload: Buffer;
}

/**
 * Decode pkt-lines from the start of `buf` until `stopAtFlush` (a flush
 * packet) or the buffer ends. Returns the packets and the offset just past
 * the last consumed packet (the rest is raw data, e.g. a packfile).
 */
export function decodePktLines(buf: Buffer, stopAtFlush = false): { packets: Pkt[]; rest: number } {
  const packets: Pkt[] = [];
  let off = 0;
  while (off + 4 <= buf.length) {
    const head = buf.subarray(off, off + 4).toString('ascii');
    if (!/^[0-9a-fA-F]{4}$/.test(head)) break;
    const len = Number.parseInt(head, 16);
    if (len === 0) {
      packets.push({ kind: 'flush', payload: Buffer.alloc(0) });
      off += 4;
      if (stopAtFlush) break;
      continue;
    }
    if (len === 1) {
      packets.push({ kind: 'delim', payload: Buffer.alloc(0) });
      off += 4;
      continue;
    }
    if (len === 2) {
      packets.push({ kind: 'response-end', payload: Buffer.alloc(0) });
      off += 4;
      continue;
    }
    if (len < 4 || off + len > buf.length) break;
    packets.push({ kind: 'data', payload: buf.subarray(off + 4, off + len) });
    off += len;
  }
  return { packets, rest: off };
}

/** Split sideband-64k payload bytes into band-1 pkt-lines (chunked). */
export function sideband(data: Buffer, chunk = 32768): Buffer {
  const parts: Buffer[] = [];
  for (let off = 0; off < data.length; off += chunk) {
    const piece = data.subarray(off, Math.min(off + chunk, data.length));
    parts.push(pktLine(Buffer.concat([Buffer.from([1]), piece])));
  }
  return Buffer.concat(parts);
}

/* ------------------------------------------------------------------ */
/* Refs                                                                */
/* ------------------------------------------------------------------ */

export interface RefEntry {
  oid: string;
  name: string;
  peeled?: string;
  symrefTarget?: string;
  unborn?: boolean;
}

function peelOid(repo: Repo, oid: string): string | undefined {
  const obj = repo.objects.get(oid);
  if (!obj) return undefined;
  if (obj.type === 'tag') {
    const tag = parseTag(obj.data);
    return peelOid(repo, tag.object);
  }
  return obj.type === 'commit' ? oid : undefined;
}

/** All refs: HEAD (symref to the default branch) + branches + tags, sorted. */
export function listRefs(repo: Repo): RefEntry[] {
  const entries: RefEntry[] = [];
  const defaultRef = `refs/heads/${repo.defaultBranch}`;
  const headOid = repo.refs.get(defaultRef);
  if (headOid) {
    entries.push({ oid: headOid, name: 'HEAD', symrefTarget: defaultRef });
  } else {
    entries.push({ oid: ZERO_OID, name: 'HEAD', symrefTarget: defaultRef, unborn: true });
  }
  const names = [...repo.refs.keys()].sort();
  for (const name of names) {
    const oid = repo.refs.get(name)!;
    const entry: RefEntry = { oid, name };
    if (name.startsWith('refs/tags/')) {
      const peeled = peelOid(repo, oid);
      if (peeled && peeled !== oid) entry.peeled = peeled;
    }
    entries.push(entry);
  }
  return entries;
}

const UPLOAD_CAPS_V0 =
  'side-band-64k multi_ack_detailed thin-pack no-progress ofs-delta shallow agent=forge/1.0 symref=HEAD:refs/heads/main object-format=sha1';
const RECEIVE_CAPS_V0 =
  'report-status report-status-v2 delete-refs side-band-64k ofs-delta agent=forge/1.0 object-format=sha1';

/** v2 capability advertisement (info/refs with Git-Protocol: version=2). */
export function v2Advertisement(): Buffer {
  return Buffer.concat([
    pktLine('version 2\n'),
    pktLine('ls-refs=unborn\n'),
    pktLine('fetch=shallow filter\n'),
    pktLine('agent=forge/1.0\n'),
    pktLine('object-format=sha1\n'),
    PKT_FLUSH,
  ]);
}

/** v0 ref advertisement (info/refs without the v2 header, or receive-pack). */
export function v0Advertisement(repo: Repo, service: 'git-upload-pack' | 'git-receive-pack'): Buffer {
  const caps = service === 'git-upload-pack' ? UPLOAD_CAPS_V0 : RECEIVE_CAPS_V0;
  const parts: Buffer[] = [pktLine(`# service=${service}\n`), PKT_FLUSH];
  const refs = listRefs(repo);
  const realRefs = refs.filter((r) => r.name !== 'HEAD');
  if (realRefs.length === 0) {
    // Empty repo: the capabilities^{} pseudo-ref carries the capabilities.
    parts.push(pktLine(`${ZERO_OID} capabilities^{}\0${caps}\n`));
  } else {
    let first = true;
    for (const ref of refs) {
      const line = first
        ? `${ref.oid} ${ref.name}\0${caps}\n`
        : `${ref.oid} ${ref.name}\n`;
      parts.push(pktLine(line));
      first = false;
      if (ref.peeled) parts.push(pktLine(`${ref.peeled} ${ref.name}^{}\n`));
    }
  }
  parts.push(PKT_FLUSH);
  return Buffer.concat(parts);
}

/* ------------------------------------------------------------------ */
/* v2 ls-refs                                                          */
/* ------------------------------------------------------------------ */

export interface LsRefsRequest {
  peel: boolean;
  symrefs: boolean;
  unborn: boolean;
  prefixes: string[];
}

/** Parse a v2 ls-refs request body into its options. */
export function parseLsRefs(body: Buffer): LsRefsRequest {
  const req: LsRefsRequest = { peel: false, symrefs: false, unborn: false, prefixes: [] };
  const { packets } = decodePktLines(body);
  for (const p of packets) {
    if (p.kind !== 'data') continue;
    const line = p.payload.toString('utf8').replace(/\n$/, '');
    if (line === 'peel') req.peel = true;
    else if (line === 'symrefs') req.symrefs = true;
    else if (line === 'unborn') req.unborn = true;
    else if (line.startsWith('ref-prefix ')) req.prefixes.push(line.slice('ref-prefix '.length));
  }
  return req;
}

/** Answer a v2 ls-refs request. */
export function handleLsRefs(repo: Repo, body: Buffer): Buffer {
  const req = parseLsRefs(body);
  const refs = listRefs(repo);
  const keep = (name: string): boolean =>
    req.prefixes.length === 0 || req.prefixes.some((prefix) => name.startsWith(prefix));
  const parts: Buffer[] = [];
  for (const ref of refs) {
    if (!keep(ref.name)) continue;
    if (ref.unborn) {
      if (!req.unborn) continue;
      const attrs = ref.symrefTarget && req.symrefs ? ` symref-target:${ref.symrefTarget}` : '';
      parts.push(pktLine(`unborn ${ref.name}${attrs}\n`));
      continue;
    }
    let line = `${ref.oid} ${ref.name}`;
    if (ref.peeled && req.peel) line += ` peeled:${ref.peeled}`;
    if (ref.symrefTarget && req.symrefs) line += ` symref-target:${ref.symrefTarget}`;
    parts.push(pktLine(`${line}\n`));
  }
  parts.push(PKT_FLUSH);
  return Buffer.concat(parts);
}

/* ------------------------------------------------------------------ */
/* Fetch (v2 + v0)                                                     */
/* ------------------------------------------------------------------ */

export interface FetchRequest {
  wants: string[];
  haves: string[];
  shallow: string[];
  depth: number | null;
  deepenSince: number | null;
  deepenNot: string[];
  filter: string | null;
}

/** Parse a v2 or v0 fetch request body (pkt-lines) into its fields. */
export function parseFetch(body: Buffer): FetchRequest {
  const req: FetchRequest = {
    wants: [],
    haves: [],
    shallow: [],
    depth: null,
    deepenSince: null,
    deepenNot: [],
    filter: null,
  };
  const { packets } = decodePktLines(body);
  for (const p of packets) {
    if (p.kind !== 'data') continue;
    let line = p.payload.toString('utf8').replace(/\n$/, '');
    // v0 puts capabilities after a NUL on the first want line.
    const nul = line.indexOf('\0');
    if (nul >= 0) line = line.slice(0, nul);
    if (line.startsWith('want ')) req.wants.push(line.slice(5).trim().split(' ')[0]);
    else if (line.startsWith('have ')) req.haves.push(line.slice(5).trim());
    else if (line.startsWith('shallow ')) req.shallow.push(line.slice(8).trim());
    else if (line.startsWith('deepen ')) {
      const n = Number.parseInt(line.slice(7).trim(), 10);
      if (Number.isFinite(n)) req.depth = n;
    } else if (line.startsWith('deepen-since ')) {
      const n = Number.parseInt(line.slice(13).trim(), 10);
      if (Number.isFinite(n)) req.deepenSince = n;
    } else if (line.startsWith('deepen-not ')) req.deepenNot.push(line.slice(11).trim());
    else if (line.startsWith('filter ')) req.filter = line.slice(7).trim();
  }
  return req;
}

interface FetchPlan {
  /** Oids to put in the pack (commits + trees + blobs, filter applied). */
  send: string[];
  /** New shallow boundary oids (deepest included commits with parents). */
  shallow: string[];
  /** Old boundaries that are no longer boundaries (their parents were sent). */
  unshallow: string[];
}

/**
 * The objects a client's `haves` already cover. A have that is a shallow
 * boundary covers only itself plus its tree/blobs (its parents were never
 * fetched); any other have covers its whole reachable history.
 */
function haveObjects(repo: Repo, req: FetchRequest): Set<string> {
  const covered = new Set<string>();
  const boundaries = new Set(req.shallow);
  for (const have of req.haves) {
    if (boundaries.has(have)) {
      const obj = repo.objects.get(have);
      if (!obj) continue;
      covered.add(have);
      if (obj.type === 'commit') {
        const tree = parseCommit(obj.data).tree;
        if (tree) for (const oid of reachableTrees(repo, tree)) covered.add(oid);
      } else if (obj.type === 'tag') {
        const target = parseTag(obj.data).object;
        const t = repo.objects.get(target);
        if (t && t.type === 'commit') {
          const tree = parseCommit(t.data).tree;
          if (tree) for (const oid of reachableTrees(repo, tree)) covered.add(oid);
        }
      }
    } else {
      for (const oid of reachable(repo, [have])) covered.add(oid);
    }
  }
  return covered;
}

/** A tree oid plus every tree/blob under it (no commits, no parents). */
function reachableTrees(repo: Repo, treeOid: string): Set<string> {
  const out = new Set<string>();
  const stack = [treeOid];
  while (stack.length > 0) {
    const oid = stack.pop()!;
    if (out.has(oid)) continue;
    const obj = repo.objects.get(oid);
    if (!obj) continue;
    out.add(oid);
    if (obj.type === 'tree') {
      for (const e of parseTree(obj.data)) stack.push(e.oid);
    }
  }
  return out;
}

/**
 * Work out what a fetch sends. `depth` is the total depth from the wants
 * (null = the whole history). Haves are never re-sent. The shallow boundary
 * is the deepest included commit that still has parents; an old boundary
 * whose parents were sent becomes unshallow.
 */
export function planFetch(repo: Repo, req: FetchRequest): FetchPlan {
  const haves = new Set(req.haves);
  const boundaries = new Set(req.shallow);
  const depth = req.depth ?? Number.POSITIVE_INFINITY;
  // Walk commits from the wants, tracking the generation (0 = the want).
  const visited = new Set<string>();
  const queue: Array<{ oid: string; gen: number }> = [];
  for (const want of req.wants) {
    if (!visited.has(want)) {
      visited.add(want);
      queue.push({ oid: want, gen: 0 });
    }
  }
  const included = new Set<string>();
  const boundary = new Set<string>();
  const direct = new Set<string>();
  while (queue.length > 0) {
    const { oid, gen } = queue.shift()!;
    const obj = repo.objects.get(oid);
    if (!obj) continue;
    if (obj.type !== 'commit') {
      direct.add(oid);
      continue;
    }
    // A have that is not a shallow boundary is fully held: stop here.
    if (haves.has(oid) && !boundaries.has(oid)) continue;
    if (gen < depth && !haves.has(oid)) included.add(oid);
    const parents = parseCommit(obj.data).parents;
    if (gen + 1 >= depth) {
      // Deepest included generation: the boundary when history continues.
      if (parents.length > 0 && gen + 1 === depth) boundary.add(oid);
      continue;
    }
    for (const parent of parents) {
      if (!visited.has(parent)) {
        visited.add(parent);
        queue.push({ oid: parent, gen: gen + 1 });
      }
    }
  }
  // Old boundaries whose parents were sent are no longer boundaries.
  const unshallow: string[] = [];
  for (const old of req.shallow) {
    const obj = repo.objects.get(old);
    if (!obj || obj.type !== 'commit') continue;
    const parents = parseCommit(obj.data).parents;
    if (parents.some((p) => included.has(p))) unshallow.push(old);
  }
  // Gather the objects: included commits, their trees/blobs, direct wants.
  const sendSet = new Set<string>();
  for (const oid of included) {
    sendSet.add(oid);
    const tree = parseCommit(repo.objects.get(oid)!.data).tree;
    if (tree) for (const t of reachableTrees(repo, tree)) sendSet.add(t);
  }
  for (const oid of direct) {
    sendSet.add(oid);
    const obj = repo.objects.get(oid);
    if (obj?.type === 'tag') {
      const target = parseTag(obj.data).object;
      sendSet.add(target);
      const t = repo.objects.get(target);
      if (t?.type === 'commit') {
        const tree = parseCommit(t.data).tree;
        if (tree) for (const x of reachableTrees(repo, tree)) sendSet.add(x);
      }
    } else if (obj?.type === 'tree') {
      for (const t of reachableTrees(repo, oid)) sendSet.add(t);
    }
  }
  // Never re-send what the client already holds.
  const covered = haveObjects(repo, req);
  for (const oid of covered) sendSet.delete(oid);
  // Apply the filter.
  for (const oid of [...sendSet]) {
    const obj = repo.objects.get(oid);
    if (!obj) {
      sendSet.delete(oid);
      continue;
    }
    if (req.filter === 'blob:none' && obj.type === 'blob') sendSet.delete(oid);
    if (req.filter === 'tree:0' && (obj.type === 'tree' || obj.type === 'blob')) sendSet.delete(oid);
  }
  // Deterministic pack order.
  const send = [...sendSet].sort();
  return { send, shallow: [...boundary].sort(), unshallow: unshallow.sort() };
}

/** The v2 fetch response: acknowledgments, shallow-info, wanted-refs, packfile. */
export function handleFetchV2(repo: Repo, body: Buffer): Buffer {
  const req = parseFetch(body);
  const plan = planFetch(repo, req);
  const parts: Buffer[] = [];
  // acknowledgments
  parts.push(pktLine('acknowledgments\n'));
  const common = req.haves.filter((h) => {
    if (!repo.objects.has(h)) return false;
    return req.wants.some((w) => w === h || isAncestor(repo, h, w));
  });
  if (common.length > 0) {
    for (const ack of common) parts.push(pktLine(`ACK ${ack}\n`));
  } else {
    parts.push(pktLine('NAK\n'));
  }
  parts.push(pktLine('ready\n'));
  parts.push(PKT_DELIM);
  // shallow-info
  if (plan.shallow.length > 0 || plan.unshallow.length > 0 || req.shallow.length > 0) {
    parts.push(pktLine('shallow-info\n'));
    for (const s of plan.shallow) parts.push(pktLine(`shallow ${s}\n`));
    for (const u of plan.unshallow) parts.push(pktLine(`unshallow ${u}\n`));
    parts.push(PKT_DELIM);
  }
  // wanted-refs
  const refs = listRefs(repo).filter((r) => r.name !== 'HEAD');
  const wanted = refs.filter((r) => req.wants.includes(r.oid));
  if (wanted.length > 0) {
    parts.push(pktLine('wanted-refs\n'));
    for (const r of wanted) parts.push(pktLine(`${r.oid} ${r.name}\n`));
    parts.push(PKT_DELIM);
  }
  // packfile
  parts.push(pktLine('packfile\n'));
  const pack = buildPack(repo, plan.send);
  parts.push(sideband(pack));
  parts.push(PKT_FLUSH);
  return Buffer.concat(parts);
}

/** The v0 fetch response: NAK/ACK + shallow lines, then the sideband pack. */
export function handleFetchV0(repo: Repo, body: Buffer): Buffer {
  const req = parseFetch(body);
  const plan = planFetch(repo, req);
  const parts: Buffer[] = [];
  const common = req.haves.filter((h) => {
    if (!repo.objects.has(h)) return false;
    return req.wants.some((w) => w === h || isAncestor(repo, h, w));
  });
  if (common.length > 0) {
    for (const ack of common) parts.push(pktLine(`ACK ${ack}\n`));
  } else {
    parts.push(pktLine('NAK\n'));
  }
  for (const s of plan.shallow) parts.push(pktLine(`shallow ${s}\n`));
  for (const u of plan.unshallow) parts.push(pktLine(`unshallow ${u}\n`));
  const pack = buildPack(repo, plan.send);
  parts.push(sideband(pack));
  parts.push(PKT_FLUSH);
  return Buffer.concat(parts);
}

/* ------------------------------------------------------------------ */
/* receive-pack (push)                                                 */
/* ------------------------------------------------------------------ */

export interface PushCommand {
  oldOid: string;
  newOid: string;
  ref: string;
}

const REFNAME_BAD = /(\.\.|[~^:?*[\]\\]|\/\/|^\.|\.$|\.lock$|@{|@\{)/;
function validRefname(ref: string): boolean {
  if (!ref.startsWith('refs/') || ref.length > 200) return false;
  if (ref.endsWith('/') || ref.includes('//')) return false;
  if (REFNAME_BAD.test(ref)) return false;
  const parts = ref.split('/');
  if (parts.some((p) => p.length === 0)) return false;
  return true;
}

/**
 * Parse a receive-pack request: pkt-line commands up to the flush, then the
 * raw pack bytes that follow. Returns the commands and the pack (or null for
 * a delete-only push, which carries no pack).
 */
export function parseReceivePack(body: Buffer): { commands: PushCommand[]; pack: Buffer | null } {
  const { packets, rest } = decodePktLines(body, true);
  const commands: PushCommand[] = [];
  for (const p of packets) {
    if (p.kind !== 'data') continue;
    let line = p.payload.toString('utf8').replace(/\n$/, '');
    const nul = line.indexOf('\0');
    if (nul >= 0) line = line.slice(0, nul);
    const parts = line.split(' ');
    if (parts.length < 3) continue;
    commands.push({ oldOid: parts[0], newOid: parts[1], ref: parts.slice(2).join(' ') });
  }
  const pack = rest < body.length ? body.subarray(rest) : null;
  return { commands, pack };
}

/**
 * Apply a receive-pack: unpack the pack, validate each ref command, apply
 * the valid ones, and build the sideband report-status response.
 * The response body is returned; the caller sets the content type.
 */
export function handleReceivePack(repo: Repo, body: Buffer): Buffer {
  const { commands, pack } = parseReceivePack(body);
  const results: Array<{ ref: string; ok: boolean; reason: string }> = [];
  let unpackOk = true;
  let unpackDetail = 'ok';

  // Unpack first: a corrupt pack fails the whole exchange, but the ref
  // reports are still sent (git reports "unpack error" and refuses each ref).
  const newObjects: Array<{ type: ObjectType; data: Buffer; oid: string }> = [];
  if (pack && pack.length > 0) {
    try {
      if (!verifyPackTrailer(pack)) throw new Error('pack trailer checksum mismatch');
      const parsed = parsePack(pack);
      newObjects.push(...parsed);
    } catch (err) {
      unpackOk = false;
      unpackDetail = err instanceof Error ? err.message : 'unpack error';
    }
  }
  if (unpackOk) {
    for (const o of newObjects) {
      repo.objects.set(o.oid, { type: o.type, data: o.data, compressed: null });
    }
  }

  for (const cmd of commands) {
    if (!unpackOk) {
      results.push({ ref: cmd.ref, ok: false, reason: 'unpack failed' });
      continue;
    }
    if (!validRefname(cmd.ref)) {
      results.push({ ref: cmd.ref, ok: false, reason: 'invalid refname' });
      continue;
    }
    const current = repo.refs.get(cmd.ref);
    // Old value must match (ZERO_OID means "must not exist").
    if (cmd.oldOid === ZERO_OID) {
      if (current !== undefined) {
        results.push({ ref: cmd.ref, ok: false, reason: 'already exists' });
        continue;
      }
    } else if (current !== cmd.oldOid) {
      results.push({ ref: cmd.ref, ok: false, reason: 'old value does not match' });
      continue;
    }
    if (cmd.newOid === ZERO_OID) {
      // Delete.
      repo.refs.delete(cmd.ref);
      results.push({ ref: cmd.ref, ok: true, reason: '' });
      continue;
    }
    // Create / update: the new object must exist and be the right kind.
    const obj = repo.objects.get(cmd.newOid);
    if (!obj) {
      results.push({ ref: cmd.ref, ok: false, reason: 'object not found' });
      continue;
    }
    if (cmd.ref.startsWith('refs/heads/') && obj.type !== 'commit') {
      results.push({ ref: cmd.ref, ok: false, reason: 'branch must point at a commit' });
      continue;
    }
    if (cmd.ref.startsWith('refs/tags/') && obj.type !== 'commit' && obj.type !== 'tag') {
      results.push({ ref: cmd.ref, ok: false, reason: 'tag must point at a commit or tag' });
      continue;
    }
    // Fast-forward: the new tip must descend from the old one (or be equal).
    if (cmd.oldOid !== ZERO_OID && cmd.newOid !== cmd.oldOid && !isAncestor(repo, cmd.oldOid, cmd.newOid)) {
      results.push({ ref: cmd.ref, ok: false, reason: 'non-fast-forward' });
      continue;
    }
    repo.refs.set(cmd.ref, cmd.newOid);
    results.push({ ref: cmd.ref, ok: true, reason: '' });
  }

  // Report-status, pkt-lined inside sideband band 1.
  const reportLines: Buffer[] = [pktLine(`unpack ${unpackOk ? 'ok' : unpackDetail}\n`)];
  for (const r of results) {
    reportLines.push(pktLine(r.ok ? `ok ${r.ref}\n` : `ng ${r.ref} ${r.reason}\n`));
  }
  const report = Buffer.concat(reportLines);
  return Buffer.concat([sideband(report), PKT_FLUSH]);
}

/** SHA-1 of a buffer (used by the pack trailer check in gitobj.ts). */
export { sha1 };
export { parseTree };
