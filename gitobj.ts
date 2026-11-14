// Git objects and packfiles, stdlib only (node:zlib, node:crypto).
// Objects are immutable and addressed by SHA-1; packs carry them over the
// wire. Delta-compressed objects are resolved on read; objects we send are
// undelted (the suite accepts that) with their zlib bytes cached per oid.

import { deflateSync, inflateSync } from 'node:zlib';
import { gitObjectId, sha1, ZERO_OID } from './util.ts';
import type { GitObject, ObjectType, Repo } from './store.ts';

export { ZERO_OID };

/* ------------------------------------------------------------------ */
/* Object construction                                                 */
/* ------------------------------------------------------------------ */

export function makeBlob(data: Buffer): { type: ObjectType; data: Buffer; oid: string } {
  return { type: 'blob', data, oid: gitObjectId('blob', data) };
}

export interface TreeEntry {
  mode: string;
  name: string;
  oid: string;
}

/** Serialize a tree: entries sorted git-style (dirs sort as name + "/"). */
export function serializeTree(entries: TreeEntry[]): Buffer {
  const sorted = [...entries].sort((a, b) => {
    const an = a.mode === '40000' ? `${a.name}/` : a.name;
    const bn = b.mode === '40000' ? `${b.name}/` : b.name;
    return an < bn ? -1 : an > bn ? 1 : 0;
  });
  const parts: Buffer[] = [];
  for (const e of sorted) {
    parts.push(Buffer.from(`${e.mode} ${e.name}\0`, 'utf8'));
    parts.push(Buffer.from(e.oid, 'hex'));
  }
  return Buffer.concat(parts);
}

export function makeTree(entries: TreeEntry[]): { type: ObjectType; data: Buffer; oid: string } {
  const data = serializeTree(entries);
  return { type: 'tree', data, oid: gitObjectId('tree', data) };
}

/** Parse a tree object into its entries. */
export function parseTree(data: Buffer): TreeEntry[] {
  const entries: TreeEntry[] = [];
  let off = 0;
  while (off < data.length) {
    const sp = data.indexOf(0x20, off);
    if (sp < 0) break;
    const nul = data.indexOf(0, sp);
    if (nul < 0 || nul + 21 > data.length) break;
    const mode = data.subarray(off, sp).toString('utf8');
    const name = data.subarray(sp + 1, nul).toString('utf8');
    const oid = data.subarray(nul + 1, nul + 21).toString('hex');
    entries.push({ mode, name, oid });
    off = nul + 21;
  }
  return entries;
}

export interface CommitInfo {
  tree: string;
  parents: string[];
  author: string;
  committer: string;
  message: string;
  /** Any other header lines, in order (e.g. gpgsig is not produced here). */
  extra: string[];
}

/** Serialize a commit. Author/committer are full "Name <email> ts tz" lines. */
export function serializeCommit(c: CommitInfo): Buffer {
  const lines: string[] = [`tree ${c.tree}`];
  for (const p of c.parents) lines.push(`parent ${p}`);
  lines.push(`author ${c.author}`);
  lines.push(`committer ${c.committer}`);
  for (const e of c.extra) lines.push(e);
  return Buffer.from(`${lines.join('\n')}\n\n${c.message}`, 'utf8');
}

export function makeCommit(c: CommitInfo): { type: ObjectType; data: Buffer; oid: string } {
  const data = serializeCommit(c);
  return { type: 'commit', data, oid: gitObjectId('commit', data) };
}

/** Parse a commit object. */
export function parseCommit(data: Buffer): CommitInfo {
  const text = data.toString('utf8');
  const sep = text.indexOf('\n\n');
  const head = sep >= 0 ? text.slice(0, sep) : text;
  const message = sep >= 0 ? text.slice(sep + 2) : '';
  const info: CommitInfo = { tree: '', parents: [], author: '', committer: '', message, extra: [] };
  for (const line of head.split('\n')) {
    if (line.startsWith('tree ')) info.tree = line.slice(5).trim();
    else if (line.startsWith('parent ')) info.parents.push(line.slice(7).trim());
    else if (line.startsWith('author ')) info.author = line.slice(7);
    else if (line.startsWith('committer ')) info.committer = line.slice(10);
    else if (line.length > 0) info.extra.push(line);
  }
  return info;
}

export interface TagInfo {
  object: string;
  type: ObjectType;
  tag: string;
  tagger: string;
  message: string;
}

export function serializeTag(t: TagInfo): Buffer {
  return Buffer.from(
    `object ${t.object}\ntype ${t.type}\ntag ${t.tag}\ntagger ${t.tagger}\n\n${t.message}`,
    'utf8',
  );
}

export function makeTag(t: TagInfo): { type: ObjectType; data: Buffer; oid: string } {
  const data = serializeTag(t);
  return { type: 'tag', data, oid: gitObjectId('tag', data) };
}

/** Parse a tag object. */
export function parseTag(data: Buffer): TagInfo {
  const text = data.toString('utf8');
  const sep = text.indexOf('\n\n');
  const head = sep >= 0 ? text.slice(0, sep) : text;
  const message = sep >= 0 ? text.slice(sep + 2) : '';
  const info: TagInfo = { object: '', type: 'commit', tag: '', tagger: '', message };
  for (const line of head.split('\n')) {
    if (line.startsWith('object ')) info.object = line.slice(7).trim();
    else if (line.startsWith('type ')) info.type = line.slice(5).trim() as ObjectType;
    else if (line.startsWith('tag ')) info.tag = line.slice(4);
    else if (line.startsWith('tagger ')) info.tagger = line.slice(7);
  }
  return info;
}

/* ------------------------------------------------------------------ */
/* Reachability                                                        */
/* ------------------------------------------------------------------ */

/** All objects reachable from `roots` (commits, their trees/blobs, tags peeled). */
export function reachable(repo: Repo, roots: string[]): Set<string> {
  const seen = new Set<string>();
  const stack: string[] = [...roots];
  while (stack.length > 0) {
    const oid = stack.pop()!;
    if (oid === ZERO_OID || seen.has(oid)) continue;
    const obj = repo.objects.get(oid);
    if (!obj) continue;
    seen.add(oid);
    if (obj.type === 'commit') {
      const c = parseCommit(obj.data);
      if (c.tree) stack.push(c.tree);
      for (const p of c.parents) stack.push(p);
    } else if (obj.type === 'tree') {
      for (const e of parseTree(obj.data)) stack.push(e.oid);
    } else if (obj.type === 'tag') {
      stack.push(parseTag(obj.data).object);
    }
  }
  return seen;
}

/** Is `ancestor` an ancestor of `descendant` (or equal)? Walks first-parent chains and merges. */
export function isAncestor(repo: Repo, ancestor: string, descendant: string): boolean {
  if (ancestor === descendant) return true;
  if (ancestor === ZERO_OID) return true;
  const seen = new Set<string>();
  const stack = [descendant];
  while (stack.length > 0) {
    const oid = stack.pop()!;
    if (oid === ZERO_OID || seen.has(oid)) continue;
    seen.add(oid);
    const obj = repo.objects.get(oid);
    if (!obj || obj.type !== 'commit') continue;
    for (const p of parseCommit(obj.data).parents) {
      if (p === ancestor) return true;
      stack.push(p);
    }
  }
  return false;
}

/* ------------------------------------------------------------------ */
/* Pack reading                                                        */
/* ------------------------------------------------------------------ */

const PACK_TYPES: Record<number, ObjectType | 'ofs_delta' | 'ref_delta'> = {
  1: 'commit',
  2: 'tree',
  3: 'blob',
  4: 'tag',
  6: 'ofs_delta',
  7: 'ref_delta',
};

/** Read a varint as used in pack object headers (MSB continuation, 7 bits/byte, little-endian groups). */
function readSizeVarint(buf: Buffer, off: number): { value: number; next: number } {
  let value = buf[off] & 0x0f;
  let shift = 4;
  let i = off;
  while (buf[i] & 0x80) {
    i += 1;
    value |= (buf[i] & 0x7f) << shift;
    shift += 7;
  }
  return { value, next: i + 1 };
}

/** Read a delta's size varint (MSB continuation, 7 bits/byte, big-endian groups). */
function readDeltaVarint(buf: Buffer, off: number): { value: number; next: number } {
  let value = 0;
  let i = off;
  for (;;) {
    const b = buf[i];
    i += 1;
    value = (value << 7) | (b & 0x7f);
    if (!(b & 0x80)) break;
  }
  return { value, next: i };
}

/** Apply a delta to a base object, returning the result bytes. */
export function applyDelta(base: Buffer, delta: Buffer): Buffer {
  const src = readDeltaVarint(delta, 0);
  if (src.value !== base.length) throw new Error('delta source size mismatch');
  const dst = readDeltaVarint(delta, src.next);
  const out: Buffer[] = [];
  let total = 0;
  let i = dst.next;
  while (i < delta.length) {
    const op = delta[i];
    i += 1;
    if (op & 0x80) {
      // Copy from base: offset/size encoded in following bytes.
      let offset = 0;
      let size = 0;
      for (let b = 0; b < 4; b++) {
        if (op & (1 << b)) {
          offset |= delta[i] << (8 * b);
          i += 1;
        }
      }
      for (let b = 0; b < 3; b++) {
        if (op & (0x10 << b)) {
          size |= delta[i] << (8 * b);
          i += 1;
        }
      }
      if (size === 0) size = 0x10000;
      out.push(base.subarray(offset, offset + size));
      total += size;
    } else if (op !== 0) {
      out.push(delta.subarray(i, i + op));
      total += op;
      i += op;
    } else {
      throw new Error('delta opcode 0');
    }
  }
  if (total !== dst.value) throw new Error('delta result size mismatch');
  return Buffer.concat(out);
}

export interface ParsedPackObject {
  type: ObjectType;
  data: Buffer;
  oid: string;
}

/**
 * Parse a whole packfile into objects (deltas resolved). Throws on any
 * structural error. The trailer checksum is verified by the caller.
 */
export function parsePack(pack: Buffer): ParsedPackObject[] {
  if (pack.length < 12 || pack.subarray(0, 4).toString('ascii') !== 'PACK') {
    throw new Error('not a packfile');
  }
  const version = pack.readUInt32BE(4);
  if (version !== 2) throw new Error(`unsupported pack version ${version}`);
  const count = pack.readUInt32BE(8);
  const objects: ParsedPackObject[] = [];
  // Deltas may reference objects later in the pack; resolve in a second pass.
  const raw: { type: ObjectType | 'ofs_delta' | 'ref_delta'; data: Buffer; baseOid?: string; baseOff?: number; index: number; start: number }[] = [];
  const startOffsets: number[] = [];
  let off = 12;
  for (let n = 0; n < count; n++) {
    const start = off;
    startOffsets.push(start);
    const first = pack[off];
    const typeNum = (first >> 4) & 0x07;
    const type = PACK_TYPES[typeNum];
    if (!type) throw new Error(`unknown pack object type ${typeNum}`);
    const size = readSizeVarint(pack, off);
    off = size.next;
    let baseOid: string | undefined;
    let baseOff: number | undefined;
    if (type === 'ofs_delta') {
      // Negative offset, variable length, big-endian with +1 per byte.
      let b = pack[off];
      off += 1;
      let offset = b & 0x7f;
      while (b & 0x80) {
        b = pack[off];
        off += 1;
        offset = ((offset + 1) << 7) | (b & 0x7f);
      }
      baseOff = start - offset;
    } else if (type === 'ref_delta') {
      baseOid = pack.subarray(off, off + 20).toString('hex');
      off += 20;
    }
    const inflated = inflateSync(pack.subarray(off));
    // inflateSync ignores trailing bytes, so it returns exactly this
    // object's data; endOfZlib locates where the stream ends.
    off = endOfZlib(pack, off);
    raw.push({ type, data: inflated, baseOid, baseOff, index: n, start });
  }
  // Resolve in order (a delta's base is always earlier in the pack).
  const byIndex = new Map<number, ParsedPackObject>();
  const byOid = new Map<string, ParsedPackObject>();
  for (const r of raw) {
    if (r.type === 'ofs_delta' || r.type === 'ref_delta') {
      let base: ParsedPackObject | undefined;
      if (r.type === 'ofs_delta' && r.baseOff !== undefined) {
        const idx = startOffsets.indexOf(r.baseOff);
        base = idx >= 0 ? byIndex.get(idx) : undefined;
      } else if (r.baseOid) {
        base = byOid.get(r.baseOid);
      }
      if (!base) throw new Error('delta base not found');
      const data = applyDelta(base.data, r.data);
      const oid = gitObjectId(base.type, data);
      const obj: ParsedPackObject = { type: base.type, data, oid };
      byIndex.set(r.index, obj);
      byOid.set(oid, obj);
      objects.push(obj);
    } else {
      const oid = gitObjectId(r.type as ObjectType, r.data);
      const obj: ParsedPackObject = { type: r.type as ObjectType, data: r.data, oid };
      byIndex.set(r.index, obj);
      byOid.set(oid, obj);
      objects.push(obj);
    }
  }
  return objects;
}

/** Find where a zlib stream starting at `off` ends (returns the offset just past it). */
function endOfZlib(buf: Buffer, off: number): number {
  // inflateSync ignores trailing bytes, so tryInflate(mid) succeeds exactly
  // when mid is at or past the stream's true end, and throws when the
  // stream is truncated. Binary-search the smallest succeeding end.
  const tryInflate = (end: number): boolean => {
    try {
      inflateSync(buf.subarray(off, end));
      return true;
    } catch {
      return false;
    }
  };
  let left = off + 1;
  let right = buf.length;
  while (left < right) {
    const mid = (left + right) >> 1;
    if (tryInflate(mid)) right = mid;
    else left = mid + 1;
  }
  return left;
}

/** Verify a pack's 20-byte trailer equals the SHA-1 of all preceding bytes. */
export function verifyPackTrailer(pack: Buffer): boolean {
  if (pack.length < 20) return false;
  const body = pack.subarray(0, pack.length - 20);
  const trailer = pack.subarray(pack.length - 20).toString('hex');
  return sha1(body) === trailer;
}

/* ------------------------------------------------------------------ */
/* Pack writing                                                        */
/* ------------------------------------------------------------------ */

/** zlib-compressed bytes of one object's raw data, cached on the object. */
export function compressedObject(obj: GitObject): Buffer {
  if (obj.compressed) return obj.compressed;
  obj.compressed = deflateSync(obj.data);
  return obj.compressed;
}

/** Write one pack object entry (undelted): type/size header + zlib data.
 *  `compressed` is the object's cached deflated bytes when the caller has
 *  them (repeated clones of one history re-deflate nothing). */
function writePackObject(type: ObjectType, data: Buffer, compressed?: Buffer): Buffer {
  const typeNum = type === 'commit' ? 1 : type === 'tree' ? 2 : type === 'blob' ? 3 : 4;
  const size = data.length;
  // Header: first byte holds type (bits 4-6) and size low 4 bits; continuation.
  const header: number[] = [];
  let s = size;
  let first = (typeNum << 4) | (s & 0x0f);
  s >>= 4;
  if (s > 0) first |= 0x80;
  header.push(first);
  while (s > 0) {
    let b = s & 0x7f;
    s >>= 7;
    if (s > 0) b |= 0x80;
    header.push(b);
  }
  // Pack entries carry the raw object data only (no "<type> <len>\0" prefix).
  const body = compressed ?? deflateSync(data);
  return Buffer.concat([Buffer.from(header), body]);
}

/**
 * Build a packfile of `oids` (undelted objects, in the given order).
 * Returns the pack bytes including the SHA-1 trailer.
 */
export function buildPack(repo: Repo, oids: string[]): Buffer {
  const parts: Buffer[] = [];
  const head = Buffer.alloc(12);
  head.write('PACK', 0, 'ascii');
  head.writeUInt32BE(2, 4);
  head.writeUInt32BE(oids.length, 8);
  parts.push(head);
  for (const oid of oids) {
    const obj = repo.objects.get(oid);
    if (!obj) continue;
    parts.push(writePackObject(obj.type, obj.data, compressedObject(obj)));
  }
  const body = Buffer.concat(parts);
  const trailer = Buffer.from(sha1(body), 'hex');
  return Buffer.concat([body, trailer]);
}
