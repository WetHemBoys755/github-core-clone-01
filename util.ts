// Shared helpers for the forge: hashing, pkt-line framing, HTTP plumbing.
// Stdlib only (node:crypto, node:http). No wall clock or randomness reaches
// any stored state: ids and timestamps come from counters and the trial seed.

import { createHash } from 'node:crypto';
import type { IncomingMessage, ServerResponse } from 'node:http';

export const JSON_TYPE = 'application/json; charset=utf-8';
export const ZERO_OID = '0000000000000000000000000000000000000000';

/** SHA-1 of bytes, hex. */
export function sha1(data: Buffer): string {
  return createHash('sha1').update(data).digest('hex');
}

/** SHA-1 of a git object: sha1("<type> <len>\0" + payload). */
export function gitObjectId(type: string, payload: Buffer): string {
  const head = Buffer.from(`${type} ${payload.length}\0`, 'utf8');
  return sha1(Buffer.concat([head, payload]));
}

/** True when `s` is a 40-hex-char object id. */
export function isSha1(s: string): boolean {
  return /^[0-9a-f]{40}$/.test(s);
}

/** An opaque Relay-style global id: base64("<Type>:<numeric id or sha>"). */
export function globalId(type: string, id: string | number): string {
  return Buffer.from(`${type}:${id}`, 'utf8').toString('base64');
}

/** Decode a global id back to its type and id, or null when malformed. */
export function parseGlobalId(id: string): { type: string; id: string } | null {
  try {
    const text = Buffer.from(id, 'base64').toString('utf8');
    const colon = text.indexOf(':');
    if (colon <= 0) return null;
    return { type: text.slice(0, colon), id: text.slice(colon + 1) };
  } catch {
    return null;
  }
}

/** A short, deterministic request id (counter-based, never a clock). */
let requestCounter = 0;
export function nextRequestId(): string {
  requestCounter += 1;
  return `req-${requestCounter.toString(16).padStart(8, '0')}`;
}

/** Read a request body fully, with a size cap. Rejects on overflow. */
export function readBody(req: IncomingMessage, maxBytes = 64 * 1024 * 1024): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    req.on('data', (chunk: Buffer) => {
      total += chunk.length;
      if (total > maxBytes) {
        reject(new Error('request body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

/** Parse a JSON body; returns null when the bytes are not valid JSON. */
export function parseJson(buf: Buffer): unknown {
  try {
    return JSON.parse(buf.toString('utf8'));
  } catch {
    return null;
  }
}

/** Send a JSON reply with the GitHub shape: content-type, request id. */
export function sendJson(
  res: ServerResponse,
  status: number,
  body: unknown,
  headers: Readonly<Record<string, string>> = {},
): void {
  const text = `${JSON.stringify(body)}\n`;
  res.writeHead(status, {
    'content-type': JSON_TYPE,
    'content-length': Buffer.byteLength(text),
    'x-github-request-id': nextRequestId(),
    ...headers,
  });
  res.end(text);
}

/** GitHub-style error body: {message}. */
export function sendError(res: ServerResponse, status: number, message: string): void {
  sendJson(res, status, { message });
}

/** The path of a request URL, or '/' when it does not parse. */
export function pathOf(req: IncomingMessage): string {
  try {
    return new URL(req.url ?? '/', 'http://entry').pathname;
  } catch {
    return '/';
  }
}

/** Query parameters of a request URL. */
export function queryOf(req: IncomingMessage): URLSearchParams {
  try {
    return new URL(req.url ?? '/', 'http://entry').searchParams;
  } catch {
    return new URLSearchParams();
  }
}

/** Decode a path segment (percent-decoding, never throws). */
export function decodeSegment(s: string): string {
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
}

/** Build a GitHub-style Link header for pagination (rel="next"/"last"). */
export function pageLinks(
  base: string,
  query: URLSearchParams,
  page: number,
  perPage: number,
  total: number,
): string {
  const lastPage = Math.max(1, Math.ceil(total / perPage));
  const parts: string[] = [];
  const mk = (p: number, rel: string): string => {
    const q = new URLSearchParams(query);
    q.set('page', String(p));
    q.set('per_page', String(perPage));
    return `<${base}?${q.toString()}>; rel="${rel}"`;
  };
  if (page < lastPage) parts.push(mk(page + 1, 'next'));
  if (lastPage > 1) parts.push(mk(lastPage, 'last'));
  if (page > 1) parts.push(mk(1, 'first'));
  if (page > 1) parts.push(mk(page - 1, 'prev'));
  return parts.join(', ');
}

/** Slice a list into one page; returns the page items and the total. */
export function paginate<T>(items: readonly T[], page: number, perPage: number): { page: T[]; total: number } {
  const p = Math.max(1, page);
  const pp = Math.max(1, Math.min(100, perPage));
  return { page: items.slice((p - 1) * pp, p * pp), total: items.length };
}

/** Encode one pkt-line: 4 hex length digits + payload. Length 0 = flush. */
export function pktLine(payload: Buffer | string): Buffer {
  const data = typeof payload === 'string' ? Buffer.from(payload, 'utf8') : payload;
  const len = data.length + 4;
  const head = Buffer.from(len.toString(16).padStart(4, '0'), 'utf8');
  return Buffer.concat([head, data]);
}

/** The pkt-line flush packet ("0000"). */
export const PKT_FLUSH = Buffer.from('0000', 'utf8');

/** The pkt-line delimiter packet ("0001"). */
export const PKT_DELIM = Buffer.from('0001', 'utf8');

/** The pkt-line response-end packet ("0002"). */
export const PKT_RESPONSE_END = Buffer.from('0002', 'utf8');

/**
 * Split a buffer into pkt-lines. Returns the lines (payloads without the
 * 4-byte length prefix) and any trailing partial bytes. A line of length 0
 * (flush) is returned as null; delim/response-end as their markers.
 */
export function splitPktLines(buf: Buffer): { lines: (Buffer | null)[]; rest: Buffer } {
  const lines: (Buffer | null)[] = [];
  let off = 0;
  while (off + 4 <= buf.length) {
    const len = parseInt(buf.subarray(off, off + 4).toString('utf8'), 16);
    if (Number.isNaN(len)) break;
    if (len === 0) {
      lines.push(null);
      off += 4;
      continue;
    }
    if (len < 4 || off + len > buf.length) break;
    lines.push(buf.subarray(off + 4, off + len));
    off += len;
  }
  return { lines, rest: buf.subarray(off) };
}

/** Base64 encode bytes (standard alphabet, padded). */
export function b64encode(data: Buffer): string {
  return data.toString('base64');
}

/** Base64 decode; returns null on invalid input. */
export function b64decode(s: string): Buffer | null {
  try {
    const clean = s.replace(/\s+/g, '');
    if (!/^[A-Za-z0-9+/]*={0,2}$/.test(clean)) return null;
    return Buffer.from(clean, 'base64');
  } catch {
    return null;
  }
}

/** Format a Date as an ISO-8601 string. */
export function iso(d: Date): string {
  return d.toISOString();
}

/** Compare two strings byte-wise (for sorting tree entries). */
export function cmpStr(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}
