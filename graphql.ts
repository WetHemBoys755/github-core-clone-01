// GitHub GraphQL v4 surface (POST /api/graphql), stdlib only.
//
// A small, purpose-built GraphQL executor for the documents gh and
// github-mcp-server actually send (see the kit's
// program/env/src/checks-graphql.ts): viewer, repository fields, the
// __type feature probes, issue/pull-request reads, cursor pagination,
// node(id:), the five mutations, invalid-field errors, 401 without a token,
// and operation batching (a JSON array of documents in one body).
//
// The executor is a real parser + resolver, not a string matcher: it lexes
// the document, parses operations/fields/arguments/fragments, resolves
// variables, validates field names against the schema (an unknown field is
// a 200 with an `errors` array, never an HTTP error), and executes the
// selection set against the same store the REST surface uses — so a mutation
// here is visible through REST and vice versa.

import { globalId, parseGlobalId } from './util.ts';
import { tick, type Issue, type PullRequest, type Repo, type Store } from './store.ts';
import { commitTree, mergeTrees, refTip } from './treeops.ts';
import { mergeBase, mergePullRequest } from './rest.ts';
import { parseTree } from './gitobj.ts';

/* ------------------------------------------------------------------ */
/* Lexer                                                               */
/* ------------------------------------------------------------------ */

type TokenKind = 'name' | 'string' | 'int' | 'float' | 'punct' | 'spread' | 'eof';
interface Token {
  kind: TokenKind;
  value: string;
}

const PUNCT = new Set(['!', '$', '(', ')', ':', '=', '@', '[', ']', '{', '}', '|', '&']);

function lex(input: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  const n = input.length;
  while (i < n) {
    const c = input[i];
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r' || c === ',') {
      i += 1;
      continue;
    }
    if (c === '#') {
      while (i < n && input[i] !== '\n') i += 1;
      continue;
    }
    if (c === '.' && input[i + 1] === '.' && input[i + 2] === '.') {
      tokens.push({ kind: 'spread', value: '...' });
      i += 3;
      continue;
    }
    if (PUNCT.has(c)) {
      tokens.push({ kind: 'punct', value: c });
      i += 1;
      continue;
    }
    if (c === '"') {
      // A string (possibly a block string """).
      if (input.startsWith('"""', i)) {
        const end = input.indexOf('"""', i + 3);
        const text = end < 0 ? input.slice(i + 3) : input.slice(i + 3, end);
        tokens.push({ kind: 'string', value: text.replace(/\\"""/g, '"""') });
        i = end < 0 ? n : end + 3;
      } else {
        let out = '';
        i += 1;
        while (i < n && input[i] !== '"') {
          if (input[i] === '\\' && i + 1 < n) {
            const esc = input[i + 1];
            if (esc === 'n') out += '\n';
            else if (esc === 't') out += '\t';
            else if (esc === 'r') out += '\r';
            else out += esc;
            i += 2;
          } else {
            out += input[i];
            i += 1;
          }
        }
        tokens.push({ kind: 'string', value: out });
        i += 1;
      }
      continue;
    }
    if (/[0-9-]/.test(c)) {
      let j = i;
      if (input[j] === '-') j += 1;
      while (j < n && /[0-9]/.test(input[j])) j += 1;
      let isFloat = false;
      if (input[j] === '.' && /[0-9]/.test(input[j + 1] ?? '')) {
        isFloat = true;
        j += 1;
        while (j < n && /[0-9]/.test(input[j])) j += 1;
      }
      if (input[j] === 'e' || input[j] === 'E') {
        isFloat = true;
        j += 1;
        if (input[j] === '+' || input[j] === '-') j += 1;
        while (j < n && /[0-9]/.test(input[j])) j += 1;
      }
      tokens.push({ kind: isFloat ? 'float' : 'int', value: input.slice(i, j) });
      i = j;
      continue;
    }
    if (/[_A-Za-z]/.test(c)) {
      let j = i;
      while (j < n && /[_A-Za-z0-9]/.test(input[j])) j += 1;
      tokens.push({ kind: 'name', value: input.slice(i, j) });
      i = j;
      continue;
    }
    throw new Error(`unexpected character ${JSON.stringify(c)} in GraphQL document`);
  }
  tokens.push({ kind: 'eof', value: '' });
  return tokens;
}

/* ------------------------------------------------------------------ */
/* Parser (AST)                                                        */
/* ------------------------------------------------------------------ */

export type ValueNode =
  | { kind: 'variable'; name: string }
  | { kind: 'string'; value: string }
  | { kind: 'int'; value: number }
  | { kind: 'float'; value: number }
  | { kind: 'boolean'; value: boolean }
  | { kind: 'null' }
  | { kind: 'enum'; value: string }
  | { kind: 'list'; values: ValueNode[] }
  | { kind: 'object'; fields: Array<{ name: string; value: ValueNode }> };

export interface FieldNode {
  kind: 'field';
  name: string;
  alias: string;
  args: Array<{ name: string; value: ValueNode }>;
  selections: SelectionNode[];
}

export interface FragmentNode {
  kind: 'fragment';
  typeCondition: string | null;
  selections: SelectionNode[];
}

export type SelectionNode = FieldNode | FragmentNode;

export interface OperationNode {
  kind: 'operation';
  operation: 'query' | 'mutation' | 'subscription';
  name: string | null;
  variables: Array<{ name: string; type: string; defaultValue: ValueNode | null }>;
  selections: SelectionNode[];
}

export interface FragmentDefNode {
  kind: 'fragmentDef';
  name: string;
  typeCondition: string;
  selections: SelectionNode[];
}

class Parser {
  private tokens: Token[];
  private pos = 0;

  constructor(input: string) {
    this.tokens = lex(input);
  }

  private peek(): Token {
    return this.tokens[this.pos];
  }

  private next(): Token {
    const t = this.tokens[this.pos];
    this.pos += 1;
    return t;
  }

  private expectPunct(value: string): void {
    const t = this.next();
    if (t.kind !== 'punct' || t.value !== value) throw new Error(`expected "${value}", got ${JSON.stringify(t.value)}`);
  }

  private expectName(): string {
    const t = this.next();
    if (t.kind !== 'name') throw new Error(`expected a name, got ${JSON.stringify(t.value)}`);
    return t.value;
  }

  private skipDirectives(): void {
    while (this.peek().kind === 'punct' && this.peek().value === '@') {
      this.next();
      this.expectName();
      if (this.peek().kind === 'punct' && this.peek().value === '(') this.parseArguments();
    }
  }

  parseDocument(): { operations: OperationNode[]; fragments: Map<string, FragmentDefNode> } {
    const operations: OperationNode[] = [];
    const fragments = new Map<string, FragmentDefNode>();
    while (this.peek().kind !== 'eof') {
      const t = this.peek();
      if (t.kind === 'punct' && t.value === '{') {
        operations.push(this.parseOperation('query', null));
      } else if (t.kind === 'name' && (t.value === 'query' || t.value === 'mutation' || t.value === 'subscription')) {
        const op = this.next().value as 'query' | 'mutation' | 'subscription';
        let name: string | null = null;
        if (this.peek().kind === 'name') name = this.next().value;
        operations.push(this.parseOperation(op, name));
      } else if (t.kind === 'name' && t.value === 'fragment') {
        this.next();
        const name = this.expectName();
        this.expectName(); // "on"
        const typeCondition = this.expectName();
        this.skipDirectives();
        const selections = this.parseSelectionSet();
        fragments.set(name, { kind: 'fragmentDef', name, typeCondition, selections });
      } else {
        throw new Error(`unexpected token ${JSON.stringify(t.value)} at the top of a GraphQL document`);
      }
    }
    return { operations, fragments };
  }

  private parseOperation(op: 'query' | 'mutation' | 'subscription', name: string | null): OperationNode {
    const variables: OperationNode['variables'] = [];
    if (this.peek().kind === 'punct' && this.peek().value === '(') {
      this.next();
      while (!(this.peek().kind === 'punct' && this.peek().value === ')')) {
        this.expectPunct('$');
        const varName = this.expectName();
        this.expectPunct(':');
        const type = this.parseType();
        let defaultValue: ValueNode | null = null;
        if (this.peek().kind === 'punct' && this.peek().value === '=') {
          this.next();
          defaultValue = this.parseValue();
        }
        this.skipDirectives();
        variables.push({ name: varName, type, defaultValue });
      }
      this.expectPunct(')');
    }
    this.skipDirectives();
    const selections = this.parseSelectionSet();
    return { kind: 'operation', operation: op, name, variables, selections };
  }

  private parseType(): string {
    let type: string;
    if (this.peek().kind === 'punct' && this.peek().value === '[') {
      this.next();
      const inner = this.parseType();
      this.expectPunct(']');
      type = `[${inner}]`;
    } else {
      type = this.expectName();
    }
    if (this.peek().kind === 'punct' && this.peek().value === '!') {
      this.next();
      type += '!';
    }
    return type;
  }

  private parseSelectionSet(): SelectionNode[] {
    this.expectPunct('{');
    const selections: SelectionNode[] = [];
    while (!(this.peek().kind === 'punct' && this.peek().value === '}')) {
      if (this.peek().kind === 'eof') throw new Error('unterminated selection set');
      selections.push(this.parseSelection());
    }
    this.expectPunct('}');
    return selections;
  }

  private parseSelection(): SelectionNode {
    if (this.peek().kind === 'spread') {
      this.next();
      if (this.peek().kind === 'name' && this.peek().value === 'on') {
        this.next();
        const typeCondition = this.expectName();
        this.skipDirectives();
        return { kind: 'fragment', typeCondition, selections: this.parseSelectionSet() };
      }
      if (this.peek().kind === 'name') {
        const name = this.next().value;
        this.skipDirectives();
        return { kind: 'fragment', typeCondition: name, selections: [] };
      }
      throw new Error('a fragment spread needs a name or "on Type"');
    }
    const first = this.expectName();
    let alias = first;
    let name = first;
    if (this.peek().kind === 'punct' && this.peek().value === ':') {
      this.next();
      name = this.expectName();
    }
    const args = this.parseArguments();
    this.skipDirectives();
    const selections =
      this.peek().kind === 'punct' && this.peek().value === '{' ? this.parseSelectionSet() : [];
    return { kind: 'field', name, alias, args, selections };
  }

  private parseArguments(): Array<{ name: string; value: ValueNode }> {
    const args: Array<{ name: string; value: ValueNode }> = [];
    if (this.peek().kind === 'punct' && this.peek().value === '(') {
      this.next();
      while (!(this.peek().kind === 'punct' && this.peek().value === ')')) {
        const name = this.expectName();
        this.expectPunct(':');
        args.push({ name, value: this.parseValue() });
      }
      this.expectPunct(')');
    }
    return args;
  }

  private parseValue(): ValueNode {
    const t = this.next();
    if (t.kind === 'punct' && t.value === '$') {
      return { kind: 'variable', name: this.expectName() };
    }
    if (t.kind === 'punct' && t.value === '[') {
      const values: ValueNode[] = [];
      while (!(this.peek().kind === 'punct' && this.peek().value === ']')) values.push(this.parseValue());
      this.expectPunct(']');
      return { kind: 'list', values };
    }
    if (t.kind === 'punct' && t.value === '{') {
      const fields: Array<{ name: string; value: ValueNode }> = [];
      while (!(this.peek().kind === 'punct' && this.peek().value === '}')) {
        const name = this.expectName();
        this.expectPunct(':');
        fields.push({ name, value: this.parseValue() });
      }
      this.expectPunct('}');
      return { kind: 'object', fields };
    }
    if (t.kind === 'string') return { kind: 'string', value: t.value };
    if (t.kind === 'int') return { kind: 'int', value: Number.parseInt(t.value, 10) };
    if (t.kind === 'float') return { kind: 'float', value: Number.parseFloat(t.value) };
    if (t.kind === 'name') {
      if (t.value === 'true') return { kind: 'boolean', value: true };
      if (t.value === 'false') return { kind: 'boolean', value: false };
      if (t.value === 'null') return { kind: 'null' };
      return { kind: 'enum', value: t.value };
    }
    throw new Error(`unexpected value token ${JSON.stringify(t.value)}`);
  }
}

/* ------------------------------------------------------------------ */
/* Schema (for validation + __type introspection)                      */
/* ------------------------------------------------------------------ */

interface TypeDef {
  kind: 'OBJECT' | 'ENUM' | 'INPUT_OBJECT' | 'SCALAR';
  fields?: string[];
  enumValues?: string[];
}

const SCHEMA: Record<string, TypeDef> = {
  Query: { kind: 'OBJECT', fields: ['viewer', 'repository', 'node', 'search', '__type', '__schema'] },
  Mutation: { kind: 'OBJECT', fields: ['createIssue', 'closeIssue', 'createPullRequest', 'mergePullRequest', 'addComment'] },
  User: { kind: 'OBJECT', fields: ['login', 'id', 'databaseId', 'name', 'email', 'avatarUrl', 'url'] },
  Repository: {
    kind: 'OBJECT',
    fields: ['id', 'databaseId', 'name', 'nameWithOwner', 'owner', 'defaultBranchRef', 'viewerPermission', 'hasIssuesEnabled', 'isPrivate', 'description', 'createdAt', 'updatedAt', 'issue', 'issues', 'pullRequests', 'pullRequest', 'labels'],
  },
  Ref: { kind: 'OBJECT', fields: ['name', 'id'] },
  Issue: { kind: 'OBJECT', fields: ['id', 'databaseId', 'number', 'title', 'state', 'body', 'author', 'createdAt', 'updatedAt', 'closedAt', 'labels', 'comments'] },
  IssueConnection: { kind: 'OBJECT', fields: ['nodes', 'edges', 'pageInfo', 'totalCount'] },
  IssueEdge: { kind: 'OBJECT', fields: ['node', 'cursor'] },
  PullRequest: {
    kind: 'OBJECT',
    fields: ['id', 'databaseId', 'number', 'title', 'body', 'state', 'merged', 'mergeable', 'headRefName', 'baseRefName', 'headRefOid', 'baseRefOid', 'mergeCommit', 'files', 'createdAt', 'updatedAt', 'closedAt', 'mergedAt', 'author'],
  },
  PullRequestConnection: { kind: 'OBJECT', fields: ['nodes', 'edges', 'pageInfo', 'totalCount'] },
  PullRequestEdge: { kind: 'OBJECT', fields: ['node', 'cursor'] },
  PullRequestChangedFile: { kind: 'OBJECT', fields: ['path', 'additions', 'deletions', 'changeType'] },
  PullRequestChangedFileConnection: { kind: 'OBJECT', fields: ['nodes', 'edges', 'pageInfo', 'totalCount'] },
  IssueComment: { kind: 'OBJECT', fields: ['id', 'databaseId', 'body', 'author', 'createdAt', 'updatedAt'] },
  IssueCommentEdge: { kind: 'OBJECT', fields: ['node', 'cursor'] },
  PageInfo: { kind: 'OBJECT', fields: ['hasNextPage', 'hasPreviousPage', 'startCursor', 'endCursor'] },
  Label: { kind: 'OBJECT', fields: ['id', 'name', 'color', 'description'] },
  LabelConnection: { kind: 'OBJECT', fields: ['nodes', 'edges', 'pageInfo', 'totalCount'] },
  LabelEdge: { kind: 'OBJECT', fields: ['node', 'cursor'] },
  IssueCommentConnection: { kind: 'OBJECT', fields: ['nodes', 'edges', 'pageInfo', 'totalCount'] },
  __Type: { kind: 'OBJECT', fields: ['name', 'kind', 'description', 'fields', 'enumValues', 'inputFields', 'interfaces', 'possibleTypes'] },
  __Field: { kind: 'OBJECT', fields: ['name', 'description', 'args', 'type', 'isDeprecated', 'deprecationReason'] },
  __EnumValue: { kind: 'OBJECT', fields: ['name', 'description', 'isDeprecated', 'deprecationReason'] },
  __Schema: { kind: 'OBJECT', fields: ['description', 'types', 'queryType', 'mutationType', 'subscriptionType', 'directives'] },
  __InputValue: { kind: 'OBJECT', fields: ['name', 'description', 'type', 'defaultValue'] },
  CreateIssuePayload: { kind: 'OBJECT', fields: ['issue', 'clientMutationId'] },
  CloseIssuePayload: { kind: 'OBJECT', fields: ['issue', 'clientMutationId'] },
  CreatePullRequestPayload: { kind: 'OBJECT', fields: ['pullRequest', 'clientMutationId'] },
  MergePullRequestPayload: { kind: 'OBJECT', fields: ['pullRequest', 'clientMutationId'] },
  AddCommentPayload: { kind: 'OBJECT', fields: ['commentEdge', 'clientMutationId'] },
  CreateIssueInput: { kind: 'INPUT_OBJECT', fields: ['repositoryId', 'title', 'body', 'clientMutationId'] },
  CloseIssueInput: { kind: 'INPUT_OBJECT', fields: ['issueId', 'clientMutationId'] },
  CreatePullRequestInput: { kind: 'INPUT_OBJECT', fields: ['repositoryId', 'baseRefName', 'headRefName', 'title', 'body', 'clientMutationId'] },
  MergePullRequestInput: { kind: 'INPUT_OBJECT', fields: ['pullRequestId', 'mergeMethod', 'clientMutationId'] },
  AddCommentInput: { kind: 'INPUT_OBJECT', fields: ['subjectId', 'body', 'clientMutationId'] },
  IssueState: { kind: 'ENUM', enumValues: ['OPEN', 'CLOSED'] },
  PullRequestState: { kind: 'ENUM', enumValues: ['OPEN', 'CLOSED', 'MERGED'] },
  MergeableState: { kind: 'ENUM', enumValues: ['MERGEABLE', 'CONFLICTING', 'UNKNOWN'] },
  PullRequestMergeMethod: { kind: 'ENUM', enumValues: ['MERGE', 'SQUASH', 'REBASE'] },
  RepositoryPermission: { kind: 'ENUM', enumValues: ['ADMIN', 'MAINTAIN', 'WRITE', 'TRIAGE', 'READ'] },
  SearchType: { kind: 'ENUM', enumValues: ['ISSUE', 'REPOSITORY', 'USER'] },
  IssueOrderField: { kind: 'ENUM', enumValues: ['CREATED_AT', 'UPDATED_AT', 'COMMENTS'] },
  OrderDirection: { kind: 'ENUM', enumValues: ['ASC', 'DESC'] },
  String: { kind: 'SCALAR' },
  Int: { kind: 'SCALAR' },
  ID: { kind: 'SCALAR' },
  Boolean: { kind: 'SCALAR' },
};

/* ------------------------------------------------------------------ */
/* Executor                                                            */
/* ------------------------------------------------------------------ */

interface Ctx {
  store: Store;
}

interface Wrapped {
  __typename: string;
  [key: string]: unknown;
}

function wrap(typename: string, data: Record<string, unknown>): Wrapped {
  return { __typename: typename, ...data };
}

type Resolver = (parent: Wrapped, args: Record<string, unknown>, ctx: Ctx) => unknown;

const RESOLVERS: Record<string, Record<string, Resolver>> = {
  Query: {
    viewer: (_p, _a, ctx) => wrap('User', { login: ctx.store.owner, id: 1000, name: ctx.store.owner, email: `${ctx.store.owner}@example.invalid` }),
    repository: (_p, args, ctx) => {
      const owner = args.owner as string;
      const name = args.name as string;
      if (owner !== ctx.store.owner) return null;
      const repo = ctx.store.getRepo(owner, name);
      if (!repo) return null;
      return wrap('Repository', { repo });
    },
    node: (_p, args, ctx) => {
      const parsed = parseGlobalId(String(args.id ?? ''));
      if (!parsed) return null;
      if (parsed.type === 'Repository') {
        const id = Number.parseInt(parsed.id, 10);
        const repo = ctx.store.repoList.find((r) => r.id === id);
        if (!repo) return null;
        return wrap('Repository', { repo });
      }
      if (parsed.type === 'Issue') {
        const found = findIssue(ctx.store, Number.parseInt(parsed.id, 10));
        if (!found) return null;
        return wrap('Issue', { repo: found.repo, issue: found.issue });
      }
      if (parsed.type === 'PullRequest') {
        const found = findPull(ctx.store, Number.parseInt(parsed.id, 10));
        if (!found) return null;
        return wrap('PullRequest', { repo: found.repo, pr: found.pr });
      }
      return null;
    },
    __type: (_p, args) => {
      const def = SCHEMA[String(args.name ?? '')];
      if (!def) return null;
      return wrap('__Type', { typeName: String(args.name), def });
    },
    __schema: () => wrap('__Schema', {}),
  },
  User: {
    login: (p) => p.login,
    id: (p) => globalId('User', String(p.id ?? p.login)),
    databaseId: (p) => p.id ?? null,
    name: (p) => p.name ?? p.login,
    email: (p) => p.email ?? null,
    avatarUrl: () => '',
    url: (p) => `/api/v3/users/${p.login}`,
  },
  Repository: {
    id: (p) => globalId('Repository', (p.repo as Repo).id),
    databaseId: (p) => (p.repo as Repo).id,
    name: (p) => (p.repo as Repo).name,
    nameWithOwner: (p) => `${(p.repo as Repo).owner}/${(p.repo as Repo).name}`,
    owner: (p, _a, ctx) => wrap('User', { login: (p.repo as Repo).owner, id: 1000, name: (p.repo as Repo).owner, email: `${(p.repo as Repo).owner}@example.invalid` }),
    defaultBranchRef: (p) => {
      const repo = p.repo as Repo;
      const tip = refTip(repo, `refs/heads/${repo.defaultBranch}`);
      if (!tip) return null;
      return wrap('Ref', { name: repo.defaultBranch });
    },
    viewerPermission: () => 'ADMIN',
    hasIssuesEnabled: () => true,
    isPrivate: (p) => (p.repo as Repo).private,
    description: (p) => (p.repo as Repo).description,
    createdAt: (p) => (p.repo as Repo).createdAt,
    updatedAt: (p) => (p.repo as Repo).updatedAt,
    issue: (p, args) => {
      const repo = p.repo as Repo;
      const issue = repo.issues.find((i) => i.number === (args.number as number));
      if (!issue) return null;
      return wrap('Issue', { repo, issue });
    },
    issues: (p, args) => {
      const repo = p.repo as Repo;
      let issues = [...repo.issues];
      const states = args.states as string[] | undefined;
      if (states && states.length > 0) {
        const wanted = new Set(states.map((s) => String(s).toLowerCase()));
        issues = issues.filter((i) => wanted.has(i.state));
      }
      const orderBy = args.orderBy as { field?: string; direction?: string } | undefined;
      const field = orderBy?.field ?? 'CREATED_AT';
      const direction = orderBy?.direction ?? 'DESC';
      issues.sort((a, b) => {
        const ka = field === 'UPDATED_AT' ? a.updatedAt : a.createdAt;
        const kb = field === 'UPDATED_AT' ? b.updatedAt : b.createdAt;
        const cmp = ka < kb ? -1 : ka > kb ? 1 : a.number - b.number;
        return direction === 'ASC' ? cmp : -cmp;
      });
      return connection(issues.map((issue) => wrap('Issue', { repo, issue })), args, 'IssueConnection');
    },
    pullRequests: (p, args) => {
      const repo = p.repo as Repo;
      let pulls = [...repo.pulls];
      const states = args.states as string[] | undefined;
      if (states && states.length > 0) {
        const wanted = new Set(states.map((s) => String(s).toUpperCase()));
        pulls = pulls.filter((pr) => wanted.has(prState(pr)));
      }
      pulls.sort((a, b) => b.number - a.number);
      return connection(pulls.map((pr) => wrap('PullRequest', { repo, pr })), args, 'PullRequestConnection');
    },
    pullRequest: (p, args) => {
      const repo = p.repo as Repo;
      const pr = repo.pulls.find((x) => x.number === (args.number as number));
      if (!pr) return null;
      return wrap('PullRequest', { repo, pr });
    },
    labels: (p, args) => {
      const repo = p.repo as Repo;
      return connection(repo.labels.map((l) => wrap('Label', { label: l })), args, 'LabelConnection');
    },
  },
  Ref: {
    name: (p) => p.name,
    id: (p) => globalId('Ref', String(p.name)),
  },
  Issue: {
    id: (p) => globalId('Issue', (p.issue as Issue).id),
    databaseId: (p) => (p.issue as Issue).id,
    number: (p) => (p.issue as Issue).number,
    title: (p) => (p.issue as Issue).title,
    state: (p) => (p.issue as Issue).state.toUpperCase(),
    body: (p) => (p.issue as Issue).body,
    author: (p, _a, ctx) => wrap('User', { login: (p.issue as Issue).user, id: 1000, name: (p.issue as Issue).user, email: `${(p.issue as Issue).user}@example.invalid` }),
    createdAt: (p) => (p.issue as Issue).createdAt,
    updatedAt: (p) => (p.issue as Issue).updatedAt,
    closedAt: (p) => (p.issue as Issue).closedAt,
    labels: (p, args) => {
      const repo = p.repo as Repo;
      const issue = p.issue as Issue;
      const labels = issue.labels
        .map((name) => repo.labels.find((l) => l.name === name))
        .filter((l): l is NonNullable<typeof l> => l !== undefined);
      return connection(labels.map((l) => wrap('Label', { label: l })), args, 'LabelConnection');
    },
    comments: (p, args) => {
      const issue = p.issue as Issue;
      return connection(issue.comments.map((c) => wrap('IssueComment', { comment: c, repo: p.repo })), args, 'IssueCommentConnection');
    },
  },
  IssueConnection: {
    nodes: (p) => (p.items as Wrapped[]).slice((p.offset as number), (p.offset as number) + (p.limit as number)),
    edges: (p) => {
      const items = (p.items as Wrapped[]).slice((p.offset as number), (p.offset as number) + (p.limit as number));
      return items.map((node, i) => wrap('IssueEdge', { node, cursor: encodeCursor((p.offset as number) + i) }));
    },
    pageInfo: (p) => pageInfo(p),
    totalCount: (p) => (p.items as Wrapped[]).length,
  },
  IssueEdge: {
    node: (p) => p.node,
    cursor: (p) => p.cursor,
  },
  PullRequest: {
    id: (p) => globalId('PullRequest', (p.pr as PullRequest).id),
    databaseId: (p) => (p.pr as PullRequest).id,
    number: (p) => (p.pr as PullRequest).number,
    title: (p) => (p.pr as PullRequest).title,
    body: (p) => (p.pr as PullRequest).body,
    state: (p) => prState(p.pr as PullRequest),
    merged: (p) => (p.pr as PullRequest).merged,
    headRefName: (p) => (p.pr as PullRequest).headRef,
    baseRefName: (p) => (p.pr as PullRequest).baseRef,
    headRefOid: (p) => (p.pr as PullRequest).headSha,
    baseRefOid: (p) => (p.pr as PullRequest).baseSha,
    mergeable: (p, _a, ctx) => prMergeable(ctx.store, p.repo as Repo, p.pr as PullRequest),
    files: (p, args) => {
      const files = pullRequestFiles(p.repo as Repo, p.pr as PullRequest);
      return connection(files.map((f) => wrap('PullRequestChangedFile', { file: f })), args, 'PullRequestChangedFileConnection');
    },
    createdAt: (p) => (p.pr as PullRequest).createdAt,
    updatedAt: (p) => (p.pr as PullRequest).updatedAt,
    closedAt: (p) => (p.pr as PullRequest).closedAt,
    mergedAt: (p) => (p.pr as PullRequest).mergedAt,
    author: (p, _a, ctx) => wrap('User', { login: ctx.store.owner, id: 1000, name: ctx.store.owner, email: `${ctx.store.owner}@example.invalid` }),
  },
  PullRequestConnection: {
    nodes: (p) => (p.items as Wrapped[]).slice((p.offset as number), (p.offset as number) + (p.limit as number)),
    edges: (p) => {
      const items = (p.items as Wrapped[]).slice((p.offset as number), (p.offset as number) + (p.limit as number));
      return items.map((node, i) => wrap('PullRequestEdge', { node, cursor: encodeCursor((p.offset as number) + i) }));
    },
    pageInfo: (p) => pageInfo(p),
    totalCount: (p) => (p.items as Wrapped[]).length,
  },
  PullRequestEdge: {
    node: (p) => p.node,
    cursor: (p) => p.cursor,
  },
  PullRequestChangedFile: {
    path: (p) => (p.file as { filename: string }).filename,
    additions: (p) => (p.file as { additions: number }).additions ?? 0,
    deletions: (p) => (p.file as { deletions: number }).deletions ?? 0,
    changeType: (p) => String((p.file as { status: string }).status ?? 'modified').toUpperCase(),
  },
  PullRequestChangedFileConnection: {
    nodes: (p) => (p.items as Wrapped[]).slice((p.offset as number), (p.offset as number) + (p.limit as number)),
    pageInfo: (p) => pageInfo(p),
    totalCount: (p) => (p.items as Wrapped[]).length,
  },
  IssueComment: {
    id: (p) => globalId('IssueComment', (p.comment as { id: number }).id),
    databaseId: (p) => (p.comment as { id: number }).id,
    body: (p) => (p.comment as { body: string }).body,
    author: (p, _a, ctx) => wrap('User', { login: (p.comment as { user: string }).user, id: 1000, name: (p.comment as { user: string }).user, email: `${(p.comment as { user: string }).user}@example.invalid` }),
    createdAt: (p) => (p.comment as { createdAt: string }).createdAt,
    updatedAt: (p) => (p.comment as { updatedAt: string }).updatedAt,
  },
  IssueCommentEdge: {
    node: (p) => p.node,
    cursor: (p) => p.cursor,
  },
  LabelConnection: {
    nodes: (p) => (p.items as Wrapped[]).slice((p.offset as number), (p.offset as number) + (p.limit as number)),
    edges: (p) => {
      const items = (p.items as Wrapped[]).slice((p.offset as number), (p.offset as number) + (p.limit as number));
      return items.map((node, i) => wrap('LabelEdge', { node, cursor: encodeCursor((p.offset as number) + i) }));
    },
    pageInfo: (p) => pageInfo(p),
    totalCount: (p) => (p.items as Wrapped[]).length,
  },
  LabelEdge: {
    node: (p) => p.node,
    cursor: (p) => p.cursor,
  },
  IssueCommentConnection: {
    nodes: (p) => (p.items as Wrapped[]).slice((p.offset as number), (p.offset as number) + (p.limit as number)),
    edges: (p) => {
      const items = (p.items as Wrapped[]).slice((p.offset as number), (p.offset as number) + (p.limit as number));
      return items.map((node, i) => wrap('IssueCommentEdge', { node, cursor: encodeCursor((p.offset as number) + i) }));
    },
    pageInfo: (p) => pageInfo(p),
    totalCount: (p) => (p.items as Wrapped[]).length,
  },
  PageInfo: {
    hasNextPage: (p) => (p.hasNextPage as boolean) ?? false,
    hasPreviousPage: (p) => (p.hasPreviousPage as boolean) ?? false,
    startCursor: (p) => (p.startCursor as string | null) ?? null,
    endCursor: (p) => (p.endCursor as string | null) ?? null,
  },
  Label: {
    id: (p) => globalId('Label', (p.label as { id: number }).id),
    name: (p) => (p.label as { name: string }).name,
    color: (p) => (p.label as { color: string }).color,
    description: (p) => (p.label as { description: string }).description,
  },
  __Type: {
    name: (p) => p.typeName,
    kind: (p) => (p.def as TypeDef).kind,
    description: () => null,
    fields: (p, args) => {
      const def = p.def as TypeDef;
      const names = def.fields ?? [];
      return names.map((name) => wrap('__Field', { name }));
    },
    enumValues: (p, _args) => {
      const def = p.def as TypeDef;
      return (def.enumValues ?? []).map((name) => wrap('__EnumValue', { name }));
    },
    inputFields: () => [],
    interfaces: () => [],
    possibleTypes: () => [],
  },
  __Field: {
    name: (p) => p.name,
    description: () => null,
    args: () => [],
    type: () => wrap('__Type', { typeName: 'String', def: SCHEMA.String! }),
    isDeprecated: () => false,
    deprecationReason: () => null,
  },
  __EnumValue: {
    name: (p) => p.name,
    description: () => null,
    isDeprecated: () => false,
    deprecationReason: () => null,
  },
  __Schema: {
    description: () => null,
    types: () => Object.keys(SCHEMA).map((name) => wrap('__Type', { typeName: name, def: SCHEMA[name]! })),
    queryType: () => wrap('__Type', { typeName: 'Query', def: SCHEMA.Query! }),
    mutationType: () => wrap('__Type', { typeName: 'Mutation', def: SCHEMA.Mutation! }),
    subscriptionType: () => null,
    directives: () => [],
  },
  __InputValue: {
    name: (p) => p.name,
    description: () => null,
    type: () => wrap('__Type', { typeName: 'String', def: SCHEMA.String! }),
    defaultValue: () => null,
  },
  Mutation: {
    createIssue: (_p, args, ctx) => {
      const input = args.input as { repositoryId?: string; title?: string; body?: string; clientMutationId?: string };
      const parsed = parseGlobalId(String(input.repositoryId ?? ''));
      if (!parsed || parsed.type !== 'Repository') throw new Error('createIssue: repositoryId is not a repository');
      const repo = ctx.store.repoList.find((r) => r.id === Number.parseInt(parsed.id, 10));
      if (!repo) throw new Error('createIssue: repository not found');
      const issue = ctx.store.createIssue(repo, input.title ?? '', input.body ?? '', []);
      return wrap('CreateIssuePayload', { issue: wrap('Issue', { repo, issue }), clientMutationId: input.clientMutationId ?? null });
    },
    closeIssue: (_p, args, ctx) => {
      const input = args.input as { issueId?: string; clientMutationId?: string };
      const parsed = parseGlobalId(String(input.issueId ?? ''));
      if (!parsed || parsed.type !== 'Issue') throw new Error('closeIssue: issueId is not an issue');
      const found = findIssue(ctx.store, Number.parseInt(parsed.id, 10));
      if (!found) throw new Error('closeIssue: issue not found');
      found.issue.state = 'closed';
      found.issue.closedAt = tick();
      found.issue.updatedAt = tick();
      return wrap('CloseIssuePayload', { issue: wrap('Issue', { repo: found.repo, issue: found.issue }), clientMutationId: input.clientMutationId ?? null });
    },
    createPullRequest: (_p, args, ctx) => {
      const input = args.input as { repositoryId?: string; baseRefName?: string; headRefName?: string; title?: string; body?: string; clientMutationId?: string };
      const parsed = parseGlobalId(String(input.repositoryId ?? ''));
      if (!parsed || parsed.type !== 'Repository') throw new Error('createPullRequest: repositoryId is not a repository');
      const repo = ctx.store.repoList.find((r) => r.id === Number.parseInt(parsed.id, 10));
      if (!repo) throw new Error('createPullRequest: repository not found');
      const headSha = refTip(repo, `refs/heads/${input.headRefName ?? ''}`);
      const baseSha = refTip(repo, `refs/heads/${input.baseRefName ?? ''}`);
      if (!headSha || !baseSha) throw new Error('createPullRequest: head or base branch not found');
      const pr = ctx.store.createPull(repo, input.title ?? '', input.body ?? '', input.headRefName ?? '', input.baseRefName ?? '', headSha, baseSha);
      return wrap('CreatePullRequestPayload', { pullRequest: wrap('PullRequest', { repo, pr }), clientMutationId: input.clientMutationId ?? null });
    },
    mergePullRequest: (_p, args, ctx) => {
      const input = args.input as { pullRequestId?: string; mergeMethod?: string; clientMutationId?: string };
      const parsed = parseGlobalId(String(input.pullRequestId ?? ''));
      if (!parsed || parsed.type !== 'PullRequest') throw new Error('mergePullRequest: pullRequestId is not a pull request');
      const found = findPull(ctx.store, Number.parseInt(parsed.id, 10));
      if (!found) throw new Error('mergePullRequest: pull request not found');
      const { repo, pr } = found;
      const result = mergePullRequest(ctx.store, repo, pr);
      if (!result.ok) throw new Error(`mergePullRequest: ${result.reason}`);
      return wrap('MergePullRequestPayload', { pullRequest: wrap('PullRequest', { repo, pr }), clientMutationId: input.clientMutationId ?? null });
    },
    addComment: (_p, args, ctx) => {
      const input = args.input as { subjectId?: string; body?: string; clientMutationId?: string };
      const parsed = parseGlobalId(String(input.subjectId ?? ''));
      if (!parsed || parsed.type !== 'Issue') throw new Error('addComment: subjectId is not an issue');
      const found = findIssue(ctx.store, Number.parseInt(parsed.id, 10));
      if (!found) throw new Error('addComment: issue not found');
      const comment = {
        id: ctx.store.nextId(),
        body: input.body ?? '',
        user: ctx.store.owner,
        createdAt: tick(),
        updatedAt: tick(),
      };
      found.issue.comments.push(comment);
      found.issue.updatedAt = tick();
      const node = wrap('IssueComment', { comment, repo: found.repo });
      return wrap('AddCommentPayload', { commentEdge: wrap('IssueCommentEdge', { node, cursor: encodeCursor(0) }), clientMutationId: input.clientMutationId ?? null });
    },
  },
  CreateIssuePayload: {
    issue: (p) => p.issue,
    clientMutationId: (p) => p.clientMutationId,
  },
  CloseIssuePayload: {
    issue: (p) => p.issue,
    clientMutationId: (p) => p.clientMutationId,
  },
  CreatePullRequestPayload: {
    pullRequest: (p) => p.pullRequest,
    clientMutationId: (p) => p.clientMutationId,
  },
  MergePullRequestPayload: {
    pullRequest: (p) => p.pullRequest,
    clientMutationId: (p) => p.clientMutationId,
  },
  AddCommentPayload: {
    commentEdge: (p) => p.commentEdge,
    clientMutationId: (p) => p.clientMutationId,
  },
  IssueCommentEdge: {
    node: (p) => p.node,
    cursor: (p) => p.cursor,
  },
};

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function prState(pr: PullRequest): string {
  if (pr.merged) return 'MERGED';
  return pr.state === 'open' ? 'OPEN' : 'CLOSED';
}

function prMergeable(store: Store, repo: Repo, pr: PullRequest): string {
  if (pr.state !== 'open') return 'UNKNOWN';
  const baseTip = refTip(repo, `refs/heads/${pr.baseRef}`);
  const headTip = refTip(repo, `refs/heads/${pr.headRef}`);
  if (!baseTip || !headTip) return 'UNKNOWN';
  const base = mergeBase(repo, baseTip, headTip);
  if (!base) return 'UNKNOWN';
  const { conflicts } = mergeTrees(repo, commitTree(repo, base), commitTree(repo, baseTip), commitTree(repo, headTip));
  return conflicts.length > 0 ? 'CONFLICTING' : 'MERGEABLE';
}

function findIssue(store: Store, id: number): { repo: Repo; issue: Issue } | null {
  for (const repo of store.repoList) {
    const issue = repo.issues.find((i) => i.id === id);
    if (issue) return { repo, issue };
  }
  return null;
}

function findPull(store: Store, id: number): { repo: Repo; pr: PullRequest } | null {
  for (const repo of store.repoList) {
    const pr = repo.pulls.find((p) => p.id === id);
    if (pr) return { repo, pr };
  }
  return null;
}

/** The changed files of a pull request (merge-base -> head). */
function pullRequestFiles(repo: Repo, pr: PullRequest): Array<Record<string, unknown>> {
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
  const out: Array<Record<string, unknown>> = [];
  for (const p of [...new Set([...baseFiles.keys(), ...headFiles.keys()])].sort()) {
    const b = baseFiles.get(p);
    const h = headFiles.get(p);
    if (b === h) continue;
    out.push({ filename: p, status: b === undefined ? 'added' : h === undefined ? 'removed' : 'modified', additions: 0, deletions: 0 });
  }
  return out;
}

function encodeCursor(index: number): string {
  return Buffer.from(`cursor:${index}`, 'utf8').toString('base64');
}

function decodeCursor(cursor: string | null | undefined): number {
  if (!cursor) return 0;
  try {
    const text = Buffer.from(cursor, 'base64').toString('utf8');
    const m = /^cursor:(\d+)$/.exec(text);
    if (m) return Number.parseInt(m[1], 10) + 1;
  } catch {
    // fall through
  }
  return 0;
}

/** A connection wrapper with cursor pagination (first/after). */
function connection(items: Wrapped[], args: Record<string, unknown>, type: string): Wrapped {
  const first = typeof args.first === 'number' ? args.first : items.length;
  const offset = decodeCursor(args.after as string | null | undefined);
  const limit = Math.max(0, Math.min(first, items.length));
  const page = items.slice(offset, offset + limit);
  const hasNextPage = offset + limit < items.length;
  const endCursor = page.length > 0 ? encodeCursor(offset + page.length - 1) : null;
  const startCursor = page.length > 0 ? encodeCursor(offset) : null;
  return wrap(type, {
    items,
    offset,
    limit,
    hasNextPage,
    hasPreviousPage: offset > 0,
    startCursor,
    endCursor,
  });
}

function pageInfo(p: Wrapped): Wrapped {
  return wrap('PageInfo', {
    hasNextPage: p.hasNextPage,
    hasPreviousPage: p.hasPreviousPage,
    startCursor: p.startCursor,
    endCursor: p.endCursor,
  });
}

/* ------------------------------------------------------------------ */
/* Execution                                                           */
/* ------------------------------------------------------------------ */

interface ExecError {
  message: string;
  path?: (string | number)[];
}

function resolveValue(node: ValueNode, variables: Record<string, unknown>): unknown {
  switch (node.kind) {
    case 'variable': {
      const v = variables[node.name];
      return v === undefined ? null : v;
    }
    case 'string':
      return node.value;
    case 'int':
    case 'float':
      return node.value;
    case 'boolean':
      return node.value;
    case 'null':
      return null;
    case 'enum':
      return node.value;
    case 'list':
      return node.values.map((v) => resolveValue(v, variables));
    case 'object': {
      const out: Record<string, unknown> = {};
      for (const f of node.fields) out[f.name] = resolveValue(f.value, variables);
      return out;
    }
  }
}

class Executor {
  private ctx: Ctx;
  private fragments: Map<string, FragmentDefNode>;
  private errors: ExecError[] = [];

  constructor(ctx: Ctx, fragments: Map<string, FragmentDefNode>) {
    this.ctx = ctx;
    this.fragments = fragments;
  }

  executeOperation(op: OperationNode, variables: Record<string, unknown>): Record<string, unknown> {
    const rootType = op.operation === 'mutation' ? 'Mutation' : 'Query';
    const root = wrap(rootType, {});
    const out: Record<string, unknown> = {};
    for (const sel of op.selections) {
      this.executeSelection(sel, root, rootType, variables, out, []);
    }
    return out;
  }

  private executeSelection(
    sel: SelectionNode,
    parent: Wrapped,
    parentType: string,
    variables: Record<string, unknown>,
    out: Record<string, unknown>,
    path: (string | number)[],
  ): void {
    if (sel.kind === 'fragment') {
      // A named fragment spread references a definition; an inline fragment
      // carries its selections directly.
      let selections = sel.selections;
      let typeCondition = sel.typeCondition;
      if (selections.length === 0 && typeCondition) {
        const def = this.fragments.get(typeCondition);
        if (!def) {
          this.errors.push({ message: `Unknown fragment "${typeCondition}"`, path });
          return;
        }
        selections = def.selections;
        typeCondition = def.typeCondition;
      }
      if (typeCondition && typeCondition !== parentType) return;
      for (const s of selections) this.executeSelection(s, parent, parentType, variables, out, path);
      return;
    }
    const fieldDef = SCHEMA[parentType];
    if (!fieldDef || !fieldDef.fields || !fieldDef.fields.includes(sel.name)) {
      this.errors.push({
        message: `Cannot query field "${sel.name}" on type "${parentType}"`,
        path: [...path, sel.alias],
      });
      return;
    }
    const typeResolvers = RESOLVERS[parentType];
    const resolver = typeResolvers?.[sel.name];
    if (!resolver) {
      this.errors.push({ message: `No resolver for field "${sel.name}" on type "${parentType}"`, path: [...path, sel.alias] });
      return;
    }
    const args: Record<string, unknown> = {};
    for (const a of sel.args) args[a.name] = resolveValue(a.value, variables);
    let value: unknown;
    try {
      value = resolver(parent, args, this.ctx);
    } catch (err) {
      this.errors.push({
        message: err instanceof Error ? err.message : String(err),
        path: [...path, sel.alias],
      });
      out[sel.alias] = null;
      return;
    }
    out[sel.alias] = this.completeValue(value, sel, variables, [...path, sel.alias]);
  }

  private completeValue(value: unknown, field: FieldNode, variables: Record<string, unknown>, path: (string | number)[]): unknown {
    if (value === null || value === undefined) return null;
    if (field.selections.length === 0) return value;
    if (Array.isArray(value)) {
      return value.map((item, i) => this.completeObject(item as Wrapped, field.selections, variables, [...path, i]));
    }
    if (typeof value === 'object') {
      return this.completeObject(value as Wrapped, field.selections, variables, path);
    }
    return value;
  }

  private completeObject(value: Wrapped, selections: SelectionNode[], variables: Record<string, unknown>, path: (string | number)[]): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    const type = value.__typename;
    for (const sel of selections) {
      this.executeSelection(sel, value, type, variables, out, path);
    }
    return out;
  }
}

export interface GraphQLRequest {
  query: string;
  variables?: Record<string, unknown>;
  operationName?: string;
}

/** Execute one GraphQL operation; returns the {data, errors} envelope. */
export function executeGraphQL(store: Store, req: GraphQLRequest): { data: Record<string, unknown> | null; errors?: ExecError[] } {
  let doc: { operations: OperationNode[]; fragments: Map<string, FragmentDefNode> };
  try {
    doc = new Parser(req.query).parseDocument();
  } catch (err) {
    return { data: null, errors: [{ message: err instanceof Error ? err.message : String(err) }] };
  }
  if (doc.operations.length === 0) {
    return { data: null, errors: [{ message: 'No operation found in the document' }] };
  }
  let op = doc.operations[0];
  if (req.operationName) {
    const named = doc.operations.find((o) => o.name === req.operationName);
    if (named) op = named;
  }
  const variables: Record<string, unknown> = { ...(req.variables ?? {}) };
  for (const v of op.variables) {
    if (!(v.name in variables) && v.defaultValue) variables[v.name] = resolveValue(v.defaultValue, {});
  }
  const executor = new Executor({ store }, doc.fragments);
  const data = executor.executeOperation(op, variables);
  const errors = (executor as unknown as { errors: ExecError[] }).errors;
  if (errors.length > 0) return { data, errors };
  return { data };
}
