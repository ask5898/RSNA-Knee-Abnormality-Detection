"""Scope-aware rewrite of global names: NAME -> C.NAME, only where the name refers
to the notebook-global (now config) value, never a parameter or local of the
same name. Comments, strings and formatting are untouched: edits are spliced in
by exact source position."""
import ast


class _Finder(ast.NodeVisitor):
    def __init__(self, names):
        self.names = set(names)
        self.scopes = [None]          # None = module scope: every name there is global
        self.hits = []                # (lineno, col_offset, end_col_offset)
        self.global_lines = []        # lines of `global X` statements that name a setting

    # --- scope bookkeeping -------------------------------------------------------
    @staticmethod
    def _locals(body_nodes, args=None):
        local, declared_global = set(), set()
        if args is not None:
            for a in args.posonlyargs + args.args + args.kwonlyargs:
                local.add(a.arg)
            if args.vararg:
                local.add(args.vararg.arg)
            if args.kwarg:
                local.add(args.kwarg.arg)
        stack = list(body_nodes)
        while stack:
            n = stack.pop()
            if isinstance(n, ast.Global):
                declared_global.update(n.names)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local.add(n.name)
                continue                      # its body is its own scope
            if isinstance(n, ast.Lambda):
                continue
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                local.add(n.id)
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    local.add((a.asname or a.name).split(".")[0])
            if isinstance(n, ast.ExceptHandler) and n.name:
                local.add(n.name)
            stack.extend(ast.iter_child_nodes(n))
        return local - declared_global

    def _is_global(self, name):
        for scope in reversed(self.scopes):
            if scope is None:
                return True
            if name in scope:
                return False
        return True

    def _visit_function(self, node):
        for d in node.decorator_list:
            self.visit(d)
        for d in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(d)                     # defaults are evaluated in the enclosing scope
        if getattr(node, "returns", None):
            self.visit(node.returns)
        self.scopes.append(self._locals(node.body, node.args))
        for stmt in node.body:
            self.visit(stmt)
        self.scopes.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Lambda(self, node):
        for d in node.args.defaults:
            self.visit(d)
        self.scopes.append(self._locals([node.body], node.args))
        self.visit(node.body)
        self.scopes.pop()

    def visit_ClassDef(self, node):
        for d in node.decorator_list + node.bases + [k.value for k in node.keywords]:
            self.visit(d)
        self.scopes.append(self._locals(node.body))
        for stmt in node.body:
            self.visit(stmt)
        self.scopes.pop()

    def _visit_comp(self, node):
        targets = set()
        for g in node.generators:
            for n in ast.walk(g.target):
                if isinstance(n, ast.Name):
                    targets.add(n.id)
        self.scopes.append(targets)
        self.generic_visit(node)
        self.scopes.pop()

    visit_ListComp = visit_SetComp = visit_GeneratorExp = visit_DictComp = _visit_comp

    def visit_Global(self, node):
        if self.names & set(node.names):
            self.global_lines.append(node.lineno)

    def visit_Name(self, node):
        if node.id in self.names and self._is_global(node.id):
            self.hits.append((node.lineno, node.col_offset, node.end_col_offset))


def rewrite(src, names, prefix="C."):
    """Returns (new_source, n_replaced, global_statement_lines)."""
    tree = ast.parse(src)
    f = _Finder(names)
    f.visit(tree)
    lines = src.splitlines(keepends=True)
    starts, pos = [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line)

    def char_col(lineno, byte_col):          # ast offsets are UTF-8 bytes
        return len(lines[lineno - 1].encode("utf-8")[:byte_col].decode("utf-8"))

    out = src
    for lineno, c0, c1 in sorted(set(f.hits), reverse=True):
        a = starts[lineno - 1] + char_col(lineno, c0)
        b = starts[lineno - 1] + char_col(lineno, c1)
        assert out[a:b] in names, (lineno, out[a:b])
        out = out[:a] + prefix + out[a:b] + out[b:]
    return out, len(set(f.hits)), f.global_lines
