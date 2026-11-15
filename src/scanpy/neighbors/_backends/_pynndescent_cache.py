"""Make pynndescent's dense NN-descent disk-cacheable for its built-in metrics.

pynndescent passes the distance function into its kernels as an argument. A jitted function
argument makes numba's cache key differ per process, so the whole NN-descent chain is recompiled
in every session (10–35 s). Here we rebuild that chain with the distance bound as a module global
(and the thread count passed in rather than read inside the kernels), write it to a real file so
numba can cache it, and route `pynndescent_.nn_descent` to it. The compiled code is the same, so
results are bit-identical; unknown metrics and pynndescent versions use the stock path.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import os
import sys
from functools import cache
from pathlib import Path

import numba
from packaging.version import Version

from ..._compat import pkg_version

# functions that take `dist`, per module; everything they call without `dist` is fine as is
_CHAIN = {
    "pynndescent.pynndescent_": (
        "generate_leaf_updates",
        "init_rp_tree",
        "init_random",
        "process_candidates",
        "nn_descent_internal",
        "nn_descent",
    ),
    "pynndescent.utils": ("generate_graph_update_array",),
}
_NEED_THREADS = {"nn_descent", "nn_descent_internal"}
_TESTED = {Version("0.7.0")}  # add versions only after checking bit-identity


class _Rewrite(ast.NodeTransformer):
    """Drop the `dist` parameter/arguments, replace `numba.get_num_threads()` with `n_threads_`."""

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        a = node.args
        names = [x.arg for x in a.args]
        if "dist" in names:
            i = names.index("dist")
            n_no_default = len(a.args) - len(a.defaults)
            del a.args[i]
            if i >= n_no_default:
                del a.defaults[i - n_no_default]
        if node.name in _NEED_THREADS:
            a.args.append(ast.arg("n_threads_"))
            a.defaults.append(ast.Constant(None))
        node.decorator_list = []
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "get_num_threads":
            return ast.Name("n_threads_", ast.Load())
        if isinstance(f, ast.Name) and any(f.id in fns for fns in _CHAIN.values()):
            node.args = [
                x for x in node.args if not (isinstance(x, ast.Name) and x.id == "dist")
            ]
            node.keywords = [k for k in node.keywords if k.arg != "dist"]
            if f.id in _NEED_THREADS:
                node.keywords.append(
                    ast.keyword("n_threads_", ast.Name("n_threads_", ast.Load()))
                )
        return node


@cache
def _stock() -> dict:
    """Get the upstream dispatchers, captured before `enable` replaces `nn_descent`."""
    return {
        fn: getattr(importlib.import_module(m), fn)
        for m, fns in _CHAIN.items()
        for fn in fns
    }


def _cache_dir() -> Path:
    base = (
        numba.config.CACHE_DIR
        or os.environ.get("XDG_CACHE_HOME")
        or Path.home() / ".cache"
    )
    return Path(base) / "scanpy_pynndescent"


_REWRITE_VERSION = "1"  # bump when `_Rewrite` or the generated header changes


def _source(dist_name: str) -> str:
    """Build the rewritten chain as module source, with `dist = pynndescent.distances.<dist_name>`."""
    parts = [
        "import importlib as _il\n",
        *(
            f"globals().update({{k: v for k, v in vars(_il.import_module({m!r})).items() if not k.startswith('__')}})\n"
            for m in _CHAIN
        ),
        f"dist = pynnd_dist.{dist_name}\n",
    ]
    for d in _stock().values():
        tree = _Rewrite().visit(ast.parse(inspect.getsource(d.py_func)))
        parts.append(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    return "\n".join(parts)


@cache
def _build(dist_name: str):
    """Write (once) and import the rewritten chain with `dist = pynndescent.distances.<dist_name>`."""
    origs = _stock()
    # key the generated file on the upstream sources, so an existing one is found
    # without regenerating (parsing and unparsing) the rewritten source every session
    key = hashlib.sha1(f"{_REWRITE_VERSION}:{dist_name}".encode())
    for m in _CHAIN:
        key.update(Path(inspect.getfile(importlib.import_module(m))).read_bytes())
    path = _cache_dir() / f"nnd_{dist_name}_{key.hexdigest()[:12]}.py"
    if not path.exists():
        src = _source(dist_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(src)
        # another process may have written it meanwhile; replacing an identical file would
        # change its mtime and invalidate the numba cache compiled from it
        if path.exists() and path.read_text() == src:
            tmp.unlink()
        else:
            tmp.replace(path)  # atomic
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    for fn, d in origs.items():  # same numba options as upstream, plus the disk cache
        opts = {k: v for k, v in d.targetoptions.items() if k != "nopython"}
        setattr(
            mod,
            fn,
            numba.njit(**opts, cache=True, locals=d.locals)(getattr(mod, fn)),  # noqa: TID251 (must reproduce pynndescent's own numba options exactly)
        )
    return mod.nn_descent


def enable() -> bool:
    """Route dense NNDescent builds through the cacheable chain. Returns whether it was applied."""
    try:
        import pynndescent.distances as pd
        import pynndescent.pynndescent_ as p
    except ImportError:
        return False
    if getattr(p.nn_descent, "_scanpy_cached", False):
        return True
    if pkg_version("pynndescent") not in _TESTED:
        return False
    stock = _stock()["nn_descent"]
    names = {
        id(v): k
        for k, v in vars(pd).items()
        if isinstance(v, numba.core.registry.CPUDispatcher)
    }
    sig = inspect.signature(stock.py_func)

    def nn_descent(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        kw = dict(bound.arguments)
        name = names.get(id(kw.pop("dist")))
        if name is None:
            return stock(*args, **kwargs)
        try:
            return _build(name)(**kw, n_threads_=numba.get_num_threads())
        # rewritten source doesn't compile against this build, the cache dir isn't writable,
        # or numba finds no writable cache locator for the generated module (RuntimeError)
        except (numba.core.errors.NumbaError, OSError, RuntimeError, SyntaxError):
            return stock(*args, **kwargs)

    nn_descent._scanpy_cached = True
    p.nn_descent = nn_descent
    return True
