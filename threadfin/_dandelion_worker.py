"""Dandelion's clone calling in a new Python process with a fixed string-hash seed (internal; not public API).

Why. ``dandelion.tl.find_clones`` (sc-dandelion 0.5.x) iterates over Python sets of sequences and of V/J groups, so
which cells form a clone, and every clone id, depend on Python's string-hash seed. An interpreter's seed is fixed
when it starts (the ``PYTHONHASHSEED`` environment variable) and cannot be changed afterwards. When the calling
session was not started with the seed that :func:`threadfin.define_clones_dandelion` needs (0 by default), the
Dandelion step therefore runs in a new interpreter that is.

How. :func:`run` pickles the job into a private temporary directory and starts::

    <this Python> [this interpreter's options] -c BOOTSTRAP <threadfin directory> <job file> <result file>

with ``PYTHONHASHSEED`` set, through :mod:`subprocess`. :mod:`multiprocessing` is not used: its "spawn" start
method re-imports the caller's ``__main__`` and fails in scripts without an ``if __name__ == "__main__":`` guard.
BOOTSTRAP registers the caller's ``threadfin`` directory as the package, without running ``threadfin/__init__.py``,
so that the child runs the caller's own copy of this module and of :mod:`threadfin.dandelion` whatever its own
``sys.path`` holds, and imports nothing else before :func:`main` has replaced its ``sys.path`` with the caller's:
pandas, Dandelion and their dependencies then come from the same places as in the caller. (Starting Python and
importing Dandelion takes about 2 s; Dandelion imports most of the scientific stack itself.)

What comes back. The return value of :func:`threadfin.dandelion._call_clones`, the progress lines it produced, and
the warnings raised while it ran (issued again in the caller); or the child's exception, raised again in the caller
with the child's traceback as its cause.

Only the standard library is imported at module level: the child imports this module before it has the caller's
``sys.path``.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import traceback
import warnings

#: run as ``python -c BOOTSTRAP <threadfin directory> <job file> <result file>``
BOOTSTRAP = (
    "import sys, types\n"
    "package = types.ModuleType('threadfin')\n"
    "package.__path__ = [sys.argv[1]]\n"
    "sys.modules['threadfin'] = package\n"
    "from threadfin._dandelion_worker import main\n"
    "sys.exit(main(sys.argv[2], sys.argv[3]))\n"
)

_PROTOCOL = pickle.HIGHEST_PROTOCOL

#: registry for warnings issued again in the caller, so that the "default" action shows each one once per session,
#: as it would have if Dandelion had run in the caller's process
_WARNING_REGISTRY: dict = {}


class ChildProcessTraceback(Exception):
    """The traceback of an exception raised in the child process; it is the cause of the exception raised again in
    the caller, so both tracebacks are shown."""


# ----------------------------------------------------------------------------------------------- caller's side

def _interpreter_options() -> list[str]:
    """This interpreter's command-line options (-O, -B, -W, -X ...), for the child. -E and -I are left out: they
    would make the child ignore PYTHONHASHSEED (:func:`_environment` drops the other PYTHON* variables instead)."""
    try:
        # what multiprocessing uses for the same purpose
        from subprocess import _args_from_interpreter_flags

        options = [o for o in _args_from_interpreter_flags() if o not in ("-E", "-I")]
    except Exception:  # noqa: BLE001 - a private helper; without it the child runs with default options
        options = []
    if sys.flags.no_user_site and "-s" not in options:
        options.append("-s")
    return options


def _environment(seed: int) -> dict:
    env = dict(os.environ)
    if sys.flags.ignore_environment:          # -E / -I: this interpreter ignores the PYTHON* variables; so does the child
        env = {k: v for k, v in env.items() if not k.upper().startswith("PYTHON")}
    env.pop("PYTHONINSPECT", None)            # would leave the child waiting at an interactive prompt
    env["PYTHONHASHSEED"] = str(seed)
    env["MPLBACKEND"] = "Agg"                 # nothing is drawn; a notebook's inline backend would need IPython
    return env


def _caller_sys_path() -> list[str]:
    """``sys.path`` with relative entries ('' is the working directory) made absolute, since the child runs in the
    temporary directory."""
    try:
        cwd = os.getcwd()
    except OSError:                           # the working directory was deleted
        cwd = None
    paths = []
    for p in sys.path:
        if not isinstance(p, str):
            continue
        if not os.path.isabs(p):
            if cwd is None:
                continue
            p = os.path.abspath(os.path.join(cwd, p))
        paths.append(p)
    return paths


def _tail(data: bytes, n_lines: int = 25) -> str:
    text = (data or b"").decode(errors="replace").replace("\r", "\n").strip()
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n_lines:])


def _raise_again(result: dict):
    exc = result.get("exception")
    if not isinstance(exc, BaseException):
        exc = RuntimeError(f"{result['type']}: {result['message']}")
    raise exc from ChildProcessTraceback(
        "raised in the Python process that called the clones with Dandelion:\n" + result["traceback"])


def run(job: dict, *, seed: int) -> dict:
    """Run ``threadfin.dandelion._call_clones(ddl, **job)`` in a new interpreter started with PYTHONHASHSEED=seed.

    Returns a dict with ``value`` (what ``_call_clones`` returned), ``messages`` (its progress lines) and
    ``dandelion_version``. Warnings raised in the child are issued again here; an exception raised in the child is
    raised again here, with the child's traceback as its cause.
    """
    advice = ("Start Python with PYTHONHASHSEED=0 (Dandelion then runs in this process), or pass hash_seed=None to "
              "call clones in this process with its own hash seed (clone ids are then not reproducible between "
              "sessions).")
    if getattr(sys, "frozen", False) or not sys.executable:
        raise RuntimeError("Dandelion's clone calls depend on Python's string-hash seed, and this Python (a frozen "
                           "or embedded interpreter) cannot start a separate Python process with a fixed seed. "
                           + advice)
    package_dir = os.path.dirname(os.path.abspath(__file__))
    setup = {"sys_path": _caller_sys_path(), "seed": int(seed)}
    with tempfile.TemporaryDirectory(prefix="threadfin_dandelion_") as tmp:
        job_path, result_path = os.path.join(tmp, "job.pickle"), os.path.join(tmp, "result.pickle")
        with open(job_path, "wb") as fh:
            pickle.dump(setup, fh, protocol=_PROTOCOL)     # standard-library objects only: read before sys.path is set
            pickle.dump(job, fh, protocol=_PROTOCOL)
        command = [sys.executable, *_interpreter_options(), "-c", BOOTSTRAP, package_dir, job_path, result_path]
        try:
            proc = subprocess.run(command, cwd=tmp, env=_environment(seed), stdin=subprocess.DEVNULL,
                                  capture_output=True, check=False)
        except OSError as exc:
            raise RuntimeError(f"Could not start a Python process for Dandelion's clone calling ({exc}). "
                               + advice) from exc
        if not os.path.exists(result_path):
            output = _tail(proc.stderr) or _tail(proc.stdout)
            raise RuntimeError(
                f"The Python process that called the clones with Dandelion stopped (exit status {proc.returncode}) "
                "before it returned a result" + (f"; its last output:\n{output}" if output else ".")
                + "\nIf it ran out of memory, call clones for fewer cells at a time (e.g. per donor). " + advice)
        with open(result_path, "rb") as fh:
            result = pickle.load(fh)
    if not result["ok"]:
        _raise_again(result)
    for category, message, filename, lineno, module in result["warnings"]:
        warnings.warn_explicit(message, category, filename, lineno, module=module, registry=_WARNING_REGISTRY)
    return result


# ------------------------------------------------------------------------------------------------ child's side

def _check_seed(seed: int) -> None:
    """The child must really run with the requested seed; sys.flags shows whether PYTHONHASHSEED=0 was applied."""
    started_with = os.environ.get("PYTHONHASHSEED")
    ok = (sys.flags.hash_randomization == 0) if seed == 0 else (
        sys.flags.hash_randomization == 1 and not sys.flags.ignore_environment and started_with == str(seed))
    if not ok:
        raise RuntimeError(f"The Python process for Dandelion did not start with PYTHONHASHSEED={seed} "
                           f"(PYTHONHASHSEED={started_with!r}, hash randomisation {sys.flags.hash_randomization}).")


def _picklable(obj) -> bool:
    try:
        pickle.loads(pickle.dumps(obj, protocol=_PROTOCOL))
    except Exception:  # noqa: BLE001 - anything that does not survive the trip is sent as text
        return False
    return True


def _portable_warnings(caught) -> list:
    """(category, message, filename, line, module) of each warning, for ``warnings.warn_explicit`` in the caller."""
    modules = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if isinstance(path, str):
            modules[os.path.normcase(os.path.abspath(path))] = name
    out = []
    for w in caught:
        category = w.category
        if not _picklable(category):          # a warning class defined where the caller cannot import it
            category = next((c for c in category.__mro__ if c.__module__ == "builtins"), UserWarning)
        module = modules.get(os.path.normcase(os.path.abspath(w.filename))) if w.filename else None
        out.append((category, str(w.message), w.filename, w.lineno, module))
    return out


def main(job_path: str, result_path: str) -> int:
    """Child's side: read the job, call the clones, write the result (or the exception) for the caller."""
    try:
        with open(job_path, "rb") as fh:
            setup = pickle.load(fh)
            sys.path[:] = setup["sys_path"]
            job = pickle.load(fh)
        _check_seed(setup["seed"])
        from threadfin.dandelion import _call_clones, _dandelion_version, _import_dandelion

        ddl = _import_dandelion()
        messages: list = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")  # every distinct warning once; the caller's filters then decide
            value = _call_clones(ddl, say=messages.append, **job)
        result = {"ok": True, "value": value, "messages": messages, "dandelion_version": _dandelion_version(ddl),
                  "warnings": _portable_warnings(caught)}
    except BaseException as exc:  # noqa: BLE001 - every failure is reported to the caller
        result = {"ok": False, "type": f"{type(exc).__module__}.{type(exc).__qualname__}", "message": str(exc),
                  "traceback": traceback.format_exc(), "exception": exc if _picklable(exc) else None}
    partial = result_path + ".part"
    with open(partial, "wb") as fh:
        pickle.dump(result, fh, protocol=_PROTOCOL)
    os.replace(partial, result_path)                  # the caller sees a complete file or none
    return 0
