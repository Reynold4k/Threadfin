"""Clone calls must not depend on the Python session's string-hash seed.

Dandelion 0.5.x numbers V/J groups, and orders the sequences it clusters, by iterating over Python sets, so its
clone calls depend on PYTHONHASHSEED: on the GSE287123 tutorial data, sessions with other seeds changed the cells of
11-13% of the clones with two or more cells and renumbered every clone. ``define_clones_dandelion`` therefore calls
clones under seed 0 by default, in a separate Python process when the session was not started with
PYTHONHASHSEED=0. These tests check that the separate process gives exactly what a session started with
PYTHONHASHSEED=0 gives, on data whose clone ids do depend on the seed (so that the check can fail), that its errors,
warnings and progress lines reach the caller, the opt-out, and the matplotlib >= 3.9 shim that lets Dandelion 0.5.x
import on Python 3.13+.

Each Dandelion call in a separate process starts a new Python (a few seconds), so the contig tables are tiny and the
separate-process tests are few. Tests that should run Dandelion in this process whatever seed pytest was started
with replace ``_session_hash_seed``, the one place where the session's seed is read.
"""

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import threadfin
import threadfin.dandelion as tfd
from threadfin import _dandelion_worker as worker

#: 12 heavy-chain V genes, so 12 V/J groups, which Dandelion numbers in the order it iterates over a set of them:
#: the clone ids then differ between hash seeds (12! orders), while every seed finds the same 3 two-cell clones
V_GENES = [f"IGHV1-{i}*01" for i in range(2, 14)]


def seed_sensitive_contigs():
    """Heavy + light contigs of 15 cells in 12 V/J groups: groups 0, 4 and 8 are two-cell clones; two donors."""
    rows = []
    for g, v in enumerate(V_GENES):
        cells = [f"g{g}a", f"g{g}b"] if g % 4 == 0 else [f"g{g}a"]
        for cell in cells:
            donor = "d1" if g < 6 else "d2"
            rows.append({"cell_id": cell, "donor": donor, "locus": "IGH", "v_call": v, "j_call": "IGHJ4*02",
                         "junction_aa": "CARDTGDYW", "umi_count": 10})
            rows.append({"cell_id": cell, "donor": donor, "locus": "IGK", "v_call": "IGKV1-5*01",
                         "j_call": "IGKJ1*01", "junction_aa": "CQQYNSYPLTF", "umi_count": 8})
    t = pd.DataFrame(rows)
    t["junction"] = ["TGT" * len(a) for a in t["junction_aa"]]
    t["productive"] = "T"
    return t


def tcr_contigs():
    """Four T cells: two pairs with identical beta and alpha junctions."""
    rows = []
    for cell, beta, alpha in (("t1", "CASSLGQGYEQYF", "CAVRDSNYQLIW"), ("t2", "CASSLGQGYEQYF", "CAVRDSNYQLIW"),
                              ("t3", "CASSPDRGNTEAFF", "CAMSGGSYIPTF"), ("t4", "CASSPDRGNTEAFF", "CAMSGGSYIPTF")):
        rows.append({"cell_id": cell, "locus": "TRB", "v_call": "TRBV7-9*01", "j_call": "TRBJ2-7*01",
                     "junction_aa": beta, "umi_count": 5})
        rows.append({"cell_id": cell, "locus": "TRA", "v_call": "TRAV1-2*01", "j_call": "TRAJ33*01",
                     "junction_aa": alpha, "umi_count": 3})
    t = pd.DataFrame(rows)
    t["junction"] = ["TGT" * len(a) for a in t["junction_aa"]]
    t["productive"] = "T"
    return t


def seed0_reference(contigs, tmp_path, **kwargs):
    """Clone ids from a Python session started with PYTHONHASHSEED=0, where Dandelion runs in-process: this one
    when pytest itself was started so, otherwise a new Python that imports this copy of Threadfin."""
    if sys.flags.hash_randomization == 0:
        ids = tfd.define_clones_dandelion(contigs, verbose=False, **kwargs)
        assert not ids.attrs["dandelion_subprocess"]
        return ids
    contigs.to_pickle(tmp_path / "contigs.pkl")
    code = ("import json, sys, pandas as pd\n"
            "from threadfin.dandelion import define_clones_dandelion\n"
            "ids = define_clones_dandelion(pd.read_pickle(sys.argv[1]), verbose=False, **json.loads(sys.argv[2]))\n"
            "assert sys.flags.hash_randomization == 0 and not ids.attrs['dandelion_subprocess']\n"
            "print(json.dumps({'index': list(ids.index), 'values': list(ids)}))\n")
    env = dict(os.environ, PYTHONHASHSEED="0")
    env["PYTHONPATH"] = os.pathsep.join([str(Path(threadfin.__file__).resolve().parents[1]),
                                         *filter(None, [env.get("PYTHONPATH")])])
    proc = subprocess.run([sys.executable, "-c", code, str(tmp_path / "contigs.pkl"), json.dumps(kwargs)],
                          env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    return pd.Series(out["values"], index=pd.Index(out["index"], name="cell_id"), dtype="object")


@pytest.fixture
def unseeded(monkeypatch):
    """Behave as in any session not started with PYTHONHASHSEED=0, whatever seed pytest was started with."""
    monkeypatch.setattr(tfd, "_session_hash_seed", lambda: None)


# ------------------------------------------------------------------------------- the seed-0 result in any session

def test_default_clone_calls_are_those_of_a_session_started_with_hash_seed_0(tmp_path, monkeypatch, capsys):
    pytest.importorskip("dandelion")
    t = seed_sensitive_contigs()
    reference = seed0_reference(t, tmp_path, donor_key="donor")
    # then as in any session not started with PYTHONHASHSEED=0, whatever seed pytest was started with
    monkeypatch.setattr(tfd, "_session_hash_seed", lambda: None)
    capsys.readouterr()
    ids = tfd.define_clones_dandelion(t, donor_key="donor", verbose=True)
    assert ids.attrs["dandelion_subprocess"] is True and ids.attrs["dandelion_hash_seed"] == 0
    assert list(ids.index) == list(reference.index)
    assert list(ids) == list(reference)                          # cell for cell, the same clone ids
    assert ids.attrs["n_clones_ge2"] == 3 and ids.attrs["dandelion_version"]
    # the progress lines of the separate process are printed here
    printed = capsys.readouterr().out
    assert "separate Python process" in printed and "  d1: " in printed and "  d2: " in printed

    # the same data under another seed: the same clones, other ids. Without this the check above could pass on
    # data that do not depend on the seed at all.
    other = tfd.define_clones_dandelion(t, donor_key="donor", hash_seed=1, verbose=False)
    assert other.attrs["dandelion_hash_seed"] == 1 and other.attrs["dandelion_subprocess"] is True
    assert list(other.index) == list(reference.index) and list(other) != list(reference)
    same_clones = pd.crosstab(other, reference)
    assert (same_clones > 0).sum(axis=1).eq(1).all() and (same_clones > 0).sum(axis=0).eq(1).all()


def test_errors_and_warnings_of_the_separate_process_reach_the_caller(unseeded):
    pytest.importorskip("dandelion")
    # Dandelion fills in its default for a chain type missing from an identity dict, and warns
    with pytest.warns(UserWarning, match="Identity value for tr-ab"):
        ids = tfd.define_clones_dandelion(tcr_contigs(), identity={"ig": 0.85}, verbose=False)
    assert ids.attrs["dandelion_subprocess"] and ids.attrs["identity"]["tr-ab"] == 1.0
    assert ids["t1"] == ids["t2"] and ids["t3"] == ids["t4"] and ids["t1"] != ids["t3"]
    # an exception raised inside Dandelion is raised here, with the same type and the child's traceback as cause
    with pytest.raises(TypeError) as info:
        tfd.define_clones_dandelion(seed_sensitive_contigs(), identity="0.9", verbose=False)
    assert isinstance(info.value.__cause__, worker.ChildProcessTraceback)
    assert "find_clones" in str(info.value.__cause__)


# ----------------------------------------------------------------------------------------- the opt-out, settings

def test_hash_seed_none_calls_clones_in_this_session_and_warns_when_its_seed_is_not_0(monkeypatch):
    pytest.importorskip("dandelion")
    t = seed_sensitive_contigs()
    monkeypatch.setattr(tfd, "_session_hash_seed", lambda: None)
    with pytest.warns(UserWarning, match="hash_seed=None"):
        ids = tfd.define_clones_dandelion(t, hash_seed=None, verbose=False)
    assert ids.attrs["dandelion_subprocess"] is False and ids.attrs["dandelion_hash_seed"] is None
    monkeypatch.setattr(tfd, "_session_hash_seed", lambda: 0)        # a session started with PYTHONHASHSEED=0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ids = tfd.define_clones_dandelion(t, hash_seed=None, verbose=False)
    assert not [w for w in caught if "hash_seed" in str(w.message)]
    assert ids.attrs["dandelion_subprocess"] is False and ids.attrs["dandelion_hash_seed"] == 0


@pytest.mark.parametrize("bad, error", [(-1, ValueError), (2 ** 32, ValueError), (1.5, TypeError),
                                        ("0", TypeError), (True, TypeError)])
def test_hash_seed_must_be_an_integer_that_pythonhashseed_accepts(bad, error):
    with pytest.raises(error, match="hash_seed"):
        tfd.define_clones_dandelion(seed_sensitive_contigs(), hash_seed=bad, verbose=False)


def test_valid_hash_seeds():
    assert tfd._check_hash_seed(None) is None
    assert tfd._check_hash_seed(np.int64(7)) == 7 and tfd._check_hash_seed(4294967295) == 4294967295


def test_session_seed_is_read_from_the_interpreter_not_the_environment(monkeypatch):
    """Setting PYTHONHASHSEED inside a running session (e.g. os.environ in a notebook) changes nothing."""
    expected = 0 if sys.flags.hash_randomization == 0 else None
    for value in ("0", "1"):
        monkeypatch.setenv("PYTHONHASHSEED", value)
        assert tfd._session_hash_seed() == expected


def test_contig_check_within_donors_works_without_read_counts(monkeypatch):
    """The donors' tables were split off before the read-count column was added, so the check stopped with
    KeyError: 'consensus_count' whenever donor_key and check_contigs=True met a table without read counts."""
    pytest.importorskip("dandelion")
    monkeypatch.setattr(tfd, "_session_hash_seed", lambda: 0)        # in this process: no new Python needed
    t = seed_sensitive_contigs()
    assert "consensus_count" not in t
    ids = tfd.define_clones_dandelion(t, donor_key="donor", check_contigs=True, verbose=False)
    assert ids.attrs["n_clones_ge2"] == 3 and ids.index.str.startswith("g").all()


# ---------------------------------------------------------------------------- the protocol, without a new Python

def test_child_errors_are_raised_again_with_their_traceback():
    with pytest.raises(KeyError) as info:
        worker._raise_again({"ok": False, "type": "builtins.KeyError", "message": "'x'",
                             "traceback": "Traceback ...\nKeyError: 'x'", "exception": KeyError("x")})
    assert "KeyError: 'x'" in str(info.value.__cause__)
    with pytest.raises(RuntimeError, match="somepackage.Odd: cannot be pickled"):       # an exception that did
        worker._raise_again({"ok": False, "type": "somepackage.Odd", "message": "cannot be pickled",  # not survive
                             "traceback": "Traceback ...", "exception": None})


def test_child_warnings_keep_their_category_and_origin():
    class LocalWarning(FutureWarning):            # cannot be pickled: sent as its nearest built-in class
        pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("from pandas", pd.errors.PerformanceWarning)
        warnings.warn("local", LocalWarning)
    sent = worker._portable_warnings(caught)
    assert [s[0] for s in sent] == [pd.errors.PerformanceWarning, FutureWarning]
    assert [s[1] for s in sent] == ["from pandas", "local"]
    assert sent[0][2] == __file__ and sent[0][4] == __name__        # file and module the warning came from


def test_the_child_refuses_to_run_with_another_seed(tmp_path):
    """The child checks that PYTHONHASHSEED really took effect; here it did not (this process), so it reports."""
    import pickle

    job, result = tmp_path / "job.pickle", tmp_path / "result.pickle"
    with open(job, "wb") as fh:
        pickle.dump({"sys_path": list(sys.path), "seed": 4294967295}, fh)
        pickle.dump({"groups": []}, fh)
    assert worker.main(str(job), str(result)) == 0
    with open(result, "rb") as fh:
        out = pickle.load(fh)
    assert not out["ok"] and "did not start with PYTHONHASHSEED=4294967295" in out["message"]


def test_child_environment_and_options():
    env = worker._environment(0)
    assert env["PYTHONHASHSEED"] == "0" and env["MPLBACKEND"] == "Agg" and "PYTHONINSPECT" not in env
    assert "-E" not in worker._interpreter_options() and "-I" not in worker._interpreter_options()
    assert all(os.path.isabs(p) for p in worker._caller_sys_path())


# ------------------------------------------------------------------------------------- matplotlib >= 3.9 shim

def test_get_cmap_is_restored_for_dandelion_when_matplotlib_removed_it(monkeypatch):
    import matplotlib
    import matplotlib.cm

    # importing Threadfin made it available, whatever the matplotlib version, so `import dandelion` works afterwards
    assert callable(matplotlib.cm.get_cmap)
    # without get_cmap, as in matplotlib >= 3.9 (the setattr first, so that the test leaves the module as it was)
    monkeypatch.setattr(matplotlib.cm, "get_cmap", None, raising=False)
    monkeypatch.delattr(matplotlib.cm, "get_cmap")
    tfd._restore_get_cmap()
    get_cmap = matplotlib.cm.get_cmap
    assert get_cmap("viridis").name == "viridis"                      # as nxviz calls it
    assert get_cmap("viridis", 8).N == 8
    assert get_cmap().name == matplotlib.rcParams["image.cmap"]
    cmap = matplotlib.colormaps["bwr"]
    assert get_cmap(cmap) is cmap
    # an existing get_cmap (matplotlib < 3.9) is left alone
    tfd._restore_get_cmap()
    assert matplotlib.cm.get_cmap is get_cmap
