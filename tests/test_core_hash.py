"""The recorded core-source baseline must match the tree: any change under src/ness (except
reference plugins) requires `python tools/update_core_hash.py`, which makes core edits visible
in review and lets `ness audit` prove to instance implementers that core is untouched."""

from ness.cli.main import main
from ness.integrity import baseline_hash, check_core_integrity, core_source_hash


def test_baseline_matches_tree():
    base = baseline_hash()
    assert base is not None, "src/ness/_core_hash.py missing: run python tools/update_core_hash.py"
    assert core_source_hash() == base, "core sources changed: run python tools/update_core_hash.py and review the diff under src/ness"


def test_report_and_strict_audit():
    rep = check_core_integrity()
    assert rep.matches is True and "MATCHES" in rep.summary()
    assert main(["audit"]) == 0
    assert main(["audit", "--strict"]) == 0


def test_hash_is_sensitive_to_core_but_not_to_reference_plugins(tmp_path):
    root = tmp_path / "ness"
    (root / "reference_plugins").mkdir(parents=True)
    (root / "a.py").write_text("x = 1\n")
    (root / "reference_plugins" / "p.py").write_text("y = 1\n")
    h1 = core_source_hash(root)
    (root / "reference_plugins" / "p.py").write_text("y = 2\n")
    assert core_source_hash(root) == h1                      # plugins are not core
    (root / "a.py").write_text("x = 2\n")
    assert core_source_hash(root) != h1                      # core is
