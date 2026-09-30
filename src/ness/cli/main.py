"""``ness`` command line: validate / run / evaluate / inspect / audit / plugins."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import __version__


def _registry():
    from ..plugin_api.registry import default_registry
    return default_registry()


def cmd_validate(args: argparse.Namespace) -> int:
    from ..composition import compile_graph
    from ..config import load_experiment
    exp = load_experiment(args.config)
    reg = _registry()
    print(f"experiment {exp.protocol_id} schema {exp.schema_version} hash {exp.protocol_hash[:12]}")
    if args.schema_only:
        for a in exp.arms:
            print(f"  arm {a.arm_id}: {len(a.composition.nodes)} nodes, spec hash {a.composition.spec_hash[:12]} (schema-only)")
        return 0
    scenario = reg.create(exp.scenario.plugin, exp.scenario.config)
    tasks = {t.task_id: reg.create(t.plugin, {**t.config, "task_id": t.task_id}) for t in exp.tasks}
    roles = frozenset.intersection(*(t.allowed_roles() for t in tasks.values()))
    spaces = {tid: t.prediction_space() for tid, t in tasks.items()}
    rc = 0
    for a in exp.arms:
        if args.arms and a.arm_id not in args.arms:
            continue
        try:
            compiled = compile_graph(a.composition, reg, scenario.observation_fields(), spaces, roles)
            to = f" training_only={compiled.training_only_nodes}" if compiled.training_only_nodes else ""
            print(f"  arm {a.arm_id}: OK  composition {compiled.composition_hash[:12]} nodes={compiled.node_ids()} diff_runtime={compiled.differentiable_runtime}{to}")
            for nid, ports in compiled.resolved_wiring().items():
                for port, srcs in ports.items():
                    print(f"      {nid}.{port} <- {', '.join(srcs)}")
        except Exception as exc:  # report every arm, fail closed overall
            rc = 1
            print(f"  arm {a.arm_id}: INVALID  {type(exc).__name__}: {exc}")
    return rc


def cmd_run(args: argparse.Namespace) -> int:
    from ..config import load_experiment
    from ..experiments.runner import run_experiment
    exp = load_experiment(args.config)
    report = run_experiment(exp, _registry(), args.out, tuple(args.arms) if args.arms else None, with_substitutions=not args.no_substitutions,
                            only_substitutions=args.only_substitutions)
    print(report.summary_table())
    if args.out:
        print(f"artifacts written under {args.out} (summary.txt, report.json, metrics/, predictions/, traces/, checkpoints/); next: ness report {args.out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from ..experiments.toy_report import build_report
    out = build_report(args.run_dir, args.title)
    print(f"report written to {out}")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    from ..visualize import render_experiment
    written, notes = render_experiment(args.config, args.out, args.arms or None, spec_only=args.spec_only, fmt=args.format, dpi=args.dpi,
                                       include_substitutions=not args.no_substitutions)
    for n in notes:
        print(f"note: {n}")
    for w in written:
        print(w)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    p = Path(args.report)
    if p.is_dir():
        summary = p / "summary.txt"
        if summary.exists():
            print(summary.read_text())
            return 0
        p = p / "report.json"
    d = json.loads(p.read_text())
    print(f"protocol {d['protocol_id']} ({d['protocol_hash'][:12]})")
    for arm, r in d["arms"].items():
        for task, t in (r.get("test") or {}).items():
            print(f"  {arm:32s} {task:14s} test loss {t['loss']:.5f}  n={int(t['n_requests'])}  updates={r['n_updates']}  manifest={str(r['manifest_id'])[:12]}")
    for arm, p in d.get("paired", {}).items():
        for task, x in p.items():
            print(f"  paired {arm:24s} {task:14s} delta={x['delta']:+.5f} vs {x['reference']}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from ..checkpoint import CheckpointStore
    store = CheckpointStore(args.root)
    ids = store.list_manifests()
    if not args.manifest:
        print("manifests:")
        for m in ids:
            print(f"  {m}")
        return 0
    idx = store.load_index(args.manifest)
    m = idx["manifest"]
    print(f"manifest {args.manifest}\n  protocol {m['protocol_id']} arm {m['arm_id']} composition {m['composition_hash'][:12]}")
    print(f"  inference {m['inference_profile']} learning {m['learning_profile']} memory {m['memory_snapshot_id']} kind {idx['kind']}")
    print(f"  dependency lock {m['dependency_lock']}")
    for c in m["components"]:
        print(f"  component {c['node_id']:28s} {c['plugin_id']}@{c['plugin_version']} kind={c['module_kind']} runtime={c['runtime']} state={c['state_hash'][:12]}")
    for nid, ports in m["extra"].get("resolved_wiring", {}).items():
        for port, srcs in ports.items():
            print(f"  wiring {nid}.{port} <- {', '.join(srcs)}")
    rt = m["extra"].get("runtime")
    if rt:
        print("  runtime identity: " + ", ".join(f"{k}={v}" for k, v in rt.items()))
    for name, spec in m["extra"].get("algorithm_specs", {}).items():
        print(f"  algorithm spec [{name}]:")
        specs = spec if (isinstance(spec, dict) and "learning_rule" not in spec and "derivative" not in spec) else {"": spec}
        for sub, sp in specs.items():
            prefix = f"    {sub}: " if sub else "    "
            for k in ("initialization", "inference", "clamps_prediction", "clamps_learning", "derivative", "readout", "learning_rule", "loss", "reductions", "evaluation"):
                if k in sp:
                    print(f"{prefix}{k} = {sp[k]}")
    return 0


def cmd_plugins(args: argparse.Namespace) -> int:
    reg = _registry()
    for d in reg.descriptors():
        missing = d.missing_dependencies()
        flag = f"  MISSING deps {list(missing)}" if missing else ""
        print(f"{d.plugin_id:40s} {d.kind:16s} v{d.version:8s} {d.provided_by}{flag}")
    if reg.discovery_errors:
        print("discovery errors:")
        for e in reg.discovery_errors:
            print(f"  {e}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from ..inference import INFERENCE_PROFILES
    from ..integrity import check_core_integrity
    from ..learning import LEARNING_PROFILES
    from ..runtime.system import dependency_lock
    print(f"ness {__version__}")
    integrity = check_core_integrity()
    print("core integrity:")
    print("  " + integrity.summary().replace("\n", "\n  "))
    print("dependency lock:")
    for k, v in dependency_lock().items():
        print(f"  {k:10s} {v}")
    print("learning profiles:")
    for c in LEARNING_PROFILES.items:
        print(f"  {c.name:26s} {c.status.value:12s} {c.evidence}")
    print("inference profiles:")
    for c in INFERENCE_PROFILES.items:
        print(f"  {c.name:26s} {c.status.value:12s} {c.evidence}")
    reg = _registry()
    print(f"plugins discovered: {len(reg.ids())}  (kinds: {sorted(set(d.kind for d in reg.descriptors()))})")
    if getattr(args, "strict", False) and integrity.matches is False:
        print("error: core sources differ from the release baseline (audit --strict)", file=sys.stderr)
        return 1
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from ..runtimes.doctor import format_doctor, run_doctor
    d = run_doctor(args.backend, args.platform, not args.no_probe, args.profile)
    print(format_doctor(d))
    if args.json:
        Path(args.json).write_text(json.dumps(d, indent=1, default=str))
        print(f"written {args.json}")
    return 1 if d["errors"] else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ness", description="NESS cross-domain research platform")
    p.add_argument("--version", action="version", version=f"ness {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="validate an experiment config and compile its arms")
    v.add_argument("config")
    v.add_argument("--arms", nargs="*")
    v.add_argument("--schema-only", action="store_true", help="structural validation without instantiating plugins")
    v.set_defaults(fn=cmd_validate)
    r = sub.add_parser("run", help="run all (or selected) arms of an experiment")
    r.add_argument("config")
    r.add_argument("--out", default=None)
    r.add_argument("--arms", nargs="*")
    r.add_argument("--no-substitutions", action="store_true", help="skip the configuration-only substitution proofs")
    r.add_argument("--only-substitutions", action="store_true", help="run only the substitution proofs, refreshing metrics/substitution_matrix.* in an existing run directory")
    r.set_defaults(fn=cmd_run)
    e = sub.add_parser("evaluate", help="summarise a run directory (or a report.json)")
    e.add_argument("report")
    e.set_defaults(fn=cmd_evaluate)
    rp = sub.add_parser("report", help="regenerate metrics, figures and the Markdown report from a run directory's saved artifacts")
    rp.add_argument("run_dir")
    rp.add_argument("--title", default="NESS toy vertical slice - run report")
    rp.set_defaults(fn=cmd_report)
    g = sub.add_parser("graph", help="draw an experiment: composition graph and sandwich view per arm, arms matrix, data-access map (needs ness[report])")
    g.add_argument("config")
    g.add_argument("--out", default=None, help="output directory (default graphs/<protocol_id>)")
    g.add_argument("--arms", nargs="*", default=None, help="draw only these arms (substitutions are then skipped)")
    g.add_argument("--spec-only", action="store_true", help="never instantiate plugins: draw from the configuration alone")
    g.add_argument("--no-substitutions", action="store_true", help="leave substitution proofs out of the arms matrix")
    g.add_argument("--format", default="png", choices=["png", "svg", "pdf"])
    g.add_argument("--dpi", type=int, default=150)
    g.set_defaults(fn=cmd_graph)
    i = sub.add_parser("inspect", help="inspect a checkpoint store / manifest")
    i.add_argument("root")
    i.add_argument("manifest", nargs="?")
    i.set_defaults(fn=cmd_inspect)
    a = sub.add_parser("audit", help="core-source integrity (release baseline), dependency locks and capability status")
    a.add_argument("--strict", action="store_true", help="exit 1 when the core sources differ from the release baseline")
    a.set_defaults(fn=cmd_audit)
    pl = sub.add_parser("plugins", help="list discovered plugins")
    pl.set_defaults(fn=cmd_plugins)
    dr = sub.add_parser("doctor", help="runtime diagnostics: versions, JAX/FabricPC bootstrap order, devices, capabilities")
    dr.add_argument("--backend", default="auto", choices=["auto", "numpy", "jax", "fabricpc"])
    dr.add_argument("--platform", default=None, choices=[None, "cpu", "gpu", "tpu"])
    dr.add_argument("--profile", default="development", choices=["development", "scientific"])
    dr.add_argument("--no-probe", action="store_true", help="skip executable FabricPC capability probes")
    dr.add_argument("--json", default=None, help="also write the machine-readable report here")
    dr.set_defaults(fn=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
