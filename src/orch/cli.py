import argparse
import sys

from orch import __version__


def cmd_version(_args: argparse.Namespace) -> int:
    print(f"orch {__version__}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    from pathlib import Path

    from orch.config import init_config
    from orch.errors import ConfigError

    try:
        cfg = init_config(Path(args.target_repo), Path(args.config))
    except ConfigError as e:
        print(f"error: {e}")
        return 1
    Path(cfg.state_dir).mkdir(exist_ok=True)
    Path(cfg.runs_dir).mkdir(exist_ok=True)
    print(f"wrote {args.config}\nnext: orch plan \"<brief>\"")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone
    from pathlib import Path

    from orch.adapters.factory import factory
    from orch.artifacts import RunDir
    from orch.config import load_config
    from orch.errors import ConfigError, PlanValidationError
    from orch.nodes.planner import run_planner

    try:
        cfg = load_config(Path(args.config))
    except ConfigError as e:
        print(f"error: {e}")
        return 1

    run_id = args.run or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = RunDir.create(Path(cfg.runs_dir), run_id)
    adapter = factory(cfg)

    try:
        plan = run_planner(
            cfg=cfg,
            adapter=adapter,
            brief=args.brief,
            source=args.source,
            run_dir=run_dir,
        )
    except PlanValidationError as e:
        invalid = run_dir.root / "plan.invalid.yaml"
        if invalid.exists():
            print(f"invalid plan written to {invalid}")
        for issue in e.issues:
            print(f"issue: {issue}")
        return 1

    risky = [task.id for task in plan.tasks if task.risky]
    print(f"plan written to {run_dir.plan_path}")
    if risky:
        print("risky tasks: " + ", ".join(risky))
    else:
        print("no risky tasks")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orch")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("version", help="print version")
    v.set_defaults(fn=cmd_version)
    i = sub.add_parser("init", help="write orch.config.yaml for a target repo")
    i.add_argument("target_repo", help="path to the target git repository")
    i.add_argument("--config", default="orch.config.yaml", help="path to write the config")
    i.set_defaults(fn=cmd_init)
    pl = sub.add_parser("plan", help="run the planner and write plan.yaml")
    pl.add_argument("brief", help="the brief to plan against")
    pl.add_argument("--from", dest="source", default=None, help="source document to plan from")
    pl.add_argument("--config", default="orch.config.yaml", help="path to the config")
    pl.add_argument("--run", default=None, help="run id (defaults to a UTC timestamp)")
    pl.set_defaults(fn=cmd_plan)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
