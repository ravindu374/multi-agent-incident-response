"""Command line for the lab: python -m oncall <command>."""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from .actions import ACTIONS
from .episodes import LabNotReady, preflight, run_episode
from .faults import FAULTS, apply_reference_fix, inject_fault, revert_all, revert_fault
from .incidents import IncidentDetector, capture_evidence
from .lab import Lab, utcnow_iso
from .oracle import HealthReport, observe


def _table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    cells = [[str(h) for h in headers]] + [[("" if v is None else str(v)) for v in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    lines = ["  ".join(v.ljust(widths[i]) for i, v in enumerate(row)).rstrip() for row in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _parse_params(pairs: list[str]) -> dict:
    params = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"bad --param {pair!r}; expected KEY=VALUE")
        try:
            params[key] = json.loads(value)
        except json.JSONDecodeError:
            params[key] = value
    return params


def _print_report(report: HealthReport) -> None:
    rows = [(c.name, "ok" if c.ok else "FAIL", c.detail) for c in report.checks]
    print(_table(rows, ("check", "status", "detail")))
    print(f"\noverall: {'HEALTHY' if report.healthy else 'UNHEALTHY'}")


def cmd_init(lab: Lab, args) -> int:
    written = lab.runtime.init(overwrite=args.force)
    print(f"runtime configs written: {', '.join(written) or 'none (already present; use --force)'}")
    return 0


def _ready(lab: Lab) -> bool:
    problems = preflight(lab)
    for problem in problems:
        print(f"not ready: {problem}")
    return not problems


def cmd_health(lab: Lab, args) -> int:
    if not _ready(lab):
        return 2
    report = observe(lab)
    _print_report(report)
    return 0 if report.healthy else 1


def cmd_watch(lab: Lab, args) -> int:
    detector = IncidentDetector()
    print(f"watching every {args.interval:.0f}s (Ctrl+C to stop)")
    try:
        while True:
            for event in detector.observe(observe(lab)):
                incident = event.incident
                stamp = utcnow_iso()
                if event.kind == "opened":
                    print(f"{stamp} OPENED {incident.incident_id}: {incident.summary}")
                    incident.evidence = capture_evidence(lab)
                elif event.kind == "updated":
                    print(f"{stamp} UPDATED {incident.incident_id}: + {', '.join(a.name for a in event.new_alerts)}")
                else:
                    print(f"{stamp} RESOLVED {incident.incident_id}")
                    lab.ledger.append("incidents", incident.model_dump())
                    lab.ledger.append("ground_truth", {
                        "incident_id": incident.incident_id,
                        "source": "watch",
                        "active_injections": list(lab.ledger.load_active().values()),
                    })
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def cmd_faults(lab: Lab, args) -> int:
    if args.action == "list":
        rows = [(f.id, f.owner, ",".join(f.symptom_tags), ",".join(f.valid_fixes), f.title)
                for f in FAULTS.values()]
        print(_table(rows, ("id", "owner", "symptom tags", "valid fixes", "title")))
        return 0
    if args.action == "actions":
        rows = [(a.name, a.risk, ",".join(a.domains), "yes" if a.reversible else "no", a.summary)
                for a in ACTIONS.values()]
        print(_table(rows, ("action", "risk", "domains", "reversible", "summary")))
        return 0
    if args.action == "revert" and args.fault_id == "all":
        print(json.dumps(revert_all(lab), indent=2))
        return 0
    fault_id = args.fault_id.upper()
    if fault_id not in FAULTS:
        raise SystemExit(f"unknown fault {args.fault_id!r}; see `python -m oncall faults list`")
    if args.action == "inject":
        record = inject_fault(lab, fault_id, _parse_params(args.param))
        print(f"injected {fault_id} as {record['injection_id']} with {record['params']}")
    elif args.action == "fix":
        print(f"applied: {apply_reference_fix(lab, fault_id)}")
    elif args.action == "revert":
        revert_fault(lab, fault_id)
        print(f"reverted {fault_id}")
    return 0


def cmd_validate(lab: Lab, args) -> int:
    fault_ids = [f.upper() for f in args.fault_ids] or list(FAULTS)
    unknown = [f for f in fault_ids if f not in FAULTS]
    if unknown:
        raise SystemExit(f"unknown fault(s): {unknown}")
    if not _ready(lab):
        return 2
    print(f"validating {len(fault_ids)} fault(s); each episode takes about 3-6 minutes")
    results = []
    for fault_id in fault_ids:
        try:
            results.append(run_episode(lab, fault_id))
        except LabNotReady as exc:
            print(f"stopping: {exc}")
            break
    rows = [(r.fault_id, r.owner, "yes" if r.detected else "NO", r.detect_s, r.title, ",".join(r.fix_actions),
             "yes" if r.recovered else "NO", r.recover_s, r.error or r.fix_error or "")
            for r in results]
    print()
    print(_table(rows, ("fault", "owner", "detected", "after_s", "first alert", "fix", "recovered", "after_s",
                        "error")))
    return 0 if results and all(r.detected and r.recovered and not r.error for r in results) else 1


def cmd_synth(lab: Lab, args) -> int:
    from collections import Counter

    from .synth import generate, write_dataset

    fault_ids = [f.upper() for f in args.faults] or None
    incidents, truths, manifest = generate(per_fault=args.per_fault, seed=args.seed, fault_ids=fault_ids)
    write_dataset(args.out, incidents, truths, manifest)
    print(f"wrote {len(incidents)} synthetic incidents to {args.out}  (splits: {manifest['splits']})\n")
    rows = []
    for fault_id in manifest["faults"]:
        mine = [t for t in truths if t["fault_id"] == fault_id]
        variants = Counter(t["variant"] for t in mine)
        rows.append((fault_id, FAULTS[fault_id].owner, len(mine), manifest["resampled_undetected"].get(fault_id, 0),
                     round(sum(t["detect_after_s"] for t in mine) / len(mine)),
                     ", ".join(f"{v} x{n}" for v, n in variants.most_common(3))
                     + (f", +{len(variants) - 3} more" if len(variants) > 3 else "")))
    print(_table(rows, ("fault", "owner", "incidents", "resampled", "mean detect s", "variants")))
    return 0


def cmd_train(lab: Lab, args) -> int:
    from .train import train_classifier

    result = train_classifier(args.data, args.out, device=args.device)
    info = result["info"]
    print(f"trained logreg on {info['incidents']} train incidents in {info['seconds']}s "
          f"(backend: {info['backend']}, {info['features']} features = {info['structured']} structured "
          f"+ {info['text_terms']} text terms, train accuracy {info['train_accuracy']:.1%})")
    print(f"saved to {result['model_path']}\n")
    rows = []
    for label, key in (("owner accuracy (top guess)", "owner_accuracy"), ("true owner on the team", "owner_on_team"),
                       ("mean team size", "mean_team_size"), ("confident share", "confident_share"),
                       ("accuracy when confident", "accuracy_when_confident")):
        fmt = (lambda v: f"{v:.2f}") if key == "mean_team_size" else (lambda v: "-" if v is None else f"{v:.1%}")
        rows.append((label, fmt(result["rules_val"]["overall"][key]), fmt(result["val"]["overall"][key])))
    print(f"validation split (n={result['val']['overall']['n']}):")
    print(_table(rows, ("", "rules", "logreg")))
    print("\nper fault, top guess right (val):")
    print(_table([(f, f"{result['rules_val']['per_fault'][f]['owner_accuracy']:.0%}", f"{s['owner_accuracy']:.0%}")
                  for f, s in result["val"]["per_fault"].items()], ("fault", "rules", "logreg")))
    print("\nstrongest features per domain:")
    for domain, feats in result["top_features"].items():
        print(f"  {domain:9} " + ", ".join(name for name, _ in feats))
    return 0


def cmd_calibrate(lab: Lab, args) -> int:
    from .route_eval import calibrate_and_compare

    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else None
    kwargs = {"alphas": alphas} if alphas else {}
    result = calibrate_and_compare(args.data, args.model, args.out, alpha=args.alpha, method=args.method,
                                   final=args.final, **kwargs)
    by = ", ".join(f"{d} {n}" for d, n in result["calibration_by_domain"].items())
    print(f"{result['classifier']} classifier; calibrated on {result['calibration_n']} incidents ({by}), "
          f"reported on {result['report_n']} ({result['split']})")
    print(f"saved to {result['saved']}: {result['saved_description']}")
    if "(pooled)" in result["saved_description"]:
        print("  note: (pooled) domains had too few calibration incidents for this alpha; they get only the "
              "average promise. More incidents per domain (or --final on all of val) fixes this.")
    print()

    def pct(v):
        return "-" if v is None else f"{v:.0%}"

    rows = [(name, f"{r['owner_on_team']:.1%}", *(pct(r["per_domain"][d]) for d in ("network", "data", "infra",
                                                                                   "security")),
             f"{r['mean_team_size']:.2f}", f"{r['mean_cost']:.2f}") for name, r in result["rows"].items()]
    print(_table(rows, ("team strategy", "owner on team", "net", "data", "infra", "sec", "team size",
                        "cost/incident")))
    print("\ncheapest strategy reaching each coverage target (overall AND in every domain):")
    for c in result["cheapest"]:
        best = f"{c['strategy']}  (cost {c['mean_cost']:.2f})" if c["strategy"] else "none reaches it"
        print(f"  >= {c['target']:.0%}: {best}")
    print("\nconformal promise: lac >= 1-alpha on average; mondrian >= 1-alpha for every owner domain "
          "(both need calibration and report incidents from the same distribution)")
    return 0


def cmd_eval_agents(lab: Lab, args) -> int:
    from .agent_eval import evaluate_agents

    result = evaluate_agents(args.data, split=args.split, reasoner=args.reasoner, model=args.model,
                             agent_ids=args.agents or None, limit=args.limit, show=args.show)
    print(f"{result['reasoner']} reasoner on {result['incidents']} incidents (split: {result['split']}); "
          f"fix targets known for {result['targets_checked']}")
    if not result["targets_checked"]:
        print("note: this dataset has no fix_targets (made by generator v1?); re-run `synth` so 'solved' can "
              "check that a fix hits the culprit")
    if result["reasoner"] == "rules":
        print("note: the rule reasoner and the synthetic generator were written from the same knowledge, "
              "so these scores show the pipeline works; they are not a research result.")
    print()

    def pct(v):
        return "-" if v is None else f"{v:.0%}"

    rows = [(a, r["own_incidents"], pct(r["solved"]), pct(r["wrong_target"]), pct(r["false_claims"]), pct(r["unsure"]),
             f"{r['mean_tools']:.1f}", "-" if r["mean_tokens"] is None else f"{r['mean_tokens']:.0f}",
             "-" if r["mean_seconds"] is None else f"{r['mean_seconds']:.1f}")
            for a, r in result["rows"].items()]
    print(_table(rows, ("agent", "own incidents", "solved", "wrong target", "false claims", "unsure", "tools",
                        "tokens", "seconds")))
    print("solved = right owner + a correct fix (valid, executable, grounded, aimed at the culprit); "
          "wrong target = right kind of fix, wrong thing")
    print("\nsolved per fault (owner agents):")
    faults = sorted({f for r in result["rows"].values() for f in r["per_fault"]})
    print(_table([(a, *(pct(r["per_fault"].get(f)) for f in faults)) for a, r in result["rows"].items()],
                 ("agent", *faults)))
    if args.show:
        print(f"\nmistakes ({len(result['mistakes'])}):" if result["mistakes"] else "\nno mistakes")
        for m in result["mistakes"]:
            print(m)
    return 0


def cmd_eval_team(lab: Lab, args) -> int:
    from .team_eval import evaluate_team

    r = evaluate_team(args.data, split=args.split, model=args.model, calibration=args.calibration,
                      reasoner=args.reasoner, ollama_model=args.ollama_model, limit=args.limit,
                      cross_check=args.cross_check, follow_pointers=not args.no_pointers,
                      escalate=not args.no_escalation, careless=args.careless,
                      second_look_below=args.second_look_below, must_propose=not args.may_abstain,
                      shy=args.shy, approval=not args.no_approval, memory=args.memory, trust=args.trust,
                      past_cases=not args.no_past_cases, precedent=not args.no_precedent, show=args.show)
    m = r["memory"]
    if m:
        print(f"history agent: {m['experiences']} past incidents in {m['path']}; agents' verified record "
              f"{m['trust']}; past incidents shown to experts {'yes' if m['past_cases'] else 'no'}; "
              f"precedent counted {'yes' if m['precedent'] else 'no'}")
    else:
        print("no history agent: agents trusted by planning numbers, no past incidents")
    print(f"{r['incidents']} incidents (split: {r['split']}); classifier {r['classifier']}"
          f"{' + conformal' if r['calibrated'] else ''}; agents: {r['reasoner']}; cross-check {r['cross_check']}; "
          f"pointers {'on' if r['follow_pointers'] else 'off'}; escalation {'on' if r['escalate'] else 'off'}; "
          f"{'every chosen agent must propose a fix' if r['must_propose'] else 'agents may abstain'}; "
          f"{'a human may approve a fix' if r['approval'] else 'no approval option'}"
          + (f"; CARELESS agents: {', '.join(r['careless'])}" if r["careless"] else "")
          + (f"; SHY agents: {', '.join(r['shy'])}" if r["shy"] else ""))
    if r["reasoner"] == "rules":
        print("note: rule agents and the generator share knowledge; this shows the rounds work, not a result.")

    def pct(v):
        return "-" if v is None else f"{v:.0%}"

    rows = [
        ("owner on team", pct(r["owner_on_team"]), "router called the owner's domain"),
        ("owner asked", pct(r["owner_asked"]), "...or an agent's pointer brought it in"),
        ("correct fix proposed", pct(r["correct_fix"]), "some agent handed in a correct fix (claim or best guess)"),
        ("no fix at all", pct(r["no_fix"]), "no agent handed in a usable fix (5c: escalate to a human)"),
        ("fixes per incident", f"{r['fixes_per_incident']:.2f}", "claims + best guesses that passed the checks"),
        ("correct claim found", pct(r["correct_claim"]), "...of which an agent also said 'it's mine'"),
        ("claims per incident", f"{r['claims_per_incident']:.2f}", f"no claim at all: {pct(r['no_claim'])}"),
        ("wrong fixes", str(r["wrong_claims"]), "wrong domain, or not aimed at the culprit"),
        ("  ...contradicted", pct(r["wrong_contradicted"]), "a reviewer objected (good)"),
        ("right fixes", str(r["right_claims"]), ""),
        ("  ...supported", pct(r["right_supported"]), "a reviewer agreed"),
        ("  ...contradicted", pct(r["right_contradicted"]), "a reviewer objected (the price of cross-checking)"),
        ("fixes reviewed", pct(r["claims_reviewed"]), "fixes that had at least one reviewer"),
        ("pointers followed", str(r["transfers"]), f"{r['transfer_rescued']} found the owner the router missed"),
        ("escalations", str(r["escalations"]), f"{r['escalation_rescued']} ended with a correct fix"),
        ("agent calls", f"{r['calls_per_incident']:.2f}", f"per incident ({r['failed_calls']} failed in total)"),
        ("cost", f"{r['cost_per_incident']:.2f}", "agent cost units per incident"),
    ]
    print(_table(rows, ("measure", "value", "meaning")))

    print("\nthe plan (Block 5c: optimizer + policy agent)")
    rows = [
        ("acts on its own (remediate)", pct(r["plan_remediate"]), "runs without asking anyone"),
        ("a human approves (approve)", pct(r["plan_approve"]),
         f"a human says yes before each stage; {r['approve_wasted']} held no correct fix (check wasted)"),
        ("escalated to a human", pct(1 - r["plan_remediate"] - r["plan_approve"]), "a human starts from zero"),
        ("plan contains a correct fix", pct(r["plan_has_fix"]), "in some stage"),
        ("...in stage 1", pct(r["first_stage_right"]), "fixed by the first thing tried"),
        ("extra actions run", f"{r['extra_actions']:.2f}", "per incident: wrong actions run before the fix"),
        ("escalated though fixable", str(r["escalated_but_fixable"]), "a correct fix existed, but acting looked too risky"),
        ("needs a human's OK", pct(r["needs_human"]), "policy agent: approval plans and escalations"),
        ("not approved", str(r["not_approved"]), "plans the policy agent refused (a human takes over)"),
    ]
    print(_table(rows, ("measure", "value", "meaning")))
    lo = r["loss"]
    print("\nrealized loss per incident (lower is better; same incidents, same agent answers)")
    print(_table([("optimizer plan", f"{lo['plan']:.2f}"), ("top answer (most confident agent's fix)", f"{lo['top_claim']:.2f}"),
                  ("always a human", f"{lo['human']:.2f}"), ("oracle (best possible)", f"{lo['oracle']:.2f}")],
                 ("strategy", "loss")))
    if args.show:
        print(f"\nincidents with a problem ({len(r['mistakes'])}):" if r["mistakes"] else "\nno problems")
        for m in r["mistakes"]:
            print(m + "\n")
    return 0


def _pct(v) -> str:
    return "-" if v is None else f"{v:.0%}"


def _memory_tables(summary: dict) -> None:
    rows = []
    for key, c in summary["fixes"].items():
        agent, verdict = key.split(" ")
        known = c.get("right", 0) + c.get("wrong", 0)
        rows.append((agent, verdict, sum(c.values()), c.get("right", 0), c.get("wrong", 0), c.get("unknown", 0),
                     _pct(c.get("right", 0) / known if known else None)))
    print(_table(rows, ("agent", "verdict", "fixes", "right", "wrong", "unknown", "right of checked")))
    rows = [(a, c.get("right", 0), c.get("wrong", 0), c.get("cannot tell", 0),
             _pct(c.get("right", 0) / (c.get("right", 0) + c.get("wrong", 0))
                  if c.get("right", 0) + c.get("wrong", 0) else None))
            for a, c in summary["reviews"].items()]
    print("\nreviews (right = supported a right fix or contradicted a wrong one):")
    print(_table(rows, ("reviewer", "right", "wrong", "cannot tell / unchecked", "right of checked")))


def cmd_remember(lab: Lab, args) -> int:
    from .memory_eval import build_memory

    store, r = build_memory(args.data, split=args.split, model=args.model, calibration=args.calibration,
                            reasoner=args.reasoner, ollama_model=args.ollama_model, mode=args.feedback,
                            noise=args.noise, limit=args.limit, seed=args.seed, approval=not args.no_approval,
                            careless=args.careless, shy=args.shy, trust=args.trust, skip_faults=args.skip_faults)
    folder = store.save(args.out)
    print(f"remembered {r['incidents']} incidents ({r['split']}, oldest first); agents: {r['reasoner']}; "
          f"classifier {r['classifier']}; feedback {r['mode']}" + (f", {r['noise']:.0%} of operator answers wrong"
                                                                     if r["noise"] else "")
          + (f"; CARELESS: {', '.join(r['careless'])}" if r["careless"] else "")
          + (f"; SHY: {', '.join(r['shy'])}" if r["shy"] else "")
          + (f"; faults left out: {', '.join(r['skip_faults'])}" if r["skip_faults"] else ""))
    print(f"saved to {folder}")
    if r["trust"] != "off":
        print(f"\nlearning as it goes (trust {r['trust']}): each incident used the memory of the ones before it")
        print(_table([(f"{c['from']}-{c['to']}", f"{c['loss']:.2f}", f"{c['cost']:.2f}") for c in r["curve"]],
                     ("incidents", "loss", "agent cost")))
        print()
    print(f"plans: {r['plans']}   feedback from: {r['sources']}  (operator = a person was involved)\n")
    print("what the memory knows about each agent's fixes (Block 6b turns these counts into trust):")
    _memory_tables(r["summary"])
    print("\nlabel check against ground truth (lab only; a wrong label would teach Block 6b something false):")
    rows = [(lab_, c.get("truly right", 0), c.get("truly wrong", 0)) for lab_, c in r["labels"].items()]
    print(_table(rows, ("label", "truly right", "truly wrong")))
    o = r["owners"]
    print(f"verified owner: right {o.get('right', 0)}, wrong {o.get('wrong', 0)}, unknown {o.get('unknown', 0)}")
    if r["reasoner"] == "rules":
        print("note: rule agents and the generator share knowledge; this shows the memory works, not a result.")
    return 0


def cmd_eval_memory(lab: Lab, args) -> int:
    from controller.memory import ExperienceStore

    from .memory_eval import evaluate_retrieval

    store = ExperienceStore.load(args.memory)
    r = evaluate_retrieval(store, args.data, split=args.split, k=args.k, limit=args.limit, model=args.model)
    print(f"{r['queries']} {r['split']} incidents searched in a memory of {r['memory']} (k={r['k']})\n")
    rows = [("nearest past case is the same fault", _pct(r["same_fault_top1"]), "near copies make this easy"),
            ("nearest past case has the right owner", _pct(r["owner_top1"]), "its verified owner"),
            (f"owner by similarity vote of {r['k']}", _pct(r["owner_vote"]), "")]
    if r["classifier"] is not None:
        rows.append(("classifier's top guess (for comparison)", _pct(r["classifier"]), "Block 2b"))
    rows += [("leave-one-fault-out: nearest has the right owner", _pct(r["lofo_owner_top1"]),
              "every past case of the same fault hidden"),
             ("leave-one-fault-out: owner by vote", _pct(r["lofo_owner_vote"]), ""),
             ("similarity of the nearest case", f"{r['mean_top1_similarity'] or 0:.2f}",
              f"leave-one-fault-out: {r['mean_lofo_top1_similarity'] or 0:.2f}")]
    print(_table(rows, ("measure", "value", "meaning")))
    print("\nper fault (leave-one-fault-out: what an unseen fault looks like to the memory):")
    rows = [(f, c["owner"], c["n"], _pct(c["same_fault"]), _pct(c["lofo_owner"]), c["lofo_lookalike"] or "-")
            for f, c in r["per_fault"].items()]
    print(_table(rows, ("fault", "owner", "n", "nearest = same fault", "LOFO right owner", "LOFO nearest fault")))
    return 0


def cmd_evaluate(lab: Lab, args) -> int:
    from .evaluate import evaluate_classifier

    classifier = None
    if args.model:
        from controller.learned import LogRegClassifier

        classifier = LogRegClassifier.load(args.model)
    result = evaluate_classifier(args.data, split=args.split, classifier=classifier)
    o = result["overall"]
    print(f"{result['method']} classifier on {args.data} (split: {result['split']}, n={o['n']})")
    print(f"  owner accuracy (top guess)   {o['owner_accuracy']:.1%}")
    print(f"  true owner on the team       {o['owner_on_team']:.1%}")
    print(f"  mean team size               {o['mean_team_size']:.2f} domains")
    print(f"  confident on                 {o['confident_share']:.1%} of incidents, "
          f"accuracy when confident {o['accuracy_when_confident'] or 0:.1%}\n")
    rows = [(f, FAULTS[f].owner, s["n"], f"{s['owner_accuracy']:.0%}", f"{s['owner_on_team']:.0%}",
             f"{s['mean_team_size']:.1f}", f"{s['confident_share']:.0%}") for f, s in result["per_fault"].items()]
    print(_table(rows, ("fault", "owner", "n", "top guess right", "owner on team", "team size", "confident")))
    print("\nconfusion (rows = true owner, columns = top guess):")
    domains = list(result["confusion"])
    print(_table([(owner, *result["confusion"][owner].values()) for owner in domains], ("", *domains)))
    return 0


def cmd_reset(lab: Lab, args) -> int:
    print(json.dumps(revert_all(lab), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m oncall", description=(
        "Lab CLI for the adaptive A2A on-call project (Phase 1: sandbox, faults, oracle; no AI)."))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="write runtime config files from runtime/defaults")
    p.add_argument("--force", action="store_true", help="overwrite existing live configs")
    p.set_defaults(fn=cmd_init)

    sub.add_parser("health", help="run every health check once").set_defaults(fn=cmd_health)

    p = sub.add_parser("watch", help="poll health continuously and log incidents")
    p.add_argument("--interval", type=float, default=5.0)
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("faults", help="list faults/actions, or inject, fix, revert one fault")
    fsub = p.add_subparsers(dest="action", required=True)
    fsub.add_parser("list", help="show the fault catalogue")
    fsub.add_parser("actions", help="show the remediation action allowlist")
    q = fsub.add_parser("inject", help="inject one fault")
    q.add_argument("fault_id")
    q.add_argument("-p", "--param", action="append", default=[], metavar="KEY=VALUE")
    q = fsub.add_parser("fix", help="apply a fault's reference fix")
    q.add_argument("fault_id")
    q = fsub.add_parser("revert", help="undo one fault, or 'all'")
    q.add_argument("fault_id")
    p.set_defaults(fn=cmd_faults)

    p = sub.add_parser("validate", help="for each fault: inject, detect, reference-fix, recover, revert")
    p.add_argument("fault_ids", nargs="*", help="default: every fault")
    p.set_defaults(fn=cmd_validate)

    sub.add_parser("reset", help="revert every fault and restore default configs").set_defaults(fn=cmd_reset)

    p = sub.add_parser("synth", help="generate a labelled synthetic incident set (no Docker needed)")
    p.add_argument("--per-fault", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("data/synthetic/dev"))
    p.add_argument("--faults", nargs="*", default=[], help="default: every fault")
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("evaluate", help="score the controller's classification on a labelled incident set")
    p.add_argument("--data", type=Path, default=Path("data/synthetic/dev"))
    p.add_argument("--split", choices=["train", "val", "test"], default=None)
    p.add_argument("--model", type=Path, default=None, help="a trained logreg model (default: the rules baseline)")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("calibrate", help="calibrate the router's conformal team and compare team strategies")
    p.add_argument("--data", type=Path, default=Path("data/synthetic/dev"))
    p.add_argument("--model", type=Path, default=None, help="trained logreg model (default: rules)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="alpha of the calibration to SAVE: 0.05 -> owner on team >= 95%%")
    p.add_argument("--method", choices=["lac", "mondrian"], default="mondrian", help="method of the saved calibration")
    p.add_argument("--alphas", default=None, help="comma-separated sweep, default 0.01,0.02,0.03,0.05,0.10,0.20")
    p.add_argument("--out", type=Path, default=Path("models/conformal.json"))
    p.add_argument("--final", action="store_true", help="calibrate on all of val, report on test (once, at the end)")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("eval-agents", help="score every expert agent's proposals against ground truth")
    p.add_argument("--data", type=Path, default=Path("data/synthetic/dev"))
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--reasoner", choices=["rules", "ollama"], default="rules")
    p.add_argument("--model", default=None, help="Ollama model override (default: each agent card's model)")
    p.add_argument("--agents", nargs="*", default=[], help="only these agents (default: all)")
    p.add_argument("--limit", type=int, default=None, help="only the first N incidents (useful with ollama)")
    p.add_argument("--show", action="store_true", help="print every wrong answer next to the right one")
    p.set_defaults(fn=cmd_eval_agents)

    p = sub.add_parser("eval-team", help="score routing + agent rounds (investigate, pointers, cross-check)")
    p.add_argument("--data", type=Path, default=Path("data/synthetic/dev"))
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--model", type=Path, default=None, help="trained logreg model (default: rules classifier)")
    p.add_argument("--calibration", type=Path, default=None, help="conformal calibration (default: fixed)")
    p.add_argument("--reasoner", choices=["rules", "ollama"], default="rules")
    p.add_argument("--ollama-model", default=None, help="Ollama model override (default: each card's model)")
    p.add_argument("--cross-check", choices=["always", "contested", "never"], default="always")
    p.add_argument("--no-pointers", action="store_true")
    p.add_argument("--no-escalation", action="store_true", help="do not ask a stronger agent when nobody claims")
    p.add_argument("--careless", nargs="*", default=[], help="make these agents over-eager (robustness test)")
    p.add_argument("--shy", nargs="*", default=[],
                   help="these agents find the fix but say 'not mine', as the small LLMs did (robustness test)")
    p.add_argument("--second-look-below", type=float, default=0.7,
                   help="ask a stronger agent when the best plan is less sure than this (0 = never)")
    p.add_argument("--may-abstain", action="store_true",
                   help="earlier design: an agent may answer 'not mine' without a fix (default: every chosen "
                        "agent must propose its best fix)")
    p.add_argument("--no-approval", action="store_true",
                   help="earlier design: no 'a human approves the fix' option; every plan is priced and scored "
                        "as if it ran on its own")
    p.add_argument("--memory", type=Path, default=None,
                   help="Block 6b: trust agents by their verified record in this experience memory (read only)")
    p.add_argument("--trust", choices=["local", "global", "off"], default="local",
                   help="agents' verified record: local = whole memory + similar past incidents; global = whole "
                        "memory only; off = planning numbers")
    p.add_argument("--no-past-cases", action="store_true", help="do not show similar past incidents to the experts")
    p.add_argument("--no-precedent", action="store_true", help="the optimizer ignores what fixed similar incidents")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--show", action="store_true", help="print the rounds of every incident that went wrong")
    p.set_defaults(fn=cmd_eval_team)

    p = sub.add_parser("remember", help="Block 6a: fill the experience memory from verified (simulated) outcomes")
    p.add_argument("--data", type=Path, default=Path("data/synthetic/dev"))
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--model", type=Path, default=None, help="trained logreg model (default: rules classifier)")
    p.add_argument("--calibration", type=Path, default=None, help="conformal calibration (default: fixed)")
    p.add_argument("--reasoner", choices=["rules", "ollama"], default="rules")
    p.add_argument("--ollama-model", default=None, help="Ollama model override (default: each card's model)")
    p.add_argument("--feedback", choices=["bandit", "full"], default="bandit",
                   help="bandit: an operator only reviews incidents a person was involved in (realistic); "
                        "full: an operator reviews every incident (upper bound)")
    p.add_argument("--noise", type=float, default=0.0, help="share of operator answers that are wrong (C3)")
    p.add_argument("--seed", type=int, default=0, help="random seed for --noise")
    p.add_argument("--no-approval", action="store_true", help="earlier design: no 'a human approves' plans")
    p.add_argument("--careless", nargs="*", default=[], help="make these agents over-eager (robustness test)")
    p.add_argument("--shy", nargs="*", default=[], help="these agents find the fix but say 'not mine'")
    p.add_argument("--skip-faults", nargs="*", default=[],
                   help="leave these faults out of the memory (to test history on kinds of incident it never saw)")
    p.add_argument("--trust", choices=["off", "local", "global"], default="off",
                   help="off: only remember (Block 6a); local/global: learn as you go, each incident using the "
                        "memory of the ones before it (Block 6b)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("models/memory"))
    p.set_defaults(fn=cmd_remember)

    p = sub.add_parser("eval-memory", help="Block 6a: does the memory find similar past incidents?")
    p.add_argument("--memory", type=Path, default=Path("models/memory"))
    p.add_argument("--data", type=Path, default=Path("data/synthetic/dev"), help="the dataset the memory came from")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--model", type=Path, default=None, help="also show this classifier's accuracy for comparison")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_eval_memory)

    p = sub.add_parser("train", help="train the learned classifier on the train split, report on val")
    p.add_argument("--data", type=Path, default=Path("data/synthetic/dev"))
    p.add_argument("--out", type=Path, default=Path("models/logreg.json"))
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="auto = GPU if PyTorch with CUDA is installed, else CPU")
    p.set_defaults(fn=cmd_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(Lab(), args)
