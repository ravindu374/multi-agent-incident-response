# Plan

## Research question

In an open pool of A2A agents whose competence overlaps and changes over time, can a controller that
(i) matches incidents to Agent Cards, (ii) uses calibrated prediction sets to decide how many agents to
involve, and (iii) learns only from verified outcomes, resolve as many incidents as consult-everyone with
far fewer agent calls, and adapt faster than static or LLM-only routers when agents join, fail, or degrade?

| | Contribution | Measured by |
|---|---|---|
| C1 (core) | Cost-aware conformal delegation: primary owner + consults chosen with a coverage guarantee | coverage vs agent calls, on the shared-symptom groups |
| C2 | Open-pool routing from Agent Cards (no retraining when agents change) | cold-start accuracy of a mid-stream Security agent; recovery after silent degradation |
| C3 | Learning from verified outcomes only, with credit for consulted agents | robustness to 20% wrong operator feedback |

## Phases (12 weeks)

| Weeks | Phase | Exit criterion |
|---|---|---|
| 1-2 | **Sandbox + fault lab (this commit)** | `python -m oncall validate` detects, fixes and reverts all 14 faults |
| 3-4 | A2A specialists: fast/strong variant per domain + one generalist; read-only MCP tools; Agent Cards; registry | any agent returns a typed `Diagnosis` + `ProposedAction` artifact for any incident |
| 5-6 | Dataset from validation episodes; baselines A (rules), B (LLM-only), C (embedding classifier), D (single generalist agent) | baseline table on the validation split |
| 7-8 | Router: hard filter -> success model -> conformal set -> team choice; controller-mediated consults; ownership transfer; validator + executor | full loop closes on single and compound faults |
| 9-10 | Routing simulator (latent agent competence); bandit learning with a verified-outcome gate; join/fail/degrade experiments | learning curves, regret |
| 11-12 | End-to-end runs, ablations, statistics, write-up, demo video | final report |

## Design decisions (settled)

- **Ownership = remediation domain.** For example, an ETL timeout caused by a CPU hog belongs to infra,
  even though the data team sees it first.
- **Success = the oracle,** never an LLM judge. An incident is resolved when every SLO check passes.
- **Ground truth comes from the injector,** not from annotating text. Incidents are built from the alerts
  and logs the fault actually produced.
- **Agents propose, the executor acts.** Actions are a typed allowlist (`oncall/actions.py`) with risk
  tiers. High-risk actions need human approval. Specialists never get write access.
- **A fix counts only if it hits the culprit.** Every proposed action must be executable (names what it
  acts on, e.g. which session), grounded (its ids appear in what the agent saw) and, in evaluation, aimed
  at the culprit the fault created (ground truth `fix_targets`). Choosing the right kind of fix for the
  wrong thing (stopping the waiting ingest session instead of the lock holder) is reported as
  "wrong target", not as solved. Checks: `controller/checks.py`, `oncall/agent_eval.py`.
- **The plan is chosen by expected loss, not by an LLM judge.** `controller/optimizer.py`: P(domain) from the
  classifier moved by agents' evidence, times P(fix | domain) from tier precision (self-reported confidence
  ignored), then the staged plan (most likely fix per unit of risk first, verify between stages) with minimum
  expected loss, or escalation to a human. Unsure (< 70%) -> a stronger agent's second look first. Without
  experience this safety costs agent calls (8.1 vs 3.6 units/incident on rule agents); learning agents'
  reliability from verified outcomes (Block 6, experience memory) is what should bring that cost down.
- **LLM agents observe before they judge, and know their domain.** In the first runs (qwen2.5:3b fast,
  llama3 strong) both models wrote "not mine" and then described their own domain's fault, because the
  answer schema asked for the verdict first. The schema now runs findings -> root_cause -> verdict ->
  points_to -> confidence -> actions (reviews: reason -> stance), and each prompt lists what the domain
  covers, taken from its specialists' Agent Card skills, plus one ownership rule (a warehouse query is a
  data-domain session even though it is also a postgres process). `agents/reasoners/ollama.py`.
- **Every agent the classifier chose must propose a fix; the optimizer combines them.** An agent can no
  longer answer "not mine" without a fix: it hands in its best fix for its domain and says, in its
  verdict, how strongly its evidence backs it (owner / unsure / not mine). Invented fixes are still dropped
  by the executable + grounded + located checks. The optimizer weighs each fix by P(domain) x tier
  precision x a verdict factor (owner 1, unsure 0.6, not mine 0.3) and every fix is cross-checked.
  Rule agents, dev100 val (280): when agents find the fix but say "not mine" (`--shy`, all 8 specialists,
  what the LLMs did), a correct fix reaches the optimizer in 99% of incidents instead of 21%, and the loss
  falls from 6.87 to 5.02; with honest agents nothing changes (3.59) except cost (8.1 -> 9.9 units: more
  reviews). The optimizer still acts in only 49% of the shy incidents (oracle 3.17), because the fixed
  verdict factor treats every "not mine" fix as weak: whether a given agent's "not mine" fixes are
  right is exactly what Block 6 learns per agent from verified outcomes. `--may-abstain` = earlier design.
  First LLM run of this design (qwen2.5:3b + llama3, 10 dev val incidents): a correct fix reached the
  optimizer in 60% of incidents (correct claims: 30% before), no incident was left without a fix (60%
  before), loss 6.22 -> 6.07 (oracle 4.99).
- **A fix never stops the component itself, a rejected fix gets one repair turn, and a failed LLM call
  is a failed call.** From that run: both models put every pid of the warehouse into stop_process, pid 1
  and Postgres's own processes included (wrong, and one such plan ran), and llama3 killed a notebook's
  queries as processes (a session: stop_session, data domain); llama3 copied the mapping placeholder
  as a key; 4 qwen calls timed out and were read as "unsure 0.00", so no fallback agent was asked.
  Now: the located check rejects pid 1 and the warehouse's `postgres` processes (`controller/checks.py`
  OWN_PROCESSES: stop_process is a `kill -9`); the LLM sees exactly why a fix was rejected and answers
  once more (the agent still checks the second answer); the prompt shows a mapping example, not a
  placeholder; answers are bounded and calls to one Ollama server go one at a time; a call that still
  fails raises ReasonerFailed, so the A2A task fails and the controller asks the domain's fallback agent.
- **Three ways to act: alone, after a human's yes, or by a human from zero.** In the LLM run, 2 of 10
  incidents went to a human from zero although data-strong's mapping fix was right and 94-96% believed:
  a high-risk fix (cost 5) only beat a P3 human (6.4) at ~98%. Yet the policy agent already demanded a
  human's OK for it, and nothing priced that. Now the policy decides WHETHER a human must be involved
  (`controller/policy.py` approval_reasons: high risk, irreversible while < 80% sure, P1 while < 60%
  sure) and the optimizer decides HOW: plan kind "approve" = a human checks each stage before it runs,
  so a wrong fix never runs and a right one runs with a fifth of its risk; cost 1 per fix checked +
  0.15 x value(priority) of waiting (a human from zero: 4 + 0.3 x value). Assumed, as for escalation,
  that the human can tell a right fix from a wrong one. Approval is offered only for plans the policy
  holds back: priced alone, a human's check would also beat many safe fixes and nothing would run alone.
  Rule agents, dev100 val (280): loss 3.59 -> 3.28 (oracle under the same rules 3.04), correct fixes
  escalated anyway 20 -> 0, 14% of plans approved, none of them without a correct fix; shy agents
  5.02 -> 4.85, the rest of that gap being trust in "not mine" fixes (Block 6). `--no-approval` =
  earlier design, which reproduces the earlier numbers exactly.
- **The memory learns only what was verified, and says "unknown" otherwise (Block 6a).** Each closed
  incident becomes an experience (`controller/memory.py`): incident text + embedding, every proposal and
  review, the plan, and the verified outcome: the executor's report (stages run, healthy again or not)
  plus the operator's postmortem only when a person was involved. A fix is right if the operator says so
  or it is the fix that resolved it; wrong if the operator says so, it ran and the incident stayed broken,
  or it is in another domain than the owner; otherwise unknown (bandit feedback: S1 and N3 have two valid
  fixes, so an untried fix is not called wrong). Rule agents, dev100 train (840, oldest first): 1248 right,
  180 wrong, 178 unknown, every one consistent with ground truth; 126 incidents had an operator. Building
  it found a matching bug: a fix naming session 556 and one naming the reporting-worker sessions counted as
  the same fix (both act on the warehouse), so the optimizer merged a wrong guess into a right claim and a
  rule reviewer supported it; `checks.same_fix` now needs an overlapping identifier of the same kind
  (rules results unchanged: 3.28 / 4.85). Retrieval (hashing embedder, cosine; dev100 val vs train memory):
  the nearest past case is the same fault 91% of the time (similarity 1.00: synthetic incidents of one
  fault are near copies), owner by a 5-vote 93% (classifier 93%); with every past case of the same fault
  hidden, the nearest has the right owner only 8% (similarity 0.67), because the catalogue gives every
  fault a look-alike owned by another domain. So memory is trusted only for incidents it has really seen
  (high similarity); Block 6b needs a similarity threshold, not a blind kNN. Caveats: the memory comes from
  the split the classifier was trained on (sharper routing than on new incidents), and the synthetic
  splits are not ordered in time (prequential runs in 6c order by `opened_at`).
- **Agents are trusted by their verified record, not by tier guesses (Block 6b).** `controller/reliability.py`
  counts, per agent, what the memory verified; `controller/trust.py` turns counts into numbers by
  shrinkage: each planning number counts as 4 imaginary cases, the whole memory moves it, and past incidents
  at least 0.85 alike (cosine) move it again (2 imaginary cases), so an agent good at one kind of incident
  and bad at another is trusted only where it is good, and a new agent keeps the planning numbers. Measured
  per agent: success (right fix when its domain owned it) for the router; fix precision per verdict for the
  optimizer; and log-likelihood ratios of its verdicts (owned vs not owned) and reviews (reviewed fix right
  vs wrong), over the whole memory only, +1 smoothing, capped at 4. Self-reported confidence stays unused.
  With a record, "nobody claimed it" no longer triggers an escalation (that rule stood in for "no fix we
  trust"); the plan's confidence decides (second look). Rule agents, dev100 val (280), memory from train
  (840, bandit feedback), loss / agent cost per incident:

  | agents | planning numbers | memory, global | memory, local |
  |---|---|---|---|
  | honest | 3.28 / 9.89 | 3.28 / 3.93 | 3.28 / 3.51 |
  | careless data-fast | 3.44 / 9.89 | 3.29 / 4.69 | 3.29 / 3.51 |
  | shy (all 8 specialists) | 4.85 / 13.32 | 3.30 / 6.60 | 3.39 / 4.31 |

  (oracle 3.04). Shy: correct fixes escalated anyway 138 -> 0 (global) / 3 (local). Local loses those 3
  (I1 incidents, the I1/D5 look-alikes) because a shy agent's "not mine" carries no ownership evidence (the
  memory measures llr ~0), and local trust sends the second look to a second infra specialist, who agrees on
  the fix but cannot settle the domain, where global trust happens to ask the generalist, who still claims.
  Choosing the second-look agent by value of information (what its answer could change) is the fix; open.
  Learning as it goes (train, oldest first, each incident decided with the memory of the ones before it):
  honest agent cost per quarter 4.34, 4.59, 4.26, 3.70 vs 10.18, 10.16, 9.52, 9.97 without the memory (same
  loss); shy loss 3.42, 3.27, 3.70, 3.40 vs 4.69, 4.91, 4.87, 4.97, cost 6.30 -> 4.00 vs ~13.3. Wrong
  feedback (C3), local trust on val: 20% of operator answers wrong: loss 3.28 / 3.30 / 3.29 (honest /
  careless / shy), cost up to 4.29 / 4.18 / 6.93 (noisier records = less trust = more checking); an
  operator on every incident with 40% wrong: loss 3.35 / 3.37 / 3.74, cost 12.4-14.1, above the no-memory
  cost: past ~20% wrong, the memory stops paying for itself. Over real A2A (syn-7-00046, D1): data-strong
  asked first (100% right in 298 verified cases, 30 on incidents like this), 2 calls / 4.2 units instead of
  5 / 10.2, plan 97% sure instead of 87%.
- **Consults go through the controller.** A specialist asks for a *capability*, never a named agent, so
  every delegation is visible to the router, budgeted and logged.
- **Information barrier.** Agents never see `lab/`, Toxiproxy, orders-api `/admin`, or warehouse
  superuser access.

## Data strategy (option C)

Develop on synthetic incidents (`python -m oncall synth`), report final results on sandbox incidents
(`python -m oncall validate`). Tune nothing on the synthetic test split. Open item: make the sandbox's
`capture_evidence` also record the per-domain evidence bundle (`controller/evidence.py`), so both sources
have the same shape, and check the generator's log/dbt formats against real captures. Sandbox ground
truth also needs `fix_targets` (the injected session's pid, the burner pids, the client IP), taken from
each fault's injection details; until then sandbox incidents are scored without the culprit check.

## Phase 1 open items

- [ ] First full `validate` run; record which alerts each fault really produces; correct `symptom_tags`.
- [ ] Calibrate I1 / D5 / S2 intensity so each fires within ~2 min.
- [ ] Confirm `docker update` memory limits work in the Codespace (I2).
- [ ] Compound faults (e.g. I1 + D3), once single faults are stable.
