"""Draw docs/figures/system-flow.svg: how one incident flows through the project's files.

    python docs/figures/make_flow_figure.py          # writes system-flow.svg next to this script

Black and white on purpose (prints cleanly). The PNG is rendered from the SVG with headless Chrome.
"""
import html
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "system-flow.svg"
W, H = 1800, 1255
SANS = "Segoe UI, Arial, Helvetica, sans-serif"
MONO = "Consolas, Menlo, 'Courier New', monospace"
INK, MID, SOFT = "#111", "#444", "#777"
COLS = [76, 322, 568, 814, 1060, 1306]          # six columns, 188 wide, 58 apart
BW = 188
els: list[str] = []


def esc(s):
    return html.escape(s, quote=False)


def text(x, y, s, size=11.5, weight=400, family=SANS, fill=INK, anchor="start", italic=False, extra=""):
    style = ' font-style="italic"' if italic else ""
    els.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
               f'fill="{fill}" text-anchor="{anchor}"{style}{extra}>{esc(s)}</text>')


def label(x, y, s, anchor="start", size=10.5):
    """Arrow label with a white halo so it stays readable over lines."""
    text(x, y, s, size=size, fill="#333", anchor=anchor, italic=True,
         extra=' stroke="#fff" stroke-width="4" stroke-linejoin="round" paint-order="stroke"')


def seg_line(x, y, segs, size=11.5):
    parts = "".join(
        f'<tspan font-family="{MONO if k == "f" else SANS}" fill="{INK if k == "f" else MID}">{esc(s)}</tspan>'
        for s, k in segs)
    els.append(f'<text x="{x}" y="{y}" font-size="{size}">{parts}</text>')


def vtext(x, cy, s, size, weight, fill):
    els.append(f'<text transform="translate({x},{cy}) rotate(-90)" text-anchor="middle" font-family="{SANS}" '
               f'font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(s)}</text>')


def rect(x, y, w, h, sw=1.6, stroke=INK, rx=6, dash=None, fill="#fff"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    els.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
               f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, sw=1, stroke=SOFT, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    els.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def step(cx, cy, n):
    els.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="#fff" stroke="{INK}" stroke-width="1.5"/>')
    text(cx, cy + 4.2, str(n), size=11.5 if n < 10 else 10.5, weight=700, anchor="middle")


def arrow(pts, dash=False, double=False, sw=1.5):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    a = f'fill="none" stroke="{INK}" stroke-width="{sw}" marker-end="url(#ah)"'
    if double:
        a += ' marker-start="url(#ah)"'
    if dash:
        a += ' stroke-dasharray="6 4"'
    els.append(f'<path d="{d}" {a}/>')


def box(x, y, w, h, title=None, n=None, lines=(), kind="main", title_size=13.5):
    """A step (bold border) or a helper file (thin border). Lines starting with ` are file names."""
    if kind == "main":
        rect(x, y, w, h)
    else:
        rect(x, y, w, h, sw=1, stroke="#555", rx=4)
    if title:
        tx = x + 10
        if n is not None:
            step(x + 20, y + 20, n)
            tx = x + 37
        text(tx, y + 25, title, size=title_size, weight=700)
        cy = y + 46
    else:
        cy = y + 20
    prev = None
    for ln in lines:
        k = "d" if isinstance(ln, str) and not ln.startswith("`") else "f"
        if prev and k != prev:
            cy += 4
        if isinstance(ln, str):
            if k == "f":
                text(x + 10, cy, ln[1:], family=MONO, size=11.5)
            else:
                text(x + 10, cy, ln, fill=MID, size=11)
        else:
            seg_line(x + 10, cy, ln)
        prev = k
        cy += 16


# ---------------------------------------------------------------- frame
els.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fff"/>')
text(30, 38, "a2a-oncall: how one incident flows through the project files", size=20, weight=700)

LANES = [("SANDBOX", "sandbox/ · Docker", 58, 280),
         ("LAB", "oncall/ · the exam board", 326, 516),
         ("CONTROLLER", "controller/ · pipeline.py runs steps 5–9", 566, 876),
         ("AGENTS", "agents/ · reached over A2A", 916, 1192)]
for name, sub, y0, y1 in LANES:
    rect(30, y0, 1490, y1 - y0, sw=1.2, stroke=SOFT, rx=8)
    line(66, y0, 66, y1)
    vtext(47, (y0 + y1) / 2, name, 12.5, 700, INK)
    vtext(61, (y0 + y1) / 2, sub, 10.5, 400, MID)

# ---------------------------------------------------------------- lane A: the practice company
text(76, 86, "The practice company: orders come in → get loaded → stored → cleaned → shown on a dashboard",
     size=12.5, fill=MID, italic=True)
box(COLS[0], 100, BW, 92, "Orders come in", lines=["`orders_api/app.py", "fake food-delivery orders"])
box(COLS[1], 100, BW, 92, "Load them", lines=["`ingest/ingest.py", "`ingest/schema.py", "pulls + checks new orders"])
box(COLS[2], 100, BW, 92, "Store them", lines=["`warehouse/init.sql", "Postgres database; also",
                                                "backup-compress.sh (CPU hog)"])
box(COLS[3], 100, BW, 92, "Clean them", lines=["`transform/runner.py", "`transform/dbt/", "raw orders → revenue table"])
box(COLS[4], 100, BW, 92, "Show them", lines=["`metrics_api/app.py", "revenue dashboard + login"])
rect(COLS[5], 100, BW, 42)
text(COLS[5] + 10, 118, "Bad actors", size=13, weight=700)
text(COLS[5] + 10, 134, "toolbox/*.py · S1 S2 D4 D5", family=MONO, size=11)
rect(COLS[5], 150, BW, 42)
text(COLS[5] + 10, 168, "Health numbers", size=13, weight=700)
text(COLS[5] + 10, 184, "prometheus/ + exporter", family=MONO, size=11)
for i in range(4):
    arrow([(COLS[i] + BW, 146), (COLS[i + 1], 146)])
# toxiproxy sits on the first two lines
rect(200, 212, 560, 50, sw=1, stroke="#555", rx=4)
text(212, 232, "toxiproxy/toxiproxy.json", family=MONO, size=11.5)
text(212, 250, "a switchboard on these two lines: the lab adds delay or cuts the line (network faults)",
     size=11, fill=MID)
line(293, 152, 293, 212, stroke="#555", dash="3 3")
line(539, 152, 539, 212, stroke="#555", dash="3 3")
rect(COLS[3], 212, 434, 50, sw=1, stroke="#555", rx=4)
text(COLS[3] + 12, 232, "runtime/defaults/*.json", family=MONO, size=11.5)
text(COLS[3] + 12, 250, "live settings the services read; faults and fixes edit them", size=11, fill=MID)

# ---------------------------------------------------------------- lane B: the lab
LB = 360
box(COLS[0], LB, BW, 136, "Break it on purpose", n=1,
    lines=["`oncall/faults.py", "`oncall/episodes.py", "14 faults, each with its", "true owner; fire drill:",
           "break → spot → fix → undo"])
box(COLS[1], LB, BW, 136, "Spot the problem", n=2,
    lines=["`oncall/oracle.py", "`oncall/incidents.py", "checks health against", "targets, opens an incident"])
box(COLS[2], LB, BW, 136, "Practice incidents",
    lines=["`oncall/synth/", "`→ data/synthetic/", "`samples/", "realistic fake incidents;", "no Docker needed"])
box(COLS[3], LB, BW, 136, "Train & calibrate",
    lines=["`oncall/train.py", "`oncall/route_eval.py", "writes models/*.json,", "used in steps 4 and 5"])
box(COLS[4], LB, BW, 136, "Answer key (locked)",
    lines=["`oncall/lab.py", "`ground_truth.jsonl", "true cause, owner, right fix;", "the controller never sees it"])
box(COLS[5], LB, BW, 136, "Mark the result",
    lines=["`oncall/evaluate.py", "`oncall/agent_eval.py", "`oncall/team_eval.py", "compares answers with",
           "the answer key"])
arrow([(170, LB), (170, 282)])
label(178, 312, "inject a fault")
arrow([(416, 282), (416, LB)])
label(424, 312, "health readings")
arrow([(COLS[4], 428), (COLS[3] + BW, 428)])
label(1031, 421, "labels", anchor="middle")
arrow([(COLS[4] + BW, 428), (COLS[5], 428)])
label(1277, 421, "labels", anchor="middle")

# ---------------------------------------------------------------- information barrier
line(30, 541, 1520, 541, sw=2, stroke=INK, dash="10 6")
rect(815, 523, 430, 36, sw=0, stroke="#fff", rx=0)
text(1030, 538, "INFORMATION BARRIER", size=12, weight=700, anchor="middle")
text(1030, 553, "the controller never sees the answer key (tests/test_controller_isolation.py)",
     size=10.5, fill=MID, anchor="middle")

# incident report crosses the barrier into step 3; practice incidents can replace the live lab
arrow([(416, 496), (416, 592), (170, 592), (170, 620)])
label(182, 610, "IncidentReport (contracts.py): alerts, logs, evidence only")
arrow([(662, 496), (662, 592), (422, 592)], dash=True)
label(670, 586, "or: practice incidents (no Docker)")

# ---------------------------------------------------------------- lane C: the controller
LC = 620
box(COLS[0], LC, BW, 130, "Read the incident", n=3,
    lines=["`intake.py", "cleans the logs, hides", "changing numbers and IPs,", "pulls out the key signals"])
box(COLS[1], LC, BW, 130, "Guess the department", n=4,
    lines=["`classifier.py · learned.py", "rules or a trained model:", "chance it is network, data,",
           "infra or security"])
box(COLS[2], LC, BW, 130, "Choose the team", n=5,
    lines=["`conformal.py · router.py", "team that should hold the", "right owner 95% of the time;",
           "main agent, helpers, backups"])
box(COLS[3], LC, BW, 130, "Ask the agents", n=6,
    lines=["`orchestrator.py", "asks agents in parallel,", "follows 'not mine' pointers,",
           "escalates, then cross-checks"])
box(COLS[4], LC, BW, 130, "Pick the best plan", n=8,
    lines=["`optimizer.py", "one staged plan with the", "lowest expected loss, or", "'call a human now'"])
box(COLS[5], LC, BW, 130, "Hand over the plan", n=10,
    lines=["`plan.py", "RemediationPlan goes out,", "ExecutionReport comes back"])
for a, b, word in [(0, 1, "features"), (1, 2, "scores"), (2, 3, "team"), (3, 4, "answers"), (4, 5, "plan")]:
    arrow([(COLS[a] + BW, 685), (COLS[b], 685)])
    label((COLS[a] + BW + COLS[b]) / 2, 678, word, anchor="middle")
# second look when the best plan is not sure enough
arrow([(1110, LC), (1110, 600), (958, 600), (958, LC)])
label(1034, 594, "not sure (< 70%)? second look by a stronger agent", anchor="middle")
# the plan is scored by the lab (development only)
arrow([(1400, LC), (1400, 496)], dash=True)
label(1392, 606, "scored by the lab", anchor="end")

LS = 786
box(COLS[0], LS, BW, 80, lines=["`evidence.py", "splits deeper evidence:", "one slice per domain"], kind="sub")
box(COLS[2], LS, BW, 80, lines=["`registry.py · cards.py", "`pool.py", "who exists, who is up,",
                                 "who may take the task"], kind="sub")
box(COLS[3], LS, BW, 80, lines=["`proposal.py · wire.py", "message forms, packed", "for A2A"], kind="sub")
box(COLS[4], LS, BW, 80, lines=["`policy.py", "the safety rules the", "policy agent applies"], kind="sub")
box(COLS[5], LS, BW, 80, lines=["`docs/HANDOVER.md", "`docs/contract/*.json", "the hand-over, written down"],
    kind="sub")
arrow([(662, LS), (662, LC + 130)])
arrow([(908, LC + 130), (908, LS)], double=True)
arrow([(1154, LC + 130), (1154, LS)], double=True)

# ---------------------------------------------------------------- lane D: the agents
arrow([(170, LS + 80), (170, 950)])
label(178, 902, "evidence slice per domain")
arrow([(908, LS + 80), (908, 990)], double=True)
label(916, 902, "tasks down; Proposal / Review back")
arrow([(1154, LS + 80), (1154, 990)], double=True)
label(1162, 902, "plan down; verdict back")

rect(76, 950, 680, 228, sw=1.4, rx=8)
step(96, 972, 7)
text(114, 977, "Expert agents investigate", size=14, weight=700)
text(330, 977, "agents/specialist.py", family=MONO, size=11.5)
for i, dom in enumerate(["network", "data", "infra", "security"]):
    for j, tier in enumerate(["fast", "strong"]):
        x, y = 92 + i * 130, 992 + j * 34
        rect(x, y, 120, 26, sw=1, stroke="#555", rx=13)
        text(x + 60, y + 17.5, f"{dom}-{tier}", family=MONO, size=11, anchor="middle")
rect(614, 992, 126, 60, sw=1, stroke="#555", rx=13)
text(677, 1026, "generalist", family=MONO, size=11, anchor="middle")
text(92, 1072, "fast = qwen2.5:3b · strong = llama3 · each has a business card in agents/cards/*.json",
     size=11, fill=MID)
box(92, 1086, 190, 80, lines=["`tools.py", "read-only evidence; only", "the tools on its own card"], kind="sub")
box(296, 1086, 240, 80, lines=["`reasoners/", [("rules.py", "f"), ("   no AI", "d")],
                               [("ollama.py", "f"), ("  local LLM", "d")],
                               [("careless.py", "f"), ("  test liar", "d")]], kind="sub")
box(550, 1086, 190, 80, lines=["`controller/checks.py", "drops fixes that can't run", "or aren't backed by evidence"],
    kind="sub")

box(COLS[3], 990, BW, 124, "A2A doorway",
    lines=["`server.py", "real A2A web server", "`local.py", "same agents, no network"])
arrow([(756, 1050), (COLS[3], 1050)], double=True)
box(COLS[4], 990, BW, 124, "Safety check", n=9,
    lines=["`policy_agent.py", "approve, needs a human,", "or refuse; silence = refuse"])

# ---------------------------------------------------------------- executor (another team)
rect(1580, 58, 200, 818, sw=1.5, dash="7 5", rx=8)
text(1680, 86, "EXECUTOR", size=15, weight=700, anchor="middle")
text(1680, 106, "another team", size=12, fill=MID, anchor="middle")
text(1680, 123, "MCP / Microsoft Fabric", size=12, fill=MID, anchor="middle")
text(1680, 140, "out of scope", size=11.5, fill=MID, anchor="middle", italic=True)
box(1596, 158, 168, 110, "In the lab",
    lines=["`oncall/intents.py", "`oncall/actions.py", "11 intents →", "13 allowed fixes"])
arrow([(1596, 214), (1522, 214)])
label(1558, 207, "apply fix", anchor="middle")
arrow([(COLS[5] + BW, 662), (1580, 662)])
arrow([(1580, 716), (COLS[5] + BW, 716)])
text(1596, 666, "RemediationPlan in", family=MONO, size=11.5)
text(1596, 720, "ExecutionReport back", family=MONO, size=11.5)
text(1596, 760, "runs the plan stage by", size=11, fill=MID)
text(1596, 776, "stage; 'escalate' means", size=11, fill=MID)
text(1596, 792, "page a human", size=11, fill=MID)

# ---------------------------------------------------------------- legend
LY = 1222
arrow([(76, LY), (116, LY)])
text(124, LY + 4, "main flow", size=11.5, fill=MID)
arrow([(220, LY), (260, LY)], dash=True)
text(268, LY + 4, "practice data or scoring (lab only)", size=11.5, fill=MID)
arrow([(500, LY), (540, LY)], double=True)
text(548, LY + 4, "request and reply over A2A", size=11.5, fill=MID)
rect(740, LY - 10, 34, 20, sw=1.5, dash="5 4", rx=3)
text(782, LY + 4, "out of scope (another team)", size=11.5, fill=MID)
step(991, LY, 1)
text(1010, LY + 4, "order of the steps", size=11.5, fill=MID)
rect(1150, LY - 10, 34, 20, rx=3)
text(1192, LY + 4, "a step", size=11.5, fill=MID)
rect(1250, LY - 10, 34, 20, sw=1, stroke="#555", rx=3)
text(1292, LY + 4, "a helper file", size=11.5, fill=MID)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
       '<defs><marker id="ah" viewBox="0 0 10 10" refX="9.5" refY="5" markerWidth="9" markerHeight="9" '
       'markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
       f'<path d="M0,0.8 L10,5 L0,9.2 z" fill="{INK}"/></marker></defs>\n'
       + "\n".join(els) + "\n</svg>\n")
OUT.write_text(svg, encoding="utf-8")
print(f"wrote {OUT}")
