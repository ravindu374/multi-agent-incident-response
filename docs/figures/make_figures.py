"""Draw every figure of the project guide, in colour, into docs/figures/*.svg.

    python docs/figures/make_figures.py          # the SVGs (used by docs/guide/project-guide.html)
    python docs/figures/make_figures.py --png    # also a 2x PNG of each, rendered with headless Chrome

Colour system (the same in every figure):
  layers   sandbox green · lab rose · controller blue · agents violet · memory teal · executor grey
  domains  network blue · data orange · infra aqua · security yellow   (dataviz reference slots 1-4,
           validated: adjacent CVD dE 11.8, every bar and chip carries a text label)
  arrows   main flow ink · lab-only dashed rose · A2A request/reply violet · memory teal
"""
import html
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SANS = "Segoe UI, Arial, Helvetica, sans-serif"
MONO = "Consolas, Menlo, 'Courier New', monospace"
INK, MID, SOFT, RULE = "#1f2933", "#52606d", "#9aa5b1", "#cbd2d9"
LAYERS = {  # accent, lane tint, header tint
    "sandbox": ("#1f7a35", "#f1f8f3", "#d3ecd9"),
    "lab": ("#a8326a", "#fcf2f6", "#f5d7e5"),
    "controller": ("#1b5fb8", "#eff5fd", "#d1e2f8"),
    "agents": ("#5b3fd0", "#f4f1fd", "#ddd5fa"),
    "memory": ("#0f7470", "#ecf7f6", "#c8eae6"),
    "executor": ("#5f6b7a", "#f5f6f8", "#e1e5ea"),
}
DOMAIN = {"network": "#257de0", "data": "#f47230", "infra": "#00b894", "security": "#f2a600"}
OWNER = {"N1": "network", "N2": "network", "N3": "network", "D1": "data", "D2": "data", "D3": "data",
         "D4": "data", "D5": "data", "I1": "infra", "I2": "infra", "I3": "infra", "I4": "infra",
         "S1": "security", "S2": "security"}
BARRIER = "#c2372f"
GOOD, WARN, BAD = "#0aac23", "#fbb315", "#dc3c3d"   # status: always shown with an icon + words
BAR_LIGHT, BAR_DARK = "#84b9f3", "#1b64be"          # one hue, two ordinal steps
A_MAIN, A_LAB, A_A2A, A_MEM, A_EXE = INK, LAYERS["lab"][0], LAYERS["agents"][0], LAYERS["memory"][0], "#5f6b7a"


def mix(c, t):
    """Blend colour c towards white by t (0 = c, 1 = white)."""
    c = c.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return "#" + "".join(f"{round(v + (255 - v) * t):02x}" for v in (r, g, b))


def esc(s):
    return html.escape(str(s), quote=False)


class Canvas:
    def __init__(self, w, h):
        self.w, self.h, self.els, self.markers = w, h, [], {}
        self.els.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="#fff"/>')

    # ---- primitives
    def text(self, x, y, s, size=11.5, weight=400, family=SANS, fill=INK, anchor="start", italic=False, extra=""):
        st = ' font-style="italic"' if italic else ""
        self.els.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
                        f'fill="{fill}" text-anchor="{anchor}"{st}{extra}>{esc(s)}</text>')

    def label(self, x, y, s, anchor="start", size=10.5, fill="#3e4c59", italic=True, weight=400):
        self.text(x, y, s, size=size, fill=fill, anchor=anchor, italic=italic, weight=weight,
                  extra=' stroke="#fff" stroke-width="4" stroke-linejoin="round" paint-order="stroke"')

    def segs(self, x, y, parts, size=11.5):
        sp = "".join(f'<tspan font-family="{MONO if k == "f" else SANS}" fill="{INK if k == "f" else MID}">'
                     f'{esc(s)}</tspan>' for s, k in parts)
        self.els.append(f'<text x="{x}" y="{y}" font-size="{size}">{sp}</text>')

    def vtext(self, x, cy, s, size, weight, fill):
        self.els.append(f'<text transform="translate({x},{cy}) rotate(-90)" text-anchor="middle" font-family="{SANS}" '
                        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(s)}</text>')

    def rect(self, x, y, w, h, fill="#fff", stroke=INK, sw=1.4, rx=8, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{opacity}"' if opacity is not None else ""
        s = f'stroke="{stroke}" stroke-width="{sw}"' if stroke else 'stroke="none"'
        self.els.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" {s}{d}{o}/>')

    def line(self, x1, y1, x2, y2, stroke=SOFT, sw=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.els.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def path(self, d, fill="none", stroke=INK, sw=1.4):
        self.els.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def circle(self, cx, cy, r, fill, stroke="none", sw=0):
        self.els.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def step(self, cx, cy, n, color, r=10.5):
        s = str(n)
        size = 11 if len(s) == 1 else (10 if s.isdigit() else 9)
        self.circle(cx, cy, r, color)
        self.text(cx, cy + size * 0.36, s, size=size, weight=700, fill="#fff", anchor="middle")

    def status(self, cx, cy, kind, r=7):
        """Status icon: good = check, warn = !, bad = cross (never colour alone)."""
        col = {"good": GOOD, "warn": WARN, "bad": BAD}[kind]
        self.circle(cx, cy, r, col)
        if kind == "good":
            self.path(f"M{cx - 3.2},{cy + 0.2} L{cx - 0.8},{cy + 2.6} L{cx + 3.4},{cy - 2.6}", stroke="#fff", sw=1.8)
        elif kind == "bad":
            self.path(f"M{cx - 2.6},{cy - 2.6} L{cx + 2.6},{cy + 2.6} M{cx + 2.6},{cy - 2.6} L{cx - 2.6},{cy + 2.6}",
                      stroke="#fff", sw=1.8)
        else:
            self.text(cx, cy + 3.6, "!", size=10.5, weight=700, fill=INK, anchor="middle")

    def arrow(self, pts, color=A_MAIN, dash=False, double=False, sw=1.6):
        mid = self.markers.setdefault(color, "ah" + color.lstrip("#"))
        d = "M" + " L".join(f"{x},{y}" for x, y in pts)
        a = f'fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#{mid})"'
        if double:
            a += f' marker-start="url(#{mid})"'
        if dash:
            a += ' stroke-dasharray="6 4"'
        self.els.append(f'<path d="{d}" {a}/>')

    # ---- composites
    def box(self, x, y, w, h, layer, title=None, n=None, lines=(), kind="main", title_size=13.5, dashed=False):
        """A step (white card, tinted header band) or a helper file (thin card). Lines starting with ` are files."""
        acc, tint, head = LAYERS[layer]
        if kind == "main":
            self.rect(x, y, w, h, fill="#fff", stroke=acc, sw=1.6, rx=8, dash="6 4" if dashed else None)
            if title:
                self.rect(x + 0.8, y + 0.8, w - 1.6, 29, fill=head, stroke=None, rx=7.2)
                self.rect(x + 0.8, y + 16, w - 1.6, 14, fill=head, stroke=None, rx=0)
                self.rect(x, y, w, h, fill="none", stroke=acc, sw=1.6, rx=8, dash="6 4" if dashed else None)
        else:
            self.rect(x, y, w, h, fill="#fff", stroke=mix(acc, 0.35), sw=1.1, rx=6)
        cy = y + 20
        if title:
            tx = x + 10
            if n is not None:
                self.step(x + 19, y + 15, n, acc)
                tx = x + 35
            self.text(tx, y + 20, title, size=title_size, weight=700, fill=acc)
            cy = y + 48
        prev = None
        for ln in lines:
            k = "d" if isinstance(ln, str) and not ln.startswith("`") else "f"
            if prev and k != prev:
                cy += 4
            if isinstance(ln, str):
                if k == "f":
                    self.text(x + 10, cy, ln[1:], family=MONO, size=11.5)
                else:
                    self.text(x + 10, cy, ln, fill=MID, size=11)
            else:
                self.segs(x + 10, cy, ln)
            prev = k
            cy += 16

    def lane(self, x, y0, w, y1, layer, name, sub):
        acc, tint, _ = LAYERS[layer]
        self.rect(x, y0, w, y1 - y0, fill=tint, stroke=mix(acc, 0.45), sw=1.2, rx=10)
        self.rect(x, y0, 36, y1 - y0, fill=acc, stroke=None, rx=10)
        self.rect(x + 22, y0, 14, y1 - y0, fill=acc, stroke=None, rx=0)
        self.vtext(x + 16, (y0 + y1) / 2, name, 12.5, 700, "#fff")
        self.vtext(x + 30, (y0 + y1) / 2, sub, 10, 400, mix(acc, 0.8))

    def chip(self, x, y, w, h, s, color, size=11, mono=True):
        self.rect(x, y, w, h, fill=mix(color, 0.84), stroke=color, sw=1.2, rx=h / 2)
        self.text(x + w / 2, y + h / 2 + size * 0.36, s, family=MONO if mono else SANS, size=size, anchor="middle")

    def badge(self, x, y, fid, w=34, h=20):
        col = DOMAIN[OWNER[fid]]
        self.rect(x, y, w, h, fill=mix(col, 0.8), stroke=col, sw=1.5, rx=6)
        self.text(x + w / 2, y + 14, fid, size=11.5, weight=700, anchor="middle")

    def bar(self, x, y, w, h, fill, horizontal=False, r=4):
        """A bar anchored to its baseline, rounded only at the data end."""
        if w <= 0 or h <= 0:
            return
        r = min(r, (h if horizontal else w) / 2, (w if horizontal else h))
        if horizontal:   # baseline on the left
            d = (f"M{x},{y} H{x + w - r} A{r},{r} 0 0 1 {x + w},{y + r} V{y + h - r} "
                 f"A{r},{r} 0 0 1 {x + w - r},{y + h} H{x} Z")
        else:            # baseline at the bottom (y + h)
            d = (f"M{x},{y + h} V{y + r} A{r},{r} 0 0 1 {x + r},{y} H{x + w - r} "
                 f"A{r},{r} 0 0 1 {x + w},{y + r} V{y + h} Z")
        self.els.append(f'<path d="{d}" fill="{fill}"/>')

    def save(self, name):
        defs = "".join(
            f'<marker id="{mid}" viewBox="0 0 10 10" refX="9.5" refY="5" markerWidth="9" markerHeight="9" '
            f'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0.8 L10,5 L0,9.2 z" fill="{col}"/>'
            f'</marker>' for col, mid in self.markers.items())
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}">\n<defs>{defs}</defs>\n' + "\n".join(self.els) + "\n</svg>\n")
        out = HERE / f"{name}.svg"
        out.write_text(svg, encoding="utf-8")
        FIGS.append((out, self.w, self.h))
        print(f"wrote {out.name}  ({self.w}x{self.h})")


FIGS: list = []


# ======================================================================= the big picture
def fig_overview():
    c = Canvas(1120, 560)
    c.text(24, 34, "The big picture: one incident, start to finish", size=19, weight=700)
    c.text(24, 54, "Each card: what happens · the hospital version in italics · the step numbers used in the rest of the guide",
           size=11.5, fill=MID)

    def card(x, y, w, h, layer, badge, title, lines, analogy, dashed=False):
        acc = LAYERS[layer][0]
        c.box(x, y, w, h, layer, title=title, n=badge, dashed=dashed)
        for i, s in enumerate(lines):
            c.text(x + 12, y + 50 + i * 17, s, size=12, fill=INK)
        c.text(x + 12, y + h - 12, analogy, size=11.5, fill=acc, italic=True)

    card(24, 238, 170, 128, "sandbox", "2", "Something breaks", ["a check fails;", "an incident opens"], "a patient arrives")
    card(244, 222, 230, 160, "controller", "3", "Controller",
         ["reads the incident,", "guesses the owner team,", "chooses the agents (4–5)"], "triage nurse")
    card(524, 222, 200, 160, "controller", "8", "Best plan",
         ["prices every option;", "lowest expected loss,", "or 'call a human'"], "head doctor")
    card(774, 238, 152, 128, "agents", "9", "Safety check", ["approve, ask a", "human, or refuse"], "safety officer")
    card(976, 238, 122, 128, "executor", "10", "Executor", ["another team", "runs the fix"], "the nurses", dashed=True)
    # history agent + memory (top)
    acc_m = LAYERS["memory"][0]
    c.box(244, 72, 480, 100, "memory", title="History agent + experience memory", n="H")
    c.text(256, 122, "H1  before deciding: similar past incidents and each agent's verified record", size=12)
    c.text(256, 140, "H2  after the fix: remembers what really worked (verified outcomes only)", size=12)
    c.text(256, 162, "the medical records", size=11.5, fill=acc_m, italic=True)
    # expert agents (bottom)
    acc_a = LAYERS["agents"][0]
    c.box(244, 432, 480, 116, "agents", title="Expert agents, reached over A2A", n="7")
    for i, dom in enumerate(DOMAIN):
        for j, tier in enumerate(("fast", "strong")):
            c.chip(256 + i * 96, 470 + j * 28, 88, 22, f"{dom}-{tier}", DOMAIN[dom], size=10)
    c.chip(642, 470, 72, 50, "generalist", acc_a, size=10)
    c.text(256, 541, "the doctors: each checks only its own domain; second opinions", size=11.5, fill=acc_a, italic=True)
    # arrows
    c.arrow([(194, 302), (244, 302)])
    c.arrow([(474, 302), (524, 302)])
    c.label(499, 294, "answers", anchor="middle")
    c.arrow([(724, 302), (774, 302)])
    c.arrow([(926, 302), (976, 302)])
    c.arrow([(360, 382), (360, 432)], color=A_A2A, double=True)
    c.label(368, 412, "tasks down · proposals + reviews up (A2A)", fill=A_A2A)
    c.arrow([(360, 172), (360, 222)], color=A_MEM)
    c.label(368, 202, "H1 look up first", fill=A_MEM)
    c.arrow([(1037, 238), (1037, 122), (724, 122)], color=A_MEM)
    c.label(1029, 190, "H2 the outcome:", fill=A_MEM, anchor="end")
    c.label(1029, 204, "did it work?", fill=A_MEM, anchor="end")
    c.arrow([(624, 432), (624, 382)], color=A_A2A)
    c.save("fig-overview")


# ======================================================================= the practice company + faults
def fig_company():
    c = Canvas(1120, 420)
    c.text(24, 34, "The practice company, and where each of the 14 faults hits", size=19, weight=700)
    c.text(24, 54, "A small food-delivery analytics platform in Docker (sandbox/). Badge colour = the team that owns the fault.",
           size=11.5, fill=MID)
    # legend
    lx = 760
    for i, (dom, col) in enumerate(DOMAIN.items()):
        x = lx + i * 88
        c.rect(x, 70, 16, 14, fill=mix(col, 0.8), stroke=col, sw=1.5, rx=4)
        c.text(x + 22, 82, dom, size=11.5)
    xs = [31, 253, 475, 697, 919]
    W, Y, H = 170, 108, 104
    names = [("Orders API", ["`orders_api/app.py", "fake food-delivery", "orders (the source)"]),
             ("Ingest", ["`ingest/ingest.py", "loads new orders", "every 15 s"]),
             ("Warehouse", ["`warehouse/init.sql", "PostgreSQL database:", "raw + clean tables"]),
             ("Transform", ["`transform/dbt/", "dbt cleans the data", "every 30 s, with tests"]),
             ("Metrics API", ["`metrics_api/app.py", "revenue dashboard", "+ analyst login"])]
    for x, (t, ls) in zip(xs, names):
        c.box(x, Y, W, H, "sandbox", title=t, lines=ls)
    for i in range(4):
        c.arrow([(xs[i] + W, Y + 62), (xs[i + 1], Y + 62)])
    # toxiproxy on the first two lines
    for i in (0, 1):
        gx = (xs[i] + W + xs[i + 1]) / 2
        c.rect(gx - 22, Y + 70, 44, 16, fill=mix(DOMAIN["network"], 0.88), stroke=DOMAIN["network"], sw=1, rx=4)
        c.text(gx, Y + 82, "toxiproxy", size=8.5, anchor="middle", fill=INK)
    # fault badges
    g1 = (xs[0] + W + xs[1]) / 2
    g2 = (xs[1] + W + xs[2]) / 2
    c.badge(g1 - 17, Y + 4, "N1")
    c.badge(g1 - 17, Y + 28, "N3")
    c.badge(g2 - 17, Y + 28, "N2")
    under = {1: [["D1", "I3"]], 2: [["D3", "D4", "D5"], ["I1", "S1"]], 3: [["D2", "I2"]], 4: [["I4", "S2"]]}
    for i, rows in under.items():
        for r, row in enumerate(rows):
            total = len(row) * 34 + (len(row) - 1) * 6
            x0 = xs[i] + (W - total) / 2
            for k, fid in enumerate(row):
                c.badge(x0 + k * 40, Y + H + 12 + r * 26, fid)
    # health checks band
    acc_l, tint_l, _ = LAYERS["lab"]
    c.rect(31, 300, 1058, 54, fill=tint_l, stroke=mix(acc_l, 0.45), sw=1.2, rx=8)
    c.text(45, 322, "Health checks watch the platform (oncall/oracle.py)", size=13, weight=700, fill=acc_l)
    c.text(45, 342, "data freshness · ingest and transform job runs · database load · dashboard speed and errors  "
                    "→  when a check fails, an incident opens (oncall/incidents.py)", size=11.5, fill=INK)
    for x in xs:
        c.line(x + W / 2, Y + H + (64 if x == xs[2] else 38), x + W / 2, 300, stroke=mix(acc_l, 0.4), dash="3 3")
    c.text(31, 392, "Faults are caused by the lab (oncall/faults.py) through Toxiproxy, the toolbox 'bad actor' scripts, "
                    "the live settings files and SQL. The answer key records the true owner.", size=11, fill=MID)
    c.save("fig-company")


# ======================================================================= the build timeline
def fig_timeline():
    c = Canvas(1120, 330)
    c.text(24, 34, "How the build went (November 2026)", size=19, weight=700)
    days = ["19 Nov", "20 Nov", "21 Nov", "22 Nov", "23 Nov", "24 Nov", "27 Nov"]
    x0, colw = 30, 152
    c.line(x0, 70, x0 + colw * 7, 70, stroke=RULE, sw=1.5)
    for i, d in enumerate(days):
        cx = x0 + i * colw + colw / 2
        c.circle(cx, 70, 4, INK)
        c.text(cx, 60, d, size=12, weight=700, anchor="middle")
    bx = x0 + 6 * colw - 4
    c.text(bx, 74, "//", size=12, weight=700, fill=MID, anchor="middle")
    items = {
        0: [("lab", "Phase 1 sandbox", "code written, not run", "warn"), ("lab", "Synthetic data", "generator: 420 incidents", "good"),
            ("controller", "Blocks 1, 2a, 2b", "intake + classifiers", "good")],
        1: [("controller", "Blocks 3 + 4", "cards, registry, router", "good"), ("agents", "Block 5a", "agents as A2A servers", "good"),
            ("controller", "Blocks 5b + 5c", "orchestrator, optimizer", "good"), ("agents", "First LLM runs", "+ guardrails", "good")],
        2: [("executor", "Flow diagram", "first version", "good")],
        3: [("sandbox", "GitHub push", "first public commit", "good"), ("memory", "Blocks 6a + 6b", "memory + trust", "good")],
        4: [("memory", "Block 6c", "history agent", "good")],
        5: [("memory", "Block 6d", "memory in PostgreSQL", "good")],
        6: [("executor", "CV + this guide", "documentation", "good")],
    }
    for i, chips in items.items():
        for k, (layer, t1, t2, st) in enumerate(chips):
            acc, tint, head = LAYERS[layer]
            x, y = x0 + i * colw + 6, 88 + k * 48
            c.rect(x, y, colw - 12, 40, fill=tint, stroke=mix(acc, 0.3), sw=1.1, rx=6)
            c.rect(x, y, 5, 40, fill=acc, stroke=None, rx=2)
            c.status(x + 18, y + 13, st, r=6.5)
            c.text(x + 30, y + 17, t1, size=11.5, weight=700, fill=acc)
            c.text(x + 13, y + 33, t2, size=10.5, fill=MID)
    c.status(40, 312, "good", r=6.5)
    c.text(52, 316, "done", size=11, fill=MID)
    c.status(110, 312, "warn", r=6.5)
    c.text(122, 316, "written but never run (needs Docker, meant for a Codespace)", size=11, fill=MID)
    lx = 520
    for i, (layer, nm) in enumerate([("lab", "lab / data"), ("controller", "controller"), ("agents", "agents"),
                                     ("memory", "memory"), ("sandbox", "repository"), ("executor", "docs")]):
        acc = LAYERS[layer][0]
        c.rect(lx + i * 98, 305, 12, 12, fill=acc, stroke=None, rx=3)
        c.text(lx + i * 98 + 17, 316, nm, size=11, fill=MID)
    c.save("fig-timeline")


# ======================================================================= what the memory changed (3 small charts)
def fig_memory_effect():
    c = Canvas(1120, 320)
    c.text(24, 34, "What the memory changed", size=19, weight=700)
    c.text(24, 54, "dev100 validation set (280 incidents), rule agents, synthetic data · memory = 840 training incidents · "
                   "lighter bar = no memory, darker bar = memory + trust + past cases", size=11.5, fill=MID)
    panels = [("Agent cost per incident", "units (fast call = 1, strong = 4)", 9.89, 2.88, "{:.2f}", "about 71% less work"),
              ("Escalations", "over the 280 incidents", 220, 2, "{:.0f}", "almost never needs a senior"),
              ("Expected loss per incident", "lower is better", 3.28, 3.28, "{:.2f}", "plan quality unchanged")]
    base, top = 262, 104
    for p, (title, unit, a, b, fmt, note) in enumerate(panels):
        px = 30 + p * 365
        c.text(px, 88, title, size=13.5, weight=700)
        c.text(px + c_width(title, 13.5) + 8, 88, unit, size=11, fill=MID)
        vmax = max(a, b) * 1.18
        c.line(px, base, px + 300, base, stroke=SOFT, sw=1.2)
        for k, (v, col, nm) in enumerate(((a, BAR_LIGHT, "no memory"), (b, BAR_DARK, "with memory"))):
            bx = px + 50 + k * 120
            h = (base - top) * v / vmax
            c.bar(bx, base - h, 74, h, col)
            c.text(bx + 37, base - h - 7, fmt.format(v), size=13, weight=700, anchor="middle")
            c.text(bx + 37, base + 17, nm, size=11.5, fill=INK, anchor="middle")
        c.text(px + 50, base + 40, note, size=11.5, fill=BAR_DARK, italic=True)
    c.save("fig-memory-effect")


def c_width(s, size):
    return len(s) * size * 0.56


# ======================================================================= the detailed flow (one page, sideways in the PDF)
def fig_flow():
    W, H = 1800, 1255
    c = Canvas(W, H)
    COLS = [76, 322, 568, 814, 1060, 1306]
    BW = 188
    c.text(30, 38, "a2a-oncall: how one incident flows through the project files", size=20, weight=700)
    c.lane(30, 58, 1490, 280, "sandbox", "SANDBOX", "sandbox/ · Docker")
    c.lane(30, 326, 1490, 516, "lab", "LAB", "oncall/ · the exam board")
    c.lane(30, 566, 1490, 876, "controller", "CONTROLLER", "controller/ · pipeline.py runs one incident")
    c.lane(30, 916, 1490, 1192, "agents", "AGENTS", "agents/ · reached over A2A")

    # --- sandbox
    c.text(76, 86, "The practice company: orders come in → get loaded → stored → cleaned → shown on a dashboard",
           size=12.5, fill=MID, italic=True)
    for x, t, ls in [(COLS[0], "Orders come in", ["`orders_api/app.py", "fake food-delivery orders"]),
                     (COLS[1], "Load them", ["`ingest/ingest.py", "`ingest/schema.py", "pulls + checks new orders"]),
                     (COLS[2], "Store them", ["`warehouse/init.sql", "Postgres database; also", "backup-compress.sh (CPU hog)"]),
                     (COLS[3], "Clean them", ["`transform/runner.py", "`transform/dbt/", "raw orders → revenue table"]),
                     (COLS[4], "Show them", ["`metrics_api/app.py", "revenue dashboard + login"])]:
        c.box(x, 100, BW, 92, "sandbox", title=t, lines=ls)
    acc_s = LAYERS["sandbox"][0]
    for y, t, f in [(100, "Bad actors", "toolbox/*.py · S1 S2 D4 D5"), (150, "Health numbers", "prometheus/ + exporter")]:
        c.rect(COLS[5], y, BW, 42, fill="#fff", stroke=acc_s, sw=1.6, rx=8)
        c.text(COLS[5] + 10, y + 18, t, size=13, weight=700, fill=acc_s)
        c.text(COLS[5] + 10, y + 34, f, family=MONO, size=11)
    for i in range(4):
        c.arrow([(COLS[i] + BW, 146), (COLS[i + 1], 146)])
    c.box(200, 212, 560, 50, "sandbox", kind="sub",
          lines=["`toxiproxy/toxiproxy.json",
                 "a switchboard on these two lines: the lab adds delay or cuts the line (network faults)"])
    c.line(293, 152, 293, 212, stroke=DOMAIN["network"], dash="3 3", sw=1.2)
    c.line(539, 152, 539, 212, stroke=DOMAIN["network"], dash="3 3", sw=1.2)
    c.box(COLS[3], 212, 434, 50, "sandbox", kind="sub",
          lines=["`runtime/defaults/*.json", "live settings the services read; faults and fixes edit them"])

    # --- lab
    LB = 360
    for x, t, n, ls in [
        (COLS[0], "Break it on purpose", 1, ["`oncall/faults.py", "`oncall/episodes.py", "14 faults, each with its",
                                              "true owner; fire drill:", "break → spot → fix → undo"]),
        (COLS[1], "Spot the problem", 2, ["`oncall/oracle.py", "`oncall/incidents.py", "checks health against",
                                          "targets, opens an incident"]),
        (COLS[2], "Practice incidents", None, ["`oncall/synth/", "`→ data/synthetic/", "`samples/",
                                               "realistic fake incidents;", "no Docker needed"]),
        (COLS[3], "Train & calibrate", None, ["`oncall/train.py", "`oncall/route_eval.py", "writes models/*.json,",
                                              "used in steps 4 and 5"]),
        (COLS[4], "Answer key (locked)", None, ["`oncall/lab.py", "`ground_truth.jsonl", "true cause, owner, right fix;",
                                                "the controller never sees it"]),
        (COLS[5], "Mark the result", None, ["`oncall/evaluate.py", "`oncall/agent_eval.py", "`oncall/team_eval.py",
                                            "compares answers with", "the answer key"])]:
        c.box(x, LB, BW, 136, "lab", title=t, n=n, lines=ls)
    c.arrow([(170, LB), (170, 282)], color=A_LAB)
    c.label(178, 312, "inject a fault", fill=A_LAB)
    c.arrow([(416, 282), (416, LB)])
    c.label(424, 312, "health readings")
    c.arrow([(COLS[4], 428), (COLS[3] + BW, 428)], color=A_LAB)
    c.label(1031, 421, "labels", anchor="middle", fill=A_LAB)
    c.arrow([(COLS[4] + BW, 428), (COLS[5], 428)], color=A_LAB)
    c.label(1277, 421, "labels", anchor="middle", fill=A_LAB)

    # --- barrier
    c.line(30, 541, 1520, 541, stroke=BARRIER, sw=2.4, dash="11 6")
    c.rect(810, 522, 440, 38, fill="#fff", stroke=BARRIER, sw=1.4, rx=8)
    c.text(1030, 538, "INFORMATION BARRIER", size=12, weight=700, fill=BARRIER, anchor="middle")
    c.text(1030, 553, "the controller never sees the answer key (tests/test_controller_isolation.py)", size=10.5,
           fill=MID, anchor="middle")
    c.arrow([(416, 496), (416, 592), (170, 592), (170, 620)])
    c.label(182, 610, "IncidentReport (contracts.py): alerts, logs, evidence only")
    c.arrow([(662, 496), (662, 592), (422, 592)], color=A_LAB, dash=True)
    c.label(670, 586, "or: practice incidents (no Docker)", fill=A_LAB)

    # --- controller
    LC = 620
    for x, t, n, ls in [
        (COLS[0], "Read the incident", 3, ["`intake.py", "cleans the logs, hides", "changing numbers and IPs,",
                                           "pulls out the key signals"]),
        (COLS[1], "Guess the department", 4, ["`classifier.py · learned.py", "rules or a trained model:",
                                              "chance it is network, data,", "infra or security"]),
        (COLS[2], "Choose the team", 5, ["`conformal.py · router.py", "team that should hold the",
                                         "right owner 95% of the time;", "main agent, helpers, backups",
                                         "+ agents' track record (H1)"]),
        (COLS[3], "Ask the agents", 6, ["`orchestrator.py", "asks agents in parallel,", "follows 'not mine' pointers,",
                                        "escalates, then cross-checks", "+ shows past cases (H1)"]),
        (COLS[4], "Pick the best plan", 8, ["`optimizer.py", "one staged plan with the", "lowest expected loss, or",
                                            "'call a human now'", "+ counts precedent (H1)"]),
        (COLS[5], "Hand over the plan", 10, ["`plan.py", "RemediationPlan goes out,", "ExecutionReport comes back"])]:
        c.box(x, LC, BW, 130, "controller", title=t, n=n, lines=ls)
    for a, b, word in [(0, 1, "features"), (1, 2, "scores"), (2, 3, "team"), (3, 4, "answers"), (4, 5, "plan")]:
        c.arrow([(COLS[a] + BW, 685), (COLS[b], 685)])
        c.label((COLS[a] + BW + COLS[b]) / 2, 678, word, anchor="middle")
    c.arrow([(1110, LC), (1110, 600), (958, 600), (958, LC)], color=LAYERS["controller"][0])
    c.label(1034, 594, "not sure (< 70%)? second look by a stronger agent", anchor="middle",
            fill=LAYERS["controller"][0])
    c.arrow([(1400, LC), (1400, 496)], color=A_LAB, dash=True)
    c.label(1392, 606, "scored by the lab", anchor="end", fill=A_LAB)
    LS = 786
    c.box(COLS[0], LS, BW, 80, "controller", kind="sub", lines=["`evidence.py", "splits deeper evidence:", "one slice per domain"])
    c.box(COLS[2], LS, BW, 80, "controller", kind="sub",
          lines=["`registry.py · cards.py", "`pool.py", "who exists, who is up,", "who may take the task"])
    c.box(COLS[3], LS, BW, 80, "controller", kind="sub", lines=["`proposal.py · wire.py", "message forms, packed", "for A2A"])
    c.box(COLS[4], LS, BW, 80, "controller", kind="sub", lines=["`policy.py", "the safety rules the", "policy agent applies"])
    c.box(COLS[5], LS, BW, 80, "memory", kind="sub",
          lines=["`history.py · memory.py", "talks to the history agent;", "the Experience record"])
    c.arrow([(1400, LC + 130), (1400, LS)], color=A_MEM)
    c.step(1424, 768, "H2", A_MEM)
    c.arrow([(662, LS), (662, LC + 130)])
    c.arrow([(908, LC + 130), (908, LS)], color=A_A2A, double=True)
    c.arrow([(1154, LC + 130), (1154, LS)], color=A_A2A, double=True)

    # --- agents
    c.arrow([(170, LS + 80), (170, 950)])
    c.label(178, 902, "evidence slice per domain")
    c.arrow([(908, LS + 80), (908, 990)], color=A_A2A, double=True)
    c.label(916, 902, "tasks down; Proposal / Review back", fill=A_A2A)
    c.arrow([(1154, LS + 80), (1154, 990)], color=A_A2A, double=True)
    c.label(1162, 902, "plan down; verdict back", fill=A_A2A)
    c.arrow([(1400, LS + 80), (1400, 990)], color=A_MEM, double=True)
    c.label(1408, 896, "H1 look up first", fill=A_MEM)
    c.label(1408, 910, "H2 remember", fill=A_MEM)
    acc_a = LAYERS["agents"][0]
    c.rect(76, 950, 680, 228, fill="#fff", stroke=acc_a, sw=1.6, rx=10)
    c.step(96, 972, 7, acc_a)
    c.text(114, 977, "Expert agents investigate", size=14, weight=700, fill=acc_a)
    c.text(330, 977, "agents/specialist.py", family=MONO, size=11.5)
    for i, dom in enumerate(DOMAIN):
        for j, tier in enumerate(("fast", "strong")):
            c.chip(92 + i * 130, 992 + j * 34, 120, 26, f"{dom}-{tier}", DOMAIN[dom])
    c.chip(614, 992, 126, 60, "generalist", acc_a)
    c.text(92, 1072, "fast = qwen2.5:3b · strong = llama3 · each has a business card in agents/cards/*.json",
           size=11, fill=MID)
    c.box(92, 1086, 190, 80, "agents", kind="sub", lines=["`tools.py", "read-only evidence; only", "the tools on its own card"])
    c.box(296, 1086, 240, 80, "agents", kind="sub",
          lines=["`reasoners/", [("rules.py", "f"), (" no AI   ", "d"), ("ollama.py", "f"), (" LLM", "d")],
                 [("careless.py", "f"), (" · ", "d"), ("shy.py", "f"), ("  test agents", "d")],
                 "over-claims / under-claims"])
    c.box(550, 1086, 190, 80, "agents", kind="sub",
          lines=["`controller/checks.py", "drops fixes that can't run", "or aren't backed by evidence"])
    c.box(COLS[3], 990, BW, 124, "agents", title="A2A doorway",
          lines=["`server.py", "real A2A web server", "`local.py", "same agents, no network"])
    c.arrow([(756, 1050), (COLS[3], 1050)], color=A_A2A, double=True)
    c.box(COLS[4], 990, BW, 124, "agents", title="Safety check", n=9,
          lines=["`policy_agent.py", "approve, needs a human's yes,", "or refuse; silence = refuse"])
    c.box(COLS[5], 990, BW, 124, "memory", title="History agent", n="H",
          lines=["`history_agent.py", "`trust.py · reliability.py", "`memory_pg.py", "similar past incidents +",
                 "each agent's verified record"])

    # --- executor
    acc_x, tint_x, _ = LAYERS["executor"]
    c.rect(1580, 58, 200, 1134, fill=tint_x, stroke=acc_x, sw=1.5, dash="7 5", rx=10)
    c.text(1680, 86, "EXECUTOR", size=15, weight=700, fill=acc_x, anchor="middle")
    c.text(1680, 106, "another team", size=12, fill=MID, anchor="middle")
    c.text(1680, 123, "MCP / Microsoft Fabric", size=12, fill=MID, anchor="middle")
    c.text(1680, 140, "out of scope", size=11.5, fill=MID, anchor="middle", italic=True)
    c.box(1596, 158, 168, 110, "executor", title="In the lab",
          lines=["`oncall/intents.py", "`oncall/actions.py", "11 intents →", "13 allowed fixes"])
    c.arrow([(1596, 214), (1522, 214)], color=A_EXE)
    c.label(1558, 207, "apply fix", anchor="middle")
    c.arrow([(COLS[5] + BW, 662), (1580, 662)])
    c.arrow([(1580, 716), (COLS[5] + BW, 716)], color=A_EXE)
    c.text(1596, 666, "RemediationPlan in", family=MONO, size=11.5)
    c.text(1596, 720, "ExecutionReport back", family=MONO, size=11.5)
    c.text(1596, 760, "runs the plan stage by", size=11, fill=MID)
    c.text(1596, 776, "stage; 'escalate' means", size=11, fill=MID)
    c.text(1596, 792, "page a human", size=11, fill=MID)
    c.box(1596, 816, 168, 64, "executor", kind="sub",
          lines=["`docs/HANDOVER.md", "`docs/contract/*.json", "the contract, written down"])

    # --- legend
    LY = 1222
    c.arrow([(76, LY), (116, LY)])
    c.text(124, LY + 4, "main flow", size=11.5, fill=MID)
    c.arrow([(210, LY), (250, LY)], color=A_LAB, dash=True)
    c.text(258, LY + 4, "lab only (practice data, scoring)", size=11.5, fill=MID)
    c.arrow([(470, LY), (510, LY)], color=A_A2A, double=True)
    c.text(518, LY + 4, "request and reply over A2A", size=11.5, fill=MID)
    c.arrow([(700, LY), (740, LY)], color=A_MEM)
    c.text(748, LY + 4, "memory: look up (H1), remember (H2)", size=11.5, fill=MID)
    c.rect(966, LY - 10, 34, 20, fill=tint_x, stroke=acc_x, sw=1.5, dash="5 4", rx=4)
    c.text(1008, LY + 4, "out of scope", size=11.5, fill=MID)
    c.line(1100, LY, 1140, LY, stroke=BARRIER, sw=2.4, dash="8 5")
    c.text(1148, LY + 4, "information barrier", size=11.5, fill=MID)
    for k, (layer, nm) in enumerate([("sandbox", "sandbox"), ("lab", "lab"), ("controller", "controller"),
                                     ("agents", "agents"), ("memory", "memory")]):
        c.step(1300 + k * 96, LY, "", LAYERS[layer][0], r=7)
        c.text(1312 + k * 96, LY + 4, nm, size=11.5, fill=MID)
    c.save("system-flow")


# ======================================================================= worked example as two sequence diagrams
def fig_sequence():
    c = Canvas(1120, 520)
    c.text(24, 34, "Incident syn-7-00046: the same fix, without and with memory", size=19, weight=700)
    c.text(24, 54, "Real run, 27 Nov 2026 (rule agents). Solid arrow = a task, dashed arrow = the answer, (n) = cost units.",
           size=11.5, fill=MID)
    lifelines = [("Controller", "controller"), ("History", "memory"), ("data-fast", "agents"), ("data-strong", "agents"),
                 ("Policy", "agents")]
    LX = [52, 150, 252, 366, 476]
    TOP, BOT = 104, 448

    def panel(x0, title, used_history, msgs, footer, good):
        c.text(x0, 86, title, size=14.5, weight=700)
        for (nm, layer), lx in zip(lifelines, LX):
            acc, tint, head = LAYERS[layer]
            faded = (nm == "History" and not used_history)
            x = x0 + lx
            c.line(x, TOP + 26, x, BOT, stroke=mix(acc, 0.85 if faded else 0.45), sw=1.4, dash="4 4")
            c.rect(x - 46, TOP, 92, 26, fill=mix(head, 0.5) if faded else head, stroke=mix(acc, 0.7) if faded else acc,
                   sw=1.3, rx=6)
            c.text(x, TOP + 17, nm, size=11.5, weight=700, fill=mix(acc, 0.6) if faded else acc, anchor="middle")
            if faded:
                c.text(x, TOP - 6, "(not used)", size=10, fill=SOFT, anchor="middle", italic=True)
        for y, a, b, txt, reply, mark in msgs:
            xa, xb = x0 + LX[a], x0 + LX[b]
            c.arrow([(xa, y), (xb - 6 if xb > xa else xb + 6, y)], dash=reply, color=MID if reply else INK, sw=1.4)
            tx = min(xa, xb) + 8
            c.label(tx, y - 6, txt, size=10.5, italic=reply, fill=INK if not reply else "#3e4c59")
            if mark:
                c.status(tx + c_width(txt, 10.5) + 8, y - 10, mark, r=6)
        c.rect(x0, BOT + 14, 520, 34, fill=mix(BAR_DARK, 0.9) if good else "#f6f7f9",
               stroke=BAR_DARK if good else SOFT, sw=1.2, rx=8)
        c.text(x0 + 260, BOT + 36, footer, size=12.5, weight=700, anchor="middle")

    left = [(146, 0, 2, "investigate (1)", False, None),
            (172, 2, 0, "not mine; guess: stop_session", True, "bad"),
            (206, 0, 3, "nobody claimed → escalate: investigate (4)", False, None),
            (232, 3, 0, "OWNER 0.90: schema drift, map fields", True, "good"),
            (266, 0, 3, "review data-fast's guess (4)", False, None),
            (292, 3, 0, "contradicts", True, None),
            (326, 0, 2, "review the claim (1)", False, None),
            (352, 2, 0, "cannot tell", True, None),
            (386, 0, 4, "check the plan (0.2)", False, None),
            (412, 4, 0, "approved, needs a human's yes", True, None)]
    right = [(146, 0, 1, "look up (0.2)", False, None),
             (172, 1, 0, "5 similar cases; data-fast weak here", True, None),
             (206, 0, 3, "investigate (4)", False, None),
             (232, 3, 0, "OWNER 0.90: schema drift, map fields", True, "good"),
             (266, 0, 4, "check the plan (0.2)", False, None),
             (292, 4, 0, "approved, needs a human's yes", True, None)]
    panel(24, "Without memory", False, left, "5 calls · 10.2 units · plan 87% sure · loss 4.88", False)
    panel(576, "With memory (history agent)", True, right, "3 calls · 4.4 units · plan 99% sure · loss 4.38", True)
    c.line(560, 76, 560, 500, stroke=RULE, sw=1)
    c.save("fig-sequence")


# ======================================================================= the learning loop
def fig_loop():
    c = Canvas(920, 446)
    c.text(15, 28, "The learning loop (Block 6): each verified outcome makes the next decision cheaper", size=15, weight=700)
    X = [15, 245, 475, 705]
    TOP, BOT, BH = 50, 270, 130
    c.box(X[0], TOP, 200, BH, "sandbox", title="New incident",
          lines=["`IncidentReport", "alerts, logs, evidence", "(e.g. syn-7-00046)"])
    c.box(X[1], TOP, 200, BH, "memory", title="Look it up", n="H1",
          lines=["`history_agent.py", "similar past incidents", "(similarity ≥ 0.85) and", "each agent's verified record"])
    c.box(X[2], TOP, 200, BH, "controller", title="Decide",
          lines=["`pipeline.py", "route with trust (step 5),", "experts see past cases (6),", "precedent in the plan (8)"])
    c.box(X[3], TOP, 200, BH, "agents", title="Act", lines=["policy check (step 9),", "then the executor runs", "the plan (step 10)"])
    for a in range(3):
        c.arrow([(X[a] + 200, TOP + 65), (X[a + 1], TOP + 65)])
    c.box(X[3], BOT, 200, BH, "executor", title="Outcome comes back",
          lines=["`ExecutionReport", "did it fix it? plus the", "operator's note when a", "person was involved"])
    c.box(X[2], BOT, 200, BH, "memory", title="Remember", n="H2",
          lines=["`memory.py", "each agent's answer marked", "right / wrong / unknown,", "from verified facts only"])
    c.box(X[1], BOT, 200, BH, "memory", title="Experience memory",
          lines=["`memory_pg.py · trust.py", "files or PostgreSQL;", "now 840 past incidents", "(dev100 train split)"])
    acc_m, tint_m, _ = LAYERS["memory"]
    c.rect(X[0], BOT, 200, BH, fill=tint_m, stroke=acc_m, sw=1.2, dash="5 4", rx=8)
    c.text(X[0] + 10, BOT + 22, "Honesty rules", size=13.5, weight=700, fill=acc_m)
    for i, s in enumerate(["only verified outcomes count", "less alike than 0.85? ignored", "old ids are never copied",
                           "scored incidents never stored"]):
        c.status(X[0] + 16, BOT + 44 + i * 20, "good", r=5.5)
        c.text(X[0] + 27, BOT + 48 + i * 20, s, size=11, fill=INK)
    c.arrow([(X[3] + 100, TOP + BH), (X[3] + 100, BOT)], color=A_EXE)
    c.arrow([(X[3], BOT + 65), (X[2] + 200, BOT + 65)], color=A_MEM)
    c.arrow([(X[2], BOT + 65), (X[1] + 200, BOT + 65)], color=A_MEM)
    c.arrow([(X[1] + 100, BOT), (X[1] + 100, TOP + BH)], color=A_MEM)
    c.label(X[1] + 108, (TOP + BH + BOT) / 2 + 4, "next incident", fill=A_MEM)
    c.text(15, 432, "Measured on synthetic data with rule agents: agent cost per incident 9.89 → 2.88, escalations "
                    "220 → 2, plan quality unchanged (loss 3.28).", size=11.5, fill=MID, italic=True)
    c.save("learning-loop")


# ======================================================================= conformal prediction, two real incidents
def fig_conformal():
    c = Canvas(1120, 350)
    c.text(24, 34, "Conformal prediction chooses the team: one team when the model is sure, more when it is not",
           size=17, weight=700)
    c.text(24, 54, "Bar = the classifier's probability for each team. Black tick = that team's cut-off (Mondrian, α = 0.05). "
                   "A team joins when its bar reaches its tick.", size=11.5, fill=MID)
    cut = {"network": 0.91, "data": 0.36, "infra": 0.42, "security": 0.45}
    cases = [("syn-7-00046", "D1: source renamed a field (owner data)", {"network": 0.02, "data": 0.96, "infra": 0.01,
                                                                         "security": 0.00},
              "team = data · 1 agent team · sure"),
             ("syn-7-00004", "I1: unknown process burns CPU (owner infra)", {"network": 0.00, "data": 0.45,
                                                                              "infra": 0.52, "security": 0.02},
              "team = infra + data · the true owner (infra) is on it")]
    for p, (iid, desc, probs, result) in enumerate(cases):
        x0 = 24 + p * 556
        c.text(x0, 88, iid, size=14, weight=700, family=MONO)
        c.text(x0 + 104, 88, desc, size=11.5, fill=MID)
        tx, tw = x0 + 86, 330
        for k, dom in enumerate(DOMAIN):
            y = 108 + k * 40
            v = probs[dom]
            c.rect(x0, y + 4, 12, 12, fill=DOMAIN[dom], stroke=None, rx=3)
            c.text(x0 + 18, y + 15, dom, size=12)
            c.rect(tx, y, tw, 20, fill="#f2f4f7", stroke=None, rx=4)
            c.bar(tx, y, max(tw * v, 0), 20, DOMAIN[dom], horizontal=True)
            c.text(tx + tw * v + 6 if v < 0.85 else tx + tw * v - 6, y + 15, f"{v:.2f}", size=12, weight=700,
                   anchor="start" if v < 0.85 else "end", fill=INK)
            cx = tx + tw * cut[dom]
            c.line(cx, y - 4, cx, y + 24, stroke=INK, sw=2.2)
            c.text(cx, y + 34, f"{cut[dom]:.2f}" + ("*" if dom == "security" else ""), size=9.5, fill=MID, anchor="middle")
            on = v >= cut[dom]
            if on:
                c.status(tx + tw + 20, y + 10, "good", r=7)
                c.text(tx + tw + 32, y + 15, "on the team", size=11.5, fill=INK)
            else:
                c.text(tx + tw + 14, y + 15, "–  not called", size=11.5, fill=SOFT)
        c.rect(x0, 274, 520, 32, fill=mix(BAR_DARK, 0.9), stroke=BAR_DARK, sw=1.2, rx=8)
        c.text(x0 + 260, 295, result, size=12.5, weight=700, anchor="middle")
    c.text(24, 334, "* security has too few calibration incidents (18 < 19 needed at α = 0.05), so it uses the shared "
                    "(pooled) cut-off. The top guess is always on the team.", size=11, fill=MID)
    c.save("fig-conformal")


# ======================================================================= the optimizer's options
def fig_loss():
    c = Canvas(1120, 300)
    c.text(24, 34, "The optimizer prices every option and picks the lowest expected loss (incident syn-7-00046)",
           size=17, weight=700)
    c.text(24, 54, "Expected loss = P(not fixed) × priority value + risk of the actions + their cost. Lower is better. "
                   "Same scale in both panels.", size=11.5, fill=MID)
    panels = [("Without memory", [("escalate to a human now", 6.40, "0.00", False),
                                  ("a human approves data-strong's fix", 4.88, "0.79", True),
                                  ("… then data-fast's guess as stage 2", 5.57, "0.85", False)]),
              ("With memory", [("escalate to a human now", 6.40, "0.00", False),
                               ("a human approves data-strong's fix", 4.38, "0.89", True)])]
    for p, (title, rows) in enumerate(panels):
        x0 = 24 + p * 556
        c.text(x0, 90, title, size=14.5, weight=700)
        lx, tx, tw = x0, x0 + 222, 230
        scale = tw / 7.0
        for k, (lab, v, pf, chosen) in enumerate(rows):
            y = 108 + k * 46
            c.text(lx, y + 16, lab, size=11.5, fill=INK, weight=700 if chosen else 400)
            c.bar(tx, y, v * scale, 22, BAR_DARK if chosen else BAR_LIGHT, horizontal=True)
            c.text(tx + v * scale + 6, y + 16, f"{v:.2f}", size=12, weight=700)
            c.text(tx + v * scale + 42, y + 16, f"P(fixed) {pf}", size=10.5, fill=MID)
            if chosen:
                c.status(lx - 0, y + 34, "good", r=6)
                c.text(lx + 11, y + 38, "chosen", size=10.5, fill=INK, weight=700)
        c.line(tx, 100, tx, 108 + len(rows) * 46 - 14, stroke=SOFT, sw=1.2)
        for t in range(0, 8):
            c.text(tx + t * scale, 108 + 3 * 46 + 6, str(t), size=9.5, fill=MID, anchor="middle")
    c.text(24, 286, "The fix is high risk (it changes what the data means), so the plan is 'a human approves'. "
                    "'Escalate now' costs 4 + 0.3 × 8 = 6.40 for a P3 incident.", size=11, fill=MID)
    c.save("fig-loss")


def render_png():
    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    profile = os.path.join(tempfile.gettempdir(), "a2a-figures-chrome-profile")
    for svg, w, h in FIGS:
        png = svg.with_suffix(".png")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={profile}",
                        "--force-device-scale-factor=2", f"--window-size={w},{h}", f"--screenshot={png}",
                        svg.as_uri()], check=True, capture_output=True)
        print(f"rendered {png.name}")


if __name__ == "__main__":
    for f in (fig_overview, fig_company, fig_timeline, fig_memory_effect, fig_flow, fig_sequence, fig_loop,
              fig_conformal, fig_loss):
        f()
    if "--png" in sys.argv:
        render_png()
