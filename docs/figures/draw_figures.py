"""Method figures (PNG + SVG) for the TrafficFlowBench entry of Team Anton.

    python docs/figures/draw_figures.py

Writes figure_1_system.*, figure_2_state.*, figure_3_queue.* next to this script. Only matplotlib is needed.
Block diagrams in the style of an architecture figure: every box names a component, lists its key design
facts, and an orange chip marks the design decision that matters most. Boxes are sized to their text.
One drawing unit is 0.1 inch.
"""
# Documentation only: draws the method figures; the reproduction pipeline does not run it.
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
# Colours: INK for titles, SUB for spec lines, GREY for column headers and plain arrows. PAL maps a
# colour name to (edge, fill) of a box; CHIP is (edge, fill) of the orange decision chip.
INK = "#17263c"; SUB = "#4f5d70"; GREY = "#5b6b80"
PAL = {  # edge, fill
    "blue": ("#1f5f99", "#eef4fb"), "purple": ("#7a5ea8", "#f2eef8"), "orange": ("#c8742a", "#fbf1e7"),
    "green": ("#12866b", "#e9f5f1"), "grey": ("#5b6b80", "#f4f6f9"), "white": ("#aab6c4", "#ffffff"),
}
CHIP = ("#c8742a", "#fdf0e3")
TS, SS, CS, HS = 16, 12, 11.5, 14         # title, spec line, chip, header (pt)
# PADX: inner left/right padding of a box, in drawing units.
PADX = 1.3
# Off-screen figure used only to measure text widths (see tw).
_mfig = plt.figure(figsize=(4, 4), dpi=100); _mren = _mfig.canvas.get_renderer()


def tw(s, size, bold=False):
    """Returns the rendered width of text `s` at font size `size` (pt), in drawing units."""
    t = _mfig.text(0, 0, s, fontsize=size, fontweight="bold" if bold else "normal")
    w = t.get_window_extent(renderer=_mren).width / _mfig.dpi * 10
    t.remove()
    return w


LT, LS, LC = TS / 72 * 10 * 1.05, SS / 72 * 10 * 1.25, 2.5   # line heights (units)


def size(title, lines=(), chip=None, ts=TS):
    """Returns (width, height) in drawing units of a box with this title, spec lines and chip.

    Args:
        title: bold title text.
        lines: spec lines under the title.
        chip: optional text of the orange chip at the bottom of the box.
        ts: title font size (pt).
    """
    w = tw(title, ts, True)
    for s in lines:
        w = max(w, tw(s, SS))
    if chip:
        w = max(w, tw(chip, CS, True) + 1.8)
    h = 1.0 + ts / 72 * 10 * 1.05 + len(lines) * LS + (0.6 + LC if chip else 0) + 1.0
    return w + 2 * PADX, h


class Box:
    """A rounded box with a bold title, spec lines and an optional orange chip.

    Attributes:
        x, y, w, h: lower-left corner and size, in drawing units.
        r, l, t, b: midpoints of the right, left, top and bottom edges (arrow anchors).
    """
    def __init__(self, ax, x, yc, w, title, lines=(), chip=None, col="blue", ts=TS, h=None):
        """Draws the box on `ax`, vertically centred on `yc`.

        Args:
            ax: matplotlib axes of the canvas.
            x: left edge.
            yc: vertical centre.
            w: width (usually the column width from colwidth).
            title, lines, chip, ts: as in size().
            col: colour name in PAL.
            h: fixed height; by default the height that fits the text.
        """
        edge, fill = PAL[col]
        _, h0 = size(title, lines, chip, ts)
        h = h or h0
        self.x, self.y, self.w, self.h = x, yc - h / 2, w, h
        ax.add_patch(FancyBboxPatch((x, self.y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=fill, ec=edge, lw=1.8))
        y = self.y + h - 1.0
        ax.text(x + PADX, y, title, ha="left", va="top", fontsize=ts, fontweight="bold", color=INK)
        y -= ts / 72 * 10 * 1.05 + 0.25
        for s in lines:
            ax.text(x + PADX, y, s, ha="left", va="top", fontsize=SS, color=SUB)
            y -= LS
        if chip:
            cw = tw(chip, CS, True) + 1.8
            cy = self.y + 0.9
            ax.add_patch(FancyBboxPatch((x + PADX, cy), cw, LC, boxstyle="round,pad=0,rounding_size=0.4",
                                        fc=CHIP[1], ec=CHIP[0], lw=1.4))
            ax.text(x + PADX + 0.9, cy + LC / 2, chip, ha="left", va="center", fontsize=CS, fontweight="bold",
                    color=CHIP[0])

    r = property(lambda s: (s.x + s.w, s.y + s.h / 2))
    l = property(lambda s: (s.x, s.y + s.h / 2))
    t = property(lambda s: (s.x + s.w / 2, s.y + s.h))
    b = property(lambda s: (s.x + s.w / 2, s.y))


def canvas(xmax, ymax):
    """Returns a figure and axes of xmax x ymax drawing units, with no margins or axis lines."""
    fig, ax = plt.subplots(figsize=(xmax / 10, ymax / 10))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def arrow(ax, p0, p1, col="grey", lw=1.9, ls="-", conn=None, label=None, lx=0, ly=0.8):
    """Draws an arrow from p0 to p1 with an optional bold label above its midpoint.

    Args:
        ax: target axes.
        p0, p1: start and end points (drawing units).
        col: colour name in PAL; lw, ls: line width and style.
        conn: optional matplotlib connection style (for bent arrows).
        label: optional text; lx, ly shift it from the midpoint.
    """
    props = dict(arrowstyle="-|>", color=PAL[col][0], lw=lw, shrinkA=1, shrinkB=1, mutation_scale=15, linestyle=ls)
    if conn:
        props["connectionstyle"] = conn
    ax.annotate("", xy=p1, xytext=p0, arrowprops=props)
    if label:
        ax.text((p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly, label, ha="center", va="bottom",
                fontsize=SS, fontweight="bold", color=PAL[col][0])


def frame(ax, x, y, w, h, s, col):
    """Draws a large rounded frame with caption `s` at its top left, around one row of boxes."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc="#fbfcfe", ec=PAL[col][0], lw=1.6))
    ax.text(x + 1.3, y + h - 1.0, s, ha="left", va="top", fontsize=HS - 1, fontweight="bold", color=PAL[col][0])


def headers(ax, y, items):
    """Writes column headers at height y; items are (x, text) pairs."""
    for x, n in items:
        ax.text(x, y, n, ha="left", va="center", fontsize=HS, fontweight="bold", color=GREY)


def notes(ax, x, y, items):
    """Writes note lines (bold lead, plain rest) downward from (x, y), 2.6 units apart."""
    for i, (bold, rest) in enumerate(items):
        t = ax.text(x, y - i * 2.6, bold, ha="left", va="center", fontsize=SS + 1, fontweight="bold", color=INK)
        ax.text(x + tw(bold, SS + 1, True) + 0.8, y - i * 2.6, rest, ha="left", va="center", fontsize=SS + 1, color=SUB)


def save(fig, name):
    """Saves the figure as PNG (200 dpi) and SVG next to this script, then closes it."""
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200 if ext == "png" else None, facecolor="white",
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def colwidth(specs):
    """Returns the widest box width among the box specs (title, lines[, chip])."""
    return max(size(*s)[0] for s in specs)


# --------------------------------------------------------------------------------------------------
def figure_1():
    """Draws figure 1: the inputs, the four task modules and the submission."""
    ins = [("Mainline detectors", ["speed · flow · occupancy", "every 5 min, 10 corridors"]),
           ("Ramp counts", ["on- and off-ramps"]),
           ("Window history", ["60 min up to the origin T"]),
           ("Road network", ["lanes · capacity · free-flow speed", "routes and a weak OD prior"]),
           ("Training months", ["Jun 2030 – Feb 2031", "complete labels"])]
    mods = [("Task 1 · Speed & flow", ["stacked trees → residual correction", "→ density projection; gap models"], "whole month", "blue"),
            ("Task 2 · Queue start", ["bottleneck choice inside the window", "→ that bottleneck's usual queue"], "data up to T", "purple"),
            ("Task 2 · Existing queue", ["three tree ensembles → weighted mean", "→ vote over co-congesting links"], "data up to T", "orange"),
            ("Task 4 · Route flows", ["least squares on link counts", "flows ≥ 0, pulled to the prior"], "whole month", "green")]
    w0 = colwidth([(t, l) for t, l in ins]); w1 = colwidth([(t, l, c) for t, l, c, _ in mods])
    w2 = colwidth([("Submission", ["6.98 M rows"]), ("Private score", ["0.89493"])])
    x0, x1 = 1, 1 + w0 + 9; x2 = x1 + w1 + 9; W = x2 + w2 + 1
    fig, ax = canvas(W, 66)
    headers(ax, 63.5, [(x0, "INPUTS"), (x1, "TASK MODULES"), (x2, "OUTPUT")])
    yi = [54, 43.5, 34.5, 24.5, 13]
    bus = x1 - 4.5
    for (t, l), y in zip(ins, yi):
        b = Box(ax, x0, y, w0, t, l, col="grey")
        ax.plot([b.r[0], bus], [y, y], color=GREY, lw=1.5)
    ax.plot([bus, bus], [yi[-1], yi[0]], color=GREY, lw=1.5)
    sub = Box(ax, x2, 38, w2, "Submission", ["6.98 M rows"], col="blue")
    sco = Box(ax, x2, 26, w2, "Private score", ["0.89493"], col="green")
    for (t, l, c, col), y in zip(mods, [53, 40.5, 28, 15.5]):
        m = Box(ax, x1, y, w1, t, l, c, col=col)
        arrow(ax, (bus, y), m.l)
        arrow(ax, m.r, (sub.l[0], sub.l[1] + (y - 34) * 0.12), col=col)
    arrow(ax, sub.b, sco.t)
    save(fig, "figure_1_system")


# --------------------------------------------------------------------------------------------------
def figure_2():
    """Draws figure 2: Task 1 for ordinary missing cells and for the 90-minute gaps."""
    C = [  # column specs used only for widths
        [("Observed cells", ["same link at t ± 1, t ± 2", "links x ± 1, x ± 2", "occupancy · time · road"]),
         ("Gap boundary", ["state just before & after", "queue state · link blocks"]),
         ("On-ramp counts", ["inside the gap"])],
        [("Five classical fills", ["tensor completion (LRTC-TNN)", "time · space · space-time interp.", "weekday × hour profile"]),
         ("Stacking", ["LightGBM weighs the five fills", "by local context"]),
         ("Gap model", ["LightGBM · 73 features", "trained on artificial 90-min gaps"], "gap flow comes from here")],
        [("Residual correction", ["LightGBM · 130 relative features", "neighbour differences, far shocks"], "one model for every link"),
         ("Second learner", ["CatBoost, same features", "adds ½ (CatBoost − LightGBM)"]),
         ("Month correction", ["speed residual on artificial gaps", "cut from the same month"], "fitted at run time"),
         ("Queue probability", ["LightGBM on ramp features", "shared by co-congesting links"], "fitted at run time")],
        [("Density projection", ["flow = density × speed", "density error learned by LightGBM", "speed & flow moved onto it"], "refit where speed < 85 % free-flow"),
         ("Two-state speed", ["P · v_queued + (1 − P) · v_free", "½ with the month correction", "+ ½ January ramp residual"])],
        [("Speed & flow", ["scored in Task 1", "and in Task 3 physics"]), ("Gap speed & flow", ["speed: two-state model", "flow: gap model"])],
    ]
    ws = [colwidth(c) for c in C]
    gap = 7
    xs = [1]
    for w in ws[:-1]:
        xs.append(xs[-1] + w + gap)
    W = xs[-1] + ws[-1] + 1
    fig, ax = canvas(W, 88)
    headers(ax, 85.5, [(xs[0], "INPUTS"), (xs[1], "FIRST ESTIMATE"), (xs[2], "LEARNED CORRECTION"),
                       (xs[3], "PHYSICS · STATES"), (xs[4], "OUTPUT")])
    frame(ax, 0, 57.8, W, 25.2, "ORDINARY MISSING CELLS · 6.6 M cells", "blue")
    frame(ax, 0, 16.5, W, 34.5, "90-MINUTE GAPS AROUND QUEUE WINDOWS · 128 k cells", "orange")
    # ordinary cells
    i1 = Box(ax, xs[0], 70.5, ws[0], *C[0][0], col="grey")
    f1 = Box(ax, xs[1], 74.5, ws[1], *C[1][0], col="blue")
    f2 = Box(ax, xs[1], 63.2, ws[1], *C[1][1], col="blue")
    c1 = Box(ax, xs[2], 74.5, ws[2], *C[2][0], col="blue")
    c2 = Box(ax, xs[2], 63.2, ws[2], *C[2][1], col="blue")
    p1 = Box(ax, xs[3], 69, ws[3], *C[3][0], col="green")
    o1 = Box(ax, xs[4], 69, ws[4], *C[4][0], col="white")
    arrow(ax, i1.r, f1.l, col="blue"); arrow(ax, f1.b, f2.t, col="blue")
    arrow(ax, f2.r, c1.l, col="blue")
    arrow(ax, c1.b, c2.t, col="blue")
    arrow(ax, c2.r, p1.l, col="blue"); arrow(ax, p1.r, o1.l, col="green")
    # gaps
    i2 = Box(ax, xs[0], 41, ws[0], *C[0][1], col="grey")
    i3 = Box(ax, xs[0], 22, ws[0], *C[0][2], col="grey")
    g1 = Box(ax, xs[1], 41, ws[1], *C[1][2], col="orange")
    m1 = Box(ax, xs[2], 41, ws[2], *C[2][2], col="orange")
    q1 = Box(ax, xs[2], 24, ws[2], *C[2][3], col="orange")
    t1 = Box(ax, xs[3], 34, ws[3], *C[3][1], col="green")
    o2 = Box(ax, xs[4], 34, ws[4], *C[4][1], col="white")
    arrow(ax, i2.r, g1.l, col="orange"); arrow(ax, g1.r, m1.l, col="orange", label="speed", ly=1.2)
    arrow(ax, m1.r, (t1.l[0], t1.l[1] + 3), col="orange")
    arrow(ax, i3.r, q1.l, col="orange")
    arrow(ax, q1.r, (t1.l[0], t1.l[1] - 3), col="orange")
    ax.text(q1.r[0] + 0.6, q1.r[1] - 1.2, "P(queued)", ha="left", va="top", fontsize=SS, fontweight="bold", color=PAL["orange"][0])
    arrow(ax, t1.r, o2.l, col="green")
    notes(ax, 1.5, 12, [
        ("Relative inputs:", "every model sees differences to neighbours, so one model serves all links and months."),
        ("Physics in the loop:", "density projections keep speed, flow and density consistent, which Task 3 rewards."),
        ("Month-specific heads:", "gap corrections are refitted on each month's own observed data (Task 1 is offline)."),
    ])
    save(fig, "figure_2_state")


# --------------------------------------------------------------------------------------------------
def figure_3():
    """Draws figure 3: Task 2 for queue-start windows and for existing-queue windows."""
    C = [
        [("Window history", ["flow at each candidate", "bottleneck, T − 60 … T"]),
         ("State at T", ["speed · flow · occupancy", "margin to the queue speed", "trends over the last hour", "neighbours 1, 2, 4 links"]),
         ("Earlier days", ["queue frequency per link", "same month, up to 5 days"]),
         ("Selector replay", ["steps the official window", "rule checked before T"], "6-hour spacing handled")],
        [("Bottleneck scores", ["load / capacity and its trend", "phase of the 30-min demand step"]),
         ("Location-free LightGBM × 6", ["traffic state only"]),
         ("Full LightGBM × 6", ["+ earlier-day priors", "+ queue geometry (spacing,", "  tail speed, wave speed)"], "retrained through Feb 2031"),
         ("Monotone LightGBM × 6", ["17 inputs decreasing (speed, margin)", "12 inputs increasing (occupancy)"], "physics as shape constraints")],
        [("Within-window choice", ["conditional logit over the", "window's candidate sites", "24 settings averaged"]),
         ("Weighted mean", ["0.25 location-free", "0.375 full", "0.375 monotone"])],
        [("Usual queue extent", ["links queued in ≥ 50 % of", "training cases at that site"], "first queue at T+30 only"),
         ("One-bottleneck corridors", ["co-congesting link groups", "mean ≥ 0.7, plus the top group"]),
         ("Link-group vote", ["mean over co-congesting links", "per step"], "queued if mean ≥ 0.45"),
         ("Empty-window fallback", ["location-free decision", "at 0.5"])],
        [("Queue-start cells", ["step T+30"]), ("Queue cells", ["steps T+5 … T+30"])],
    ]
    ws = [colwidth(c) for c in C]
    gap = 7
    xs = [1]
    for w in ws[:-1]:
        xs.append(xs[-1] + w + gap)
    W = xs[-1] + ws[-1] + 1
    fig, ax = canvas(W, 100)
    headers(ax, 97.5, [(xs[0], "INPUTS  (up to T)"), (xs[1], "MODELS"), (xs[2], "CHOOSE · COMBINE"),
                        (xs[3], "DECISION"), (xs[4], "OUTPUT")])
    frame(ax, 0, 67, W, 28.5, "QUEUE START · no queue in the last hour", "purple")
    frame(ax, 0, 17.5, W, 48, "EXISTING QUEUE · a queue is visible at T", "orange")
    # queue start
    a0 = Box(ax, xs[0], 84, ws[0], *C[0][0], col="grey")
    a1 = Box(ax, xs[1], 84, ws[1], *C[1][0], col="purple")
    a2 = Box(ax, xs[2], 84, ws[2], *C[2][0], col="purple")
    a3 = Box(ax, xs[3], 84, ws[3], *C[3][0], col="purple")
    a4 = Box(ax, xs[3], 72.8, ws[3], *C[3][1], col="purple")
    a5 = Box(ax, xs[4], 78.5, ws[4], *C[4][0], col="white")
    arrow(ax, a0.r, a1.l, col="purple"); arrow(ax, a1.r, a2.l, col="purple")
    arrow(ax, a2.r, a3.l, col="purple", label="site", ly=1.0)
    arrow(ax, a0.b, a4.l, col="purple", lw=1.5, conn="angle,angleA=-90,angleB=180,rad=0")
    arrow(ax, a3.r, (a5.l[0], a5.l[1] + 1.5), col="purple"); arrow(ax, a4.r, (a5.l[0], a5.l[1] - 1.5), col="purple")
    # existing queue
    b0 = Box(ax, xs[0], 55.5, ws[0], *C[0][1], col="grey")
    b1 = Box(ax, xs[0], 40.5, ws[0], *C[0][2], col="grey")
    b2 = Box(ax, xs[0], 26, ws[0], *C[0][3], col="grey")
    m1 = Box(ax, xs[1], 57.5, ws[1], *C[1][1], col="orange")
    m2 = Box(ax, xs[1], 43.5, ws[1], *C[1][2], col="orange")
    m3 = Box(ax, xs[1], 26, ws[1], *C[1][3], col="green")
    c1 = Box(ax, xs[2], 43.5, ws[2], *C[2][1], col="orange")
    d1 = Box(ax, xs[3], 43.5, ws[3], *C[3][2], col="orange")
    d2 = Box(ax, xs[3], 26.5, ws[3], *C[3][3], col="white")
    o1 = Box(ax, xs[4], 43.5, ws[4], *C[4][1], col="white")
    bus = xs[1] - 3.5
    for b in (b0, b1, b2):
        ax.plot([b.r[0], bus], [b.r[1]] * 2, color=GREY, lw=1.5)
    ax.plot([bus, bus], [m3.l[1], m1.l[1]], color=GREY, lw=1.5)
    for m in (m1, m2, m3):
        arrow(ax, (bus, m.l[1]), m.l, lw=1.5)
    arrow(ax, m1.r, (c1.l[0], c1.l[1] + 3), col="orange"); arrow(ax, m2.r, c1.l, col="orange")
    arrow(ax, m3.r, (c1.l[0], c1.l[1] - 3), col="green")
    arrow(ax, c1.r, d1.l, col="orange")
    arrow(ax, d1.r, o1.l, col="orange")
    arrow(ax, d1.b, d2.t, ls="--", lw=1.5)
    ax.text(d1.b[0] + 1, (d1.b[1] + d2.t[1]) / 2, "window empty", fontsize=SS, color=GREY, va="center")
    notes(ax, 1.5, 13, [
        ("Online rule:", "each window reads data up to T; every model is trained on the training months."),
        ("Group decisions:", "links that congest together are decided together, so a queue grows and clears as a block."),
    ])
    save(fig, "figure_3_queue")


if __name__ == "__main__":
    figure_1(); figure_2(); figure_3()
    print("wrote", sorted(p.name for p in OUT.glob("figure_*")))
