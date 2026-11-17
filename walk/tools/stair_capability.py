"""Stair capability from ``terrain_eval.py --terrain stairs_*`` results: per policy and course, success per riser x tread
cell (trials pooled over the given files, i.e. evaluation seeds), and per tread the **border**: the riser at which a
logistic fit of success over riser height crosses 80 % and 50 %, with a bootstrap 90 % interval (trials resampled
within each cell). Risers and treads are the nominal ones (the row's and the column's; the irregular course's mean
actual values are kept in the JSON's cells). Success = ``success_on_strip``: all four feet on the landing without a fall, never more than
``--side`` off the line.

    python -B tools/stair_capability.py --out runs/eval/stairs/capability \\
        cand=runs/eval/stairs/cand.climb.s11.json,runs/eval/stairs/cand.climb.s12.json \\
        flat=runs/eval/stairs/flat.climb.s11.json,...

Writes ``<out>.json`` (cells and borders), ``<out>.md`` (tables) and ``<out>.png`` (heat maps, one per policy and course).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

STAIR_RISERS_MM = tuple(range(10, 75, 5))  # = mp3_walk_policy.terrain.STAIR_RISERS_MM (nominal riser per row)

RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # one hue, light -> dark


def load(files: list[str]) -> dict:
    """{course: {(tread_tag, level): {"riser_mm", "tread_mm", "ok": [...], "fall": [...], "climbed": [...], "steps"}}}"""
    out: dict = defaultdict(dict)
    for f in files:
        d = json.loads(Path(f).read_text())
        course = d["options"]["terrain"]
        meta = {(c["terrain"], c["level"]): c for c in d["cells"]}
        t = d["trials"]
        for i, (tag, lvl) in enumerate(zip(t["terrain"], t["level"])):
            c = meta[(tag, lvl)]
            cell = out[course].setdefault((tag, lvl), {"riser_mm": float(STAIR_RISERS_MM[lvl]),
                                                        "tread_mm": int(tag.rsplit("_t", 1)[1]),
                                                        "riser_mean_mm": c["riser_mm"], "tread_mean_mm": c["tread_mm"],
                                                        "steps": c["steps"], "ok": [], "fall": [], "climbed": [],
                                                        "leg": [], "files": set()})
            cell["ok"].append(t["success_on_strip"][i])
            cell["fall"].append(t["fall"][i])
            cell["climbed"].append(t["risers_climbed"][i])
            cell["leg"].append(t["leg_contact_s"][i] > 0)
            cell["files"].add(f)
    return out


def fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """(a, b) of p = 1 / (1 + exp(-(a + b x))) by maximum likelihood (a weak ridge keeps separable data finite)."""
    if y.min() == y.max():
        return None

    def nll(p):
        z = p[0] + p[1] * x
        return np.sum(np.logaddexp(0, z) - y * z) + 1e-4 * p[1] ** 2

    r = minimize(nll, x0=[0.0, -0.1], method="BFGS")
    return float(r.x[0]), float(r.x[1])


def crossing(ab, p: float) -> float:
    a, b = ab
    return (np.log(p / (1 - p)) - a) / b


def border(cells: list[dict], p: float, rng: np.random.Generator, n_boot: int = 400) -> dict:
    """The riser (mm) where the fitted success crosses p, its bootstrap 90 % interval, and the highest riser whose cell
    rate is >= p (the empirical border)."""
    xs = np.concatenate([np.full(len(c["ok"]), c["riser_mm"]) for c in cells])
    ys = np.concatenate([np.asarray(c["ok"], float) for c in cells])
    emp = [c["riser_mm"] for c in sorted(cells, key=lambda c: c["riser_mm"]) if np.mean(c["ok"]) >= p]
    lo_r, hi_r = min(c["riser_mm"] for c in cells), max(c["riser_mm"] for c in cells)
    res = {"p": p, "empirical_mm": max(emp) if emp else None}
    ab = fit_logistic(xs, ys)
    if ab is None or ab[1] >= 0:
        res.update(fit_mm=None, ci90_mm=None, note="no decreasing transition in range")
        return res
    res["fit_mm"] = round(float(np.clip(crossing(ab, p), lo_r - 5, hi_r + 5)), 1)
    boots = []
    for _ in range(n_boot):
        bx, by = [], []
        for c in cells:
            k = rng.integers(0, len(c["ok"]), len(c["ok"]))
            bx.append(np.full(len(k), c["riser_mm"]))
            by.append(np.asarray(c["ok"], float)[k])
        bab = fit_logistic(np.concatenate(bx), np.concatenate(by))
        if bab is not None and bab[1] < 0:
            boots.append(np.clip(crossing(bab, p), lo_r - 5, hi_r + 5))
    if boots:
        res["ci90_mm"] = [round(float(np.percentile(boots, 5)), 1), round(float(np.percentile(boots, 95)), 1)]
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policies", nargs="+", help="NAME=file1.json,file2.json,... (terrain_eval stair results)")
    ap.add_argument("--out", required=True, help="output stem")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    report, md = {}, []
    for spec in args.policies:
        name, files = spec.split("=", 1)
        data = load(files.split(","))
        report[name] = {}
        for course, cells in sorted(data.items()):
            tags = sorted({k[0] for k in cells}, key=lambda k: int(k.rsplit("_t", 1)[1]))
            levels = sorted({k[1] for k in cells})
            rows, borders = [], {}
            for tag in tags:
                col = [cells[(tag, l)] for l in levels if (tag, l) in cells]
                borders[tag] = {"tread_mm": col[0]["tread_mm"], "b80": border(col, 0.8, rng), "b50": border(col, 0.5, rng)}
            for l in levels:
                cs = [cells.get((tag, l)) for tag in tags]
                rows.append({"level": l, "riser_mm": next(c["riser_mm"] for c in cs if c),
                             "success": [None if c is None else round(float(np.mean(c["ok"])), 3) for c in cs],
                             "fall": [None if c is None else round(float(np.mean(c["fall"])), 3) for c in cs],
                             "climbed": [None if c is None else round(float(np.mean(c["climbed"])), 2) for c in cs],
                             "leg_contact": [None if c is None else round(float(np.mean(c["leg"])), 3) for c in cs],
                             "n": [0 if c is None else len(c["ok"]) for c in cs]})
            report[name][course] = {"tags": tags, "rows": rows, "borders": borders,
                                    "actual_mm": {f"{t}/L{l}": [cells[(t, l)]["riser_mean_mm"], cells[(t, l)]["tread_mean_mm"]]
                                                  for (t, l) in sorted(cells)},
                                    "files": sorted({f for c in cells.values() for f in c["files"]})}
            md.append(f"### {name}: {course}\n")
            md.append("| riser (mm) | " + " | ".join(f"tread {borders[t]['tread_mm']}" for t in tags) + " |")
            md.append("|---|" + "---|" * len(tags))
            for r in rows:
                md.append(f"| {r['riser_mm']:.0f} | " + " | ".join(
                    "–" if s is None else f"{100 * s:.0f} % (f {100 * f:.0f})" for s, f in zip(r["success"], r["fall"])) + " |")
            md.append("| **80 % border** | " + " | ".join(_b(borders[t]["b80"]) for t in tags) + " |")
            md.append("| **50 % border** | " + " | ".join(_b(borders[t]["b50"]) for t in tags) + " |")
            md.append(f"\nTrials per cell: {rows[0]['n'][0]}; success % (falls %); borders: fit (90 % interval) [highest "
                      "cell at or above the rate].\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".json").write_text(json.dumps(report, indent=1))
    Path(args.out + ".md").write_text("\n".join(md) + "\n")
    plot(report, args.out + ".png")
    print("\n".join(md))
    return 0


def _b(b: dict) -> str:
    fit = "–" if b.get("fit_mm") is None else f"{b['fit_mm']:.1f}"
    ci = "" if not b.get("ci90_mm") else f" ({b['ci90_mm'][0]:.0f}–{b['ci90_mm'][1]:.0f})"
    emp = "–" if b["empirical_mm"] is None else f"{b['empirical_mm']:.0f}"
    return f"{fit}{ci} [{emp}]"


def plot(report: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    panels = [(n, c) for n in report for c in report[n]]
    cols = min(3, len(panels))
    nrows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(4.2 * cols, 4.6 * nrows), squeeze=False)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax, (name, course) in zip(axes.ravel(), panels):
        r = report[name][course]
        grid = np.array([[np.nan if s is None else s for s in row["success"]] for row in r["rows"]])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
        ax.set_xticks(range(len(r["tags"])), [str(r["borders"][t]["tread_mm"]) for t in r["tags"]], fontsize=8)
        ax.set_yticks(range(len(r["rows"])), [f"{row['riser_mm']:.0f}" for row in r["rows"]], fontsize=8)
        for i, row in enumerate(r["rows"]):
            for j, s in enumerate(row["success"]):
                if s is not None:
                    ax.text(j, i, f"{100 * s:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if s > 0.55 else "#333333")
        ax.set_xlabel("tread (mm)", fontsize=9)
        ax.set_ylabel("riser (mm)", fontsize=9)
        ax.set_title(f"{name}: {course}", fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Stair success (%), all four feet on the landing without a fall", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    raise SystemExit(main())
