#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare COLMAP sparse models (baseline vs hybrid) and judge whether hybrid gains are real
or mostly driven by outliers / weak tracks.

Usage:
  python compare_colmap_sfm.py \
    --baseline /path/to/pass1/sparse/0 \
    --hybrid   /path/to/pass2/sparse/0 \
    --outdir   /path/to/output_report

Outputs:
- summary.json
- summary.md
- hist_tracklen.png (optional if matplotlib installed)
"""

import argparse
import json
import os
import math
from collections import Counter, defaultdict

# Import from our utils directory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from read_write_model import read_model


def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)


def robust_stats(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return {}
    xs_sorted = sorted(xs)

    def pct(p):
        k = (len(xs_sorted) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return xs_sorted[int(k)]
        return xs_sorted[f] * (c - k) + xs_sorted[c] * (k - f)

    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "median": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "min": xs_sorted[0],
        "max": xs_sorted[-1],
    }


def analyze_model(model_dir):
    """Analyze a COLMAP sparse model directory."""
    # Detect file extension
    if os.path.exists(os.path.join(model_dir, "cameras.bin")):
        ext = ".bin"
    elif os.path.exists(os.path.join(model_dir, "cameras.txt")):
        ext = ".txt"
    else:
        raise FileNotFoundError(f"No COLMAP model files found in {model_dir}")

    # read_model returns dicts: cameras, images, points3D
    cams, imgs, pts = read_model(model_dir, ext=ext)

    n_images = len(imgs)
    n_points = len(pts)

    # track length per point (number of observations)
    track_lens = []
    reproj_errors = []

    # COLMAP points3D structure: point.error is mean reprojection error for that point
    # point.image_ids gives track length
    for pid, p in pts.items():
        tl = len(p.image_ids)
        track_lens.append(tl)
        reproj_errors.append(float(p.error))

    # classify points by track length
    tl_counter = Counter(track_lens)

    def frac_ge(k):
        if not track_lens:
            return 0.0
        return sum(1 for t in track_lens if t >= k) / len(track_lens)

    # coverage proxy: how many point observations per image
    obs_per_image = defaultdict(int)
    for pid, p in pts.items():
        for iid in p.image_ids:
            obs_per_image[iid] += 1
    obs_loads = list(obs_per_image.values())

    # find potential outlier points by reproj error (top 1%)
    rs = robust_stats(reproj_errors)
    outlier_threshold = rs.get("p99", None)
    n_outlier_pts = 0
    if outlier_threshold is not None:
        n_outlier_pts = sum(1 for e in reproj_errors if e >= outlier_threshold)

    report = {
        "n_registered_images": n_images,
        "n_points3D": n_points,
        "track_length_stats": robust_stats(track_lens),
        "reproj_error_stats": robust_stats(reproj_errors),
        "frac_tracklen_ge_3": frac_ge(3),
        "frac_tracklen_ge_5": frac_ge(5),
        "frac_tracklen_ge_10": frac_ge(10),
        "frac_tracklen_ge_15": frac_ge(15),
        "obs_load_per_image_stats": robust_stats(obs_loads),
        "outlier_reproj_threshold_p99": outlier_threshold,
        "n_points_reproj_outliers_top1pct": n_outlier_pts,
        "track_length_histogram": dict(sorted(tl_counter.items())),
        # Store raw data for plotting
        "_track_lens": track_lens,
        "_reproj_errors": reproj_errors,
    }
    return report


def compare(b, h):
    """Compare baseline and hybrid reports."""

    def delta(a, c):
        if a is None or c is None:
            return None
        return c - a

    def rel(a, c):
        if a is None or c is None or a == 0:
            return None
        return (c - a) / a

    out = {}
    keys = [
        "n_registered_images",
        "n_points3D",
        "frac_tracklen_ge_3",
        "frac_tracklen_ge_5",
        "frac_tracklen_ge_10",
        "frac_tracklen_ge_15",
        "n_points_reproj_outliers_top1pct",
    ]
    for k in keys:
        out[k] = {
            "baseline": b.get(k),
            "hybrid": h.get(k),
            "delta": delta(b.get(k), h.get(k)),
            "rel": rel(b.get(k), h.get(k)),
        }

    # include key robust stats comparisons
    for statkey in ["track_length_stats", "reproj_error_stats", "obs_load_per_image_stats"]:
        out[statkey] = {}
        for subk in ["mean", "median", "p90", "p95", "p99"]:
            bv = b.get(statkey, {}).get(subk)
            hv = h.get(statkey, {}).get(subk)
            out[statkey][subk] = {
                "baseline": bv,
                "hybrid": hv,
                "delta": delta(bv, hv),
                "rel": rel(bv, hv),
            }

    return out


def verdict(comp):
    """
    Heuristic verdict:
    - Real gain if:
      * registered images not worse
      * strong-track fractions (>=10) improve meaningfully
      * reproj median/p95 not worse too much
      * outlier count not exploding
    - Outlier illusion if:
      * points/obs increase but strong-track fraction doesn't,
        and reproj p95/p99 worsens + outliers increase.
    """
    msgs = []
    scores = {"good": 0, "bad": 0, "neutral": 0}

    # Check registered images
    reg_delta = comp["n_registered_images"]["delta"]
    reg_ok = reg_delta is not None and reg_delta >= 0
    if reg_ok:
        msgs.append("Registered images: not worse (good)")
        scores["good"] += 1
    else:
        msgs.append("Registered images: worse (red flag)")
        scores["bad"] += 2

    # Check 3D points increase
    pts_rel = comp["n_points3D"]["rel"]
    if pts_rel is not None and pts_rel > 0.02:
        msgs.append(f"3D points: +{pts_rel*100:.1f}% increase")
        scores["good"] += 1
    elif pts_rel is not None and pts_rel < -0.02:
        msgs.append(f"3D points: {pts_rel*100:.1f}% decrease")
        scores["bad"] += 1

    # Check strong track fractions
    strong5_rel = comp["frac_tracklen_ge_5"]["rel"]
    strong10_rel = comp["frac_tracklen_ge_10"]["rel"]

    strong5_ok = strong5_rel is not None and strong5_rel >= -0.05
    strong10_ok = strong10_rel is not None and strong10_rel >= -0.05

    if strong10_rel is not None:
        if strong10_rel >= 0.05:
            msgs.append(f"Strong tracks (>=10): +{strong10_rel*100:.1f}% improved (good)")
            scores["good"] += 2
        elif strong10_rel >= -0.05:
            msgs.append(f"Strong tracks (>=10): {strong10_rel*100:.1f}% (stable)")
            scores["neutral"] += 1
        else:
            msgs.append(f"Strong tracks (>=10): {strong10_rel*100:.1f}% decreased (concerning)")
            scores["bad"] += 1

    # Check reprojection error
    reproj_med_rel = comp["reproj_error_stats"]["median"]["rel"]
    reproj_p95_rel = comp["reproj_error_stats"]["p95"]["rel"]

    reproj_bad = ((reproj_med_rel is not None and reproj_med_rel > 0.10) or
                  (reproj_p95_rel is not None and reproj_p95_rel > 0.15))
    reproj_good = ((reproj_med_rel is not None and reproj_med_rel < 0.03) and
                   (reproj_p95_rel is not None and reproj_p95_rel < 0.05))

    if reproj_bad:
        msgs.append("Reprojection error: worsened noticeably (potential geometric instability)")
        scores["bad"] += 2
    elif reproj_good:
        msgs.append("Reprojection error: stable or improved (good)")
        scores["good"] += 1
    else:
        msgs.append("Reprojection error: slightly increased but within acceptable range")
        scores["neutral"] += 1

    # Check outliers
    outliers_rel = comp["n_points_reproj_outliers_top1pct"]["rel"]
    outliers_bad = outliers_rel is not None and outliers_rel > 0.30

    if outliers_bad:
        msgs.append(f"High-error outliers: +{outliers_rel*100:.1f}% increase (outlier illusion risk)")
        scores["bad"] += 2
    else:
        if outliers_rel is not None:
            msgs.append(f"Outlier count: {outliers_rel*100:+.1f}% (acceptable)")
        scores["neutral"] += 1

    # Check mean track length
    tl_mean_rel = comp["track_length_stats"]["mean"]["rel"]
    if tl_mean_rel is not None:
        if tl_mean_rel >= 0.05:
            msgs.append(f"Mean track length: +{tl_mean_rel*100:.1f}% improved (good)")
            scores["good"] += 1
        elif tl_mean_rel <= -0.05:
            msgs.append(f"Mean track length: {tl_mean_rel*100:.1f}% decreased (concerning)")
            scores["bad"] += 1

    # Final verdict
    if scores["bad"] >= 3:
        label = "LIKELY_OUTLIER_ILLUSION"
    elif scores["good"] >= 3 and scores["bad"] <= 1:
        label = "LIKELY_REAL_GAIN"
    elif scores["good"] > scores["bad"]:
        label = "PROBABLE_REAL_GAIN"
    elif scores["bad"] > scores["good"]:
        label = "QUESTIONABLE_GAIN"
    else:
        label = "MIXED_UNCLEAR"

    return label, msgs, scores


def write_markdown(outpath, b_rep, h_rep, comp, label, msgs, scores):
    """Write comparison report as markdown."""

    def fmt(x, is_pct=False):
        if x is None:
            return "NA"
        if isinstance(x, float):
            if is_pct:
                return f"{x*100:.2f}%"
            return f"{x:.4f}"
        return str(x)

    def fmt_rel(x):
        if x is None:
            return "NA"
        return f"{x*100:+.2f}%"

    md = []
    md.append("# COLMAP Sparse Model Comparison\n")
    md.append(f"**Verdict:** `{label}`\n")
    md.append(f"Score breakdown: Good={scores['good']}, Bad={scores['bad']}, Neutral={scores['neutral']}\n")

    md.append("## Analysis Summary\n")
    for m in msgs:
        md.append(f"- {m}")

    md.append("\n## Key Metrics\n")
    md.append("| Metric | Baseline | Hybrid | Delta | Relative |")
    md.append("|--------|----------|--------|-------|----------|")

    for k in ["n_registered_images", "n_points3D"]:
        row = comp[k]
        md.append(f"| {k} | {fmt(row['baseline'])} | {fmt(row['hybrid'])} | {fmt(row['delta'])} | {fmt_rel(row['rel'])} |")

    md.append("\n## Track Length Quality\n")
    md.append("| Metric | Baseline | Hybrid | Delta | Relative |")
    md.append("|--------|----------|--------|-------|----------|")

    for k in ["frac_tracklen_ge_3", "frac_tracklen_ge_5", "frac_tracklen_ge_10", "frac_tracklen_ge_15"]:
        row = comp[k]
        md.append(f"| {k} | {fmt(row['baseline'], True)} | {fmt(row['hybrid'], True)} | {fmt(row['delta'])} | {fmt_rel(row['rel'])} |")

    md.append("\n## Reprojection Error (px)\n")
    md.append("| Stat | Baseline | Hybrid | Delta | Relative |")
    md.append("|------|----------|--------|-------|----------|")
    for subk in ["mean", "median", "p90", "p95", "p99"]:
        row = comp["reproj_error_stats"][subk]
        md.append(f"| {subk} | {fmt(row['baseline'])} | {fmt(row['hybrid'])} | {fmt(row['delta'])} | {fmt_rel(row['rel'])} |")

    md.append("\n## Track Length Stats\n")
    md.append("| Stat | Baseline | Hybrid | Delta | Relative |")
    md.append("|------|----------|--------|-------|----------|")
    for subk in ["mean", "median", "p90", "p95", "p99"]:
        row = comp["track_length_stats"][subk]
        md.append(f"| {subk} | {fmt(row['baseline'])} | {fmt(row['hybrid'])} | {fmt(row['delta'])} | {fmt_rel(row['rel'])} |")

    md.append("\n## Observations per Image\n")
    md.append("| Stat | Baseline | Hybrid | Delta | Relative |")
    md.append("|------|----------|--------|-------|----------|")
    for subk in ["mean", "median", "p90", "p95"]:
        row = comp["obs_load_per_image_stats"][subk]
        md.append(f"| {subk} | {fmt(row['baseline'])} | {fmt(row['hybrid'])} | {fmt(row['delta'])} | {fmt_rel(row['rel'])} |")

    md.append("\n---")
    md.append("*Generated by compare_colmap_sfm.py*")

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def try_plots(outdir, b_rep, h_rep):
    """Generate comparison plots if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # Track length histogram comparison
    b_tl = b_rep["_track_lens"]
    h_tl = h_rep["_track_lens"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Track length distribution
    ax1 = axes[0]
    max_tl = min(max(max(b_tl), max(h_tl)), 50)  # Cap at 50 for visibility
    bins = range(2, max_tl + 2)
    ax1.hist([t for t in b_tl if t <= max_tl], bins=bins, alpha=0.7, label=f"Baseline (n={len(b_tl)})", color='blue')
    ax1.hist([t for t in h_tl if t <= max_tl], bins=bins, alpha=0.7, label=f"Hybrid (n={len(h_tl)})", color='orange')
    ax1.set_xlabel("Track length (# images per point)")
    ax1.set_ylabel("Count")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_title("Track Length Distribution")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Reprojection error distribution
    ax2 = axes[1]
    b_re = b_rep["_reproj_errors"]
    h_re = h_rep["_reproj_errors"]
    max_re = min(max(max(b_re), max(h_re)), 5.0)  # Cap at 5px for visibility
    bins_re = [i * 0.1 for i in range(int(max_re * 10) + 2)]
    ax2.hist([e for e in b_re if e <= max_re], bins=bins_re, alpha=0.7, label=f"Baseline (n={len(b_re)})", color='blue')
    ax2.hist([e for e in h_re if e <= max_re], bins=bins_re, alpha=0.7, label=f"Hybrid (n={len(h_re)})", color='orange')
    ax2.set_xlabel("Reprojection error (px)")
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.set_title("Reprojection Error Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_histograms.png"), dpi=150)
    plt.close()

    print(f"  Saved comparison_histograms.png")


def main():
    ap = argparse.ArgumentParser(description="Compare COLMAP sparse models (baseline vs hybrid)")
    ap.add_argument("--baseline", required=True, help="Path to baseline sparse/0 directory")
    ap.add_argument("--hybrid", required=True, help="Path to hybrid sparse/0 directory")
    ap.add_argument("--outdir", required=True, help="Output directory for reports")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    print(f"\n{'='*60}")
    print("COLMAP Sparse Model Comparison")
    print(f"{'='*60}")
    print(f"Baseline: {args.baseline}")
    print(f"Hybrid:   {args.hybrid}")
    print(f"Output:   {args.outdir}")

    print("\n[1] Analyzing baseline model...")
    b_rep = analyze_model(args.baseline)
    print(f"    Images: {b_rep['n_registered_images']}, Points: {b_rep['n_points3D']}")

    print("\n[2] Analyzing hybrid model...")
    h_rep = analyze_model(args.hybrid)
    print(f"    Images: {h_rep['n_registered_images']}, Points: {h_rep['n_points3D']}")

    print("\n[3] Comparing models...")
    comp = compare(b_rep, h_rep)
    label, msgs, scores = verdict(comp)

    # Remove raw data before saving to JSON
    b_rep_json = {k: v for k, v in b_rep.items() if not k.startswith('_')}
    h_rep_json = {k: v for k, v in h_rep.items() if not k.startswith('_')}

    out = {
        "baseline": b_rep_json,
        "hybrid": h_rep_json,
        "comparison": comp,
        "verdict": {"label": label, "reasons": msgs, "scores": scores},
    }

    json_path = os.path.join(args.outdir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"    Saved {json_path}")

    md_path = os.path.join(args.outdir, "summary.md")
    write_markdown(md_path, b_rep, h_rep, comp, label, msgs, scores)
    print(f"    Saved {md_path}")

    print("\n[4] Generating plots...")
    try_plots(args.outdir, b_rep, h_rep)

    print(f"\n{'='*60}")
    print(f"VERDICT: {label}")
    print(f"{'='*60}")
    for m in msgs:
        print(f"  - {m}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
