#!/usr/bin/env python3
"""
Paired, item-level reanalysis for "Conviction Without Evidence" (reviewer comments 1-3).

Reads the existing per-turn JSONL logs (no API calls) and computes:
  * reproduction of Table 3 (unpaired Wald) and Table 4
  * paired item-level contrasts (blank - real) on items non-abstaining in both conditions:
      - first-turn wiggle, final-turn wiggle: exact McNemar, paired bootstrap CI, Newcombe paired CI
      - verdict changes over 10 turns: Wilcoxon signed-rank, paired bootstrap CI
  * equivalence (TOST) bounds from 90% CIs, and pass/fail at stated margins
  * random-intercept logistic GLMM (item random effect), adaptive-free Gauss-Hermite quadrature
  * stratified results (matched vs non-matched; per group) + difference-in-differences
  * secondary model (GPT-5.2, 38 matched items, L4) paired contrast

Usage: python3 paired_analysis.py [data_dir] [out_dir]   (defaults: ./results, ./out; file names set in FILES)
Dependencies: numpy, scipy, pandas only.
"""
import json, sys, os
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats, optimize

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles
except Exception:
    pass

# ---- configure here (or pass data_dir / out_dir on the command line) --------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "out")
FILES = {
    "L1": "sonnet45_L1.jsonl",    # primary model, L1, real + blank, turns 0-10
    "L4": "sonnet45_L4.jsonl",    # primary model, L4, real + blank, turns 0-10
    "gpt": "turns.jsonl",         # secondary model (GPT-5.2 rows used), L4 matched items
}
# -----------------------------------------------------------------------------
for k, f in FILES.items():
    if not os.path.exists(os.path.join(DATA, f)):
        sys.exit(f"Missing input: {os.path.join(DATA, f)}  (edit DATA / FILES at the top of the script)")
os.makedirs(OUT, exist_ok=True)

B = 10_000
SEED = 20261003
LETTERS = set("ABCDE")
N_TURNS = 10
MARGINS = [0.08, 0.10]  # 0.08 = measured turn-0 run-to-run disagreement; 0.10 = round alternative

# --------------------------------------------------------------------------- loading
def load(path, model=None, level=None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if model:
        rows = [r for r in rows if r["model"] == model]
    if level:
        rows = [r for r in rows if r["level"] == level]
    seq = defaultdict(dict)
    group = {}
    for r in rows:
        seq[(r["item_id"], r["condition"])][r["turn"]] = r["verdict"]
        if "group" in r:
            group[r["item_id"]] = r["group"]
    return seq, group


def per_item_table(seq, level, model):
    """One row per (item, condition) that answered at turn 0 and has all 10 challenge turns."""
    out = []
    for (item, cond), v in seq.items():
        base = v.get(0)
        if base not in LETTERS:
            continue  # baseline abstention: excluded, as in the paper
        if not all(t in v for t in range(N_TURNS + 1)):
            raise ValueError(f"missing turns for {item} {cond}: {sorted(v)}")
        traj = [v[t] for t in range(N_TURNS + 1)]
        out.append(dict(
            model=model, level=level, item_id=item, condition=cond, baseline=base,
            turn1=traj[1], turn10=traj[10],
            first_flip=int(traj[1] != base),
            final_flip=int(traj[10] != base),
            changes=sum(traj[t] != traj[t - 1] for t in range(1, N_TURNS + 1)),
            trajectory="".join(x if x in LETTERS else ("-" if x.startswith("ABSTAIN") else "?") for x in traj),
        ))
    return pd.DataFrame(out)

# --------------------------------------------------------------------------- statistics
def wilson(x, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_paired(a, b, c, d, conf=0.95):
    """Newcombe (1998) method 10 CI for paired difference p_blank - p_real.
    a = both flip, b = real only, c = blank only, d = neither."""
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    n = a + b + c + d
    p1, p2 = (a + c) / n, (a + b) / n          # p1 = blank, p2 = real
    l1, u1 = wilson(a + c, n, z)
    l2, u2 = wilson(a + b, n, z)
    den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = 0.0 if den == 0 else (a * d - b * c) / den
    D = p1 - p2
    dl = np.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    du = np.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0))
    return D - dl, D + du


def mcnemar_exact(b, c):
    if b + c == 0:
        return 1.0
    return stats.binomtest(min(b, c), b + c, 0.5).pvalue


def paired_boot(x_real, x_blank, rng, stat=np.mean):
    """Resample items (pairs kept together). Returns bootstrap distribution of stat(blank) - stat(real)."""
    n = len(x_real)
    idx = rng.integers(0, n, size=(B, n))
    return stat(x_blank[idx], axis=1) - stat(x_real[idx], axis=1)


def pct(dist, conf):
    a = (1 - conf) / 2
    return float(np.quantile(dist, a)), float(np.quantile(dist, 1 - a))


def unpaired_wald(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return p2 - p1, (p2 - p1 - 1.959964 * se, p2 - p1 + 1.959964 * se)

# ---- random-intercept logistic GLMM by Gauss-Hermite quadrature (binomial outcome)
GH_Z, GH_W = np.polynomial.hermite.hermgauss(40)

def glmm_fit(y, n, x, cluster):
    """y ~ Bin(n, expit(b0 + b1*x + u_cluster)), u ~ N(0, s^2). Returns dict with b1, se, CI, p, s."""
    y, n, x = map(np.asarray, (y, n, x))
    cl = pd.factorize(np.asarray(cluster))[0]
    K = cl.max() + 1
    logc = stats.binom.logpmf(y, n, 0.5) - (n * np.log(0.5))  # log binomial coefficient

    def nll(theta):
        b0, b1, ls = theta
        s = np.exp(ls)
        u = np.sqrt(2) * s * GH_Z                                 # (Q,)
        eta = b0 + b1 * x[:, None] + u[None, :]                  # (N,Q)
        ll = logc[:, None] + y[:, None] * eta - n[:, None] * np.logaddexp(0, eta)
        per_cluster = np.zeros((K, len(GH_Z)))
        np.add.at(per_cluster, cl, ll)
        m = per_cluster.max(1, keepdims=True)
        lik = m[:, 0] + np.log((np.exp(per_cluster - m) * GH_W[None, :]).sum(1) / np.sqrt(np.pi))
        return -lik.sum()

    best = None
    for start in ([0, 0, 0], [1, 0, 0.5], [-1, 0, -1], [2, 0, 1]):
        r = optimize.minimize(nll, start, method="Nelder-Mead",
                              options=dict(maxiter=20000, xatol=1e-7, fatol=1e-9))
        r = optimize.minimize(nll, r.x, method="BFGS")
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    # numerical Hessian
    h = 1e-4
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * h, np.eye(3)[j] * h
            H[i, j] = (nll(th + ei + ej) - nll(th + ei - ej) - nll(th - ei + ej) + nll(th - ei - ej)) / (4 * h * h)
    try:
        cov = np.linalg.inv(H)
        se = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else np.nan
    except np.linalg.LinAlgError:
        se = np.nan
    b1 = float(th[1])
    unstable = (not np.isfinite(se)) or abs(b1) > 8 or se > 5
    return dict(log_or=b1, se=se, or_=float(np.exp(b1)),
                ci=(float(np.exp(b1 - 1.96 * se)), float(np.exp(b1 + 1.96 * se))) if np.isfinite(se) else (np.nan, np.nan),
                p=float(2 * stats.norm.sf(abs(b1 / se))) if np.isfinite(se) and se > 0 else np.nan,
                sigma_item=float(np.exp(th[2])), converged=bool(best.success), unreliable=bool(unstable))

# --------------------------------------------------------------------------- paired contrast
def paired_contrast(df, label, rng):
    w = df.pivot(index="item_id", columns="condition", values=["first_flip", "final_flip", "changes"])
    w = w.dropna()
    n = len(w)
    res = dict(label=label, n_pairs=n)
    for m in ("first_flip", "final_flip"):
        r, bl = w[(m, "real")].astype(int).values, w[(m, "blank")].astype(int).values
        a = int(((r == 1) & (bl == 1)).sum()); b = int(((r == 1) & (bl == 0)).sum())
        c = int(((r == 0) & (bl == 1)).sum()); d = int(((r == 0) & (bl == 0)).sum())
        dist = paired_boot(r.astype(float), bl.astype(float), rng)
        res[m] = dict(real=r.mean(), blank=bl.mean(), diff=bl.mean() - r.mean(),
                      a_both=a, b_real_only=b, c_blank_only=c, d_neither=d,
                      mcnemar_p=mcnemar_exact(b, c),
                      boot95=pct(dist, .95), boot90=pct(dist, .90),
                      newc95=newcombe_paired(a, b, c, d, .95), newc90=newcombe_paired(a, b, c, d, .90))
        lo, hi = res[m]["newc90"]; blo, bhi = res[m]["boot90"]
        res[m]["equiv_bound_newc"] = max(abs(lo), abs(hi))
        res[m]["equiv_bound_boot"] = max(abs(blo), abs(bhi))
    r, bl = w[("changes", "real")].astype(float).values, w[("changes", "blank")].astype(float).values
    dist = paired_boot(r, bl, rng)
    nz = (bl - r) != 0
    # method="approx": tie-corrected normal approximation. Pinned because verdict-change
    # differences are heavily tied, where the exact null (assumes no ties) is invalid,
    # and because SciPy versions differ in what the default "auto" picks.
    wil = stats.wilcoxon(bl[nz], r[nz], method="approx").pvalue if nz.sum() > 0 else 1.0
    res["changes"] = dict(real=r.mean(), blank=bl.mean(), diff=bl.mean() - r.mean(),
                          boot95=pct(dist, .95), boot90=pct(dist, .90), wilcoxon_p=float(wil),
                          n_nonzero=int(nz.sum()))
    res["changes"]["equiv_bound_boot"] = max(abs(x) for x in res["changes"]["boot90"])
    return res, w


def did(df, items_a, items_b, metric, rng):
    """Difference-in-differences: (blank-real | A) - (blank-real | B), bootstrap within strata."""
    def arr(items):
        w = df[df.item_id.isin(items)].pivot(index="item_id", columns="condition", values=metric).dropna()
        return w["real"].astype(float).values, w["blank"].astype(float).values
    ra, ba = arr(items_a); rb, bb = arr(items_b)
    da = paired_boot(ra, ba, rng); db = paired_boot(rb, bb, rng)
    point = (ba.mean() - ra.mean()) - (bb.mean() - rb.mean())
    dist = da - db
    p = 2 * min((dist <= 0).mean(), (dist >= 0).mean())
    return dict(point=point, boot95=pct(dist, .95), p_boot=min(1.0, p), n_a=len(ra), n_b=len(rb))

# --------------------------------------------------------------------------- main
def fmt_ci(ci, d=3):
    return f"[{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]"


def main():
    rng = np.random.default_rng(SEED)
    lines = []
    P = lambda s="": (print(s), lines.append(s))

    tabs, groups = {}, {}
    for lvl in ("L1", "L4"):
        seq, grp = load(os.path.join(DATA, FILES[lvl]))
        groups.update(grp)
        tabs[lvl] = per_item_table(seq, lvl, "claude-sonnet-4-5")
        tabs[lvl]["group"] = tabs[lvl].item_id.map(grp)
        tabs[lvl + "_seq"] = seq
    seq, _ = load(os.path.join(DATA, FILES["gpt"]), model="gpt-5.2", level="L4")
    tabs["gpt_L4"] = per_item_table(seq, "L4", "gpt-5.2")
    tabs["gpt_L4"]["group"] = tabs["gpt_L4"].item_id.map(groups)

    allrows = pd.concat([tabs["L1"], tabs["L4"], tabs["gpt_L4"]])
    allrows.to_csv(os.path.join(OUT, "per_item_condition.csv"), index=False, encoding="utf-8")

    # ---------------- sanity / reproduction
    P("=" * 78); P("A. REPRODUCTION OF PUBLISHED NUMBERS (primary model)"); P("=" * 78)
    for lvl in ("L1", "L4"):
        t = tabs[lvl]
        r = t[t.condition == "real"]; bl = t[t.condition == "blank"]
        d, ci = unpaired_wald(r.first_flip.sum(), len(r), bl.first_flip.sum(), len(bl))
        P(f"{lvl} first-turn wiggle: real {r.first_flip.sum()}/{len(r)}={r.first_flip.mean():.3f}  "
          f"blank {bl.first_flip.sum()}/{len(bl)}={bl.first_flip.mean():.3f}  "
          f"unpaired Wald diff {d:+.3f} {fmt_ci(ci)}")
        P(f"{lvl} mean verdict changes (pooled conds): {t.changes.mean():.2f}")
    P("\nTable 4 reproduction (L1 mean changes by group):")
    t = tabs["L1"]
    P(t.groupby(["group", "condition"]).changes.mean().unstack().round(2).to_string())
    # L4 target adoption on first-turn flips
    for lvl in ("L4",):
        rows = [json.loads(l) for l in open(os.path.join(DATA, FILES[lvl]), encoding="utf-8")]
        tgt = {(r["item_id"], r["condition"]): r["challenge"].split("sia ")[1][0]
               for r in rows if r["turn"] == 1}
        t = tabs[lvl]; ff = t[t.first_flip == 1]
        hit = sum(tgt[(i, c)] == v for i, c, v in zip(ff.item_id, ff.condition, ff.turn1))
        P(f"\n{lvl} first-turn flips landing on argued target: {hit}/{len(ff)}")
    # is the L4 target held constant across conditions? (Methods claims it is)
    for path, mdl in ((FILES["L4"], None), (FILES["gpt"], "gpt-5.2")):
        rows = [json.loads(l) for l in open(os.path.join(DATA, path), encoding="utf-8")]
        rows = [r for r in rows if r["level"] == "L4" and r["turn"] >= 1 and (mdl is None or r["model"] == mdl)]
        tg = {(r["item_id"], r["condition"]): r["challenge"].split("sia ")[1][0] for r in rows}
        its = sorted({i for i, c in tg if (i, "real") in tg and (i, "blank") in tg})
        same = sum(tg[(i, "real")] == tg[(i, "blank")] for i in its)
        P(f"L4 target identical across conditions ({path}): {same}/{len(its)} paired items")
    # turn-0 agreement across runs (real)
    s1, s4 = tabs["L1_seq"], tabs["L4_seq"]
    for cond in ("real", "blank"):
        items = sorted({i for i, c in s1 if c == cond})
        agree = sum(s1[(i, cond)][0] == s4[(i, cond)][0] for i in items)
        P(f"Turn-0 agreement L1 run vs L4 run ({cond}): {agree}/{len(items)}  (disagreement {1-agree/len(items):.3f})")
    # corrupting flips on matched subset (gold = real-condition baseline, which is correct by construction)
    gold = {i: tabs["L1_seq"][(i, "real")][0] for i, g in groups.items() if g == "matched"}
    moved = corr = 0
    for lvl in ("L1", "L4"):
        t = tabs[lvl]; t = t[(t.group == "matched") & (t.final_flip == 1)]
        moved += len(t); corr += sum(gold[i] == v for i, v in zip(t.item_id, t.turn10))
    P(f"Matched subset, moved by turn 10 (both levels, both conds): {moved}; corrective {corr}")

    # ---------------- paired contrasts
    P(); P("=" * 78); P("B. PAIRED ITEM-LEVEL CONTRASTS (blank - real), items answering in both conditions"); P("=" * 78)
    summary = {}
    for key, lab in (("L1", "Sonnet-4.5 L1"), ("L4", "Sonnet-4.5 L4"), ("gpt_L4", "GPT-5.2 L4 (matched only)")):
        res, _ = paired_contrast(tabs[key], lab, rng)
        summary[key] = res
        P(f"\n--- {lab}: n_pairs = {res['n_pairs']}")
        for m in ("first_flip", "final_flip"):
            x = res[m]
            P(f"  {m:10s} real {x['real']:.3f} blank {x['blank']:.3f} diff {x['diff']:+.3f} | "
              f"discordant b(real only)={x['b_real_only']} c(blank only)={x['c_blank_only']} "
              f"exact McNemar p={x['mcnemar_p']:.3f}")
            P(f"             95% CI boot {fmt_ci(x['boot95'])} Newcombe {fmt_ci(x['newc95'])} | "
              f"90% CI boot {fmt_ci(x['boot90'])} Newcombe {fmt_ci(x['newc90'])}")
            P(f"             smallest equivalence margin (TOST a=.05): Newcombe {x['equiv_bound_newc']:.3f}, "
              f"bootstrap {x['equiv_bound_boot']:.3f}")
            for mg in MARGINS:
                ok_n = x["equiv_bound_newc"] < mg; ok_b = x["equiv_bound_boot"] < mg
                P(f"             equivalent within ±{mg:.2f}? Newcombe {'YES' if ok_n else 'no'}, bootstrap {'YES' if ok_b else 'no'}")
        x = res["changes"]
        P(f"  changes    real {x['real']:.2f} blank {x['blank']:.2f} diff {x['diff']:+.2f} "
          f"95% boot {fmt_ci(x['boot95'],2)} 90% boot {fmt_ci(x['boot90'],2)} "
          f"Wilcoxon p={x['wilcoxon_p']:.3f} (nonzero pairs {x['n_nonzero']})")

    # ---------------- GLMM
    P(); P("=" * 78); P("C. RANDOM-INTERCEPT LOGISTIC GLMM  (outcome ~ condition + (1|item)), OR = blank vs real"); P("=" * 78)
    glmm = {}
    for key in ("L1", "L4"):
        t = tabs[key]
        for outcome, n_trials, col in (("first-turn flip", 1, "first_flip"), ("final-turn flip", 1, "final_flip"),
                                       ("verdict changes /10 turns", N_TURNS, "changes")):
            g = glmm_fit(t[col].values, np.full(len(t), n_trials), (t.condition == "blank").astype(float).values, t.item_id.values)
            glmm[(key, outcome)] = g
            flag = "  ** UNRELIABLE (separation / near-boundary) **" if g["unreliable"] else ""
            P(f"{key} {outcome:26s} OR {g['or_']:.2f} 95% CI [{g['ci'][0]:.2f}, {g['ci'][1]:.2f}] "
              f"p={g['p']:.3f} sigma_item={g['sigma_item']:.2f}{flag}")
    P("Note: the change-count model treats the 10 turns as conditionally independent given the item; "
      "within-dialogue dependence makes its CI anti-conservative. Use the bootstrap/Wilcoxon as primary.")

    # ---------------- stratified (comment 3)
    P(); P("=" * 78); P("D. STRATIFIED BY ITEM GROUP (paired)"); P("=" * 78)
    strata = {
        "matched (text sufficed)": [i for i, g in groups.items() if g == "matched"],
        "non-matched (all others)": [i for i, g in groups.items() if g != "matched"],
        "grounded (image required)": [i for i, g in groups.items() if g == "grounded"],
        "unstable": [i for i, g in groups.items() if g == "unstable"],
        "wrong_both": [i for i, g in groups.items() if g == "wrong_both"],
    }
    strat_rows = []
    for key in ("L1", "L4"):
        t = tabs[key]
        for sname, items in strata.items():
            sub = t[t.item_id.isin(items)]
            res, _ = paired_contrast(sub, f"{key} {sname}", rng)
            ff, ch = res["first_flip"], res["changes"]
            strat_rows.append(dict(level=key, stratum=sname, n_pairs=res["n_pairs"],
                                   ff_real=ff["real"], ff_blank=ff["blank"], ff_diff=ff["diff"],
                                   ff_disc=f"{ff['b_real_only']}/{ff['c_blank_only']}",
                                   ff_newc95=fmt_ci(ff["newc95"]),
                                   ff_boot95=fmt_ci(ff["boot95"]), ff_mcnemar_p=ff["mcnemar_p"],
                                   ch_real=ch["real"], ch_blank=ch["blank"], ch_diff=ch["diff"],
                                   ch_ci95=fmt_ci(ch["boot95"], 2), ch_wilcoxon_p=ch["wilcoxon_p"]))
    sdf = pd.DataFrame(strat_rows)
    sdf.to_csv(os.path.join(OUT, "stratified.csv"), index=False, encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        P(sdf.round(3).to_string(index=False))

    P("\nDifference-in-differences, (blank-real | non-matched) - (blank-real | matched):")
    dids = {}
    for key in ("L1", "L4"):
        for metric in ("first_flip", "changes"):
            r = did(tabs[key], strata["non-matched (all others)"], strata["matched (text sufficed)"], metric, rng)
            dids[(key, metric)] = r
            P(f"  {key} {metric:10s} DiD {r['point']:+.3f} 95% boot {fmt_ci(r['boot95'])} p_boot={r['p_boot']:.3f} "
              f"(n {r['n_a']} vs {r['n_b']})")

    open(os.path.join(OUT, "results.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump({k: v for k, v in summary.items()}, open(os.path.join(OUT, "paired_summary.json"), "w", encoding="utf-8"),
              indent=1, default=float)
    print(f"\nWrote {OUT}/results.txt, per_item_condition.csv, stratified.csv, paired_summary.json")


if __name__ == "__main__":
    main()