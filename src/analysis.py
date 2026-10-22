"""Goalkeeper shot-stopping: goals prevented, save % by shot location/type, distribution, rankings.

Three expected-goals baselines for on-target shots are compared:
  raw   StatsBomb pre-shot xG as published. Every on-target shot is scored with its pre-shot xG,
        which ignores that the shot was on target, so it under-states goals and gives negative totals.
  otxg  "on-target xG": StatsBomb xG recalibrated to on-target shots with a logistic regression on
        logit(xG), penalty and header flags. Main measure: it keeps StatsBomb's shot quality but asks
        "how often does a shot of this quality, once on target, go in?".
  psxg  a simple placement-aware model that also uses where the shot crossed/was stopped in the goal mouth.
        StatsBomb records a saved shot's end location where the keeper stopped it, so this is a rough check only.
Model probabilities are out-of-fold (5-fold, grouped by match), so no shot is scored by a model that saw it.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
P, CH = ROOT / "data" / "processed", ROOT / "charts"
MIN_SHOTS = 60      # on-target shots faced to be ranked
CREDIT = "Data: StatsBomb Open Data, 2015/16 Premier League, La Liga, Serie A, Ligue 1"


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def oof_predict(X, y, groups):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        mdl = LogisticRegression(C=10, max_iter=2000).fit(X[tr], y[tr])
        pred[te] = mdl.predict_proba(X[te])[:, 1]
    return pred


def savefig(fig, name):
    fig.savefig(CH / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_rate(df, by):
    g = df.groupby(by, observed=True).agg(shots=("goal", "size"), goals=("goal", "sum"), mean_xg=("xg", "mean"),
                                          otxg=("p_otxg", "sum"))
    g["save_pct"] = 100 * (1 - g.goals / g.shots)
    g["expected_save_pct"] = 100 * (1 - g.otxg / g.shots)
    return g.reset_index()


def main():
    CH.mkdir(exist_ok=True)
    s = pd.read_csv(P / "shots_faced.csv")
    passes = pd.read_csv(P / "gk_passes.csv.gz")
    apps = pd.read_csv(P / "gk_matches.csv")
    y = s.goal.astype(int).values
    head = (s.body_part == "Head").astype(int)
    pen = s.is_penalty.astype(int)
    X1 = np.column_stack([logit(s.xg), pen, head])
    off_c = (s.mouth_y - 40).abs() / 4           # 0 = centre of goal, 1 = post
    X2 = np.column_stack([logit(s.xg), pen, head, off_c, s.mouth_z / 2.67, off_c * s.mouth_z / 2.67,
                          off_c ** 2, (s.mouth_z / 2.67) ** 2])
    s["p_otxg"] = oof_predict(X1, y, s.match_id)
    s["p_psxg"] = oof_predict(X2, y, s.match_id)
    model_eval = {name: {"auc": round(roc_auc_score(y, p), 3), "log_loss": round(log_loss(y, p), 4),
                         "brier": round(brier_score_loss(y, p), 4), "expected_goals": round(float(p.sum()), 1)}
                  for name, p in [("raw_statsbomb_xg", s.xg), ("otxg", s.p_otxg), ("psxg", s.p_psxg)]}
    s.round(4).to_csv(P / "shots_scored.csv", index=False)

    kf = {"on_target_shots": int(len(s)), "goals": int(y.sum()), "save_pct": round(100 * (1 - y.mean()), 1),
          "raw_xg_on_target": round(s.xg.sum(), 1), "models": model_eval,
          "keepers": int(s.keeper_name.nunique()),
          "matches": int(s.match_id.nunique())}
    kf["save_pct_by_league"] = (100 * (1 - s.groupby("league").goal.mean())).round(1).to_dict()

    # ---------- save % by location and type ----------
    tables = {}
    for col in ["zone", "subtype", "body_part", "play_pattern", "height", "side"]:
        tables[col] = save_rate(s, col)
    s["xg_band"] = pd.cut(s.xg, [0, 0.05, 0.1, 0.2, 0.35, 0.6, 1], include_lowest=True)
    tables["xg_band"] = save_rate(s, "xg_band")
    mouth = save_rate(s[~s.is_penalty], ["height", "side"])
    pd.concat([t.rename(columns={t.columns[0]: "group"}).assign(split=k) for k, t in tables.items()]).round(3).to_csv(
        P / "save_pct_by_group.csv", index=False)
    mouth.round(3).to_csv(P / "save_pct_goal_mouth.csv", index=False)
    kf["save_pct_by_zone"] = tables["zone"].set_index("zone").save_pct.round(1).to_dict()
    kf["save_pct_by_type"] = tables["subtype"].set_index("subtype").save_pct.round(1).to_dict()
    kf["save_pct_by_height"] = tables["height"].set_index("height").save_pct.round(1).to_dict()
    kf["save_pct_header_vs_foot"] = tables["body_part"].set_index("body_part").save_pct.round(1).to_dict()

    # ---------- keeper ratings ----------
    s["var_otxg"] = s.p_otxg * (1 - s.p_otxg)
    k = s.groupby(["keeper_name", "keeper_team", "league"]).agg(
        on_target=("goal", "size"), goals=("goal", "sum"), penalties=("is_penalty", "sum"),
        raw_xg=("xg", "sum"), otxg=("p_otxg", "sum"), psxg=("p_psxg", "sum"), var=("var_otxg", "sum"),
        mean_xg=("xg", "mean")).reset_index()
    k = k.merge(apps.rename(columns={"keeper": "_k", "team": "keeper_team"})[["keeper_name", "keeper_team", "matches"]],
                on=["keeper_name", "keeper_team"], how="left")
    k["save_pct"] = 100 * (1 - k.goals / k.on_target)
    k["gp_raw_xg"] = k.raw_xg - k.goals
    k["gp_otxg"] = k.otxg - k.goals
    k["gp_psxg"] = k.psxg - k.goals
    k["gp_per100"] = 100 * k.gp_otxg / k.on_target
    k["gp_ci90_low"] = k.gp_otxg - 1.645 * np.sqrt(k["var"])
    k["gp_ci90_high"] = k.gp_otxg + 1.645 * np.sqrt(k["var"])
    k["z"] = k.gp_otxg / np.sqrt(k["var"])
    # empirical-Bayes shrinkage of goals prevented per shot towards zero (the league average)
    rate, rvar = k.gp_otxg / k.on_target, k["var"] / k.on_target ** 2
    big = k.on_target >= 30
    tau2 = max(rate[big].var() - rvar[big].mean(), 0)
    k["gp_per100_shrunk"] = 100 * rate * tau2 / (tau2 + rvar)
    k["reliability_weight"] = tau2 / (tau2 + rvar)
    k = k.sort_values("gp_otxg", ascending=False)
    k.drop(columns="var").round(3).to_csv(P / "keeper_ratings.csv", index=False)
    r = k[k.on_target >= MIN_SHOTS].copy()
    kf["ranked_keepers"] = int(len(r))
    kf["min_on_target_for_ranking"] = MIN_SHOTS
    kf["tau_per100"] = round(100 * np.sqrt(tau2), 2)
    cols = ["keeper_name", "keeper_team", "on_target", "goals", "save_pct", "otxg", "gp_otxg", "gp_ci90_low",
            "gp_ci90_high", "gp_per100_shrunk", "gp_raw_xg", "gp_psxg"]
    kf["top5"] = r[cols].head(5).round(2).values.tolist()
    kf["bottom5"] = r[cols].tail(5).round(2).values.tolist()
    kf["significant_at_90pct"] = int(((r.gp_ci90_low > 0) | (r.gp_ci90_high < 0)).sum())
    kf["spearman_otxg_vs_psxg_gp"] = round(stats.spearmanr(r.gp_otxg, r.gp_psxg)[0], 3)
    kf["spearman_otxg_vs_raw_gp"] = round(stats.spearmanr(r.gp_otxg / r.on_target, r.gp_raw_xg / r.on_target)[0], 3)
    kf["raw_xg_gp_all_negative_share"] = round(100 * (r.gp_raw_xg < 0).mean(), 1)

    # ---------- split-half reliability ----------
    s = s.sort_values(["date", "match_id"])
    s["order"] = s.groupby("keeper_name").match_id.transform(lambda m: pd.factorize(m)[0])
    s["half"] = np.where(s.order % 2 == 0, "A", "B")
    h = s.groupby(["keeper_name", "half"]).agg(n=("goal", "size"), gp=("p_otxg", "sum"), g=("goal", "sum"))
    h["rate"] = (h.gp - h.g) / h.n
    h = h.unstack("half").dropna()
    h = h[(h[("n", "A")] >= 30) & (h[("n", "B")] >= 30)]
    rel = stats.pearsonr(h[("rate", "A")], h[("rate", "B")])
    sv = stats.pearsonr(1 - h[("g", "A")] / h[("n", "A")], 1 - h[("g", "B")] / h[("n", "B")])
    kf["split_half"] = {"keepers": int(len(h)), "r_goals_prevented_per_shot": round(rel[0], 3),
                        "p": float(f"{rel[1]:.2g}"), "r_save_pct": round(sv[0], 3)}

    # ---------- distribution ----------
    passes["long"] = passes.length_m >= 36.6   # 40 yards
    d = passes.groupby(["keeper_name", "team"]).agg(passes=("complete", "size"), completion=("complete", "mean"),
            long_share=("long", "mean"), mean_length_m=("length_m", "mean")).reset_index()
    gkk = passes[passes.pass_type == "Goal Kick"].groupby(["keeper_name", "team"]).agg(
        goal_kicks=("long", "size"), goal_kick_long_share=("long", "mean")).reset_index()
    d = d.merge(gkk, on=["keeper_name", "team"], how="left").merge(
        apps[["keeper_name", "team", "matches", "league"]], on=["keeper_name", "team"], how="left")
    d["passes_per_match"] = d.passes / d.matches
    d = d[d.matches >= 15]
    for c in ["completion", "long_share", "goal_kick_long_share"]:
        d[c] = 100 * d[c]
    d.round(2).sort_values("completion", ascending=False).to_csv(P / "keeper_distribution.csv", index=False)
    kf["distribution"] = {"keepers_15plus_matches": int(len(d)),
                          "completion_pct_range": [round(d.completion.min(), 1), round(d.completion.max(), 1)],
                          "most_accurate": [(a, b, round(c, 1), round(e, 1)) for a, b, c, e in [tuple(d.sort_values("completion").iloc[-1][["keeper_name", "team", "completion", "long_share"]])]][0],
                          "least_accurate": [(a, b, round(c, 1), round(e, 1)) for a, b, c, e in [tuple(d.sort_values("completion").iloc[0][["keeper_name", "team", "completion", "long_share"]])]][0],
                          "rho_long_share_vs_completion": round(stats.spearmanr(d.long_share, d.completion)[0], 3)}
    lg = passes.groupby("league").agg(completion=("complete", "mean"), long_share=("long", "mean"))
    kf["distribution_by_league"] = (100 * lg).round(1).to_dict(orient="index")
    dd = d.merge(k[["keeper_name", "keeper_team", "gp_per100"]].rename(columns={"keeper_team": "team"}), on=["keeper_name", "team"])
    kf["rho_distribution_vs_shot_stopping"] = round(stats.spearmanr(dd.completion, dd.gp_per100)[0], 3)

    with open(P / "key_findings.json", "w") as f:
        json.dump(kf, f, indent=2, default=float)

    # ---------- charts ----------
    pitch = VerticalPitch(pitch_type="statsbomb", half=True, line_color="#444")
    fig, axes = pitch.draw(ncols=2, figsize=(14, 7))
    np_s = s[~s.is_penalty]
    bs = pitch.bin_statistic(np_s.x, np_s.y, values=(~np_s.goal).astype(float), statistic="mean", bins=(12, 8))
    cnt = pitch.bin_statistic(np_s.x, np_s.y, statistic="count", bins=(12, 8))
    bs["statistic"] = np.where(cnt["statistic"] >= 20, bs["statistic"] * 100, np.nan)
    pitch.heatmap(bs, ax=axes[0], cmap="RdYlGn", vmin=40, vmax=100, edgecolors="white")
    pitch.label_heatmap(bs, ax=axes[0], str_format="{:.0f}%", fontsize=8, color="black",
                        exclude_nan=True)
    axes[0].set_title("Save % on on-target shots by shot location\n(non-penalty, bins with 20+ shots)")
    cnt["statistic"] = np.where(cnt["statistic"] > 0, cnt["statistic"], np.nan)
    pitch.heatmap(cnt, ax=axes[1], cmap="Blues", edgecolors="white")
    pitch.label_heatmap(cnt, ax=axes[1], str_format="{:.0f}", fontsize=8, color="black", exclude_nan=True)
    axes[1].set_title("Number of on-target shots")
    fig.text(0.02, 0.02, CREDIT, fontsize=8, color="#555")
    savefig(fig, "01_save_pct_by_location.png")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = ["Left third", "Centre", "Right third"]
    grid = mouth.pivot(index="height", columns="side", values="save_pct").reindex(["High", "Middle", "Low"])[order]
    n = mouth.pivot(index="height", columns="side", values="shots").reindex(["High", "Middle", "Low"])[order]
    im = ax.imshow(grid.values, cmap="RdYlGn", vmin=40, vmax=100, aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{grid.values[i, j]:.0f}%\n(n={int(n.values[i, j]):,})", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(3), grid.columns); ax.set_yticks(range(3), grid.index)
    ax.set_title("Save % by where the shot reached the goal\n(non-penalty; left/right as seen by the shooter)")
    fig.colorbar(im, ax=ax, label="Save %")
    fig.text(0.01, -0.03, CREDIT + ". Saved shots use the point where they were stopped.", fontsize=7, color="#555")
    savefig(fig, "02_save_pct_goal_mouth.png")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for ax, (col, title) in zip(axes, [("zone", "Shot zone"), ("subtype", "Shot type"),
                                       ("body_part", "Body part"), ("xg_band", "StatsBomb xG band")]):
        t = tables[col]
        t = t[t.shots >= 30]
        lab = t[col].astype(str)
        ax.bar(lab, t.save_pct, color="#2a9d8f")
        for i, (v, nn) in enumerate(zip(t.save_pct, t.shots)):
            ax.text(i, v + 1, f"{v:.0f}%\n{nn:,}", ha="center", fontsize=8)
        ax.set_ylim(0, 105); ax.set_title(title); ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Save % (on-target shots)")
    fig.suptitle("Save % by shot location and type (labels: save %, shots)", y=1.02)
    fig.text(0.01, -0.06, CREDIT, fontsize=7, color="#555")
    fig.tight_layout(); savefig(fig, "03_save_pct_by_shot_type.png")

    rk = pd.concat([r.head(12), r.tail(12)]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 9))
    yy = np.arange(len(rk))
    ax.errorbar(rk.gp_otxg, yy, xerr=[rk.gp_otxg - rk.gp_ci90_low, rk.gp_ci90_high - rk.gp_otxg], fmt="o",
                color="#1d3557", ecolor="#a8dadc", elinewidth=3, capsize=0)
    ax.set_yticks(yy, [f"{a} ({b}, n={c})" for a, b, c in zip(rk.keeper_name, rk.keeper_team, rk.on_target)], fontsize=8)
    ax.axvline(0, color="grey", lw=1); ax.axhline(11.5, color="#ccc", ls="--")
    ax.set_xlabel("Goals prevented vs on-target xG (bar = approx. 90% interval from shot luck alone)")
    ax.set_title(f"Best and worst 12 shot-stoppers, 2015/16 (min {MIN_SHOTS} on-target shots faced)")
    ax.text(0.01, -0.07, CREDIT, transform=ax.transAxes, fontsize=7, color="#555")
    savefig(fig, "04_goals_prevented_ranking.png")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    kk = k[k.on_target >= 10]
    ax.scatter(kk.on_target, kk.gp_per100, s=18, alpha=0.6, color="#457b9d")
    nn = np.arange(10, kk.on_target.max() + 5)
    sd = 100 * np.sqrt((s.var_otxg.mean()) / nn)
    ax.plot(nn, 1.645 * sd, "--", color="#e63946", label="90% range expected from luck alone")
    ax.plot(nn, -1.645 * sd, "--", color="#e63946")
    for row in kk[kk.z.abs() > 1.9].itertuples():
        ax.annotate(row.keeper_name, (row.on_target, row.gp_per100), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("On-target shots faced"); ax.set_ylabel("Goals prevented per 100 on-target shots")
    ax.set_title("Sample size matters: most keepers sit inside the luck-only funnel")
    ax.legend(); ax.text(0.01, -0.1, CREDIT, transform=ax.transAxes, fontsize=7, color="#555")
    savefig(fig, "05_sample_size_funnel.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(100 * h[("rate", "A")], 100 * h[("rate", "B")], color="#6a4c93", alpha=0.7)
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("Goals prevented per 100 shots, odd-numbered matches")
    ax.set_ylabel("Goals prevented per 100 shots, even-numbered matches")
    ax.set_title(f"Split-half reliability: r = {rel[0]:.2f} ({len(h)} keepers, 30+ shots in each half)")
    ax.text(0.01, -0.12, CREDIT, transform=ax.transAxes, fontsize=7, color="#555")
    savefig(fig, "06_split_half_reliability.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"Premier League": "#3d195b", "La Liga": "#ee8707", "Serie A": "#008fd7", "Ligue 1": "#2a9d8f"}
    for lg_name, g in d.groupby("league"):
        ax.scatter(g.long_share, g.completion, s=g.passes_per_match * 4, alpha=0.65, color=colors.get(lg_name), label=lg_name)
    for row in d[(d.completion > d.completion.quantile(0.93)) | (d.completion < d.completion.quantile(0.07))].itertuples():
        ax.annotate(row.keeper_name, (row.long_share, row.completion), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("Share of passes 40+ yards (%)"); ax.set_ylabel("Pass completion (%)")
    ax.set_title("Goalkeeper distribution: long-ball share vs completion (size = passes per match, 15+ matches)")
    ax.legend(); ax.text(0.01, -0.1, CREDIT, transform=ax.transAxes, fontsize=7, color="#555")
    savefig(fig, "07_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    cb = tables["xg_band"]
    xs = np.arange(len(cb))
    ax.bar(xs - 0.2, 100 * cb.mean_xg, 0.4, label="Mean StatsBomb xG (pre-shot)", color="#adb5bd")
    ax.bar(xs + 0.2, 100 - cb.save_pct, 0.4, label="Actual conversion of on-target shots", color="#e76f51")
    ax.set_xticks(xs, cb.xg_band.astype(str), rotation=20); ax.set_ylabel("%")
    ax.set_title("Why raw xG understates goals on on-target shots\n(so it is recalibrated before rating keepers)")
    ax.legend(); ax.text(0.01, -0.15, CREDIT, transform=ax.transAxes, fontsize=7, color="#555")
    savefig(fig, "08_xg_calibration.png")
    print(json.dumps(kf, indent=1, default=float))


if __name__ == "__main__":
    main()
