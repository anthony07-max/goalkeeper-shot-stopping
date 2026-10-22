"""Streamlit app: goalkeeper shot-stopping explorer (StatsBomb Open Data, 2015/16 big-four leagues).
Run the pipeline first (./run_pipeline.sh), then: streamlit run app/streamlit_app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from mplsoccer import VerticalPitch

P = Path(__file__).resolve().parents[1] / "data" / "processed"
st.set_page_config(page_title="Goalkeeper shot-stopping", layout="wide")


@st.cache_data
def load():
    return (pd.read_csv(P / "keeper_ratings.csv"), pd.read_csv(P / "shots_scored.csv"),
            pd.read_csv(P / "keeper_distribution.csv"))


ratings, shots, dist = load()
st.title("Goalkeeper shot-stopping, 2015/16")
st.caption("Premier League, La Liga, Serie A and Ligue 1. Data: StatsBomb Open Data (free for non-commercial "
           "use, credit StatsBomb). Goals prevented = on-target xG (StatsBomb xG recalibrated to on-target shots) "
           "minus goals conceded. Own goals are excluded.")

leagues = st.sidebar.multiselect("Leagues", sorted(ratings.league.unique()), default=sorted(ratings.league.unique()))
min_n = st.sidebar.slider("Minimum on-target shots faced", 10, 150, 60, 5)
metric = st.sidebar.radio("Rank by", ["gp_otxg", "gp_per100_shrunk", "gp_psxg", "save_pct"],
                          format_func={"gp_otxg": "Goals prevented (on-target xG)",
                                       "gp_per100_shrunk": "Goals prevented per 100 shots (shrunk)",
                                       "gp_psxg": "Goals prevented (placement model)", "save_pct": "Save %"}.get)
r = ratings[ratings.league.isin(leagues) & (ratings.on_target >= min_n)].sort_values(metric, ascending=False)

tab1, tab2, tab3 = st.tabs(["Rankings", "Keeper profile", "Distribution"])
with tab1:
    st.write(f"{len(r)} keepers. Intervals show the range expected from shot luck alone (90%). "
             "Keepers whose interval crosses zero cannot be told apart from average on this sample.")
    fig = px.scatter(r, x="on_target", y="gp_per100", hover_name="keeper_name", color="league",
                     hover_data=["keeper_team", "goals", "save_pct"],
                     labels={"on_target": "On-target shots faced", "gp_per100": "Goals prevented per 100 shots"})
    fig.add_hline(y=0, line_color="grey")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(r[["keeper_name", "keeper_team", "league", "matches", "on_target", "goals", "save_pct", "otxg",
                    "gp_otxg", "gp_ci90_low", "gp_ci90_high", "gp_per100_shrunk", "gp_psxg", "gp_raw_xg"]].round(2),
                 hide_index=True, width="stretch")

with tab2:
    opts = r.keeper_name + " (" + r.keeper_team + ")"
    pick = st.selectbox("Keeper", opts.tolist())
    if pick:
        row = r[opts == pick].iloc[0]
        ks = shots[(shots.keeper_name == row.keeper_name) & (shots.keeper_team == row.keeper_team)]
        a, b, c, d = st.columns(4)
        a.metric("On-target shots", int(row.on_target))
        b.metric("Save %", f"{row.save_pct:.1f}%", f"{100 * (1 - ks.p_otxg.mean()) - row.save_pct:+.1f} expected gap",
                 delta_color="inverse")
        c.metric("Goals prevented", f"{row.gp_otxg:+.1f}", f"90% luck range {row.gp_ci90_low:+.1f} to {row.gp_ci90_high:+.1f}",
                 delta_color="off")
        d.metric("Shrunk per 100 shots", f"{row.gp_per100_shrunk:+.2f}")
        c1, c2 = st.columns(2)
        with c1:
            pitch = VerticalPitch(pitch_type="statsbomb", half=True)
            fig, ax = pitch.draw(figsize=(6, 5))
            sv, gl = ks[~ks.goal], ks[ks.goal]
            pitch.scatter(sv.x, sv.y, s=sv.xg * 600 + 15, color="#2a9d8f", alpha=0.6, ax=ax, label="Saved")
            pitch.scatter(gl.x, gl.y, s=gl.xg * 600 + 15, color="#e63946", alpha=0.8, ax=ax, label="Goal")
            ax.legend(loc="lower left"); ax.set_title("Shots faced (size = StatsBomb xG)")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.add_patch(plt.Rectangle((36, 0), 8, 2.67, fill=False, lw=3))
            ax.scatter(ks.loc[~ks.goal, "end_y"], ks.loc[~ks.goal, "end_z"], color="#2a9d8f", alpha=0.6, label="Saved")
            ax.scatter(ks.loc[ks.goal, "end_y"], ks.loc[ks.goal, "end_z"], color="#e63946", alpha=0.8, label="Goal")
            ax.set_xlim(34, 46); ax.set_ylim(0, 3.2); ax.set_title("Goal mouth (shooter's view)"); ax.legend()
            st.pyplot(fig)
            by = ks.groupby("zone").agg(shots=("goal", "size"), goals=("goal", "sum"), expected=("p_otxg", "sum"))
            by["save_pct"] = 100 * (1 - by.goals / by.shots)
            by["expected_save_pct"] = 100 * (1 - by.expected / by.shots)
            st.dataframe(by.round(1), width="stretch")

with tab3:
    dd = dist[dist.league.isin(leagues)]
    fig = px.scatter(dd, x="long_share", y="completion", size="passes_per_match", color="league", hover_name="keeper_name",
                     hover_data=["team", "mean_length_m", "goal_kick_long_share"],
                     labels={"long_share": "Passes 40+ yards (%)", "completion": "Pass completion (%)"})
    st.plotly_chart(fig, width="stretch")
    st.dataframe(dd.round(1), hide_index=True, width="stretch")
