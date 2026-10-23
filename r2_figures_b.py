"""RESS R2, step 3: manuscript figures 6-9 and the seasonal tables, from R2 data.

  Fig 6  SHAP (bar + beeswarm) of the decision-time hybrid models, inbound and outbound
  Fig 7  CPA comparison: centre CPA, signed Delta_SD, truncated CPA-SD, hull polygon; by encounter type
  Fig 8  quarterly boxplots on the 3,243 log-confirmed transfers + Tables 6/7 (means, Kruskal-Wallis)
  Fig 9  worked relocation case (COSCO PORTUGAL, 2023-11-08) on real AIS traffic at t_dec

SHAP is exact for the hybrid: TreeExplainer on the RF stage plus the linear SHAP of the
Ridge residual stage (coef * standardised deviation); the one-hot ship type is summed
into one feature. The outbound figure uses the single hybrid (the route ensemble's
classifier-switched regressors have no single additive explanation).

Output: RESS_JRESS-D-26-03347/figures_R2/*.png, results/revision_r2_step3/figures_b_numbers.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("FEATURE_SET", "dispatch")  # decision at pilot-boat dispatch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import kruskal

import revision_r2_validation as v
import train_models as tm
from src.ml.model import HybridDistanceRegressor

ROOT = Path(__file__).parent
FIG = ROOT / "RESS_JRESS-D-26-03347" / "figures_R2"
OUT = ROOT / "results" / "revision_r2_step3"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "black": "#000000", "grey": "#999999"}
DIR_STYLE = {"inbound": dict(color=OI["blue"], hatch="", marker="o"),
             "outbound": dict(color=OI["orange"], hatch="///", marker="s")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                     "savefig.dpi": 300, "figure.dpi": 100})
W = 17 / 2.54  # 17 cm print width

LABELS = {
    "ref_dist_to_breakwater_m": "Ship distance to breakwater",
    "ref_bearing_from_breakwater_sin": "Ship bearing from breakwater (sin)",
    "ref_bearing_from_breakwater_cos": "Ship bearing from breakwater (cos)",
    "ref_sog_kn": "Ship SOG",
    "ref_cog_sin": "Ship COG (sin)", "ref_cog_cos": "Ship COG (cos)",
    "traffic_0_5nm": "Vessels within 0.5 NM", "traffic_1nm": "Vessels within 1 NM",
    "traffic_2nm": "Vessels within 2 NM", "nearest_third_m": "Nearest third-party vessel",
    "vessel_draft_m": "Ship draught", "wind_speed_mps": "Wind speed", "wave_height_m": "Significant wave height",
    "hour_kst": "Hour of day (KST)", "month": "Month", "lead_min": "Decision lead time",
    "vessel_ship_type_class": "Ship type",
    "ship_visible": "Ship visible in AIS at dispatch",
    "pilot_dist_to_breakwater_m": "Pilot-boat distance to breakwater",
    "pilot_bearing_sin": "Pilot-boat bearing (sin)", "pilot_bearing_cos": "Pilot-boat bearing (cos)",
    "pilot_sog_kn": "Pilot-boat SOG", "pilot_cog_sin": "Pilot-boat COG (sin)", "pilot_cog_cos": "Pilot-boat COG (cos)",
    "pilot_ship_range_m": "Pilot boat–ship range",
    "pilot_to_ship_bearing_sin": "Bearing pilot boat→ship (sin)", "pilot_to_ship_bearing_cos": "Bearing pilot boat→ship (cos)",
    "intercept_dist_to_breakwater_m": "Kinematic intercept distance", "intercept_time_min": "Kinematic intercept time",
}


def _numbers() -> dict:
    p = OUT / "figures_b_numbers.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------- Fig 6 SHAP
def fig_shap(nums: dict) -> None:
    import shap
    df = v.load()
    nums["shap"] = {}
    for direction in ("inbound", "outbound"):
        sub = df[df[tm.DIRECTION_COL].str.lower() == direction].reset_index(drop=True)
        sub, _ = v.impute(sub, sub.head(1))
        X, y = sub[v.FEATS], sub[tm.TARGET].to_numpy(float)
        m = HybridDistanceRegressor(numeric_features=tm.NUM_FEATURES, categorical_features=tm.CATEGORICAL_FEATURES,
                                    rf_params=tm.RF_PARAMS, ridge_alpha=1.0).fit(X, y)
        Xe = m.preprocessor.transform(X)
        names = list(m.preprocessor.get_feature_names_out())
        sv_rf = shap.TreeExplainer(m.rf_model).shap_values(Xe, check_additivity=False)
        sc, rd = m.linear_model.named_steps["scale"], m.linear_model.named_steps["ridge"]
        sv_lin = (Xe - sc.mean_) / np.where(sc.scale_ == 0, 1, sc.scale_) * rd.coef_
        sv = sv_rf + sv_lin
        # collapse one-hot ship type into one feature; values shown are the numeric inputs
        num_idx = [i for i, n in enumerate(names) if n.startswith("num__")]
        cat_idx = [i for i, n in enumerate(names) if n.startswith("cat__")]
        feats = [n.split("__", 1)[1] for n in np.array(names)[num_idx]] + ["vessel_ship_type_class"]
        S = np.column_stack([sv[:, num_idx], sv[:, cat_idx].sum(axis=1)])
        Xv = np.column_stack([Xe[:, num_idx], np.full(len(Xe), np.nan)])
        mean_abs = np.abs(S).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        nums["shap"][direction] = {
            "model": "hybrid RF-Ridge (single), full direction set",
            "n": int(len(sub)),
            "top5_mean_abs_shap_m": {LABELS[feats[i]]: round(float(mean_abs[i]), 1) for i in order[:5]},
            "all_mean_abs_shap_m": {LABELS[feats[i]]: round(float(mean_abs[i]), 1) for i in order},
        }
        lab = [LABELS[f] for f in feats]
        top = order[:12][::-1]
        # bar
        fig, ax = plt.subplots(figsize=(W * 0.62, 3.8))
        ax.barh(range(len(top)), mean_abs[top], color=DIR_STYLE[direction]["color"], edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(len(top)), [lab[i] for i in top])
        ax.set_xlabel("Mean |SHAP| (m of meeting distance)")
        ax.set_title(f"{direction.capitalize()} (n = {len(sub):,})")
        ax.grid(axis="x", lw=0.3, alpha=0.5)
        fig.tight_layout()
        fig.savefig(FIG / f"figure_08{'a' if direction == 'inbound' else 'b'}_shap_bar_{direction}.png")
        plt.close(fig)
        # beeswarm (manual, so colours are the colour-blind-safe blue-orange ramp)
        fig, ax = plt.subplots(figsize=(W * 0.62, 3.8))
        rng = np.random.default_rng(0)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("oi", [OI["blue"], "#dddddd", OI["verm"]])
        for k, i in enumerate(top):
            x = S[:, i]
            val = Xv[:, i]
            if np.all(np.isnan(val)):
                c = np.full(len(x), 0.5)
            else:
                lo, hi = np.nanpercentile(val, [5, 95])
                c = np.clip((val - lo) / (hi - lo if hi > lo else 1), 0, 1)
            ax.scatter(x, k + rng.uniform(-0.3, 0.3, len(x)), c=c, cmap=cmap, s=3, lw=0, rasterized=True)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(range(len(top)), [lab[i] for i in top])
        ax.set_xlabel("SHAP value (m)")
        ax.set_title(f"{direction.capitalize()}")
        sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
        cb.set_ticks([0, 1], labels=["low", "high"]); cb.set_label("Feature value", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(FIG / f"figure_08{'c' if direction == 'inbound' else 'd'}_shap_summary_{direction}.png")
        plt.close(fig)


# ---------------------------------------------------------------- Fig 7 CPA
def fig_cpa(nums: dict) -> None:
    c = pd.read_csv(ROOT / "results/revision_r2_step2/cpa_hull/event_hull_cpa.csv")
    c = c.dropna(subset=["d_cpa_centre_m", "clr_length"])
    c["cpa_sd"] = c.clr_length.clip(lower=0)
    fig, axs = plt.subplots(1, 3, figsize=(W, 3.9), gridspec_kw={"width_ratios": [1.1, 1, 1.2]})
    # (a) ECDF of the clearance definitions
    ax = axs[0]
    series = [("Centre CPA $D_{CPA}$", c.d_cpa_centre_m, OI["black"], "-"),
              ("Signed $\\Delta_{SD}$ (length)", c.clr_length, OI["verm"], "--"),
              ("Hull polygon clearance", c.clr_polygon, OI["blue"], "-."),
              ("Truncated CPA-SD = max(0, $\\Delta_{SD}$)", c.cpa_sd, OI["green"], ":")]
    for lab, s, col, ls in series:
        x = np.sort(s.dropna().to_numpy())
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, ls=ls, lw=1.4, label=lab)
    ax.axvline(0, color=OI["grey"], lw=0.6)
    ax.axvline(100, color=OI["grey"], lw=0.6, ls=":")
    ax.set_xlim(-600, 2000)
    ax.set_xlabel("Clearance at TCPA (m)")
    ax.set_ylabel("Cumulative share of transfers")
    ax.set_title("(a) Clearance definitions")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=7, frameon=False, ncol=1)
    # (b) signed Delta_SD vs polygon clearance for Delta_SD < 300 m
    ax = axs[1]
    z = c[c.clr_length < 300]
    inter = z.polygon_intersects.astype(bool)
    ax.scatter(z.clr_length[~inter], z.clr_polygon[~inter], s=8, marker="o", facecolors="none",
               edgecolors=OI["blue"], lw=0.6, label="Hulls separate")
    ax.scatter(z.clr_length[inter], z.clr_polygon[inter], s=12, marker="x", color=OI["verm"], lw=0.8,
               label="Hull polygons intersect")
    ax.plot([-600, 300], [-600, 300], color=OI["grey"], lw=0.6, ls="--")
    ax.axvline(0, color=OI["grey"], lw=0.6)
    ax.set_xlim(-600, 300); ax.set_ylim(-50, 1000)
    ax.set_xlabel("Signed $\\Delta_{SD}$ (m)")
    ax.set_ylabel("Hull polygon clearance (m)")
    ax.set_title("(b) $\\Delta_{SD}$ < 0 is not overlap")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=7, frameon=False)
    # (c) share < 100 m by encounter type and hull model
    ax = axs[2]
    types = ["crossing", "head_on", "overtaking", "stationary"]
    models = [("clr_length", "Length", OI["verm"], "///"), ("clr_heading_weighted", "Heading-weighted", OI["orange"], "..."),
              ("clr_polygon", "Polygon", OI["blue"], ""), ("clr_beam", "Beam", OI["sky"], "xx")]
    enc = c.encounter.astype(str).str.lower()
    xb = np.arange(len(types))
    for k, (col, lab, colr, h) in enumerate(models):
        vals = [100 * (c.loc[enc == t, col] < 100).mean() for t in types]
        ax.bar(xb + (k - 1.5) * 0.2, vals, 0.2, color=colr, hatch=h, edgecolor="black", lw=0.4, label=lab)
    ax.set_xticks(xb, [f"{ {'crossing': 'Cross', 'head_on': 'Head-on', 'overtaking': 'Overt.', 'stationary': 'Stat.'}[t]}\n{int((enc == t).sum())}" for t in types], fontsize=7)
    ax.set_ylabel("Share with clearance < 100 m (%)")
    ax.set_title("(c) By encounter type")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "figure_05_cpa_geometry.png")
    plt.close(fig)
    d = c.clr_length
    nums["cpa"] = {
        "n": int(len(c)), "median_centre_cpa_m": round(float(c.d_cpa_centre_m.median()), 0),
        "median_signed_delta_sd_m": round(float(d.median()), 0),
        "delta_sd_lt0": int((d < 0).sum()), "delta_sd_lt100": int((d < 100).sum()),
        "truncated_cpa_sd_zero_pct": round(100 * float((c.cpa_sd == 0).mean()), 1),
        "polygon_intersects_given_delta_lt0": f"{int(c[d < 0].polygon_intersects.astype(bool).sum())}/{int((d < 0).sum())}",
        "share_lt100_by_type_pct": {t: {lab: round(100 * float((c.loc[enc == t, col] < 100).mean()), 1)
                                        for col, lab, _, _ in models} for t in types},
    }


# ---------------------------------------------------------------- Fig 8 + Tables 6/7
def fig_seasonal(nums: dict) -> None:
    d = pd.read_csv(ROOT / "data/processed/r2/modelling_r2.csv", low_memory=False)
    c = pd.read_csv(ROOT / "results/revision_r2_step2/cpa_hull/event_hull_cpa.csv")[["event_id", "clr_length"]]
    d = d.merge(c, on="event_id", how="left")
    t = pd.to_datetime(d.meeting_time.astype(str).str[:19])
    d["quarter"] = t.dt.to_period("Q").astype(str)
    d["dist_km"] = d.dist_to_breakwater_m / 1000
    d["ukc"] = pd.to_numeric(d.ukc_m, errors="coerce")
    d["delta_sd"] = d.clr_length
    d["cpa_sd"] = d.clr_length.clip(lower=0)
    d["wind"] = pd.to_numeric(d.wind_speed_mps, errors="coerce")
    d["hs"] = pd.to_numeric(d.wave_height_m, errors="coerce")
    d["dur"] = pd.to_numeric(d.duration_min, errors="coerce")
    quarters = sorted(d.quarter.unique())
    vars_ = [("dist_km", "Meeting distance (km)"), ("ukc", "UKC (m)"), ("delta_sd", "Signed $\\Delta_{SD}$ (m)"),
             ("wind", "Wind speed (m/s)"), ("hs", "Significant wave height (m)"), ("dur", "Session duration (min)")]
    fig, axs = plt.subplots(2, 3, figsize=(W, 5.6))
    for ax, (col, lab) in zip(axs.flat, vars_):
        for k, direction in enumerate(("inbound", "outbound")):
            g = d[d.traffic_direction == direction]
            data = [g.loc[g.quarter == q, col].dropna().to_numpy() for q in quarters]
            pos = np.arange(len(quarters)) + (k - 0.5) * 0.36
            bp = ax.boxplot(data, positions=pos, widths=0.32, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=1.0))
            for b in bp["boxes"]:
                b.set_facecolor(DIR_STYLE[direction]["color"]); b.set_hatch(DIR_STYLE[direction]["hatch"])
                b.set_alpha(0.85); b.set_edgecolor("black"); b.set_linewidth(0.5)
        ax.set_xticks(range(len(quarters)), [q.replace("Q", "\nQ") for q in quarters])
        ax.set_title(lab)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=DIR_STYLE[k]["color"], hatch=DIR_STYLE[k]["hatch"], edgecolor="black",
                             label=k.capitalize()) for k in ("inbound", "outbound")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "figure_06_seasonal_summary.png")
    plt.close(fig)

    # Table 6: quarterly means (all transfers and by direction)
    agg = {"n": ("event_id", "size"), "dist_km": ("dist_km", "mean"), "ukc_m": ("ukc", "mean"),
           "delta_sd_m": ("delta_sd", "mean"), "delta_sd_median_m": ("delta_sd", "median"),
           "cpa_sd_trunc_m": ("cpa_sd", "mean"), "wind_mps": ("wind", "mean"), "hs_m": ("hs", "mean"),
           "duration_min": ("dur", "mean")}
    t6 = d.groupby("quarter").agg(**agg).round(2)
    t6.to_csv(OUT / "table6_quarterly_means.csv")
    t6d = d.groupby(["traffic_direction", "quarter"]).agg(**agg).round(2)
    t6d.to_csv(OUT / "table6_quarterly_means_by_direction.csv")
    # Table 7: Kruskal-Wallis across quarters, epsilon^2 = H / (n - 1)
    rows = []
    for direction in ("all", "inbound", "outbound"):
        g = d if direction == "all" else d[d.traffic_direction == direction]
        for col, lab in vars_ + [("cpa_sd", "Truncated CPA-SD (m)")]:
            groups = [g.loc[g.quarter == q, col].dropna() for q in quarters]
            groups = [x for x in groups if len(x) > 0]
            n = sum(len(x) for x in groups)
            H, p = kruskal(*groups)
            rows.append({"direction": direction, "variable": lab.replace("$\\Delta_{SD}$", "Delta_SD"), "n": n,
                         "H": round(float(H), 2), "p": float(f"{p:.3g}"), "epsilon2": round(float(H / (n - 1)), 4)})
    t7 = pd.DataFrame(rows)
    t7.to_csv(OUT / "table7_kruskal_wallis.csv", index=False)
    q1 = d[d.quarter == "2024Q1"]
    rest = d[d.quarter != "2024Q1"]
    nums["seasonal"] = {
        "quarters": quarters, "n_by_quarter": d.quarter.value_counts().sort_index().to_dict(),
        "table6_all": t6.reset_index().to_dict("records"),
        "table7": rows,
        "claims_check": {
            "Q1 peak wind/wave": {"wind_mean_by_q": t6.wind_mps.to_dict(), "hs_mean_by_q": t6.hs_m.to_dict()},
            "Q1 longer durations": {"dur_mean_q1": round(float(q1.dur.mean()), 2), "dur_mean_rest": round(float(rest.dur.mean()), 2)},
            "UKC narrow band": {"ukc_mean_by_q": t6.ukc_m.to_dict()},
            "Delta_SD variability Q2-Q3": {"delta_sd_iqr_by_q": d.groupby("quarter").delta_sd.apply(
                lambda s: round(float(s.quantile(.75) - s.quantile(.25)), 0)).to_dict()},
        },
    }


# ---------------------------------------------------------------- Fig 9 relocation case
def fig_case(nums: dict) -> None:
    from matplotlib.patches import FancyArrow
    # Case chosen by fixed criteria from the causal (dispatch) run: inbound, feasible recommendation,
    # ship under way (>= 5 kn) at dispatch, at least 10 min before the transfer, the ML anchor fails the collision gate, relocation 0.5-2.5 km
    # ahead along the ship's course; largest ship among those that qualify.
    import glob as _g
    feats = pd.read_csv(ROOT / "data/processed/r2/dispatch_features_r2.csv", parse_dates=["t_dispatch"])
    mod = pd.read_csv(ROOT / "data/processed/r2/modelling_r2.csv", usecols=["event_id", "traffic_direction", "vessel_ship_name", "meeting_time"])
    st = pd.read_csv(ROOT.parent / "Static_merged_bbox_result_gokhan.csv", encoding="cp949")
    st.columns = ["mmsi", "name", "type", "imo", "call", "A", "B", "C", "D", "draft", "gt"]
    st = st.drop_duplicates("mmsi").set_index("mmsi")
    best = None
    for f in sorted(_g.glob(str(ROOT / "data/processed/meeting_point/optimized_meeting_points_*_r2verified_causal.csv"))):
        oo = pd.read_csv(f).merge(mod, on="event_id", suffixes=("_ais", "")).merge(feats[["event_id", "ref_sog_kn", "ref_cog_sin", "ref_cog_cos"]], on="event_id")
        cc = pd.read_csv(f.replace("optimized_meeting_points_", "candidates_"), low_memory=False,
                         usecols=["event_id", "candidate_source", "gate_reason", "collision_risk_class"])
        an_bad = set(cc[(cc.candidate_source == "ml_anchor") & cc.gate_reason.str.contains("collision_risk_high", na=False)].event_id)
        oo = oo[(oo.traffic_direction == "inbound") & (oo.recommendation_status == "feasible") & (oo.ref_sog_kn >= 5)
                & oo.event_id.isin(an_bad) & oo.recommended_dist_to_anchor_m.between(500, 2500) & (oo.lead_min >= 10)]
        for r in oo.itertuples():
            brg = np.degrees(np.arctan2((r.recommended_lon - r.vessel_rewind_lon) * np.cos(np.radians(r.vessel_rewind_lat)),
                                        r.recommended_lat - r.vessel_rewind_lat)) % 360
            cog = np.degrees(np.arctan2(r.ref_cog_sin, r.ref_cog_cos)) % 360
            if abs((brg - cog + 180) % 360 - 180) > 60:
                continue
            loa = float(st.loc[int(r.vessel_mmsi), ["A", "B"]].sum()) if int(r.vessel_mmsi) in st.index else 0.0
            if best is None or loa > best[0]:
                best = (loa, r, f)
    loa, r, ofile = best
    eid = r.event_id
    sinfo = st.loc[int(r.vessel_mmsi)] if int(r.vessel_mmsi) in st.index else None
    case = {"event_id": eid, "vessel": {"name": str(r.vessel_ship_name), "loa_m": loa,
                                        "gt": float(sinfo["gt"]) if sinfo is not None and pd.notna(sinfo["gt"]) else None},
            "decision_time_kst": str(pd.Timestamp(r.rewind_time).tz_localize(None)),
            "meeting_time_kst": str(pd.Timestamp(str(r.meeting_time)[:19])),
            "lead_min": float(r.lead_min)}
    o = pd.read_csv(ofile)
    o = o[o.event_id == eid].iloc[0]
    cand = pd.read_csv(ofile.replace("optimized_meeting_points_", "candidates_"), low_memory=False)
    cand = cand[cand.event_id == eid]
    t_dec = pd.Timestamp(case["decision_time_kst"])
    T = pd.Timestamp(case["meeting_time_kst"])
    week_csv = ROOT / "data/processed/ais_master/weeks" / (Path(ofile).name.split("optimized_meeting_points_")[1].split("_r2verified")[0].replace("_week", "-week").replace("_", "-", 1) + ".csv")
    a = pd.concat(ch[(pd.to_datetime(ch.datetime) >= t_dec - pd.Timedelta(minutes=25)) &
                     (pd.to_datetime(ch.datetime) <= T + pd.Timedelta(minutes=5))]
                  for ch in pd.read_csv(week_csv,
                                        usecols=["mmsi", "datetime", "lat", "lon", "spd", "crs"], chunksize=2_000_000))
    a["mmsi"] = a.mmsi.astype("int64").astype(str); a["datetime"] = pd.to_datetime(a.datetime)
    ship, pilot = str(int(o.vessel_mmsi)), str(int(o.pilot_mmsi))
    snap = a[(a.datetime - t_dec).abs() <= pd.Timedelta(minutes=1)].sort_values("datetime").drop_duplicates("mmsi", keep="last")
    third = snap[~snap.mmsi.isin([ship, pilot])]
    lat0 = (o.anchor_lat + o.recommended_lat) / 2
    k = np.cos(np.radians(lat0)) * 111.32

    def dist_km(la, lo, la2, lo2):
        return np.hypot((lo - lo2) * k, (la - la2) * 111.32)

    def stats(la, lo):
        dd = dist_km(third.lat.to_numpy(), third.lon.to_numpy(), la, lo)
        return {"vessels_0_5nm": int((dd <= 0.926).sum()), "vessels_1nm": int((dd <= 1.852).sum()),
                "vessels_2nm": int((dd <= 3.704).sum()), "nearest_vessel_m": round(float(dd.min() * 1000), 0)}

    an = cand[cand.candidate_source == "ml_anchor"].iloc[0]
    rec = cand[(np.isclose(cand.candidate_lat, o.recommended_lat)) & (np.isclose(cand.candidate_lon, o.recommended_lon))].iloc[0]
    before, after = stats(o.anchor_lat, o.anchor_lon), stats(o.recommended_lat, o.recommended_lon)
    reloc = float(dist_km(o.anchor_lat, o.anchor_lon, o.recommended_lat, o.recommended_lon) * 1000)
    actual_off = float(dist_km(o.actual_meeting_lat, o.actual_meeting_lon, o.recommended_lat, o.recommended_lon) * 1000)
    nums["case"] = {
        "event_id": eid, "ship": case["vessel"], "meeting_kst": case["meeting_time_kst"], "decision_kst": case["decision_time_kst"],
        "lead_min": case["lead_min"],
        "anchor": {**before, "collision_class": an.collision_risk_class, "gate": an.gate_reason, "ukc_m": round(float(an.ukc_m), 1),
                   "dist_to_breakwater_m": round(float(an.dist_to_breakwater_m), 0)},
        "recommended": {**after, "collision_class": rec.collision_risk_class, "gate": rec.gate_reason, "ukc_m": round(float(rec.ukc_m), 1),
                        "source": rec.candidate_source, "dist_to_breakwater_m": round(float(rec.dist_to_breakwater_m), 0)},
        "hs_m": float(an.wave_height_m) if pd.notna(an.wave_height_m) else None,
        "relocation_m": round(reloc, 0), "actual_meeting_offset_from_recommended_m": round(actual_off, 0),
        "candidates": int(len(cand)), "candidates_pass": int(cand.gate_pass.astype(str).str.lower().eq("true").sum()),
    }

    fig = plt.figure(figsize=(W, 5.6))
    ax = fig.add_axes([0.1, 0.1, 0.45, 0.86])
    near = cand[dist_km(cand.candidate_lat, cand.candidate_lon, o.anchor_lat, o.anchor_lon) <= 3.0]
    trk = a[a.mmsi.isin([ship, pilot])]
    pts = pd.concat([near[["candidate_lat", "candidate_lon"]].rename(columns={"candidate_lat": "lat", "candidate_lon": "lon"}),
                     trk[["lat", "lon"]],
                     pd.DataFrame({"lat": [o.actual_meeting_lat, o.recommended_lat], "lon": [o.actual_meeting_lon, o.recommended_lon]})])
    pad = 0.006
    xlim = (pts.lon.min() - pad, pts.lon.max() + pad)
    ylim = (pts.lat.min() - pad, pts.lat.max() + pad)
    vis = third[(third.lon.between(*xlim)) & (third.lat.between(*ylim))]
    from enc_basemap import draw_enc
    draw_enc(ax, (xlim[0], xlim[1], ylim[0], ylim[1]), aids=True, pilot=True)
    # tracks
    for mm, col, ls, lab in ((ship, OI["black"], "-", "Ship track, t_dec−25 min → meeting"),
                             (pilot, OI["green"], "--", "Pilot-boat track, same window")):
        tr = a[a.mmsi == mm].sort_values("datetime")
        ax.plot(tr.lon, tr.lat, color=col, ls=ls, lw=1.3, label=lab, zorder=3)
        p0 = tr[tr.datetime <= t_dec].tail(1)
        if len(p0):
            ax.scatter(p0.lon, p0.lat, s=26, marker="o", color=col, edgecolor="white", lw=0.6, zorder=6)
    nums["case"]["ship_track_after_t_dec_m"] = round(float(dist_km(*a[(a.mmsi == ship) & (a.datetime >= t_dec)].sort_values("datetime")[["lat", "lon"]].iloc[0],
                                                               *a[(a.mmsi == ship) & (a.datetime >= t_dec)].sort_values("datetime")[["lat", "lon"]].iloc[-1]) * 1000), 0)
    ok = cand.gate_pass.astype(str).str.lower().eq("true")
    ax.scatter(cand.candidate_lon[ok], cand.candidate_lat[ok], s=10, marker="o", facecolors="none", edgecolors=OI["sky"], lw=0.6,
               label="Candidate, passes gates", zorder=2)
    ax.scatter(cand.candidate_lon[~ok], cand.candidate_lat[~ok], s=10, marker="x", color=OI["grey"], lw=0.6,
               label="Candidate, fails a gate", zorder=2)
    ax.scatter(vis.lon, vis.lat, s=28, marker="^", color=OI["purple"], edgecolor="black", lw=0.4, label="Third-party vessels at t_dec", zorder=4)
    ax.scatter([o.anchor_lon], [o.anchor_lat], s=90, marker="D", color=OI["verm"], edgecolor="black", lw=0.6, label="ML anchor at dispatch (fails collision gate)", zorder=5)
    ax.scatter([o.recommended_lon], [o.recommended_lat], s=140, marker="*", color=OI["yellow"], edgecolor="black", lw=0.7, label="Recommended point", zorder=6)
    ax.scatter([o.actual_meeting_lon], [o.actual_meeting_lat], s=50, marker="P", color=OI["blue"], edgecolor="black", lw=0.5, label="Observed meeting point", zorder=5)
    ax.annotate("", xy=(o.recommended_lon, o.recommended_lat), xytext=(o.anchor_lon, o.anchor_lat),
                arrowprops=dict(arrowstyle="->", color=OI["verm"], lw=1.2), zorder=5)
    for r_km, ls in ((0.926, ":"), (1.852, "--")):
        th = np.linspace(0, 2 * np.pi, 120)
        ax.plot(o.anchor_lon + r_km * np.sin(th) / k, o.anchor_lat + r_km * np.cos(th) / 111.32, color=OI["verm"], lw=0.5, ls=ls)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect(1 / np.cos(np.radians(lat0)))
    ax.grid(lw=0.3, alpha=0.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.3f}°E"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.3f}°N"))
    ax.tick_params(axis="x", labelrotation=30)
    # scale bar (1 km) and north arrow
    x0, y0 = xlim[0] + 0.1 * (xlim[1] - xlim[0]), ylim[0] + 0.06 * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + 1 / k], [y0, y0], color="black", lw=2.5)
    ax.text(x0 + 0.5 / k, y0 + 0.012 * (ylim[1] - ylim[0]), "1 km", ha="center", va="bottom", fontsize=8.5)
    xn, yn = xlim[1] - 0.07 * (xlim[1] - xlim[0]), ylim[1] - 0.16 * (ylim[1] - ylim[0])
    ax.annotate("N", xy=(xn, yn + 0.1 * (ylim[1] - ylim[0])), xytext=(xn, yn), ha="center", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), fontsize=7.5, frameon=False)
    # before/after table
    tax = fig.add_axes([0.6, 0.05, 0.38, 0.36]); tax.axis("off")
    rows = [["", "Anchor", "Recomm."],
            ["Collision class", str(an.collision_risk_class), str(rec.collision_risk_class)],
            ["Nearest vessel (m)", f"{before['nearest_vessel_m']:.0f}", f"{after['nearest_vessel_m']:.0f}"],
            ["Vessels ≤0.5/1/2 NM", f"{before['vessels_0_5nm']}/{before['vessels_1nm']}/{before['vessels_2nm']}",
             f"{after['vessels_0_5nm']}/{after['vessels_1nm']}/{after['vessels_2nm']}"],
            ["UKC (m)", f"{an.ukc_m:.1f}", f"{rec.ukc_m:.1f}"],
            ["Hs (m)", f"{an.wave_height_m:.1f}", f"{rec.wave_height_m:.1f}"],
            ["To breakwater (km)", f"{an.dist_to_breakwater_m / 1000:.2f}", f"{rec.dist_to_breakwater_m / 1000:.2f}"]]
    tb = tax.table(cellText=rows, loc="center", cellLoc="center", colWidths=[0.5, 0.22, 0.3])
    tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.25)
    for (r, cc), cell in tb.get_celld().items():
        cell.set_linewidth(0.3)
        if r == 0: cell.set_text_props(fontweight="bold")
    tax.set_title(f"{case['vessel']['name']}, {case['vessel']['loa_m']:.0f} m, inbound\n"
                  f"t_dec {t_dec:%Y-%m-%d %H:%M} KST\nlead {case['lead_min']:.1f} min, relocation {reloc / 1000:.2f} km",
                  fontsize=8, pad=2)
    fig.savefig(FIG / "figure_07_relocation_mechanism.png")
    plt.close(fig)


def main() -> None:
    nums = _numbers()
    fig_shap(nums)
    fig_cpa(nums)
    fig_seasonal(nums)
    fig_case(nums)
    (OUT / "figures_b_numbers.json").write_text(json.dumps(nums, indent=2, default=str))
    print(json.dumps({k: nums[k] for k in ("cpa", "case")}, indent=1, default=str))
    print(json.dumps(nums["shap"]["inbound"]["top5_mean_abs_shap_m"]), json.dumps(nums["shap"]["outbound"]["top5_mean_abs_shap_m"]))


if __name__ == "__main__":
    main()
