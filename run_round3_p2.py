import json, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

SEED = 2027; N_REP = 10; K_TOP = 80; N_DRAW = 2000; REF_YEAR = 2020; ALPHA = 0.10
CONV = 83.0 / 252.164  # USD/MMBtu -> INR/Mcal at INR 83/USD
DATA = Path("data"); OUT = Path("results"); FIGS = Path("manuscript/figures")
FILES = ["CEEW_subcritical_with_ws_price.csv", "CEEW_supercritical_with_ws_price.csv", "gem_with_ws_price.csv",
         "gem_predicted_subcritical.csv", "gem_predicted_supercritical.csv", "30_to_5zones.csv"]

a = pd.read_csv(DATA / FILES[0], encoding="latin1"); b = pd.read_csv(DATA / FILES[1], encoding="latin1")
a["boiler"] = "subcritical"; b["boiler"] = "supercritical"
ceew = pd.concat([a, b], ignore_index=True)

# CR L8: compute CEEW's own INR/Mcal column mean programmatically (both subcritical-only and combined)
ceew_inr_col = [c for c in ceew.columns if "inr" in c.lower() and "mcal" in c.lower()]
ceew_sub_inr_mcal_mean = float(a[ceew_inr_col[0]].dropna().mean()) if ceew_inr_col else float((a["coal_price"] * CONV).mean())
ceew_inr_mcal_mean = float(ceew[ceew_inr_col[0]].dropna().mean()) if ceew_inr_col else float((ceew["coal_price"] * CONV).mean())

D = pd.DataFrame({"plant": ceew["Plant name"].str.strip().str.upper(), "unit": ceew["Unit ID"],
                  "state": ceew["State"].str.strip().str.upper(),
                  "ownership": ceew["Ownership"], "status": ceew["Plant status"],
                  "capacity": ceew["Capacity"], "age": ceew["Age"], "plf": ceew["Actual avg PLF"],
                  "water_stress": ceew["bws_score"], "coal_price": ceew["coal_price"], "region": ceew["Region"],
                  "boiler": ceew["boiler"], "shr": ceew["Actual SHR"], "target_shr": ceew["Target SHR"]})
NUM = ["capacity", "age", "plf", "water_stress", "coal_price"]; CAT = ["region", "boiler"]

# Residual SD of Target SHR + PLF + PLF^2 & PLF quantiles (CR L7)
Z = np.c_[D.target_shr, D.plf, D.plf**2]
lr_des = LinearRegression().fit(Z, D.shr)
res_sd_design = float((D.shr - lr_des.predict(Z)).std(ddof=1))
plf_min, plf_max = float(D.plf.min()), float(D.plf.max())
plf_p05, plf_p95 = float(D.plf.quantile(0.05)), float(D.plf.quantile(0.95))
plf_med = float(D.plf.median())
plf_capw = float(np.average(D.plf, weights=D.capacity))

# Fit final linear Ridge model on all D (identical to main pipeline) to get s_hat at actual PLF
prep_lin = ColumnTransformer([("n", StandardScaler(), NUM), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
pipe_lin = Pipeline([("p", prep_lin), ("m", Ridge())])
gs_final_lin = GridSearchCV(pipe_lin, {"m__alpha": [0.1, 1, 10, 100]},
                            cv=GroupKFold(5, shuffle=True, random_state=SEED),
                            scoring="neg_mean_absolute_error").fit(D[NUM + CAT], D.shr.values, groups=D.plant.values)
FINAL_LIN = gs_final_lin.best_estimator_
D["pred_actual_plf"] = FINAL_LIN.predict(D[NUM + CAT])
D["obs_resid"] = D["shr"] - D["pred_actual_plf"]

# Load GEM fleet
gem = pd.read_csv(DATA / "gem_with_ws_price.csv", encoding="latin1")
zones = pd.read_csv(DATA / "30_to_5zones.csv"); zmap = dict(zip(zones.iloc[:, 0].str.strip(), zones.iloc[:, 1].str.strip()))
F0 = gem[(gem.Status == "operating") & gem["Combustion technology"].isin(["subcritical", "supercritical"])].copy()
F0["state"] = F0["Subnational unit (province, state)"].str.replace(" state", "", regex=False).str.strip()
F_raw = pd.DataFrame({"tracker": F0["Tracker ID"], "plant": F0["Plant"].str.upper(), "state": F0.state,
                      "capacity": F0["Capacity (MW)"], "year": F0["Year"], "age": REF_YEAR - F0["Year"],
                      "water_stress": F0.bws_score, "coal_price": F0.coal_price,
                      "region": F0.state.map(zmap), "boiler": F0["Combustion technology"]})
n_total_gem_sub_sup = len(F_raw)
n_miss_price = int(F_raw.coal_price.isna().sum())
n_miss_age = int(F_raw.age.isna().sum())
F = F_raw.dropna().reset_index(drop=True)
n_dropped = n_total_gem_sub_sup - len(F)
n_post_2020 = int((F.year > 2020).sum())
n_eq_2020 = int((F.year == 2020).sum())
F.loc[F.age < 0, "age"] = 0

# === Nonlinear PLF specification (1/PLF clipped at 0.15 + boiler interactions) under 10x5 plant-grouped CV ===
def add_nl(df):
    out = df.copy()
    p_clip = np.clip(out["plf"].values, 0.15, 1.0)
    out["inv_plf"] = 1.0 / p_clip
    is_sup = (out["boiler"] == "supercritical").astype(float)
    out["sup_plf"] = is_sup * out["plf"].values
    out["sup_inv_plf"] = is_sup * out["inv_plf"].values
    return out

D_nl = add_nl(D)
NUM_NL = NUM + ["inv_plf", "sup_plf", "sup_inv_plf"]
prep_nl = ColumnTransformer([("n", StandardScaler(), NUM_NL), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
pipe_nl = Pipeline([("p", prep_nl), ("m", Ridge())])

nl_maes = []
for rep in range(N_REP):
    seed = SEED + 101 * rep
    errs = []
    for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(D_nl, D_nl.shr.values, D_nl.plant.values):
        gs = GridSearchCV(pipe_nl, {"m__alpha": [0.1, 1, 10, 100]},
                          cv=GroupKFold(5, shuffle=True, random_state=seed),
                          scoring="neg_mean_absolute_error").fit(D_nl.iloc[tr][NUM_NL + CAT], D_nl.shr.values[tr], groups=D_nl.plant.values[tr])
        pred = gs.best_estimator_.predict(D_nl.iloc[te][NUM_NL + CAT])
        errs.append(np.abs(D_nl.shr.values[te] - pred))
    nl_maes.append(np.concatenate(errs).mean())
nl_mae_mean, nl_mae_std = float(np.mean(nl_maes)), float(np.std(nl_maes, ddof=1))

gs_final_nl = GridSearchCV(pipe_nl, {"m__alpha": [0.1, 1, 10, 100]},
                           cv=GroupKFold(5, shuffle=True, random_state=SEED),
                           scoring="neg_mean_absolute_error").fit(D_nl[NUM_NL + CAT], D_nl.shr.values, groups=D_nl.plant.values)
FINAL_NL = gs_final_nl.best_estimator_

def fleet_capw_shr(model, cols, plf_val, boiler=None):
    sub = F if boiler is None else F[F.boiler == boiler]
    Xf = add_nl(sub.assign(plf=plf_val))[cols]
    s = model.predict(Xf)
    return float(np.average(s, weights=sub.capacity.values))

nl_shr_50 = fleet_capw_shr(FINAL_NL, NUM_NL + CAT, 0.50)
nl_shr_med = fleet_capw_shr(FINAL_NL, NUM_NL + CAT, plf_med)
nl_shr_capw = fleet_capw_shr(FINAL_NL, NUM_NL + CAT, plf_capw)
nl_sub_30 = fleet_capw_shr(FINAL_NL, NUM_NL + CAT, 0.30, "subcritical")
nl_sub_90 = fleet_capw_shr(FINAL_NL, NUM_NL + CAT, 0.90, "subcritical")
nl_sup_30 = fleet_capw_shr(FINAL_NL, NUM_NL + CAT, 0.30, "supercritical")
nl_sup_90 = fleet_capw_shr(FINAL_NL, NUM_NL + CAT, 0.90, "supercritical")

# === Condition L2: Fallback threshold using UNIQUE UNITS (g.idx.nunique() >= 30) ===
P_all = pd.concat([pd.read_pickle(p) for p in sorted((OUT / "runs").glob("rep*_grouped_f*.pkl"))], ignore_index=True)
P_ridge = P_all[P_all.model == "Ridge"].copy()
P_ridge["boiler"] = D.boiler.values[P_ridge.idx]
P_ridge["region"] = D.region.values[P_ridge.idx]
P_ridge["resid"] = P_ridge.y - P_ridge.pred

pool_b = {b: P_ridge[P_ridge.boiler == b].resid.values for b in ("subcritical", "supercritical")}
pool_br = {}
cell_unit_counts = {}
for (b, reg), g in P_ridge.groupby(["boiler", "region"]):
    n_u = int(g["idx"].nunique())
    cell_unit_counts[f"{b}_{reg}"] = n_u
    # Pool to boiler-level residuals whenever fewer than 30 unique labelled units exist in the cell
    pool_br[(b, reg)] = g.resid.values if n_u >= 30 else pool_b[b]

# === Condition L3 (Refined): Strip generic plant words including THERMAL, POWER, SUPER before overlap ===
STOP_TOKENS = {
    "SUPER", "THERMAL", "POWER", "STATION", "PLANT", "STPS", "STPP", "TPS", "TPP",
    "MEGA", "COAL", "PROJECT", "LIMITED", "LTD", "PVT", "PRIVATE", "EXTN",
    "EXTENSION", "PHASE", "STAGE", "UNIT", "ENERGY", "GENERATION", "GEN",
    "CORPORATION", "CORP", "NTPC", "CAPTIVE", "CPP", "IPP", "UMPP"
}
def norm_name(s):
    s = re.sub(r"[^A-Z0-9]+", " ", str(s).upper()).strip()
    toks = [t for t in s.split() if t not in STOP_TOKENS and not t.isdigit()]
    return " ".join(toks)

def norm_state(s):
    s = str(s).upper().replace(" STATE", "").strip()
    aliases = {"ORISSA": "ODISHA", "CHATTISGARH": "CHHATTISGARH", "UTTARANCHAL": "UTTARAKHAND",
               "TAMILNADU": "TAMIL NADU", "NCT OF DELHI": "DELHI"}
    return aliases.get(s, s)

fl_prev = pd.read_csv(OUT / "fleet_predictions.csv")
fl_prev["plant_norm"] = fl_prev["plant"].map(norm_name)
fl_prev["state_norm"] = fl_prev["state"].map(norm_state)
D_m = D.copy()
D_m["plant_norm"] = D_m["plant"].map(norm_name)
D_m["state_norm"] = D_m["state"].map(norm_state)

def token_overlap(a_str, b_str):
    # Overlap coefficient: |A & B| / min(|A|, |B|) over normalised non-generic plant tokens
    ta, tb = set(a_str.split()), set(b_str.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))

matched_obs_resid = np.full(len(fl_prev), np.nan)
matched_obs_raw = np.full(len(fl_prev), np.nan)
matched_pairs = []
used_ceew = set()
for i, row in fl_prev.iterrows():
    cand = D_m[(D_m["state_norm"] == row["state_norm"]) &
               (D_m["boiler"] == row["boiler"]) &
               (np.abs(D_m["capacity"] - row["capacity"]) <= 10.0)]
    best_j, best_score = None, (0.0, 999.0)
    for j, crow in cand.iterrows():
        if j in used_ceew:
            continue
        ov = token_overlap(row["plant_norm"], crow["plant_norm"])
        cap_diff = abs(row["capacity"] - crow["capacity"])
        if ov >= 0.5 and (ov > best_score[0] or (ov == best_score[0] and cap_diff < best_score[1])):
            best_score = (ov, cap_diff); best_j = j
    if best_j is not None:
        crow = D_m.loc[best_j]
        matched_obs_resid[i] = crow["obs_resid"]
        matched_obs_raw[i] = crow["shr"]
        used_ceew.add(best_j)
        matched_pairs.append(dict(
            gem_tracker=row["tracker"], gem_plant=row["plant"], gem_state=row["state"],
            gem_boiler=row["boiler"], gem_capacity_mw=row["capacity"],
            ceew_plant=crow["plant"], ceew_unit=crow["unit"], ceew_capacity_mw=crow["capacity"],
            ceew_actual_plf=crow["plf"], ceew_obs_shr=crow["shr"],
            ceew_pred_actual_plf=crow["pred_actual_plf"], obs_resid=crow["obs_resid"],
            overlap_score=best_score[0], cap_diff_mw=best_score[1]
        ))

pd.DataFrame(matched_pairs).to_csv(OUT / "matched_ceew_gem_pairs.csv", index=False)

F_merged = fl_prev.copy()
F_merged["obs_resid"] = matched_obs_resid
F_merged["obs_shr_raw"] = matched_obs_raw
is_labelled = F_merged["obs_resid"].notna().values
n_labelled_matched = int(is_labelled.sum())
match_rate_ceew_pct = float(100.0 * n_labelled_matched / len(D))
match_rate_fleet_pct = float(100.0 * n_labelled_matched / len(F_merged))

# Pre-CMT (b) Preferred Fix: place matched units on consistent PLF = 0.5 basis: shr50 + (obs - s_hat at actual PLF)
base_shr_plf50 = np.where(is_labelled, F_merged["shr50"].values + F_merged["obs_resid"].values, F_merged["shr50"].values)

def excess_cost_inr_cr(shr, df, ref_override=None):
    if ref_override is None:
        ref = pd.Series(shr).groupby(df.boiler.values).transform(lambda v: np.quantile(v, 0.10)).values
    else:
        ref = ref_override
    gen_kwh = df.capacity.values * 1000.0 * 0.5 * 8760.0
    return np.clip(shr - ref, 0, None) * gen_kwh / 1000.0 * (df.coal_price.values * CONV) / 1e7

# Decomposition of cost steps: (1) conversion-only predicted SHR, (2) + observed-residual spread holding benchmark fixed, (3) + benchmark shift
ref_pred = pd.Series(F_merged["shr50"].values).groupby(F_merged.boiler.values).transform(lambda v: np.quantile(v, 0.10)).values
excess_conv_only_cr = float(excess_cost_inr_cr(F_merged["shr50"].values, F_merged).sum())
excess_spread_fixed_ref_cr = float(excess_cost_inr_cr(base_shr_plf50, F_merged, ref_override=ref_pred).sum())
F_merged["excess_cr_inr"] = excess_cost_inr_cr(base_shr_plf50, F_merged)
excess_total_point_cr = float(F_merged["excess_cr_inr"].sum())
step_spread_cr = excess_spread_fixed_ref_cr - excess_conv_only_cr
step_bench_cr = excess_total_point_cr - excess_spread_fixed_ref_cr
step_total_pct = float(100.0 * (excess_total_point_cr - excess_conv_only_cr) / excess_conv_only_cr)

point_top = set(np.argsort(-F_merged["excess_cr_inr"].values)[:K_TOP])

rng = np.random.default_rng(SEED)
plants = F_merged.plant.values
inc = np.zeros(len(F_merged)); jac = []; tot = []
for _ in range(N_DRAW):
    eps = np.zeros(len(F_merged))
    # Matched labelled units carry eps = 0 (PLF-adjusted observed SHR); unlabelled units draw 1 plant-shared residual
    unlab = ~is_labelled
    for (b, reg), grp_idx in F_merged[unlab].groupby(["boiler", "region"]).groups.items():
        grp_idx = np.asarray(grp_idx)
        pl = np.unique(plants[grp_idx])
        pool = pool_br.get((b, reg), pool_b[b])
        draw = dict(zip(pl, rng.choice(pool, len(pl))))
        eps[grp_idx] = [draw[p] for p in plants[grp_idx]]
    ec = excess_cost_inr_cr(base_shr_plf50 + eps, F_merged)
    top = set(np.argsort(-ec)[:K_TOP])
    inc[list(top)] += 1
    jac.append(len(top & point_top) / len(top | point_top))
    tot.append(ec.sum())

F_merged["p_top_r2"] = inc / N_DRAW
F_merged["is_matched"] = is_labelled
top80_matched = int(sum(is_labelled[i] for i in point_top))
robust90_mask = F_merged.p_top_r2.values >= 0.9
robust90_matched = int((robust90_mask & is_labelled).sum())
src_top = set(np.argsort(-excess_cost_inr_cr(F_merged.src50.fillna(F_merged.shr50).values, F_merged))[:K_TOP])
top80 = F_merged.nlargest(K_TOP, "excess_cr_inr")

# Condition L4: Separate PLF-50 offset vs 6-point average offset
S = pd.read_csv(OUT / "table_plf_sensitivity.csv")
offsets_all, offsets_50 = [], []
for bt in ("subcritical", "supercritical"):
    u = S[S.boiler == bt].set_index("plf")["shr_capw"]
    for p in (50, 70, 90):
        sub_f = F_merged[(F_merged.boiler == bt) & F_merged[f"src{p}"].notna()]
        src_val = float(np.average(sub_f[f"src{p}"], weights=sub_f.capacity))
        ridge_val = float(u.loc[p / 100.0])
        diff = src_val - ridge_val
        offsets_all.append(diff)
        if p == 50:
            offsets_50.append(diff)
src_offset_kcal = float(np.mean(offsets_all))
src_offset_50_kcal = float(np.mean(offsets_50))

fc = (F_merged["shr50"] / 1000.0) * (F_merged["coal_price"] * CONV)
sp_sub = float(stats.spearmanr(F_merged[F_merged.boiler=="subcritical"].shr50, F_merged[F_merged.boiler=="subcritical"].src50)[0])
sp_sup = float(stats.spearmanr(F_merged[F_merged.boiler=="supercritical"].shr50, F_merged[F_merged.boiler=="supercritical"].src50)[0])

r2_stats = {
    "res_sd_design": res_sd_design,
    "plf_obs_min": plf_min, "plf_obs_max": plf_max,
    "plf_p05": plf_p05, "plf_p95": plf_p95,
    "ceew_sub_inr_mcal_mean": ceew_sub_inr_mcal_mean,
    "ceew_inr_mcal_mean": ceew_inr_mcal_mean,
    "src_offset_kcal": src_offset_kcal,
    "src_offset_50_kcal": src_offset_50_kcal,
    "nl_mae_mean": nl_mae_mean, "nl_mae_std": nl_mae_std,
    "nl_shr_50": nl_shr_50, "nl_shr_med": nl_shr_med, "nl_shr_capw": nl_shr_capw,
    "nl_sub_30": nl_sub_30, "nl_sub_90": nl_sub_90, "nl_sup_30": nl_sup_30, "nl_sup_90": nl_sup_90,
    "nl_bias_med": nl_shr_50 - nl_shr_med, "nl_bias_med_pct": 100*(nl_shr_50 - nl_shr_med)/nl_shr_med,
    "nl_bias_capw": nl_shr_50 - nl_shr_capw, "nl_bias_capw_pct": 100*(nl_shr_50 - nl_shr_capw)/nl_shr_capw,
    "n_dropped": n_dropped, "n_miss_price": n_miss_price, "n_miss_age": n_miss_age,
    "n_post_2020": n_post_2020, "n_eq_2020": n_eq_2020,
    "n_labelled_matched": n_labelled_matched,
    "match_rate_ceew_pct": match_rate_ceew_pct,
    "match_rate_fleet_pct": match_rate_fleet_pct,
    "top80_matched": top80_matched,
    "robust90_matched": robust90_matched,
    "cell_unit_counts": cell_unit_counts,
    "k": K_TOP, "fleet_units": len(F_merged),
    "jaccard_mean": float(np.mean(jac)), "jaccard_p05": float(np.quantile(jac, 0.05)),
    "robust_ge90": int(robust90_mask.sum()),
    "robust_ge50": int((F_merged.p_top_r2 >= 0.5).sum()),
    "ever_in_top": int((F_merged.p_top_r2 > 0).sum()),
    "point_top_with_p_lt50": int(sum(F_merged.p_top_r2.values[i] < 0.5 for i in point_top)),
    "overlap_with_source_top": len(point_top & src_top),
    "excess_conv_only_cr": excess_conv_only_cr,
    "excess_total_point_cr": excess_total_point_cr,
    "step_spread_cr": step_spread_cr,
    "step_bench_cr": step_bench_cr,
    "step_total_pct": step_total_pct,
    "excess_total_p05_cr": float(np.quantile(tot, 0.05)),
    "excess_total_p95_cr": float(np.quantile(tot, 0.95)),
    "fuel_cost_mean": float(fc.mean()), "fuel_cost_med": float(fc.median()),
    "fuel_cost_p10": float(np.percentile(fc, 10)), "fuel_cost_p90": float(np.percentile(fc, 90)),
    "coal_price_inr_mean": float((F_merged.coal_price * CONV).mean()),
    "spearman_sub": sp_sub, "spearman_sup": sp_sup,
    "n_sub_fleet": int((F_merged.boiler == "subcritical").sum()),
    "n_sup_fleet": int((F_merged.boiler == "supercritical").sum()),
    "top80_sub": int((top80.boiler == "subcritical").sum()),
    "top80_sup": int((top80.boiler == "supercritical").sum()),
    "top80_SR": int((top80.region == "SR").sum()),
    "top80_WR": int((top80.region == "WR").sum()),
    "top80_NR": int((top80.region == "NR").sum()),
    "top80_ER": int((top80.region == "ER").sum()),
    "top80_age_med": float(top80.age.median()),
    "top80_cap_med": float(top80.capacity.median()),
    "top80_price_med": float((top80.coal_price * CONV).median()),
}
json.dump(r2_stats, open(OUT / "round2_p2_stats.json", "w"), indent=2)
F_merged.to_csv(OUT / "fleet_predictions_r2.csv", index=False)

# Regenerate Fig 3 (fig_fleet.pdf) with updated p_top_r2
plt.rcParams.update({"savefig.dpi": 300, "font.size": 8})
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.6))
s = S[S.boiler != "all"]
plf_grid = np.linspace(0.30, 0.90, 13)
for bt, col, lab in (("subcritical", "#3182bd", "subcritical"), ("supercritical", "#e6550d", "supercritical")):
    u = s[s.boiler == bt]
    axs[0].plot(u.plf, u.shr_capw, color=col, lw=1.3, label=f"{lab} (linear)")
    nl_curve = [fleet_capw_shr(FINAL_NL, NUM_NL + CAT, pv, bt) for pv in plf_grid]
    axs[0].plot(plf_grid, nl_curve, color=col, lw=1.1, ls="--", label=f"{lab} (1/PLF)")
    for p in (50, 70, 90):
        sub_f = F_merged[(F_merged.boiler == bt) & F_merged[f"src{p}"].notna()]
        axs[0].plot(p / 100, np.average(sub_f[f"src{p}"], weights=sub_f.capacity), "x", color=col, ms=4)
axs[0].axvline(plf_med, color="grey", lw=.7, ls=":"); axs[0].axvline(0.5, color="k", lw=.7, ls="--")
axs[0].set_xlabel("assumed PLF"); axs[0].set_ylabel("capacity-weighted SHR (kcal/kWh)")
axs[0].legend(frameon=False, fontsize=5.3, loc="upper right")
v = np.sort(F_merged.p_top_r2.values)[::-1][:200]
axs[1].plot(np.arange(1, len(v) + 1), v, color="#3182bd"); axs[1].axvline(K_TOP, color="k", lw=.7, ls="--")
axs[1].set_xlabel("unit rank by P(top-80)"); axs[1].set_ylabel("P(in top-80)")
plt.tight_layout()
for d in (OUT, FIGS): plt.savefig(d / "fig_fleet.pdf")
plt.savefig(OUT / "fig_fleet.png"); plt.close()

print("P2 Round 4 refined stats:", json.dumps({k: r2_stats[k] for k in [
    "n_labelled_matched", "match_rate_ceew_pct", "match_rate_fleet_pct",
    "top80_matched", "robust90_matched", "robust_ge90", "robust_ge50",
    "point_top_with_p_lt50", "ever_in_top", "jaccard_mean", "jaccard_p05",
    "overlap_with_source_top", "excess_conv_only_cr", "excess_total_point_cr",
    "step_spread_cr", "step_bench_cr", "step_total_pct",
    "excess_total_p05_cr", "excess_total_p95_cr", "src_offset_50_kcal", "src_offset_kcal",
    "ceew_sub_inr_mcal_mean", "ceew_inr_mcal_mean"
]}, indent=2))
