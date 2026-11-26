import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer
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
D = pd.DataFrame({"plant": ceew["Plant name"].str.strip().str.upper(), "unit": ceew["Unit ID"], "state": ceew["State"],
                  "ownership": ceew["Ownership"], "status": ceew["Plant status"],
                  "capacity": ceew["Capacity"], "age": ceew["Age"], "plf": ceew["Actual avg PLF"],
                  "water_stress": ceew["bws_score"], "coal_price": ceew["coal_price"], "region": ceew["Region"],
                  "boiler": ceew["boiler"], "shr": ceew["Actual SHR"], "target_shr": ceew["Target SHR"]})
NUM = ["capacity", "age", "plf", "water_stress", "coal_price"]; CAT = ["region", "boiler"]

# Residual SD of Target SHR + PLF + PLF^2
Z = np.c_[D.target_shr, D.plf, D.plf**2]
lr_des = LinearRegression().fit(Z, D.shr)
res_sd_design = float((D.shr - lr_des.predict(Z)).std(ddof=1))
plf_min, plf_max = float(D.plf.min()), float(D.plf.max())
plf_med = float(D.plf.median())
plf_capw = float(np.average(D.plf, weights=D.capacity))

# Load GEM fleet and count exact post-2020 units vs year==2020
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

# === R2-L1: Nonlinear PLF specification (1/PLF + boiler x PLF + boiler x 1/PLF) under 10x5 plant-grouped CV ===
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

# Fit final nonlinear model on all D_nl
gs_final_nl = GridSearchCV(pipe_nl, {"m__alpha": [0.1, 1, 10, 100]},
                           cv=GroupKFold(5, shuffle=True, random_state=SEED),
                           scoring="neg_mean_absolute_error").fit(D_nl[NUM_NL + CAT], D_nl.shr.values, groups=D_nl.plant.values)
FINAL_NL = gs_final_nl.best_estimator_

def fleet_capw_shr(model, cols, plf_val):
    Xf = add_nl(F.assign(plf=plf_val))[cols]
    s = model.predict(Xf)
    return float(np.average(s, weights=F.capacity.values))

nl_shr_50 = fleet_capw_shr(FINAL_NL, NUM_NL + CAT, 0.50)
nl_shr_med = fleet_capw_shr(FINAL_NL, NUM_NL + CAT, plf_med)
nl_shr_capw = fleet_capw_shr(FINAL_NL, NUM_NL + CAT, plf_capw)
nl_shr_obs_lo = fleet_capw_shr(FINAL_NL, NUM_NL + CAT, max(0.30, round(plf_min, 2)))
nl_shr_obs_hi = fleet_capw_shr(FINAL_NL, NUM_NL + CAT, min(0.90, round(plf_max, 2)))

# === R2-L2: Ranking uncertainty using ALL 10 repeats, stratified by boiler x region (pool to boiler if <30),
#            and using OBSERVED SHR for labelled units ===
P_all = pd.concat([pd.read_pickle(p) for p in sorted((OUT / "runs").glob("rep*_grouped_f*.pkl"))], ignore_index=True)
P_ridge = P_all[P_all.model == "Ridge"].copy()
P_ridge["boiler"] = D.boiler.values[P_ridge.idx]
P_ridge["region"] = D.region.values[P_ridge.idx]
P_ridge["resid"] = P_ridge.y - P_ridge.pred

# Build pools: boiler x region if n >= 30, else boiler
pool_b = {b: P_ridge[P_ridge.boiler == b].resid.values for b in ("subcritical", "supercritical")}
pool_br = {}
for (b, reg), g in P_ridge.groupby(["boiler", "region"]):
    pool_br[(b, reg)] = g.resid.values if len(g) >= 30 else pool_b[b]

# Match labelled CEEW units to GEM fleet by normalized plant name + boiler + closest capacity (within 10 MW)
fl_prev = pd.read_csv(OUT / "fleet_predictions.csv")
# Merge observed SHR for labelled units that match GEM plant + boiler + capacity
ceew_lookup = D.groupby(["plant", "boiler", "capacity"])["shr"].mean().reset_index()
F_merged = fl_prev.merge(ceew_lookup, on=["plant", "boiler", "capacity"], how="left")
# For any plant-level match with slight capacity rounding, keep exact match
is_labelled = F_merged["shr"].notna().values
n_labelled_matched = int(is_labelled.sum())
base_shr = np.where(is_labelled, F_merged["shr"].values, F_merged["shr50"].values)

def excess_cost_inr_cr(shr, df):
    ref = pd.Series(shr).groupby(df.boiler.values).transform(lambda v: np.quantile(v, 0.10)).values
    gen_kwh = df.capacity.values * 1000.0 * 0.5 * 8760.0
    return np.clip(shr - ref, 0, None) * gen_kwh / 1000.0 * (df.coal_price.values * CONV) / 1e7

F_merged["excess_cr_inr"] = excess_cost_inr_cr(base_shr, F_merged)
point_top = set(np.argsort(-F_merged["excess_cr_inr"].values)[:K_TOP])

rng = np.random.default_rng(SEED)
plants = F_merged.plant.values
inc = np.zeros(len(F_merged)); jac = []; tot = []
for _ in range(N_DRAW):
    eps = np.zeros(len(F_merged))
    # Only perturb UNLABELLED units; labelled units keep observed SHR (eps = 0)
    unlab = ~is_labelled
    for (b, reg), grp_idx in F_merged[unlab].groupby(["boiler", "region"]).groups.items():
        grp_idx = np.asarray(grp_idx)
        pl = np.unique(plants[grp_idx])
        pool = pool_br.get((b, reg), pool_b[b])
        draw = dict(zip(pl, rng.choice(pool, len(pl))))
        eps[grp_idx] = [draw[p] for p in plants[grp_idx]]
    ec = excess_cost_inr_cr(base_shr + eps, F_merged)
    top = set(np.argsort(-ec)[:K_TOP])
    inc[list(top)] += 1
    jac.append(len(top & point_top) / len(top | point_top))
    tot.append(ec.sum())

F_merged["p_top_r2"] = inc / N_DRAW
src_top = set(np.argsort(-excess_cost_inr_cr(F_merged.src50.fillna(F_merged.shr50).values, F_merged))[:K_TOP])
top80 = F_merged.nlargest(K_TOP, "excess_cr_inr")

# Boiler Spearman + fuel cost/kWh
fc = (F_merged["shr50"] / 1000.0) * (F_merged["coal_price"] * CONV)
sp_sub = float(stats.spearmanr(F_merged[F_merged.boiler=="subcritical"].shr50, F_merged[F_merged.boiler=="subcritical"].src50)[0])
sp_sup = float(stats.spearmanr(F_merged[F_merged.boiler=="supercritical"].shr50, F_merged[F_merged.boiler=="supercritical"].src50)[0])

r2_stats = {
    "res_sd_design": res_sd_design,
    "plf_obs_min": plf_min, "plf_obs_max": plf_max,
    "nl_mae_mean": nl_mae_mean, "nl_mae_std": nl_mae_std,
    "nl_shr_50": nl_shr_50, "nl_shr_med": nl_shr_med, "nl_shr_capw": nl_shr_capw,
    "nl_bias_med": nl_shr_50 - nl_shr_med, "nl_bias_med_pct": 100*(nl_shr_50 - nl_shr_med)/nl_shr_med,
    "nl_bias_capw": nl_shr_50 - nl_shr_capw, "nl_bias_capw_pct": 100*(nl_shr_50 - nl_shr_capw)/nl_shr_capw,
    "n_dropped": n_dropped, "n_miss_price": n_miss_price, "n_miss_age": n_miss_age,
    "n_post_2020": n_post_2020, "n_eq_2020": n_eq_2020,
    "n_labelled_matched": n_labelled_matched,
    "k": K_TOP, "fleet_units": len(F_merged),
    "jaccard_mean": float(np.mean(jac)), "jaccard_p05": float(np.quantile(jac, 0.05)),
    "robust_ge90": int((F_merged.p_top_r2 >= 0.9).sum()),
    "robust_ge50": int((F_merged.p_top_r2 >= 0.5).sum()),
    "ever_in_top": int((F_merged.p_top_r2 > 0).sum()),
    "point_top_with_p_lt50": int(sum(F_merged.p_top_r2.values[i] < 0.5 for i in point_top)),
    "overlap_with_source_top": len(point_top & src_top),
    "excess_total_point_cr": float(F_merged["excess_cr_inr"].sum()),
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

# === R2-L11: Regenerate Figs 1, 2, 3 with "Two-model variant" and y-axis units on Fig 3 ===
A = pd.read_csv(OUT / "table_accuracy.csv"); COV = pd.read_csv(OUT / "table_coverage.csv")
S = pd.read_csv(OUT / "table_plf_sensitivity.csv")
order = A[A.cv == "grouped"].sort_values("MAE_mean").model.tolist()
lbl_map = {"TwoModel": "Two-model\nvariant"}
plt.rcParams.update({"savefig.dpi": 300, "font.size": 8})

# Fig 1
fig, ax = plt.subplots(figsize=(4.8, 2.7)); x = np.arange(len(order)); w = 0.38
for j, (cv, col) in enumerate((("random", "#9ecae1"), ("grouped", "#3182bd"))):
    t = A[A.cv == cv].set_index("model").loc[order]
    ax.bar(x + (j - .5) * w, t.MAE_mean, w, yerr=t.MAE_std, color=col,
           label={"random": "random unit folds", "grouped": "plant-grouped folds"}[cv], capsize=2)
ax.set_xticks(x); ax.set_xticklabels([lbl_map.get(m, m) for m in order], fontsize=7.5)
ax.set_ylabel("MAE (kcal/kWh)"); ax.legend(frameon=False, fontsize=7)
plt.tight_layout()
for d in (OUT, FIGS): plt.savefig(d / "fig_cv_gap.pdf")
plt.savefig(OUT / "fig_cv_gap.png"); plt.close()

# Fig 2
fig, (a0, a1) = plt.subplots(1, 2, figsize=(5.0, 2.7), gridspec_kw={"width_ratios": [1, 1.7]}, sharey=True)
STY = (("random", "pooled", "o", "#e6550d", "random-fold calibration"),
       ("grouped", "pooled", "s", "#3182bd", "plant-grouped calibration"),
       ("grouped", "mondrian", "^", "#31a354", "plant-grouped, per boiler"))
for j, (cal, mode, mk, col, lab) in enumerate(STY):
    t = COV[(COV.by == "all") & (COV.calibration == cal) & (COV["mode"] == mode)].set_index("model").loc[order]
    a0.plot(np.arange(len(order)) + (j - 1) * .15, t.coverage, mk, color=col, ms=4, label=lab)
a0.set_xticks(range(len(order))); a0.set_xticklabels([lbl_map.get(m, m).replace("\n", " ") for m in order], rotation=55, ha="right", fontsize=6.5)
a0.set_ylabel("coverage on new plants"); a0.set_title("all units, by model", fontsize=7.5)
t = COV[(COV.model == "Ridge") & COV.by.isin(["boiler", "region", "ownership"]) & (COV.level != "NER")]
labs = t[["by", "level"]].drop_duplicates().values
NM = {"subcritical": "Sub", "supercritical": "Super", "CENTRAL SECTOR": "Central", "PVT SECTOR": "Private", "STATE SECTOR": "State"}
for j, (cal, mode, mk, col, lab) in enumerate(STY):
    v = [t[(t.calibration == cal) & (t["mode"] == mode) & (t.by == b_) & (t.level == l)].coverage.mean() for b_, l in labs]
    a1.plot(np.arange(len(labs)) + (j - 1) * .15, v, mk, color=col, ms=4)
a1.set_xticks(range(len(labs))); a1.set_xticklabels([NM.get(l, l) for _, l in labs], rotation=55, ha="right", fontsize=6.5)
a1.set_title("Ridge, by subgroup", fontsize=7.5)
for ax in (a0, a1): ax.axhline(1 - ALPHA, color="k", lw=.7, ls="--")
a0.legend(frameon=False, fontsize=5.8, loc="lower left"); plt.tight_layout()
for d in (OUT, FIGS): plt.savefig(d / "fig_coverage.pdf")
plt.savefig(OUT / "fig_coverage.png"); plt.close()

# Fig 3
fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.5))
s = S[S.boiler != "all"]
for bt, col in (("subcritical", "#3182bd"), ("supercritical", "#e6550d")):
    u = s[s.boiler == bt]; axs[0].plot(u.plf, u.shr_capw, color=col, label=bt)
    for p in (50, 70, 90):
        sub_f = F_merged[(F_merged.boiler == bt) & F_merged[f"src{p}"].notna()]
        axs[0].plot(p / 100, np.average(sub_f[f"src{p}"], weights=sub_f.capacity), "x", color=col)
axs[0].axvline(plf_med, color="grey", lw=.7, ls=":"); axs[0].axvline(0.5, color="k", lw=.7, ls="--")
axs[0].set_xlabel("assumed PLF"); axs[0].set_ylabel("capacity-weighted SHR (kcal/kWh)")
axs[0].legend(frameon=False, fontsize=6)
v = np.sort(F_merged.p_top_r2.values)[::-1][:200]
axs[1].plot(np.arange(1, len(v) + 1), v, color="#3182bd"); axs[1].axvline(K_TOP, color="k", lw=.7, ls="--")
axs[1].set_xlabel("unit rank by P(top-80)"); axs[1].set_ylabel("P(in top-80)")
plt.tight_layout()
for d in (OUT, FIGS): plt.savefig(d / "fig_fleet.pdf")
plt.savefig(OUT / "fig_fleet.png"); plt.close()

print("P2 Round 2 stats:", json.dumps(r2_stats, indent=2))
