import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r'''
# Paper 2 - Uncertainty-aware heat-rate estimation for India's coal fleet
ICECRF-2027, Track 1, topic 5 (*AI for industry transformation*). Re-analysis of the open dataset of Ding et al. (Zenodo 13921645; Data in Brief 2025).

**Question.** The public dataset extends unit-level station heat rate (SHR) from 540 CEEW units to the full GEM fleet with ML. How accurate is that extension for *new plants*, how uncertain is it, how much does the 50 % load-factor assumption matter, and is a retrofit priority list built on it stable?
''')
md(r'''
## 0. Fixed design decisions (pre-specified)
| Item | Decision |
|---|---|
| Labelled data | CEEW subcritical (462) + supercritical (78) units; target = `Actual SHR` (kcal/kWh). |
| Features | Only those also available for GEM units: capacity (MW), age (y), plant load factor (PLF), basin water stress, delivered coal price (INR/Mcal), power region, boiler design. `Target SHR`, coal use, efficiency and cost columns are excluded (unavailable in GEM and/or algebraic functions of the target). |
| Models | Ridge, k-nearest neighbours, random forest, gradient boosting, and a two-model replication of the source pipeline (GBM for subcritical, KNN for supercritical). |
| Tuning | Small grids (Ridge 4, KNN 8, RF 4, GBM 4 settings), inner 5-fold CV of the same type as the outer CV (nested). |
| Outer validation | 10 repeats x 5 folds, (a) random unit-level folds, (b) folds grouped by plant. |
| Primary model | Lowest mean MAE under plant-grouped CV. |
| Conformal intervals | Cross-conformal absolute-residual intervals at 90 % nominal; residuals from an inner 5-fold CV on the outer-training part, either random or plant-grouped; pooled or Mondrian (per boiler design). Coverage evaluated on plant-grouped outer test folds. |
| Fleet | GEM units with status *operating* and technology sub/supercritical; age = 2020 - commissioning year; region from state. |
| Ranking | Excess annual fuel cost vs the 10th-percentile predicted SHR of the same boiler design, at PLF = 0.5; top-k with k = 80 (~10 % of fleet); uncertainty by plant-level block resampling of grouped-CV residuals, 2,000 draws. |
''')
code(r'''
import os, sys, json, time, hashlib, platform, warnings, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sklearn, scipy, matplotlib
print({k: v.__version__ for k, v in dict(numpy=np, pandas=pd, sklearn=sklearn, scipy=scipy, matplotlib=matplotlib).items()}, "| Python", sys.version.split()[0])
SEED = 2027; N_REP = int(os.environ.get("N_REP", 10)); N_JOBS = int(os.environ.get("N_JOBS", 4))
MAX_SEC = float(os.environ.get("MAX_SECONDS", 1e9)); ALPHA = 0.10; K_TOP = 80; N_DRAW = 2000; REF_YEAR = 2020
DATA = Path(os.environ.get("DATA_DIR", "data")); OUT = Path("results"); (OUT / "runs").mkdir(parents=True, exist_ok=True)
ZEN = "https://zenodo.org/records/13921645/files/"
FILES = ["CEEW_subcritical_with_ws_price.csv", "CEEW_supercritical_with_ws_price.csv", "gem_with_ws_price.csv",
         "gem_predicted_subcritical.csv", "gem_predicted_supercritical.csv", "30_to_5zones.csv"]
DATA.mkdir(exist_ok=True)
for f in FILES:
    if not (DATA / f).exists():
        urllib.request.urlretrieve(ZEN + f + "?download=1", DATA / f)
json.dump({f: hashlib.sha256((DATA / f).read_bytes()).hexdigest() for f in FILES}, open(OUT / "data_sha256.json", "w"), indent=2)
''')
code(r'''
a = pd.read_csv(DATA / FILES[0], encoding="latin1"); b = pd.read_csv(DATA / FILES[1], encoding="latin1")
a["boiler"] = "subcritical"; b["boiler"] = "supercritical"
ceew = pd.concat([a, b], ignore_index=True)
D = pd.DataFrame({"plant": ceew["Plant name"].str.strip().str.upper(), "unit": ceew["Unit ID"], "state": ceew["State"],
                  "ownership": ceew["Ownership"], "status": ceew["Plant status"],
                  "capacity": ceew["Capacity"], "age": ceew["Age"], "plf": ceew["Actual avg PLF"],
                  "water_stress": ceew["bws_score"], "coal_price": ceew["coal_price"], "region": ceew["Region"],
                  "boiler": ceew["boiler"], "shr": ceew["Actual SHR"], "target_shr": ceew["Target SHR"]})
NUM = ["capacity", "age", "plf", "water_stress", "coal_price"]; CAT = ["region", "boiler"]
print(D.shape, "plants:", D.plant.nunique(), "| units per plant max:", D.plant.value_counts().max())
# the target is almost a deterministic function of design SHR and PLF (documented, not used as feature)
from sklearn.linear_model import LinearRegression
Z = np.c_[D.target_shr, D.plf, D.plf**2]; r2_design = LinearRegression().fit(Z, D.shr).score(Z, D.shr)
summ = D.groupby("boiler").agg(units=("shr", "size"), plants=("plant", "nunique"), shr_mean=("shr", "mean"), shr_sd=("shr", "std"),
                               cap_med=("capacity", "median"), age_med=("age", "median"), plf_med=("plf", "median")).reset_index()
summ.to_csv(OUT / "table_data.csv", index=False)
json.dump({"r2_design_plf": r2_design, "n_units": len(D), "n_plants": int(D.plant.nunique()),
           "max_units_per_plant": int(D.plant.value_counts().max())}, open(OUT / "data_facts.json", "w"), indent=2)
print("R2 of Actual SHR on (design SHR, PLF, PLF^2):", round(r2_design, 3)); summ.round(2)
''')
code(r'''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin, clone

def prep(cat=CAT): return ColumnTransformer([("n", StandardScaler(), NUM), ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
SPECS = {
 "Ridge": (Pipeline([("p", prep()), ("m", Ridge())]), {"m__alpha": [0.1, 1, 10, 100]}),
 "KNN":   (Pipeline([("p", prep()), ("m", KNeighborsRegressor())]), {"m__n_neighbors": [3, 5, 10, 20], "m__weights": ["uniform", "distance"]}),
 "RF":    (Pipeline([("p", prep()), ("m", RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=1))]),
           {"m__min_samples_leaf": [1, 5], "m__max_features": [0.5, 1.0]}),
 "GBM":   (Pipeline([("p", prep()), ("m", GradientBoostingRegressor(random_state=SEED))]),
           {"m__n_estimators": [200], "m__max_depth": [2, 3], "m__learning_rate": [0.05, 0.1]}),
}
class TwoModel(BaseEstimator, RegressorMixin):
    """Replication of the source pipeline: separate models per boiler design (GBM sub, KNN super), each tuned internally."""
    def __init__(self, inner="random", seed=SEED): self.inner, self.seed = inner, seed
    def fit(self, X, y, groups=None):
        self.m_ = {}
        for bt, (name) in (("subcritical", "GBM"), ("supercritical", "KNN")):
            k = (X.boiler == bt).values; est, grid = SPECS[name]
            est = clone(est); est.set_params(p=prep(["region"]))
            g = groups[k] if groups is not None else None
            cv = GroupKFold(5, shuffle=True, random_state=self.seed) if self.inner == "grouped" else KFold(5, shuffle=True, random_state=self.seed)
            gs = GridSearchCV(est, grid, cv=cv, scoring="neg_mean_absolute_error").fit(X[k], y[k], groups=g)
            self.m_[bt] = gs.best_estimator_
        return self
    def predict(self, X):
        out = np.empty(len(X))
        for bt, m in self.m_.items():
            k = (X.boiler == bt).values
            if k.any(): out[k] = m.predict(X[k])
        return out
MODELS = list(SPECS) + ["TwoModel"]
X_ALL, y_ALL, g_ALL = D[NUM + CAT], D.shr.values, D.plant.values
''')
code(r'''
def splitter(kind, seed): return GroupKFold(5, shuffle=True, random_state=seed) if kind == "grouped" else KFold(5, shuffle=True, random_state=seed)

def fit_tuned(name, Xtr, ytr, gtr, kind, seed):
    if name == "TwoModel": return TwoModel(inner=kind, seed=seed).fit(Xtr, ytr, groups=gtr), {}
    est, grid = SPECS[name]
    gs = GridSearchCV(clone(est), grid, cv=splitter(kind, seed), scoring="neg_mean_absolute_error").fit(Xtr, ytr, groups=gtr)
    return gs.best_estimator_, gs.best_params_

def oof_residuals(model, Xtr, ytr, gtr, kind, seed):
    """inner 5-fold out-of-fold absolute residuals with fixed (already tuned) hyper-parameters"""
    res = np.empty(len(ytr))
    for itr, ite in splitter(kind, seed + 7).split(Xtr, ytr, gtr):
        if isinstance(model, TwoModel):          # keep the tuned sub-models' settings, refit on the inner-training part
            m = TwoModel(); m.m_ = {}
            for bt, sub in model.m_.items():
                k = (Xtr.iloc[itr].boiler == bt).values; m.m_[bt] = clone(sub).fit(Xtr.iloc[itr][k], ytr[itr][k])
        else: m = clone(model).fit(Xtr.iloc[itr], ytr[itr])
        res[ite] = np.abs(ytr[ite] - m.predict(Xtr.iloc[ite]))
    return res

def cq(r, alpha=ALPHA):
    n = len(r); return np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")

def run_one(rep, kind, fold_only):
    f = OUT / "runs" / f"rep{rep}_{kind}_f{fold_only}.pkl"
    if f.exists(): return
    seed = SEED + 101 * rep; rows = []
    for fold, (tr, te) in enumerate(splitter(kind, seed).split(X_ALL, y_ALL, g_ALL)):
        if fold != fold_only: continue
        Xtr, Xte, ytr, yte, gtr = X_ALL.iloc[tr], X_ALL.iloc[te], y_ALL[tr], y_ALL[te], g_ALL[tr]
        for name in MODELS:
            m, bp = fit_tuned(name, Xtr, ytr, gtr, kind, seed)
            pred = m.predict(Xte)
            rec = pd.DataFrame({"rep": rep, "cv": kind, "fold": fold, "model": name, "idx": te, "y": yte, "pred": pred})
            if kind == "grouped":   # conformal calibration, two inner schemes x pooled/Mondrian
                for cal in ("random", "grouped"):
                    r = oof_residuals(m, Xtr, ytr, gtr, cal, seed)
                    rec[f"q_{cal}_pooled"] = cq(r)
                    bt_tr = Xtr.boiler.values; qm = {b: cq(r[bt_tr == b]) for b in np.unique(bt_tr)}
                    rec[f"q_{cal}_mondrian"] = Xte.boiler.map(qm).values
            rows.append(rec)
    pd.concat(rows).to_pickle(f)

from joblib import Parallel, delayed
jobs = [(r, k, f) for r in range(N_REP) for k in ("grouped", "random") for f in range(5) if not (OUT / "runs" / f"rep{r}_{k}_f{f}.pkl").exists()]
print(len(jobs), "jobs to run"); t0 = time.time()
for i in range(0, len(jobs), N_JOBS):
    if time.time() - t0 > MAX_SEC: print("time budget reached - re-run to resume"); break
    Parallel(n_jobs=N_JOBS)(delayed(run_one)(r, k, f) for r, k, f in jobs[i:i + N_JOBS]); print(f"{min(i+N_JOBS, len(jobs))}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
''')
md("## Accuracy under random vs plant-grouped validation")
code(r'''
from scipy import stats
P = pd.concat([pd.read_pickle(p) for p in sorted((OUT / "runs").glob("rep*_f*.pkl"))], ignore_index=True)
P["boiler"] = D.boiler.values[P.idx]; P["err"] = P.pred - P.y
def mets(g):
    e = g.err.values; y = g.y.values
    return pd.Series({"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e**2).mean()), "MAPE": 100 * np.abs(e / y).mean(),
                      "R2": 1 - (e**2).sum() / ((y - y.mean())**2).sum()})
M = P.groupby(["cv", "model", "rep"]).apply(mets).reset_index()
A = M.groupby(["cv", "model"]).agg(["mean", "std"]).drop(columns="rep"); A.columns = ["_".join(c) for c in A.columns]
A = A.reset_index(); A.to_csv(OUT / "table_accuracy.csv", index=False)
Mb = P.groupby(["cv", "model", "boiler", "rep"]).apply(mets).reset_index().groupby(["cv", "model", "boiler"])[["MAE", "MAPE", "R2"]].mean().reset_index()
Mb.to_csv(OUT / "table_accuracy_by_boiler.csv", index=False)
PRIMARY = A[A.cv == "grouped"].sort_values("MAE_mean").model.iloc[0]
# optimism: grouped minus random MAE, paired by repeat
W = M.pivot_table(index=["model", "rep"], columns="cv", values="MAE").reset_index(); W["gap"] = W.grouped - W.random
GAP = W.groupby("model").agg(gap_mean=("gap", "mean"), gap_sd=("gap", "std"), rand=("random", "mean"), grp=("grouped", "mean")).reset_index()
GAP["gap_pct"] = 100 * GAP.gap_mean / GAP.rand; GAP.to_csv(OUT / "table_optimism.csv", index=False)
# model comparison under grouped CV: Nadeau-Bengio corrected test of each model vs primary on repeat-level MAE
G = M[M.cv == "grouped"].pivot_table(index="rep", columns="model", values="MAE"); cmp = []
for mname in G.columns:
    if mname == PRIMARY: continue
    d = (G[mname] - G[PRIMARY]).values; n = len(d); se = np.sqrt((1/n + 0.25) * d.var(ddof=1)); t = d.mean() / se
    cmp.append(dict(model=mname, dMAE=d.mean(), p_nb=2 * stats.t.sf(abs(t), n - 1)))
CMP = pd.DataFrame(cmp); CMP.to_csv(OUT / "table_model_vs_primary.csv", index=False)
print("PRIMARY:", PRIMARY); display(A.round(3)); display(GAP.round(2)); display(CMP.round(4))
''')
md("## Conformal intervals: does the calibration scheme matter for new plants?")
code(r'''
Q = P[(P.cv == "grouped")].copy(); Q = Q.merge(D[["region", "ownership"]], left_on="idx", right_index=True)
cov = []
for c in ["q_random_pooled", "q_random_mondrian", "q_grouped_pooled", "q_grouped_mondrian"]:
    cal, mode = c.split("_")[1:]
    Q["hit"] = (np.abs(Q.err) <= Q[c]); Q["width"] = 2 * Q[c]
    for by in [None, "boiler", "region", "ownership"]:
        grp = ["model"] + ([by] if by else [])
        t = Q.groupby(grp).agg(coverage=("hit", "mean"), width=("width", "mean"), n=("hit", "size")).reset_index()
        t["calibration"], t["mode"], t["by"] = cal, mode, by or "all"
        if by: t = t.rename(columns={by: "level"})
        else: t["level"] = "all"
        cov.append(t)
COV = pd.concat(cov, ignore_index=True); COV.n = (COV.n / N_REP).round().astype(int); COV.to_csv(OUT / "table_coverage.csv", index=False)
COV[(COV.model == PRIMARY)].pivot_table(index=["by", "level"], columns=["calibration", "mode"], values="coverage").round(3)
''')
md("## Fleet extension: load-factor sensitivity and retrofit ranking")
code(r'''
gem = pd.read_csv(DATA / "gem_with_ws_price.csv", encoding="latin1")
zones = pd.read_csv(DATA / "30_to_5zones.csv"); zmap = dict(zip(zones.iloc[:, 0].str.strip(), zones.iloc[:, 1].str.strip()))
F = gem[(gem.Status == "operating") & gem["Combustion technology"].isin(["subcritical", "supercritical"])].copy()
F["state"] = F["Subnational unit (province, state)"].str.replace(" state", "", regex=False).str.strip()
F = pd.DataFrame({"tracker": F["Tracker ID"], "plant": F["Plant"].str.upper(), "state": F.state, "capacity": F["Capacity (MW)"],
                  "age": REF_YEAR - F["Year"], "water_stress": F.bws_score, "coal_price": F.coal_price,
                  "region": F.state.map(zmap), "boiler": F["Combustion technology"]})
print("fleet units:", len(F), "| missing:", F.isna().sum()[lambda s: s > 0].to_dict())
F = F.dropna().reset_index(drop=True); F.loc[F.age < 0, "age"] = 0
FINAL, FINAL_BP = fit_tuned(PRIMARY, X_ALL, y_ALL, g_ALL, "grouped", SEED)
plf_grid = np.round(np.arange(0.30, 0.91, 0.05), 2); S = []
for p in plf_grid:
    Xf = F.assign(plf=p)[NUM + CAT]; s = FINAL.predict(Xf)
    for bt in ("subcritical", "supercritical", "all"):
        k = (F.boiler == bt).values if bt != "all" else np.ones(len(F), bool)
        S.append(dict(plf=p, boiler=bt, shr_capw=np.average(s[k], weights=F.capacity[k]), shr_mean=s[k].mean()))
S = pd.DataFrame(S); S.to_csv(OUT / "table_plf_sensitivity.csv", index=False)
# compare with the source dataset's own predictions (MMBtu/MWh -> kcal/kWh)
KC = 252.164
src = pd.concat([pd.read_csv(DATA / "gem_predicted_subcritical.csv"), pd.read_csv(DATA / "gem_predicted_supercritical.csv")])
src = src.rename(columns={"Tracker ID": "tracker"})[["tracker", "Predicted_HR_50.0", "Predicted_HR_70.0", "Predicted_HR_90.0"]]
F["shr50"] = FINAL.predict(F.assign(plf=0.5)[NUM + CAT]); F = F.merge(src, on="tracker", how="left")
for p in (50, 70, 90): F[f"src{p}"] = F[f"Predicted_HR_{p}.0"] * KC
agree = {"n_matched": int(F.src50.notna().sum()), "spearman_50": F[["shr50", "src50"]].corr("spearman").iloc[0, 1],
         "mean_diff_50": (F.shr50 - F.src50).mean(), "mae_50": (F.shr50 - F.src50).abs().mean(),
         "ceew_median_plf": D.plf.median(), "ceew_capw_plf": np.average(D.plf, weights=D.capacity)}
json.dump(agree, open(OUT / "fleet_agreement.json", "w"), indent=2, default=float); print(agree)
S[S.boiler == "all"].round(1)
''')
code(r'''
# Retrofit ranking: excess annual fuel cost at PLF 0.5 relative to the 10th-percentile SHR of the same boiler design
PLF0 = 0.5
def excess_cost(shr, F):
    ref = pd.Series(shr).groupby(F.boiler.values).transform(lambda v: np.quantile(v, 0.10)).values
    gen_kwh = F.capacity.values * 1000 * PLF0 * 8760
    return np.clip(shr - ref, 0, None) * gen_kwh / 1000 * F.coal_price.values / 1e7      # INR crore per year
F["excess_cr"] = excess_cost(F.shr50.values, F)
point_top = set(np.argsort(-F.excess_cr.values)[:K_TOP])
# residual pool: plant-grouped out-of-fold residuals of the primary model (repeat 0) by boiler design
R0 = P[(P.cv == "grouped") & (P.model == PRIMARY) & (P.rep == 0)]; pool = {b: (R0.y - R0.pred)[R0.boiler == b].values for b in ("subcritical", "supercritical")}
rng = np.random.default_rng(SEED); plants = F.plant.values; up = np.unique(plants); inc = np.zeros(len(F)); jac = []; tot = []
for _ in range(N_DRAW):
    eps = np.empty(len(F))
    for b in ("subcritical", "supercritical"):
        k = (F.boiler == b).values; pl = np.unique(plants[k]); draw = dict(zip(pl, rng.choice(pool[b], len(pl))))   # one residual per plant (block)
        eps[k] = [draw[p] for p in plants[k]]
    ec = excess_cost(F.shr50.values + eps, F); top = set(np.argsort(-ec)[:K_TOP]); inc[list(top)] += 1
    jac.append(len(top & point_top) / len(top | point_top)); tot.append(ec.sum())
F["p_top"] = inc / N_DRAW
src_top = set(np.argsort(-excess_cost(F.src50.fillna(F.shr50).values, F))[:K_TOP])
RANK = {"k": K_TOP, "fleet_units": len(F), "jaccard_mean": float(np.mean(jac)), "jaccard_p05": float(np.quantile(jac, .05)),
        "robust_ge90": int((F.p_top >= .9).sum()), "robust_ge50": int((F.p_top >= .5).sum()), "ever_in_top": int((F.p_top > 0).sum()),
        "point_top_with_p_lt50": int(sum(F.p_top.values[i] < .5 for i in point_top)),
        "overlap_with_source_top": len(point_top & src_top), "excess_total_point_cr": float(F.excess_cr.sum()),
        "excess_total_p05_cr": float(np.quantile(tot, .05)), "excess_total_p95_cr": float(np.quantile(tot, .95))}
json.dump(RANK, open(OUT / "ranking.json", "w"), indent=2); F.to_csv(OUT / "fleet_predictions.csv", index=False); RANK
''')
code(r'''
import matplotlib
if "google.colab" not in sys.modules: matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"savefig.dpi": 300, "font.size": 8})
order = A[A.cv == "grouped"].sort_values("MAE_mean").model.tolist()
fig, ax = plt.subplots(figsize=(4.6, 2.6)); x = np.arange(len(order)); w = 0.38
for j, (cv, col) in enumerate((("random", "#9ecae1"), ("grouped", "#3182bd"))):
    t = A[A.cv == cv].set_index("model").loc[order]
    ax.bar(x + (j - .5) * w, t.MAE_mean, w, yerr=t.MAE_std, color=col, label={"random": "random unit folds", "grouped": "plant-grouped folds"}[cv], capsize=2)
ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylabel("MAE (kcal/kWh)"); ax.legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "fig_cv_gap.pdf"); plt.savefig(OUT / "fig_cv_gap.png"); plt.show()

fig, (a0, a1) = plt.subplots(1, 2, figsize=(4.9, 2.6), gridspec_kw={"width_ratios": [1, 1.7]}, sharey=True)
STY = (("random", "pooled", "o", "#e6550d", "random-fold calibration"), ("grouped", "pooled", "s", "#3182bd", "plant-grouped calibration"),
       ("grouped", "mondrian", "^", "#31a354", "plant-grouped, per boiler"))
for j, (cal, mode, mk, col, lab) in enumerate(STY):
    t = COV[(COV.by == "all") & (COV.calibration == cal) & (COV["mode"] == mode)].set_index("model").loc[order]
    a0.plot(np.arange(len(order)) + (j - 1) * .15, t.coverage, mk, color=col, ms=4, label=lab)
a0.set_xticks(range(len(order))); a0.set_xticklabels(order, rotation=60, fontsize=7); a0.set_ylabel("coverage on new plants"); a0.set_title("all units, by model", fontsize=7.5)
t = COV[(COV.model == PRIMARY) & COV.by.isin(["boiler", "region", "ownership"]) & (COV.level != "NER")]
labs = t[["by", "level"]].drop_duplicates().values
NM = {"subcritical": "Sub", "supercritical": "Super", "CENTRAL SECTOR": "Central", "PVT SECTOR": "Private", "STATE SECTOR": "State"}
for j, (cal, mode, mk, col, lab) in enumerate(STY):
    v = [t[(t.calibration == cal) & (t["mode"] == mode) & (t.by == b_) & (t.level == l)].coverage.mean() for b_, l in labs]
    a1.plot(np.arange(len(labs)) + (j - 1) * .15, v, mk, color=col, ms=4)
a1.set_xticks(range(len(labs))); a1.set_xticklabels([NM.get(l, l) for _, l in labs], rotation=60, fontsize=7); a1.set_title(f"{PRIMARY}, by subgroup", fontsize=7.5)
for ax in (a0, a1): ax.axhline(1 - ALPHA, color="k", lw=.7, ls="--")
a0.legend(frameon=False, fontsize=6, loc="lower left"); plt.tight_layout()
plt.savefig(OUT / "fig_coverage.pdf"); plt.savefig(OUT / "fig_coverage.png"); plt.show()

fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.4))
s = S[S.boiler != "all"]
for bt, col in (("subcritical", "#3182bd"), ("supercritical", "#e6550d")):
    u = s[s.boiler == bt]; axs[0].plot(u.plf, u.shr_capw, color=col, label=bt)
    for p in (50, 70, 90):
        axs[0].plot(p / 100, np.average(F[F.boiler == bt][f"src{p}"].dropna(), weights=F[(F.boiler == bt) & F[f"src{p}"].notna()].capacity), "x", color=col)
axs[0].axvline(D.plf.median(), color="grey", lw=.7, ls=":"); axs[0].axvline(0.5, color="k", lw=.7, ls="--")
axs[0].set_xlabel("assumed PLF"); axs[0].set_ylabel("capacity-weighted SHR"); axs[0].legend(frameon=False, fontsize=6)
v = np.sort(F.p_top.values)[::-1][:200]; axs[1].plot(np.arange(1, len(v) + 1), v, color="#3182bd"); axs[1].axvline(K_TOP, color="k", lw=.7, ls="--")
axs[1].set_xlabel("unit rank by P(top-80)"); axs[1].set_ylabel("P(in top-80)")
plt.tight_layout(); plt.savefig(OUT / "fig_fleet.pdf"); plt.savefig(OUT / "fig_fleet.png"); plt.show()
json.dump({"python": sys.version.split()[0], "platform": platform.platform(), "seed": SEED, "n_rep": N_REP, "primary": PRIMARY,
           "primary_params": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in FINAL_BP.items()},
           "packages": {k: v.__version__ for k, v in dict(numpy=np, pandas=pd, sklearn=sklearn, scipy=scipy).items()}},
          open(OUT / "environment.json", "w"), indent=2)
''')
nb.cells = C; nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "Paper2_CoalHeatRate.ipynb"); print("ok", len(C))
