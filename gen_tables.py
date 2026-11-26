import pandas as pd, numpy as np, json
R = "results/"; A = pd.read_csv(R+"table_accuracy.csv"); G = pd.read_csv(R+"table_optimism.csv"); B = pd.read_csv(R+"table_accuracy_by_boiler.csv")
CV = pd.read_csv(R+"table_coverage.csv"); CMP = pd.read_csv(R+"table_model_vs_primary.csv"); Dt = pd.read_csv(R+"table_data.csv")
S = pd.read_csv(R+"table_plf_sensitivity.csv"); facts = json.load(open(R+"data_facts.json")); fl = json.load(open(R+"fleet_agreement.json"))
rk_old = json.load(open(R+"ranking.json")); r2 = json.load(open(R+"round2_p2_stats.json")); env = json.load(open(R+"environment.json")); P = env["primary"]
mac = {}
def M(k, v): mac[k] = v
f0 = lambda x: f"{x:,.0f}"; f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"
M("NUnits", str(facts["n_units"])); M("NPlants", str(facts["n_plants"])); M("MaxUnits", str(facts["max_units_per_plant"]))
M("RtwoDesign", f3(facts["r2_design_plf"])); M("ResSdDesign", f0(r2["res_sd_design"])); M("NRep", str(env["n_rep"])); M("Primary", P)
a = A.set_index(["cv", "model"]); g = G.set_index("model"); b = B.set_index(["cv", "model", "boiler"])
for m in ["Ridge", "KNN", "RF", "GBM", "TwoModel"]:
    t = m.replace("TwoModel", "Two")
    M(f"MAErand{t}", f1(a.loc[("random", m), "MAE_mean"])); M(f"MAEgrp{t}", f1(a.loc[("grouped", m), "MAE_mean"]))
    M(f"Gap{t}", f"{g.loc[m, 'gap_pct']:.0f}"); M(f"RtwoGrp{t}", f3(a.loc[("grouped", m), "R2_mean"]))
    M(f"SupRtwoRand{t}", f2(b.loc[("random", m, "supercritical"), "R2"])); M(f"SupRtwoGrp{t}", f2(b.loc[("grouped", m, "supercritical"), "R2"]))
    M(f"MAPEgrp{t}", f2(a.loc[("grouped", m), "MAPE_mean"]))
    for cal in ("random", "grouped"):
        for mode in ("pooled", "mondrian"):
            v = CV[(CV.model == m) & (CV.by == "all") & (CV.calibration == cal) & (CV["mode"] == mode)].iloc[0]
            M(f"Cov{t}{cal.capitalize()}{mode.capitalize()}", f3(v.coverage)); M(f"Wid{t}{cal.capitalize()}{mode.capitalize()}", f0(v.width))
c = CMP.set_index("model")
M("PvalGBM", f"{c.loc['GBM', 'p_nb']:.4f}"); M("DmaeGBM", f"{c.loc['GBM', 'dMAE']:.1f}")
M("PvalTwo", f"{c.loc['TwoModel', 'p_nb']:.4f}"); M("DmaeTwo", f"{c.loc['TwoModel', 'dMAE']:.1f}")
M("DmaeRF", f"{c.loc['RF', 'dMAE']:.1f}"); M("PvalRF", f"{c.loc['RF', 'p_nb']:.1e}")
M("DmaeKNN", f"{c.loc['KNN', 'dMAE']:.1f}"); M("PvalKNN", f"{c.loc['KNN', 'p_nb']:.1e}")
sub = lambda by, lv, cal, mode: f3(CV[(CV.model == P) & (CV.by == by) & (CV.level == lv) & (CV.calibration == cal) & (CV["mode"] == mode)].coverage.iloc[0])
M("CovSupPooled", sub("boiler", "supercritical", "grouped", "pooled")); M("CovSupMond", sub("boiler", "supercritical", "grouped", "mondrian"))
M("CovSRMond", sub("region", "SR", "grouped", "mondrian")); M("CovPvtMond", sub("ownership", "PVT SECTOR", "grouped", "mondrian"))
M("NFleet", str(r2["fleet_units"])); M("SpearSrc", f3(fl["spearman_50"])); M("MeanDiffSrc", f0(abs(fl["mean_diff_50"]))); M("MAESrc", f0(fl["mae_50"]))
M("SpearSub", f3(r2["spearman_sub"])); M("SpearSup", f3(r2["spearman_sup"]))
M("NSubFleet", str(r2["n_sub_fleet"])); M("NSupFleet", str(r2["n_sup_fleet"]))
M("PLFmedian", f2(fl["ceew_median_plf"])); M("PLFcapw", f3(fl["ceew_capw_plf"]))
M("PlfObsMin", f"{r2['plf_obs_min']:.4f}"); M("PlfObsMax", f2(r2["plf_obs_max"]))
M("PlfPFive", f2(r2["plf_p05"])); M("PlfPNinetyFive", f2(r2["plf_p95"]))
M("CeewSubInrMean", f2(r2["ceew_sub_inr_mcal_mean"]))
M("CeewInrMean", f2(r2["ceew_inr_mcal_mean"]))
M("SrcOffsetFifty", f0(r2["src_offset_50_kcal"]))
M("SrcOffsetKcal", f0(r2["src_offset_kcal"]))
s = S[S.boiler == "all"].set_index("plf").shr_capw
M("SHRthirty", f0(s.loc[0.3])); M("SHRninety", f0(s.loc[0.9])); M("SHRfifty", f0(s.loc[0.5]))
M("SHRSpreadKcal", f0(s.loc[0.3] - s.loc[0.9]))
smed = np.interp(fl["ceew_median_plf"], s.index.values, s.values); M("SHRmedPLF", f0(smed)); M("SHRfiftyOver", f0(s.loc[0.5] - smed))
M("SHRfiftyOverPct", f1(100 * (s.loc[0.5] - smed) / smed)); M("SHRrangePct", f1(100 * (s.loc[0.3] - s.loc[0.9]) / s.loc[0.9]))
scw = np.interp(fl["ceew_capw_plf"], s.index.values, s.values); M("SHRcapwPLF", f0(scw)); M("SHRfiftyOverCapw", f0(s.loc[0.5] - scw))
M("SHRfiftyOverCapwPct", f1(100 * (s.loc[0.5] - scw) / scw))
M("SlopeTenth", f0((s.loc[0.3] - s.loc[0.9]) / 6))
# Nonlinear PLF model macros
M("NlMaeMean", f1(r2["nl_mae_mean"])); M("NlMaeStd", f1(r2["nl_mae_std"]))
M("NlShrFifty", f0(r2["nl_shr_50"])); M("NlShrMed", f0(r2["nl_shr_med"])); M("NlShrCapw", f0(r2["nl_shr_capw"]))
M("NlBiasMed", f1(r2["nl_bias_med"])); M("NlBiasMedPct", f1(r2["nl_bias_med_pct"]))
M("NlBiasCapw", f1(r2["nl_bias_capw"])); M("NlBiasCapwPct", f1(r2["nl_bias_capw_pct"]))
M("NlSubThirty", f0(r2["nl_sub_30"])); M("NlSubNinety", f0(r2["nl_sub_90"]))
M("NlSupThirty", f0(r2["nl_sup_30"])); M("NlSupNinety", f0(r2["nl_sup_90"]))
# Approximate CEEW-GEM matching & decomposition macros (Pre-CMT a-c + CR)
M("NMatchedCeew", str(r2["n_labelled_matched"]))
M("MatchCeewPct", f1(r2["match_rate_ceew_pct"]))
M("MatchFleetPct", f1(r2["match_rate_fleet_pct"]))
M("TopMatched", str(r2["top80_matched"]))
M("RobustMatched", str(r2["robust90_matched"]))
M("ExcessConvOnly", f0(r2["excess_conv_only_cr"]))
M("StepSpreadCr", f0(r2["step_spread_cr"]))
M("StepBenchCr", f0(r2["step_bench_cr"]))
M("StepTotalPct", f1(r2["step_total_pct"]))
# Dropped/age/fuel-cost/top-80 macros
M("NDropped", str(r2["n_dropped"])); M("NMissPrice", str(r2["n_miss_price"])); M("NMissAge", str(r2["n_miss_age"]))
M("NPostTwenty", str(r2["n_post_2020"])); M("NEqTwenty", str(r2["n_eq_2020"])); M("NAgeZeroTot", str(r2["n_post_2020"] + r2["n_eq_2020"]))
M("CoalPriceInrMean", f2(r2["coal_price_inr_mean"]))
M("FuelMean", f2(r2["fuel_cost_mean"])); M("FuelMed", f2(r2["fuel_cost_med"])); M("FuelPten", f2(r2["fuel_cost_p10"])); M("FuelPninety", f2(r2["fuel_cost_p90"]))
M("Ktop", str(r2["k"])); M("RobustNinety", str(r2["robust_ge90"])); M("RobustHalf", str(r2["robust_ge50"])); M("PointLtHalf", str(r2["point_top_with_p_lt50"]))
M("JacMean", f2(r2["jaccard_mean"])); M("JacLow", f2(r2["jaccard_p05"])); M("EverTop", str(r2["ever_in_top"])); M("OverlapSrc", str(r2["overlap_with_source_top"]))
M("ExcessPoint", f0(r2["excess_total_point_cr"])); M("ExcessLo", f0(r2["excess_total_p05_cr"])); M("ExcessHi", f0(r2["excess_total_p95_cr"]))
M("ExcessRaw", f0(rk_old["excess_total_point_cr"]))
M("TopSub", str(r2["top80_sub"])); M("TopSup", str(r2["top80_sup"]))
M("TopSR", str(r2["top80_SR"])); M("TopWR", str(r2["top80_WR"])); M("TopNR", str(r2["top80_NR"])); M("TopER", str(r2["top80_ER"]))
M("TopAgeMed", f1(r2["top80_age_med"])); M("TopCapMed", f0(r2["top80_cap_med"])); M("TopPriceMed", f2(r2["top80_price_med"]))
for k in ["sklearn", "numpy", "pandas", "scipy"]: M("Ver" + k.capitalize(), env["packages"][k])
M("VerPython", env["python"])
dt = Dt.set_index("boiler")
for bt, t in (("subcritical", "Sub"), ("supercritical", "Sup")):
    M(f"N{t}", str(int(dt.loc[bt, "units"]))); M(f"P{t}", str(int(dt.loc[bt, "plants"]))); M(f"SHR{t}", f0(dt.loc[bt, "shr_mean"])); M(f"SD{t}", f0(dt.loc[bt, "shr_sd"]))
open("manuscript/numbers.tex", "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))

# Table 1 data
L = [r"\begin{table}[t]", r"\centering", r"\caption{Cleaned Council on Energy, Environment and Water (CEEW) coal units by boiler design (medians for capacity, age, and plant load factor, PLF; station heat rate, SHR, in kcal/kWh).}", r"\label{tab:data}", r"\small",
     r"\begin{tabularx}{\textwidth}{@{}l *{6}{>{\raggedleft\arraybackslash}X}@{}}", r"\toprule",
     r"Boiler & Units & Plants & SHR mean (SD) & Capacity (MW) & Age (y) & PLF \\", r"\midrule"]
for bt in ("subcritical", "supercritical"):
    r = dt.loc[bt]; L.append(f"{bt.capitalize()} & {int(r.units)} & {int(r.plants)} & {r.shr_mean:,.0f} ({r.shr_sd:.0f}) & {r.cap_med:.0f} & {r.age_med:.1f} & {r.plf_med:.2f} \\\\")
L += [r"\bottomrule", r"\end{tabularx}", r"\smallskip\raggedright\scriptsize Source: authors' analysis of CEEW records packaged in Ding et al.~\cite{ding2025dataset,ding2024zenodo}.", r"\end{table}"]; open("manuscript/tab_data.tex", "w").write("\n".join(L))

# Table 2 accuracy
order = A[A.cv == "grouped"].sort_values("MAE_mean").model.tolist(); NM = {"TwoModel": "Two-model variant"}
L = [r"\begin{table}[t]", r"\centering", r"\caption{Accuracy under random unit-level and plant-grouped cross-validation (CV; mean $\pm$ SD over " + mac["NRep"] + r" repeats of 5-fold CV; MAE in kcal/kWh). Optimism = relative increase in MAE when whole plants are held out. Sup.\ $R^2$ = $R^2$ on supercritical units only.}", r"\label{tab:acc}", r"\scriptsize",
     r"\begin{tabularx}{\textwidth}{@{}l *{6}{>{\raggedleft\arraybackslash}X}@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{Random folds} & \multicolumn{4}{c}{Plant-grouped folds} \\ \cmidrule(lr){2-3}\cmidrule(l){4-7}",
     r"Model & MAE & Sup.\ $R^2$ & MAE & $R^2$ & Sup.\ $R^2$ & Optimism \\", r"\midrule"]
for m in order:
    L.append(f"{NM.get(m, m)} & ${a.loc[('random', m), 'MAE_mean']:.1f}\\pm{a.loc[('random', m), 'MAE_std']:.1f}$ & {b.loc[('random', m, 'supercritical'), 'R2']:.2f} & "
             f"${a.loc[('grouped', m), 'MAE_mean']:.1f}\\pm{a.loc[('grouped', m), 'MAE_std']:.1f}$ & {a.loc[('grouped', m), 'R2_mean']:.3f} & {b.loc[('grouped', m, 'supercritical'), 'R2']:.2f} & {g.loc[m, 'gap_pct']:.0f}\\% \\\\")
L += [r"\bottomrule", r"\end{tabularx}", r"\smallskip\raggedright\scriptsize Source: authors' nested CV analysis of Ding et al.~\cite{ding2025dataset,ding2024zenodo}.", r"\end{table}"]; open("manuscript/tab_acc.tex", "w").write("\n".join(L))

# Table 3 coverage
L = [r"\begin{table}[t]", r"\centering", r"\caption{Empirical coverage of nominal 90\% jackknife-style CV-residual intervals on held-out plants, by how calibration residuals were obtained. Width columns report mean interval width (kcal/kWh); Mondrian width corresponds to per-boiler calibration.}", r"\label{tab:cov}", r"\scriptsize",
     r"\begin{tabularx}{\textwidth}{@{}l *{5}{>{\raggedleft\arraybackslash}X}@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{Random-fold calibration} & \multicolumn{3}{c}{Plant-grouped calibration} \\ \cmidrule(lr){2-3}\cmidrule(l){4-6}",
     r"Model & Pooled & Pooled width & Pooled & Per boiler & Mondrian width \\", r"\midrule"]
def cv(m, cal, mode, col="coverage"): return CV[(CV.model == m) & (CV.by == "all") & (CV.calibration == cal) & (CV["mode"] == mode)][col].iloc[0]
for m in order:
    L.append(f"{NM.get(m, m)} & {cv(m,'random','pooled'):.3f} & {cv(m,'random','pooled','width'):.0f} & {cv(m,'grouped','pooled'):.3f} & {cv(m,'grouped','mondrian'):.3f} & {cv(m,'grouped','mondrian','width'):.0f} \\\\")
L += [r"\bottomrule", r"\end{tabularx}", r"\smallskip\raggedright\scriptsize Source: authors' grouped-CV residual audit of Ding et al.~\cite{ding2025dataset,ding2024zenodo}.", r"\end{table}"]; open("manuscript/tab_cov.tex", "w").write("\n".join(L))
print("P2 gen_tables.py R4 done:", mac["ExcessConvOnly"], mac["ExcessPoint"], mac["RobustNinety"], mac["RobustMatched"])
