"""
Additional statistical validation cells (Kruskal-Wallis + Games-Howell with
effect sizes). 
"""
import warnings
warnings.filterwarnings("ignore")

from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg

df_raw = pd.read_excel(
    "chlorophyll_readings.xlsx", sheet_name="Sheet 1",
    usecols=["Site_ID", "Time", "Chl Temp (°C)", "Chlorophyll (µg/L)"],
)
df_raw.columns = ["Site_ID", "Time", "chl_temp", "chlorophyll"]
df_raw["Time"] = pd.to_datetime(df_raw["Time"], errors="coerce")
df_raw = df_raw.dropna(subset=["chlorophyll"]).reset_index(drop=True)
sites = sorted(df_raw.Site_ID.unique())
k = len(sites)


# ---------------------------------------------------------------
# Multi-lag effective sample size: n_eff = n / (1 + 2 * sum(rho_k)),
# summing lags until |rho_k| < 1.96/sqrt(n) (or n//4, max 150 lags)
# ---------------------------------------------------------------
def autocorr(x, lag):
    xc = x - x.mean()
    d = np.sum(xc ** 2)
    return np.sum(xc[:-lag] * xc[lag:]) / d if d > 1e-9 else 0.0


def multilag_neff(x):
    n = len(x)
    if np.std(x) < 1e-9:
        return float(n)
    thr = 1.96 / np.sqrt(n)
    cum = 0.0
    for lag in range(1, min(n // 4, 150) + 1):
        r = autocorr(x, lag)
        if abs(r) < thr:
            break
        cum += r
    return max(n / (1 + 2 * cum), 2.0)


stat = {}
for s in sites:
    x = df_raw[df_raw.Site_ID == s]["chlorophyll"].values.astype(float)
    stat[s] = dict(n=len(x), neff=multilag_neff(x), mean=x.mean(), var=x.var(ddof=1))

# ---------------------------------------------------------------
# 1) Kruskal-Wallis (rank-based, no distributional assumptions)
#    NOTE: operates on raw readings, so it addresses distributional
#    assumptions only, not non-independence.
# ---------------------------------------------------------------
groups = [df_raw[df_raw.Site_ID == s]["chlorophyll"].values for s in sites]
H, p_kw = stats.kruskal(*groups)
n_total = sum(len(g) for g in groups)
eps_sq = (H - k + 1) / (n_total - k)
print(f"Kruskal-Wallis: H = {H:.2f}, df = {k - 1}, p = {p_kw:.3e}, epsilon^2 = {eps_sq:.3f}")

# ---------------------------------------------------------------
# 2) Games-Howell, raw N (pingouin) - for reference / repository only
# ---------------------------------------------------------------
gh_raw = pg.pairwise_gameshowell(
    dv="chlorophyll", between="Site_ID", data=df_raw[["Site_ID", "chlorophyll"]]
)
print(f"Games-Howell (raw N): {(gh_raw['pval'] < 0.05).sum()} of {len(gh_raw)} pairs significant")

# ---------------------------------------------------------------
# 3) Games-Howell using effective N (reported in the manuscript)
#    Standard Games-Howell statistic with n_eff substituted for n in the
#    standard errors and Welch-Satterthwaite df; simultaneous 95% CIs from
#    the studentized range distribution. Hedges' g uses raw SDs and Ns.
# ---------------------------------------------------------------
rows = []
for a, b in combinations(sites, 2):
    A, B = stat[a], stat[b]
    diff = A["mean"] - B["mean"]
    va, vb = A["var"] / A["neff"], B["var"] / B["neff"]
    se2 = va + vb
    if se2 < 1e-12:                       # both series constant
        p, ci_lo, ci_hi = (0.0 if abs(diff) > 0 else 1.0), diff, diff
        p = 1.0 if abs(diff) < 1e-12 else p
    else:
        dfw = se2 ** 2 / (va ** 2 / (A["neff"] - 1) + vb ** 2 / (B["neff"] - 1))
        q = abs(diff) / np.sqrt(se2 / 2)
        p = stats.studentized_range.sf(q, k, dfw)
        half = stats.studentized_range.ppf(0.95, k, dfw) * np.sqrt(se2 / 2)
        ci_lo, ci_hi = diff - half, diff + half
    na, nb = A["n"], B["n"]
    sp = np.sqrt(((na - 1) * A["var"] + (nb - 1) * B["var"]) / (na + nb - 2))
    g = (diff / sp) * (1 - 3 / (4 * (na + nb) - 9)) if sp > 1e-9 else np.nan
    rows.append(dict(A=a, B=b, mean_diff=diff, ci95_lo=ci_lo, ci95_hi=ci_hi,
                     p_effN=p, hedges_g=g))

gh_eff = pd.DataFrame(rows)
sig = gh_eff[gh_eff["p_effN"] < 0.05]
print(f"Games-Howell (effective N): {len(sig)} of {len(gh_eff)} pairs significant")
print("Non-significant pairs:")
print(gh_eff[gh_eff["p_effN"] >= 0.05].round(3).to_string(index=False))
ag = sig["hedges_g"].abs()
print(f"|Hedges g| among significant pairs: median = {ag.median():.2f}, "
      f"IQR = [{ag.quantile(.25):.2f}, {ag.quantile(.75):.2f}], min = {ag.min():.2f}")

gh_eff.to_csv("games_howell_effectN_with_effect_sizes.csv", index=False)
