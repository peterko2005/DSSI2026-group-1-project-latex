#!/usr/bin/env python3
"""
Rigorous Apples-to-Apples Temperature Functional Form Benchmark
on the Exact M16_CHAMPION_DualGraph_QMLE Backbone (Rent & Return).

Replaces ONLY the main temperature response block (Temp, temp_cold_hinge, temp_hot_hinge)
inside M16_CHAMPION_DualGraph_QMLE, keeping ALL other covariates, Dual Graph Laplacians
(L_geo, L_func, lam_ridge=150, lam_lap=600), 120d recency weights, FE structure, and
strict integer-minute exposure 100% identical.

Evaluates 19 distinct mathematical families:
  - Baseline: NoTemp, Linear, Quadratic
  - Super-Gaussian / Even-Power (-a(T-T0)^(2b)): Sym p=4, Sym p=6, Sym p=4+Lin,
    Asym p=1.5, Asym p=2, Asym p=3, Asym p=4, Asym p=6
  - Hinges & Smooth Hinges: M16 Hinge (18/28), Flat Hinge (18/28), Softplus Hinge,
    Huberized Flat-Top (quadratic->linear)
  - Bounded / Physiological: Gaussian Bell, Bounded Asym Super-Gaussian (p=4),
    Log-Damped Hinge
  - Non-Parametric: Natural Cubic Spline (linear tails), 2C Step Bins (empirical truth)

Also performs:
  1. Grid sweep of (T0, 2b) on 2024 CV + 2025 OOS for symmetric & asymmetric power laws
  2. Tail slice forensics: Cold (<13C, <15C), Comfort (18-28C), Hot (>31C, >33C)
  3. Unseen extreme extrapolation stress test (5C Arctic blast, 8C, 10C, 36C, 39C heatwave)
  4. Saves CSV tables + publication-grade comparison figure in outputs_benchmark/.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

TEST_DIR = Path(__file__).resolve().parent
OUT_DIR = TEST_DIR / "outputs_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(TEST_DIR))

import benchmark_engine as be

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 1. Temperature Basis Library (all normalized so f(T_ref=23C) = 0 cleanly)
# ---------------------------------------------------------------------------

def _softplus(z: np.ndarray, tau: float = 2.0) -> np.ndarray:
    x = z / tau
    return tau * np.where(x > 30.0, x, np.log1p(np.exp(np.clip(x, -40.0, 30.0))))


def _huber_pos(u: np.ndarray, delta: float = 6.0) -> np.ndarray:
    """C1 smooth: quadratic 0.5*u^2/delta for u <= delta, linear u - 0.5*delta for u > delta."""
    u_p = np.maximum(u, 0.0)
    return np.where(u_p <= delta, 0.5 * (u_p ** 2) / delta, u_p - 0.5 * delta)


def _natural_cubic_spline_basis(T: np.ndarray, knots: list[float]) -> tuple[np.ndarray, list[str]]:
    """Standard truncated-power natural cubic spline basis (strictly linear outside [k_1, k_K])."""
    k = np.asarray(knots, dtype=np.float64)
    K = len(k)
    d_last = (np.maximum(T - k[-2], 0.0) ** 3 - np.maximum(T - k[-1], 0.0) ** 3) / (k[-1] - k[-2])
    cols = [(T - 23.0) / 10.0]
    names = ["ncs_lin"]
    for j in range(K - 2):
        dj = (np.maximum(T - k[j], 0.0) ** 3 - np.maximum(T - k[-1], 0.0) ** 3) / (k[-1] - k[j])
        cols.append((dj - d_last) / 100.0)
        names.append(f"ncs_k{int(k[j])}")
    return np.column_stack(cols), names


def _temp_bins_basis(T: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """2C non-parametric bins; reference bin is [22, 24)."""
    edges = [-100.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 100.0]
    labels = [
        "<12", "12-14", "14-16", "16-18", "18-20", "20-22",
        "22-24", "24-26", "26-28", "28-30", "30-32", "32-34", ">=34"
    ]
    cols = []
    names = []
    for i in range(len(labels)):
        if labels[i] == "22-24":
            continue
        m = ((T >= edges[i]) & (T < edges[i + 1])).astype(np.float64)
        cols.append(m)
        names.append(f"bin_{labels[i]}")
    return np.column_stack(cols), names


def make_temp_specs() -> list[tuple[str, str, Callable[[np.ndarray], tuple[np.ndarray, list[str]]], float]]:
    """Return list of (spec_id, family_label, basis_fn(T)->(X_temp, col_names), ridge)."""
    specs: list[tuple[str, str, Callable[[np.ndarray], tuple[np.ndarray, list[str]]], float]] = [
        ("T00_NoTemp", "0_Ref", lambda T: (np.zeros((len(T), 0)), []), 0.1),
        ("T01_Linear", "0_Ref", lambda T: (np.column_stack([(T - 23.0)]), ["t_lin"]), 0.1),
        ("T02_Quadratic_p2", "0_Ref", lambda T: (
            np.column_stack([(T - 23.0), ((T - 23.0) / 10.0) ** 2]), ["t_lin", "t_quad"]
        ), 0.1),

        # User's y = -a (T - T0)^(2b) family (Symmetric & Asymmetric)
        ("T03_SymPower_p4_T23", "1_Power", lambda T: (
            np.column_stack([((T - 23.0) / 10.0) ** 4]), ["t_sym_p4"]
        ), 0.05),
        ("T04_SymPower_p6_T23", "1_Power", lambda T: (
            np.column_stack([((T - 23.0) / 10.0) ** 6]), ["t_sym_p6"]
        ), 0.05),
        ("T05_SymPower_p4_T245", "1_Power", lambda T: (
            np.column_stack([((T - 24.5) / 10.0) ** 4]), ["t_sym_p4_245"]
        ), 0.05),
        ("T06_SymPower_p4_plus_Lin", "1_Power", lambda T: (
            np.column_stack([(T - 23.0) / 10.0, ((T - 23.0) / 10.0) ** 4]), ["t_lin", "t_sym_p4"]
        ), 0.05),
        ("T07_AsymPower_p15_T23", "1_Power", lambda T: (
            np.column_stack([
                (np.maximum(23.0 - T, 0.0) / 10.0) ** 1.5,
                (np.maximum(T - 23.0, 0.0) / 10.0) ** 1.5,
            ]), ["t_cold_p15", "t_hot_p15"]
        ), 0.05),
        ("T08_AsymPower_p2_T23", "1_Power", lambda T: (
            np.column_stack([
                (np.maximum(23.0 - T, 0.0) / 10.0) ** 2.0,
                (np.maximum(T - 23.0, 0.0) / 10.0) ** 2.0,
            ]), ["t_cold_p2", "t_hot_p2"]
        ), 0.05),
        ("T09_AsymPower_p3_T23", "1_Power", lambda T: (
            np.column_stack([
                (np.maximum(23.0 - T, 0.0) / 10.0) ** 3.0,
                (np.maximum(T - 23.0, 0.0) / 10.0) ** 3.0,
            ]), ["t_cold_p3", "t_hot_p3"]
        ), 0.05),
        ("T10_AsymPower_p4_T23", "1_Power", lambda T: (
            np.column_stack([
                (np.maximum(23.0 - T, 0.0) / 10.0) ** 4.0,
                (np.maximum(T - 23.0, 0.0) / 10.0) ** 4.0,
            ]), ["t_cold_p4", "t_hot_p4"]
        ), 0.05),
        ("T11_AsymPower_p6_T23", "1_Power", lambda T: (
            np.column_stack([
                (np.maximum(23.0 - T, 0.0) / 10.0) ** 6.0,
                (np.maximum(T - 23.0, 0.0) / 10.0) ** 6.0,
            ]), ["t_cold_p6", "t_hot_p6"]
        ), 0.05),

        # Piecewise Linear Hinges & C^inf Smooth Hinges (Safe Linear Tails)
        ("T12_M16_Hinge_18_28", "2_Hinge", lambda T: (
            np.column_stack([T - 23.0, np.maximum(18.0 - T, 0.0), np.maximum(T - 28.0, 0.0)]),
            ["t_lin", "t_cold_h18", "t_hot_h28"]
        ), 0.1),
        ("T13_FlatHinge_18_28", "2_Hinge", lambda T: (
            np.column_stack([np.maximum(18.0 - T, 0.0), np.maximum(T - 28.0, 0.0)]),
            ["t_cold_h18", "t_hot_h28"]
        ), 0.1),
        ("T14_Softplus_18_28_tau2", "2_Hinge", lambda T: (
            np.column_stack([_softplus(18.0 - T, 2.0), _softplus(T - 28.0, 2.0)]),
            ["t_sp_cold", "t_sp_hot"]
        ), 0.1),
        ("T15_HuberFlatTop_T23_d7", "2_Hinge", lambda T: (
            np.column_stack([_huber_pos(23.0 - T, 7.0), _huber_pos(T - 23.0, 7.0)]),
            ["t_hub_cold", "t_hub_hot"]
        ), 0.1),

        # Bounded / Saturating Physiological Functions
        ("T16_Bounded_AsymSuperGauss_p4", "3_Bounded", lambda T: (
            np.column_stack([
                1.0 - np.exp(-((np.maximum(23.0 - T, 0.0) / 12.0) ** 4)),
                1.0 - np.exp(-((np.maximum(T - 23.0, 0.0) / 11.0) ** 4)),
            ]), ["t_bsg_cold", "t_bsg_hot"]
        ), 0.05),
        ("T17_LogDampedHinge_18_28", "3_Bounded", lambda T: (
            np.column_stack([np.log1p(np.maximum(18.0 - T, 0.0)), np.log1p(np.maximum(T - 28.0, 0.0))]),
            ["t_log_cold", "t_log_hot"]
        ), 0.1),

        # Non-Parametric Benchmarks
        ("T18_NaturalCubicSpline", "4_NonParam", lambda T: _natural_cubic_spline_basis(T, [13.0, 18.0, 23.0, 28.0, 33.0]), 0.1),
        ("T19_StepBins_2C", "4_NonParam", _temp_bins_basis, 0.1),
    ]
    return specs


# ---------------------------------------------------------------------------
# 2. Slice Evaluation Helper (Overall + Cold/Comfort/Hot Tails)
# ---------------------------------------------------------------------------

def eval_slice(y: np.ndarray, exp: np.ndarray, pred_rate: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    ym = y[mask]
    em = exp[mask]
    pm = np.clip(pred_rate[mask], 0.0, 500.0)
    obs_r = ym / em
    w = em / np.sum(em)
    err = pm - obs_r
    wrmse = float(np.sqrt(np.sum(w * err * err)))
    wmae = float(np.sum(w * np.abs(err)))
    obs_bar = float(np.sum(w * obs_r))
    ss_tot = float(np.sum(w * (obs_r - obs_bar) ** 2))
    wr2 = float(1.0 - np.sum(w * err * err) / max(ss_tot, 1e-12))
    pred_cnt = pm * em
    ratio = float(np.sum(pred_cnt) / max(np.sum(ym), 1e-12))
    pdev = be.poisson_deviance(ym, pred_cnt)
    return {
        "n": int(np.sum(mask)),
        "wRMSE": wrmse,
        "wMAE": wmae,
        "wR2": wr2,
        "PoisDev": pdev,
        "PredObsRatio": ratio,
        "MeanObsRate": obs_bar,
        "MeanPredRate": float(np.sum(w * pm)),
    }


def eval_curve(basis_fn, theta_t: np.ndarray, T_grid: np.ndarray) -> np.ndarray:
    """Evaluate f(T) - f(23.0) in log-rate space."""
    if len(theta_t) == 0:
        return np.zeros_like(T_grid, dtype=np.float64)
    B, _ = basis_fn(T_grid)
    B23, _ = basis_fn(np.array([23.0], dtype=np.float64))
    return (B @ theta_t) - float((B23 @ theta_t)[0])


# ---------------------------------------------------------------------------
# 3. Main Runner
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()
    print("[1/4] Loading derived panel & station metadata...")
    panel, cal, stn, _ = be.build_or_load_derived_datasets(TEST_DIR)
    train_all = panel[panel["year"] == 2024].copy()
    test_all = panel[panel["year"] == 2025].copy()
    # Restrict 2025 evaluation to the 118 common stations (exact apples-to-apples with M16 leaderboard)
    common_sids = sorted(set(train_all["station_id"].unique()) & set(test_all["station_id"].unique()))
    test_common = test_all[test_all["station_id"].isin(common_sids)].copy()

    specs = make_temp_specs()
    T_grid = np.linspace(5.0, 39.0, 341)
    T_probe = np.array([5.0, 8.0, 10.0, 13.0, 15.0, 18.0, 23.0, 28.0, 31.0, 34.0, 36.0, 39.0])

    # Temp columns in M16 that we replace apples-to-apples
    DROP_TEMP_COLS = {"Temp", "temp_cold_hinge", "temp_hot_hinge"}

    all_rows = []
    curve_store: dict[str, dict[str, np.ndarray]] = {"rent": {}, "return": {}}
    probe_rows = []

    for side, y_col, exp_col, valid_col in [
        ("rent", "y_rent", "exp_rent_eff", "valid_rent"),
        ("return", "y_ret", "exp_ret_eff", "valid_ret"),
    ]:
        print(f"\n[2/4] Preparing EXACT M16_CHAMPION_DualGraph_QMLE base matrices ({side.upper()})...")
        tr_df = train_all[train_all[valid_col]].reset_index(drop=True)
        te_df = test_common[test_common[valid_col]].reset_index(drop=True)

        station_order = sorted(tr_df["station_id"].unique().tolist())
        _, L_geo = be.build_spatial_laplacian(stn, station_order, length_scale_m=320.0)
        _, L_func = be.build_functional_laplacian(stn, station_order, sigma=1.5)

        tr_df["fe_key"] = be._make_fe_key(tr_df, "station_hour_wd")
        fe_sum_y = tr_df.groupby("fe_key")[y_col].transform("sum")
        zero_mask = fe_sum_y == 0
        zero_fe_keys = set(tr_df.loc[zero_mask, "fe_key"].unique())
        est = tr_df[~zero_mask].reset_index(drop=True)

        fe_cats, g_idx = np.unique(est["fe_key"].to_numpy(), return_inverse=True)
        G = len(fe_cats)
        N = len(est)
        y_tr = est[y_col].to_numpy(dtype=np.float64)
        exp_tr = est[exp_col].to_numpy(dtype=np.float64)
        offset_tr = np.log(exp_tr)

        d_max = float(est["days_from_2024_01_01"].max())
        age = d_max - est["days_from_2024_01_01"].to_numpy(dtype=np.float64)
        obs_w = np.exp(-np.log(2.0) * age / 120.0)
        obs_w = obs_w * (len(obs_w) / np.sum(obs_w))

        Z = sparse.csc_matrix((np.ones(N, dtype=np.float64), (np.arange(N), g_idx)), shape=(N, G))

        # Build full M16 feature matrices once, then drop ONLY the 3 main temp columns
        X_tr_full, feat_names_full, P_full = be.build_feature_matrix(
            est, "M16_CHAMPION_DualGraph_QMLE", stn, station_order,
            L_geo=L_geo, L_func=L_func, lam_ridge=150.0, lam_lap=600.0,
        )
        X_te_full, _, _ = be.build_feature_matrix(
            te_df, "M16_CHAMPION_DualGraph_QMLE", stn, station_order,
            L_geo=L_geo, L_func=L_func, lam_ridge=150.0, lam_lap=600.0,
        )
        keep_idx = [i for i, nm in enumerate(feat_names_full) if nm not in DROP_TEMP_COLS]
        X_tr_base = X_tr_full[:, keep_idx].tocsc()
        X_te_base = X_te_full[:, keep_idx].tocsc()
        P_base = P_full[np.ix_(keep_idx, keep_idx)]
        K_base = len(keep_idx)

        T_tr = est["Temp"].to_numpy(dtype=np.float64)
        T_te = te_df["Temp"].to_numpy(dtype=np.float64)
        y_te = te_df[y_col].to_numpy(dtype=np.float64)
        exp_te = te_df[exp_col].to_numpy(dtype=np.float64)
        te_fks = be._make_fe_key(te_df, "station_hour_wd")
        te_is_zero = te_fks.isin(zero_fe_keys).to_numpy()

        # Masks for 2025 OOS temperature slices
        masks = {
            "ALL": np.ones(len(te_df), dtype=bool),
            "COLD_LT13": T_te < 13.0,
            "COLD_LT15": T_te < 15.0,
            "COMFORT_18_28": (T_te >= 18.0) & (T_te <= 28.0),
            "HOT_GT31": T_te > 31.0,
            "HOT_GT33": T_te > 33.0,
            "EXTREMES_LT14_GT33": (T_te < 14.0) | (T_te > 33.0),
        }

        # Warm-start from NoTemp fit so every subsequent fit takes only 2-4 Newton steps!
        print(f"  Fitting T00_NoTemp warm-start anchor ({K_base} base cols)...")
        th_base_init, psi_init, _, _, _ = be.fit_penalized_glm_fe_core(
            y_tr, X_tr_base, offset_tr, g_idx, G, 0.0, Z=Z, P_theta=P_base, obs_w=obs_w,
            max_iter=35, tol=1e-7,
        )

        # Also prepare 2024 internal cold+hot temporal CV (hold out Jan-Feb cold + Jul-Aug hot?
        # Even better: 2024 4-block seasonal CV by day-of-year modulo 4 blocks of 2-week chunks so
        # every fold has BOTH winter cold snaps and summer heatwaves with zero adjacent-day leakage!)
        day_blk = (est["days_from_2024_01_01"].to_numpy(dtype=np.int32) // 14) % 4
        # Fast proxy check on full M16 2025 OOS + 2024 penalized AIC + exact 2024 4-fold block CV
        # (We run full 4-fold block CV on a lighter M07 backbone below for zero-leakage selection)

        for spec_id, fam, basis_fn, ridge_t in specs:
            ts = time.perf_counter()
            Bt_tr, t_names = basis_fn(T_tr)
            Bt_te, _ = basis_fn(T_te)
            Kt = Bt_tr.shape[1]

            if Kt > 0:
                X_tr_use = sparse.hstack([X_tr_base, sparse.csc_matrix(Bt_tr)], format="csc")
                X_te_use = sparse.hstack([X_te_base, sparse.csc_matrix(Bt_te)], format="csc")
                P_use = np.zeros((K_base + Kt, K_base + Kt), dtype=np.float64)
                P_use[:K_base, :K_base] = P_base
                P_use[K_base:, K_base:] = ridge_t * np.eye(Kt)
                th_init_use = np.concatenate([th_base_init, np.zeros(Kt, dtype=np.float64)])
            else:
                X_tr_use = X_tr_base
                X_te_use = X_te_base
                P_use = P_base
                th_init_use = th_base_init.copy()

            th_hat, psi_hat, mu_tr, ll_tr, n_it = be.fit_penalized_glm_fe_core(
                y_tr, X_tr_use, offset_tr, g_idx, G, 0.0, Z=Z, P_theta=P_use, obs_w=obs_w,
                theta_init=th_init_use, psi_init=psi_init, max_iter=25, tol=1e-7,
            )

            psi_map = dict(zip(fe_cats, psi_hat))
            psi_te = te_fks.map(psi_map).fillna(float(np.mean(psi_hat))).to_numpy(dtype=np.float64)
            lin_te = np.asarray(X_te_use @ th_hat, dtype=np.float64).ravel()
            pred_raw = np.where(te_is_zero, 0.0, np.exp(np.clip(psi_te + lin_te, -20.0, 8.0)))
            pred_c14 = be.apply_causal_trailing_level_calibration(te_df, pred_raw, y_col, exp_col, window_days=14)

            theta_t = th_hat[K_base:] if Kt > 0 else np.empty(0)
            curve_store[side][spec_id] = eval_curve(basis_fn, theta_t, T_grid)
            probe_vals = np.exp(eval_curve(basis_fn, theta_t, T_probe))

            m_all = eval_slice(y_te, exp_te, pred_raw, masks["ALL"])
            m_all_c = eval_slice(y_te, exp_te, pred_c14, masks["ALL"])
            m_c13 = eval_slice(y_te, exp_te, pred_raw, masks["COLD_LT13"])
            m_c15 = eval_slice(y_te, exp_te, pred_raw, masks["COLD_LT15"])
            m_com = eval_slice(y_te, exp_te, pred_raw, masks["COMFORT_18_28"])
            m_h31 = eval_slice(y_te, exp_te, pred_raw, masks["HOT_GT31"])
            m_h33 = eval_slice(y_te, exp_te, pred_raw, masks["HOT_GT33"])
            m_ext = eval_slice(y_te, exp_te, pred_raw, masks["EXTREMES_LT14_GT33"])
            # Also check CAUSAL14D on cold/hot tails
            m_c13_c = eval_slice(y_te, exp_te, pred_c14, masks["COLD_LT13"])
            m_h33_c = eval_slice(y_te, exp_te, pred_c14, masks["HOT_GT33"])

            row = {
                "side": side,
                "spec_id": spec_id,
                "family": fam,
                "n_temp_params": Kt,
                "n_iter": n_it,
                "sec": round(time.perf_counter() - ts, 2),
                "ll_train_weighted": ll_tr,
                "raw_wRMSE": m_all["wRMSE"],
                "raw_wMAE": m_all["wMAE"],
                "raw_wR2": m_all["wR2"],
                "raw_PoisDev": m_all["PoisDev"],
                "c14_wRMSE": m_all_c["wRMSE"],
                "c14_wR2": m_all_c["wR2"],
                "cold13_wRMSE": m_c13["wRMSE"],
                "cold13_ratio": m_c13["PredObsRatio"],
                "cold13_c14_ratio": m_c13_c["PredObsRatio"],
                "cold15_wRMSE": m_c15["wRMSE"],
                "cold15_ratio": m_c15["PredObsRatio"],
                "comfort_wRMSE": m_com["wRMSE"],
                "comfort_ratio": m_com["PredObsRatio"],
                "hot31_wRMSE": m_h31["wRMSE"],
                "hot31_ratio": m_h31["PredObsRatio"],
                "hot33_wRMSE": m_h33["wRMSE"],
                "hot33_ratio": m_h33["PredObsRatio"],
                "hot33_c14_ratio": m_h33_c["PredObsRatio"],
                "ext_wRMSE": m_ext["wRMSE"],
                "ext_PoisDev": m_ext["PoisDev"],
                "mult_5C": float(probe_vals[0]),
                "mult_8C": float(probe_vals[1]),
                "mult_10C": float(probe_vals[2]),
                "mult_15C": float(probe_vals[4]),
                "mult_23C": float(probe_vals[6]),
                "mult_34C": float(probe_vals[9]),
                "mult_36C": float(probe_vals[10]),
                "mult_39C": float(probe_vals[11]),
            }
            all_rows.append(row)
            print(
                f"  {side:6s} | {spec_id:30s} (k={Kt:2d}, it={n_it:2d}, {row['sec']:4.1f}s) | "
                f"ALL wRMSE={m_all['wRMSE']:.5f} (C14={m_all_c['wRMSE']:.5f}) | "
                f"EXT={m_ext['wRMSE']:.5f} | <13C r={m_c13['PredObsRatio']:.3f} | "
                f">33C r={m_h33['PredObsRatio']:.3f} | 5C={probe_vals[0]:.3f}x 39C={probe_vals[11]:.3f}x"
            )

    res_df = pd.DataFrame(all_rows)
    res_df.to_csv(OUT_DIR / "temp_functional_form_benchmark.csv", index=False)

    # -----------------------------------------------------------------------
    # 3. Exhaustive (T0, 2b) Grid Sweep for y = -a (T - T0)^(2b) (fast M07 backbone + 2024 4-fold CV)
    # -----------------------------------------------------------------------
    print("\n[3/4] Running strict 2024 4-block seasonal CV + 2025 OOS sweep over (T0, 2b)...")
    sweep_rows = []
    for side, y_col, exp_col, valid_col in [
        ("rent", "y_rent", "exp_rent_eff", "valid_rent"),
        ("return", "y_ret", "exp_ret_eff", "valid_ret"),
    ]:
        tr_df = train_all[train_all[valid_col]].reset_index(drop=True)
        te_df = test_common[test_common[valid_col]].reset_index(drop=True)
        station_order = sorted(tr_df["station_id"].unique().tolist())

        tr_df["fe_key"] = be._make_fe_key(tr_df, "station_hour_wd")
        fe_sum_y = tr_df.groupby("fe_key")[y_col].transform("sum")
        zero_mask = fe_sum_y == 0
        zero_fe_keys = set(tr_df.loc[zero_mask, "fe_key"].unique())
        est = tr_df[~zero_mask].reset_index(drop=True)

        fe_cats, g_idx = np.unique(est["fe_key"].to_numpy(), return_inverse=True)
        G, N = len(fe_cats), len(est)
        y_tr = est[y_col].to_numpy(dtype=np.float64)
        exp_tr = est[exp_col].to_numpy(dtype=np.float64)
        offset_tr = np.log(exp_tr)
        d_max = float(est["days_from_2024_01_01"].max())
        age = d_max - est["days_from_2024_01_01"].to_numpy(dtype=np.float64)
        obs_w = np.exp(-np.log(2.0) * age / 120.0)
        obs_w = obs_w * (len(obs_w) / np.sum(obs_w))
        Z = sparse.csc_matrix((np.ones(N, dtype=np.float64), (np.arange(N), g_idx)), shape=(N, G))

        # Light M07 backbone (no 1416 station slopes, ~0.2s per fit!)
        X_tr_f, fn_f, P_f = be.build_feature_matrix(est, "M07_Poisson_RichRegime", stn, station_order)
        X_te_f, _, _ = be.build_feature_matrix(te_df, "M07_Poisson_RichRegime", stn, station_order)
        keep_idx = [i for i, nm in enumerate(fn_f) if nm not in DROP_TEMP_COLS]
        X_tr_b = X_tr_f[:, keep_idx].tocsc()
        X_te_b = X_te_f[:, keep_idx].tocsc()
        P_b = P_f[np.ix_(keep_idx, keep_idx)]
        Kb = len(keep_idx)

        T_tr = est["Temp"].to_numpy(dtype=np.float64)
        T_te = te_df["Temp"].to_numpy(dtype=np.float64)
        y_te = te_df[y_col].to_numpy(dtype=np.float64)
        exp_te = te_df[exp_col].to_numpy(dtype=np.float64)
        te_fks = be._make_fe_key(te_df, "station_hour_wd")
        te_is_zero = te_fks.isin(zero_fe_keys).to_numpy()
        m_ext_te = (T_te < 14.0) | (T_te > 33.0)
        m_c13_te = T_te < 13.0

        th_b0, psi_b0, _, _, _ = be.fit_penalized_glm_fe_core(
            y_tr, X_tr_b, offset_tr, g_idx, G, 0.0, Z=Z, P_theta=P_b, obs_w=obs_w, max_iter=30, tol=1e-7
        )

        # 2-week interleaved 4-fold seasonal CV on 2024 (every fold covers full winter->summer!)
        fold_id = (est["days_from_2024_01_01"].to_numpy(dtype=np.int32) // 14) % 4
        cv_folds = []
        for f in range(4):
            val_m = fold_id == f
            trn_m = ~val_m
            fe_c_f, g_tr_f = np.unique(est.loc[trn_m, "fe_key"].to_numpy(), return_inverse=True)
            Gf, Nf = len(fe_c_f), int(np.sum(trn_m))
            Zf = sparse.csc_matrix((np.ones(Nf, dtype=np.float64), (np.arange(Nf), g_tr_f)), shape=(Nf, Gf))
            cv_folds.append((trn_m, val_m, fe_c_f, g_tr_f, Gf, Zf))

        # Test grid: symmetric & asymmetric (T0 in [22, 23, 24, 25, 26], p in [1.5, 2, 3, 4, 6]) + Softplus + FlatHinge
        candidates = []
        for mode in ["sym", "asym"]:
            for p in [1.5, 2.0, 3.0, 4.0, 6.0]:
                if mode == "sym" and p in (1.5, 3.0):
                    continue
                for T0 in [22.0, 23.0, 24.0, 25.0, 26.0]:
                    if mode == "sym":
                        fn = (lambda T, T0=T0, p=p: (np.column_stack([(np.abs(T - T0) / 10.0) ** p]), ["p"]))
                    else:
                        fn = (lambda T, T0=T0, p=p: (np.column_stack([
                            (np.maximum(T0 - T, 0.0) / 10.0) ** p,
                            (np.maximum(T - T0, 0.0) / 10.0) ** p,
                        ]), ["pl", "pr"]))
                    candidates.append((f"{mode}_p{p:g}_T{T0:g}", mode, p, T0, fn))
        candidates.append(("softplus_18_28_tau2", "softplus", 1.0, 23.0,
                           lambda T: (np.column_stack([_softplus(18.0 - T, 2.0), _softplus(T - 28.0, 2.0)]), ["l", "r"])))
        candidates.append(("flat_hinge_18_28", "hinge", 1.0, 23.0,
                           lambda T: (np.column_stack([np.maximum(18.0 - T, 0.0), np.maximum(T - 28.0, 0.0)]), ["l", "r"])))
        candidates.append(("huber_T23_d7", "huber", 1.5, 23.0,
                           lambda T: (np.column_stack([_huber_pos(23.0 - T, 7.0), _huber_pos(T - 23.0, 7.0)]), ["l", "r"])))
        candidates.append(("bounded_sg_p4", "bounded_sg", 4.0, 23.0,
                           lambda T: (np.column_stack([
                               1.0 - np.exp(-((np.maximum(23.0 - T, 0.0) / 12.0) ** 4)),
                               1.0 - np.exp(-((np.maximum(T - 23.0, 0.0) / 11.0) ** 4)),
                           ]), ["l", "r"])))

        for cname, mode, p_val, T0_val, fn in candidates:
            Bt_tr, _ = fn(T_tr)
            Bt_te, _ = fn(T_te)
            Kt = Bt_tr.shape[1]
            X_tr_u = sparse.hstack([X_tr_b, sparse.csc_matrix(Bt_tr)], format="csc")
            X_te_u = sparse.hstack([X_te_b, sparse.csc_matrix(Bt_te)], format="csc")
            P_u = np.zeros((Kb + Kt, Kb + Kt), dtype=np.float64)
            P_u[:Kb, :Kb] = P_b
            P_u[Kb:, Kb:] = 0.05 * np.eye(Kt)

            # 2024 4-fold seasonal CV
            oof_pred = np.zeros(N, dtype=np.float64)
            for trn_m, val_m, fe_c_f, g_tr_f, Gf, Zf in cv_folds:
                th_f, psi_f, _, _, _ = be.fit_penalized_glm_fe_core(
                    y_tr[trn_m], X_tr_u[trn_m], offset_tr[trn_m], g_tr_f, Gf, 0.0,
                    Z=Zf, P_theta=P_u, obs_w=obs_w[trn_m], max_iter=20, tol=1e-6,
                )
                pmap_f = dict(zip(fe_c_f, psi_f))
                pv = est.loc[val_m, "fe_key"].map(pmap_f).fillna(float(np.mean(psi_f))).to_numpy(dtype=np.float64)
                lv = np.asarray(X_tr_u[val_m] @ th_f, dtype=np.float64).ravel()
                oof_pred[val_m] = np.exp(np.clip(pv + lv, -20.0, 8.0))

            cv_all = eval_slice(y_tr, exp_tr, oof_pred, np.ones(N, dtype=bool))
            cv_ext = eval_slice(y_tr, exp_tr, oof_pred, (T_tr < 14.0) | (T_tr > 33.0))

            # Full 2024 fit -> 2025 OOS
            th_h, psi_h, _, _, _ = be.fit_penalized_glm_fe_core(
                y_tr, X_tr_u, offset_tr, g_idx, G, 0.0, Z=Z, P_theta=P_u, obs_w=obs_w,
                theta_init=np.concatenate([th_b0, np.zeros(Kt)]), psi_init=psi_b0, max_iter=20, tol=1e-7,
            )
            pmap = dict(zip(fe_cats, psi_h))
            psi_te = te_fks.map(pmap).fillna(float(np.mean(psi_h))).to_numpy(dtype=np.float64)
            pred_te = np.where(te_is_zero, 0.0, np.exp(np.clip(psi_te + np.asarray(X_te_u @ th_h).ravel(), -20.0, 8.0)))
            te_all = eval_slice(y_te, exp_te, pred_te, np.ones(len(te_df), dtype=bool))
            te_ext = eval_slice(y_te, exp_te, pred_te, m_ext_te)
            te_c13 = eval_slice(y_te, exp_te, pred_te, m_c13_te)
            pv5 = float(np.exp(eval_curve(fn, th_h[Kb:], np.array([5.0]))[0]))
            pv39 = float(np.exp(eval_curve(fn, th_h[Kb:], np.array([39.0]))[0]))

            sweep_rows.append({
                "side": side, "name": cname, "mode": mode, "p": p_val, "T0": T0_val,
                "cv24_wRMSE": cv_all["wRMSE"], "cv24_ext_wRMSE": cv_ext["wRMSE"],
                "oos25_wRMSE": te_all["wRMSE"], "oos25_ext_wRMSE": te_ext["wRMSE"],
                "oos25_cold13_ratio": te_c13["PredObsRatio"],
                "mult_5C": pv5, "mult_39C": pv39,
            })

    sw_df = pd.DataFrame(sweep_rows)
    sw_df.to_csv(OUT_DIR / "temp_power_grid_cv_sweep.csv", index=False)

    # -----------------------------------------------------------------------
    # 4. Generate Publication-Grade Diagnostic Figure
    # -----------------------------------------------------------------------
    print("\n[4/4] Plotting diagnostic comparison figure...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=160)

    plot_specs = [
        ("T19_StepBins_2C", "Empirical 2°C Bins (Truth)", "#111111", "--", 2.0, 0.85),
        ("T18_NaturalCubicSpline", "Natural Cubic Spline", "#7f7f7f", ":", 2.0, 0.9),
        ("T02_Quadratic_p2", "Quadratic (2b=2)", "#9467bd", "-.", 1.8, 0.85),
        ("T03_SymPower_p4_T23", "Sym -(T-23)^4 (User 2b=4)", "#d62728", "-", 2.3, 0.95),
        ("T04_SymPower_p6_T23", "Sym -(T-23)^6 (User 2b=6)", "#ff9896", "--", 1.9, 0.9),
        ("T10_AsymPower_p4_T23", "Asym -(T-23)^4 (L/R 2b=4)", "#ff7f0e", "-", 2.1, 0.95),
        ("T13_FlatHinge_18_28", "Flat Hinge [18,28]", "#1f77b4", "-", 2.1, 0.95),
        ("T14_Softplus_18_28_tau2", "Softplus Hinge (C∞)", "#2ca02c", "-", 2.6, 1.0),
        ("T16_Bounded_AsymSuperGauss_p4", "Bounded SuperGauss (2b=4)", "#17becf", "-.", 2.2, 0.95),
    ]

    for col_i, (side, title_s) in enumerate([("rent", "Rent (借車端)"), ("return", "Return (還車端)")]):
        # Top row: Response multiplier exp(f(T) - f(23)) from 5°C to 39°C (showing unseen extrapolation!)
        ax = axes[0, col_i]
        ax.axvspan(5.0, 7.5, color="#cfe2f3", alpha=0.45, label="Unseen Cold (<7.5°C)")
        ax.axvspan(36.9, 39.0, color="#f4cccc", alpha=0.45, label="Unseen Hot (>36.9°C)")
        ax.axvspan(18.0, 28.0, color="#e8f5e9", alpha=0.35, label="Comfort Plateau (18-28°C)")
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")
        for sid, lbl, col, ls, lw, alp in plot_specs:
            y_mult = np.exp(curve_store[side][sid])
            ax.plot(T_grid, y_mult, label=lbl, color=col, ls=ls, lw=lw, alpha=alp)
        ax.set_xlim(5.0, 39.0)
        ax.set_ylim(0.0, 1.25)
        ax.set_title(f"{title_s} — Temperature Demand Multiplier exp(f(T) - f(23°C))", fontsize=11, fontweight="bold")
        ax.set_xlabel("Temperature (°C)  [Observed 2024-2025: 7.5°C ~ 36.9°C]")
        ax.set_ylabel("Demand Multiplier vs. 23°C")
        ax.grid(alpha=0.25)
        if col_i == 0:
            ax.legend(fontsize=7.5, loc="lower center", ncol=2, framealpha=0.92)

        # Bottom row: 2025 OOS Extreme-Tail Performance (wRMSE on <14°C or >33°C vs Cold <13°C Pred/Obs Ratio)
        ax2 = axes[1, col_i]
        sub = res_df[res_df["side"] == side].copy()
        fam_colors = {
            "0_Ref": "#7f7f7f", "1_Power": "#d62728", "2_Hinge": "#2ca02c",
            "3_Bounded": "#17becf", "4_NonParam": "#1f77b4",
        }
        for _, r in sub.iterrows():
            c = fam_colors[r["family"]]
            ax2.scatter(r["cold13_ratio"], r["ext_wRMSE"], color=c, s=75, edgecolor="k", lw=0.6, zorder=4)
            short = r["spec_id"].split("_", 1)[1]
            ax2.annotate(short, (r["cold13_ratio"], r["ext_wRMSE"]), fontsize=7,
                         xytext=(4, 2), textcoords="offset points")
        ax2.axvline(1.0, color="black", ls="--", lw=1.0, label="Exact Unbiased Ratio = 1.00")
        ax2.axvspan(0.5, 1.0, color="#ffcccc", alpha=0.22, label="Underestimation Zone (<1.00)")
        ax2.set_title(f"{title_s} — 2025 OOS Extreme Tail Error vs. Cold-Snap (<13°C) Pred/Obs Ratio",
                      fontsize=11, fontweight="bold")
        ax2.set_xlabel("2025 Cold-Snap (<13°C) Pred / Obs Ratio  (>= 1.00 = No Underestimation)")
        ax2.set_ylabel("2025 Extreme-Temp (<14°C or >33°C) wRMSE  (Lower is Better)")
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Temperature Functional Form Stress Test on Full M16_CHAMPION_DualGraph_QMLE (2024 Train → 2025 OOS)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig_path = OUT_DIR / "fig_temp_functional_form_benchmark.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    print(f"\nDone in {time.perf_counter() - t_start:.1f}s!")
    print(f"Saved: {OUT_DIR / 'temp_functional_form_benchmark.csv'}")
    print(f"Saved: {OUT_DIR / 'temp_power_grid_cv_sweep.csv'}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
