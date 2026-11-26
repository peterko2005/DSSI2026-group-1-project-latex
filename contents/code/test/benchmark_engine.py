"""
Modular benchmark engine for NTU YouBike 2.0 latent demand rate modeling.

Strictly read-only with respect to `econ-5166-student-kit/data/processed/`:
all derived datasets, figures, and tables are created exclusively inside `test/`.
"""
from __future__ import annotations

import hashlib
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, sparse, special

# ---------------------------------------------------------------------------
# 1. Portable relative path resolution & read-only SHA-256 verification
# ---------------------------------------------------------------------------

INPUT_SPECS: dict[str, tuple[str, str]] = {
    "stations": (
        "stations_in_field.csv",
        "fbf0c1797ddbbe59de4852b077ce79dc8243ccc59f08caf4cff0ddfa5eba1ce4",
    ),
    "flow_hourly": (
        "youbike_station_hourly_flow/youbike_station_hourly_flow_2024_2025.parquet",
        "f2e21013892c58b9b5f93e347b4defb37999608ff45022003a0426840092dd23",
    ),
    "rental_2024": (
        "youbike_rental/youbike_rental_in_field_2024.parquet",
        "5c91e9b439f421bc41d7c247508bc9bdd01f7bac4e11e8887243cb6053d8b3b5",
    ),
    "rental_2025": (
        "youbike_rental/youbike_rental_in_field_2025.parquet",
        "d059c1457763ff8af3a327f52ddcc37262df2aa6370d1db6f7550b812b612a2e",
    ),
    "trough_2024": (
        "youbike_trough_hourly/youbike_trough_hourly_2024.parquet",
        "30e2e4381422c0b38062474c6037263c390b0b2c5ec763133ecafc81cea11760",
    ),
    "trough_2025": (
        "youbike_trough_hourly/youbike_trough_hourly_2025.parquet",
        "1532b91fc70e08dc3aa54248435a7b7dc0d3cf955743308fb8558a8225a15ffc",
    ),
    "weather": (
        "weather/weather_taipei_hourly_2024_2025.parquet",
        "c8e7fe27119fc322aa134f6f58d7ece5db7597209dcc622531316c0d83362b2a",
    ),
    "cal_2024": (
        "calendar_2024.csv",
        "0a803eeac7a1fea84c06eda5e7ed6f4fb2f086d0ae494a7af583c44be91544be",
    ),
    "cal_2025": (
        "calendar_2025.csv",
        "9dfb926aa48aeb770e7029b236221f78136885b793037a0e815f4ae2689f2686",
    ),
}

SEMESTER_RAW_RANGES = [
    ("2024-02-19", "2024-06-07"),  # 112-2
    ("2024-09-02", "2024-12-20"),  # 113-1
    ("2025-02-17", "2025-06-06"),  # 113-2
    ("2025-09-01", "2025-12-19"),  # 114-1
]
OPENING_RANGES = [
    ("2024-02-19", "2024-02-23"),
    ("2024-09-02", "2024-09-06"),
    ("2025-02-17", "2025-02-21"),
    ("2025-09-01", "2025-09-05"),
]
MIDTERM_RANGES = [
    ("2024-04-08", "2024-04-12"),
    ("2024-10-21", "2024-10-25"),
    ("2025-04-07", "2025-04-11"),
    ("2025-10-20", "2025-10-24"),
]
FINAL_EXAM_RANGES = [
    ("2024-06-03", "2024-06-07"),
    ("2024-12-16", "2024-12-20"),
    ("2025-06-02", "2025-06-06"),
    ("2025-12-15", "2025-12-19"),
]
CNY_FULL_RANGES = [
    ("2024-02-08", "2024-02-14"),
    ("2025-01-25", "2025-02-02"),
]

LANDMARKS = {
    "sports_center": (25.02166, 121.53608),   # NTU Sports Center (綜合體育館)
    "campus_core": (25.01890, 121.53750),     # Boya / Royal Palm Blvd core
    "mrt_gongguan": (25.01491, 121.53438),    # MRT Gongguan Exit 2
    "mrt_techbldg": (25.02605, 121.54360),    # MRT Technology Building
    "mrt_taipower": (25.02073, 121.52820),    # MRT Taipower Building
    "dorms_east": (25.01637, 121.54535),      # Changxing / Male Dorms
}


def resolve_dssi_root(start: Path | None = None) -> Path:
    cwd = (start or Path.cwd()).resolve()
    for p in (cwd, *cwd.parents):
        if (p / "AGENTS.md").is_file() and (p / "data" / "processed").is_dir():
            return p
    for rel in (
        "../../../../../../../econ-5166-student-kit",
        "../../../../../../econ-5166-student-kit",
    ):
        cand = (cwd / rel).resolve()
        if (cand / "AGENTS.md").is_file() and (cand / "data" / "processed").is_dir():
            return cand
    for p in cwd.parents:
        cand = p / "econ-5166-student-kit"
        if (cand / "AGENTS.md").is_file() and (cand / "data" / "processed").is_dir():
            return cand
    raise FileNotFoundError("Unable to locate `econ-5166-student-kit/` via relative path search.")


def verify_all_inputs(dssi_root: Path) -> dict[str, Path]:
    data_processed = dssi_root / "data" / "processed"
    paths: dict[str, Path] = {}
    for key, (rel_path, expected_sha) in INPUT_SPECS.items():
        full_p = data_processed / rel_path
        if not full_p.is_file():
            raise FileNotFoundError(f"Missing required read-only input: {full_p}")
        actual = hashlib.sha256(full_p.read_bytes()).hexdigest()
        if actual != expected_sha:
            raise ValueError(f"SHA-256 mismatch for {rel_path}: {actual} != {expected_sha}")
        paths[key] = full_p
    return paths


def haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# 2. Enriched Calendar & Derived Data Generation (exclusively in test/derived_data/)
# ---------------------------------------------------------------------------

def _flag_ranges(date_series: pd.Series, ranges: list[tuple[str, str]]) -> np.ndarray:
    mask = np.zeros(len(date_series), dtype=bool)
    for s0, s1 in ranges:
        mask |= (date_series >= s0) & (date_series <= s1)
    return mask.astype(np.int8)


def build_enriched_calendar(input_paths: dict[str, Path], out_csv: Path) -> pd.DataFrame:
    """Build strictly ex-ante enriched 2024-2025 calendar inside test/derived_data/."""
    cal = pd.concat(
        [pd.read_csv(input_paths["cal_2024"]), pd.read_csv(input_paths["cal_2025"])],
        ignore_index=True,
    ).sort_values("date").reset_index(drop=True)
    assert len(cal) == 731 and not cal["date"].duplicated().any()

    dt = pd.to_datetime(cal["date"])
    cal["year"] = dt.dt.year.astype(np.int16)
    cal["month"] = dt.dt.month.astype(np.int8)
    cal["day_of_year"] = dt.dt.dayofyear.astype(np.int16)
    cal["days_in_year"] = np.where(cal["year"] % 4 == 0, 366, 365)

    cal["semester_raw"] = _flag_ranges(cal["date"], SEMESTER_RAW_RANGES)
    cal["opening"] = _flag_ranges(cal["date"], OPENING_RANGES)
    cal["midterm"] = _flag_ranges(cal["date"], MIDTERM_RANGES)
    cal["final_exam"] = _flag_ranges(cal["date"], FINAL_EXAM_RANGES)
    cal["exam"] = np.maximum(cal["midterm"], cal["final_exam"]).astype(np.int8)

    cal["sem_buffer_pre"] = 0
    cal["sem_buffer_post"] = 0
    cal["semester_week"] = 0
    for s0, s1 in SEMESTER_RAW_RANGES:
        t0, t1 = pd.Timestamp(s0), pd.Timestamp(s1)
        m_pre = (dt >= t0 - pd.Timedelta(days=2)) & (dt < t0)
        m_post = (dt > t1) & (dt <= t1 + pd.Timedelta(days=7))
        m_in = (dt >= t0) & (dt <= t1)
        cal.loc[m_pre, "sem_buffer_pre"] = 1
        cal.loc[m_post, "sem_buffer_post"] = 1
        cal.loc[m_in, "semester_week"] = ((dt[m_in] - t0).dt.days // 7 + 1).astype(np.int8)

    assert (cal.loc[cal["semester_raw"] == 1, "semester"] == 1).all()
    assert (cal.loc[cal["opening"] == 1, "semester_raw"] == 1).all()
    assert (cal.loc[cal["exam"] == 1, "semester_raw"] == 1).all()
    assert not ((cal["opening"] == 1) & (cal["exam"] == 1)).any()

    cal["is_winter_break"] = ((cal["semester"] == 0) & (cal["month"].isin([1, 2, 12]))).astype(np.int8)
    cal["is_summer_break"] = ((cal["semester"] == 0) & (cal["month"].isin([6, 7, 8]))).astype(np.int8)
    cal["sem_progress"] = np.where(cal["semester_week"] > 0, (cal["semester_week"] - 1.0) / 15.0, 0.0)

    reason = cal["day_off_reason"].astype(str)
    cal["cal_correction_note"] = ""

    # Derived calendar audit corrections (upstream files remain 100% untouched read-only!):
    # 1) 2024-10-04 (Fri): Taipei resumed normal work & school (Krathon Taipei typhoon leave was
    #    Oct 2-3 only; actual rides on Oct 4 were 25,298 vs 0-546 on real typhoon days). Upstream
    #    calendar_2024.csv miscoded it as weekday=0 / typhoon, which comprised 86% of valid typhoon
    #    rows and flipped the typhoon slope to +42%. Correct cleanly inside derived calendar only.
    m_1004 = cal["date"] == "2024-10-04"
    cal.loc[m_1004, "weekday"] = 1
    cal.loc[m_1004, "cal_correction_note"] = "Taipei resumed normal school/work (Krathon leave was Oct 2-3 only)"

    cal["is_typhoon"] = (reason.str.contains("颱風", regex=False) & (~m_1004)).astype(np.int8)
    cal["is_cny"] = _flag_ranges(cal["date"], CNY_FULL_RANGES)
    cal["is_makeup_workday"] = (reason == "補班").astype(np.int8)
    # 2) 2025-05-01 (Thu): Labor Day was omitted from upstream calendar_2025 day_off_reason.
    m_labor25 = (cal["date"] == "2025-05-01") & (reason != "勞動節")
    cal.loc[m_labor25, "cal_correction_note"] = "Tagged May 1 Labor Day consistently with 2024-05-01"
    cal["is_labor_day"] = ((reason == "勞動節") | (cal["date"].str.endswith("-05-01"))).astype(np.int8)
    # 3) 2025-11-15 (Sat): Upstream used "校慶" vs 2024's "校慶停課"; unify both.
    m_anniv25 = (cal["date"] == "2025-11-15") & (reason == "校慶")
    cal.loc[m_anniv25, "cal_correction_note"] = "Unified NTU Anniversary label with 2024-11-15"
    cal["is_ntu_anniv_off"] = reason.isin(["校慶停課", "校慶"]).astype(np.int8)
    cal["is_national_holiday"] = (
        (cal["weekday"] == 0)
        & (cal["day_of_week"] <= 5)
        & (cal["is_typhoon"] == 0)
        & (cal["is_cny"] == 0)
        & (cal["is_ntu_anniv_off"] == 0)
    ).astype(np.int8)

    off_arr = (cal["weekday"] == 0).to_numpy()
    run_id = np.cumsum(np.concatenate([[True], off_arr[1:] != off_arr[:-1]]))
    run_len = pd.Series(off_arr).groupby(run_id).transform("sum").to_numpy()
    cal["is_long_weekend"] = (off_arr & (run_len >= 3) & (cal["is_typhoon"] == 0) & (cal["is_cny"] == 0)).astype(np.int8)
    cal["is_pre_holiday_workday"] = ((cal["weekday"] == 1) & (cal["weekday"].shift(-1, fill_value=1) == 0)).astype(np.int8)

    cal["is_friday_work"] = ((cal["day_of_week"] == 5) & (cal["weekday"] == 1)).astype(np.int8)
    cal["is_saturday_off"] = ((cal["day_of_week"] == 6) & (cal["weekday"] == 0)).astype(np.int8)
    cal["is_sunday_off"] = ((cal["day_of_week"] == 7) & (cal["weekday"] == 0)).astype(np.int8)

    day_regime = np.zeros(len(cal), dtype=np.int8)
    day_regime[cal["is_friday_work"] == 1] = 1
    day_regime[cal["is_saturday_off"] == 1] = 2
    day_regime[(cal["weekday"] == 0) & (cal["is_saturday_off"] == 0)] = 3
    cal["day_regime_4"] = day_regime

    cal["event_tier1"] = (cal["event_tier"] == 1).astype(np.int8)
    cal["event_tier2"] = (cal["event_tier"] == 2).astype(np.int8)
    scope = cal["event_scope"].astype(str)
    cal["event_campus"] = (scope == "campus").astype(np.int8)
    cal["event_venue"] = (scope == "venue").astype(np.int8)
    cal["event_both"] = (scope == "both").astype(np.int8)
    cal["event_any_venue"] = scope.isin(["venue", "both"]).astype(np.int8)
    cal["event_any_campus"] = scope.isin(["campus", "both"]).astype(np.int8)

    # Strictly ex-ante Taipei Free-30-Min Policy (2024-02-28) & bounded saturation curves
    policy_start = pd.Timestamp("2024-02-28")
    days_since_policy = np.maximum((dt - policy_start).dt.days.to_numpy(dtype=np.float64), 0.0)
    cal["is_free30_policy"] = (dt >= policy_start).astype(np.int8)
    cal["policy_ramp_60d"] = np.where(dt >= policy_start, 1.0 - np.exp(-days_since_policy / 60.0), 0.0)
    cal["policy_ramp_120d"] = np.where(dt >= policy_start, 1.0 - np.exp(-days_since_policy / 120.0), 0.0)
    cal["policy_ramp_180d"] = np.where(dt >= policy_start, 1.0 - np.exp(-days_since_policy / 180.0), 0.0)
    cal["days_from_2024_01_01"] = (dt - pd.Timestamp("2024-01-01")).dt.days.astype(np.int16)

    ang = 2.0 * np.pi * (cal["day_of_year"] - 1.0) / cal["days_in_year"]
    cal["sin_doy1"] = np.sin(ang)
    cal["cos_doy1"] = np.cos(ang)
    cal["sin_doy2"] = np.sin(2.0 * ang)
    cal["cos_doy2"] = np.cos(2.0 * ang)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cal.to_csv(out_csv, index=False)
    return cal


def _build_minute_floored_trough_year(
    year: int, input_paths: dict[str, Path], sid_to_idx: dict[str, int]
) -> pd.DataFrame:
    orig = pd.read_parquet(
        input_paths[f"trough_{year}"],
        columns=["station_id", "datetime", "trough_missing_reason"],
    )
    orig["station_id"] = orig["station_id"].astype(str)
    orig = orig.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    r = pd.read_parquet(
        input_paths[f"rental_{year}"],
        columns=["StationID", "SrcUpdateTime", "AvailableRentBikes", "AvailableReturnBikes", "ServiceStatus"],
    )
    r["StationID"] = r["StationID"].astype(str)
    r["min_ts"] = r["SrcUpdateTime"].dt.tz_localize(None).dt.floor("min")
    r = r.sort_values(["StationID", "SrcUpdateTime"]).drop_duplicates(["StationID", "min_ts"], keep="last")

    y0 = pd.Timestamp(f"{year}-01-01 00:00:00")
    n_hours = 8784 if year % 4 == 0 else 8760
    r["m_int"] = ((r["min_ts"] - y0).dt.total_seconds() // 60).astype(np.int32)
    r["is_full"] = (r["AvailableReturnBikes"] == 0).astype(np.int8)
    r["is_stopped"] = (r["ServiceStatus"] == "停止營運").astype(np.int8)

    hour_m = np.arange(n_hours, dtype=np.int32) * 60
    n_st = len(sid_to_idx)
    out_trough = np.full((n_st, n_hours), np.nan, dtype=np.float64)
    out_exp_rent = np.full((n_st, n_hours), np.nan, dtype=np.float64)
    out_exp_ret = np.full((n_st, n_hours), np.nan, dtype=np.float64)
    out_min_stop = np.full((n_st, n_hours), np.nan, dtype=np.float64)
    out_n_upd = np.zeros((n_st, n_hours), dtype=np.int32)

    for sid, grp in r.groupby("StationID", sort=False):
        si = sid_to_idx[sid]
        m_arr = grp["m_int"].to_numpy(dtype=np.int32)
        b_arr = grp["AvailableRentBikes"].to_numpy(dtype=np.int32)
        f_arr = grp["is_full"].to_numpy(dtype=np.int8)
        st_arr = grp["is_stopped"].to_numpy(dtype=np.int8)

        snap_idx = np.searchsorted(m_arr, hour_m, side="right") - 1
        u_start = snap_idx + 1
        u_end = np.searchsorted(m_arr, hour_m + 60, side="left")
        out_n_upd[si] = u_end - u_start

        valid_h = snap_idx >= 0
        no_upd = valid_h & (u_start == u_end)
        idx0 = snap_idx[no_upd]
        out_trough[si, no_upd] = b_arr[idx0]
        out_exp_rent[si, no_upd] = 0.0
        out_exp_ret[si, no_upd] = (1.0 - f_arr[idx0]) * 60.0
        out_min_stop[si, no_upd] = st_arr[idx0] * 60.0

        for h in np.flatnonzero(valid_h & (u_start < u_end)):
            s0, us, ue = snap_idx[h], u_start[h], u_end[h]
            h0, h1 = h * 60, (h + 1) * 60
            k = ue - us
            durs = np.empty(k + 1, dtype=np.int32)
            durs[0] = m_arr[us] - h0
            if k > 1:
                durs[1:k] = m_arr[us + 1 : ue] - m_arr[us : ue - 1]
            durs[k] = h1 - m_arr[ue - 1]

            bs = np.empty(k + 1, dtype=np.int32)
            bs[0] = b_arr[s0]
            bs[1:] = b_arr[us:ue]

            fs = np.empty(k + 1, dtype=np.int8)
            fs[0] = f_arr[s0]
            fs[1:] = f_arr[us:ue]

            sts = np.empty(k + 1, dtype=np.int8)
            sts[0] = st_arr[s0]
            sts[1:] = st_arr[us:ue]

            tb = int(bs.min())
            out_trough[si, h] = tb
            out_exp_rent[si, h] = 60 - int(durs[bs == tb].sum())
            out_exp_ret[si, h] = 60 - int(durs[fs == 1].sum())
            out_min_stop[si, h] = int(durs[sts == 1].sum())

    res = pd.DataFrame({
        "station_id": orig["station_id"],
        "datetime": orig["datetime"],
        "trough_missing_reason": orig["trough_missing_reason"],
        "trough_bikes": out_trough.ravel(),
        "exp_rent_eff": out_exp_rent.ravel(),
        "exp_ret_eff": out_exp_ret.ravel(),
        "minutes_stopped": out_min_stop.ravel(),
        "n_updates": out_n_upd.ravel(),
    })
    miss = res["trough_missing_reason"].notna()
    for c in ["trough_bikes", "exp_rent_eff", "exp_ret_eff", "minutes_stopped"]:
        res.loc[miss, c] = np.nan

    ok = ~miss
    assert np.all(np.isclose(res.loc[ok, "exp_rent_eff"], np.round(res.loc[ok, "exp_rent_eff"])))
    assert np.all(np.isclose(res.loc[ok, "exp_ret_eff"], np.round(res.loc[ok, "exp_ret_eff"])))
    return res


def build_or_load_derived_datasets(
    test_dir: Path, force_rebuild: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dssi_root = resolve_dssi_root(test_dir)
    input_paths = verify_all_inputs(dssi_root)

    derived_dir = test_dir / "derived_data"
    derived_dir.mkdir(parents=True, exist_ok=True)
    cal_path = derived_dir / "calendar_enriched_2024_2025.csv"
    panel_path = derived_dir / "panel_minute_floored_2024_2025.parquet"
    stn_path = derived_dir / "station_continuous_features.csv"

    stations_raw = pd.read_csv(input_paths["stations"], dtype={"StationUID": str, "StationID": str})
    stations_meta = (
        stations_raw.sort_values("last_seen")
        .drop_duplicates("StationID", keep="last")
        .assign(station_name_short=lambda d: d["StationNameZh_tw"].str.removeprefix("YouBike2.0_"))
        [["StationID", "StationUID", "station_name_short", "PositionLat", "PositionLon", "first_seen", "last_seen"]]
        .rename(columns={"StationID": "station_id"})
        .sort_values("station_id")
        .reset_index(drop=True)
    )
    assert len(stations_meta) == 120
    sid_to_idx = {s: i for i, s in enumerate(stations_meta["station_id"].tolist())}

    if force_rebuild or not (cal_path.is_file() and panel_path.is_file() and stn_path.is_file()):
        t0 = time.perf_counter()
        cal = build_enriched_calendar(input_paths, cal_path)

        tr24 = _build_minute_floored_trough_year(2024, input_paths, sid_to_idx)
        tr25 = _build_minute_floored_trough_year(2025, input_paths, sid_to_idx)
        trough = pd.concat([tr24, tr25], ignore_index=True)

        flow = pd.read_parquet(
            input_paths["flow_hourly"],
            columns=["station_id", "datetime", "rent_total", "rent_invalid", "return_total", "return_invalid"],
        )
        flow["station_id"] = flow["station_id"].astype(str)
        panel = flow.merge(trough, on=["station_id", "datetime"], how="inner", validate="one_to_one")
        del flow, trough

        wx = pd.read_parquet(input_paths["weather"]).sort_values("datetime").reset_index(drop=True)
        wx["dt_ts"] = wx["datetime"].dt.tz_localize(None) - pd.Timedelta(hours=1)
        wx["dt_aligned"] = wx["dt_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
        wx["date_w"] = wx["dt_ts"].dt.strftime("%Y-%m-%d")
        wx["hour_w"] = wx["dt_ts"].dt.hour

        wx["rain_1h_fill0"] = wx["rain_1h_mm"].fillna(0.0)
        wx["rain_lag1"] = wx["rain_1h_fill0"].shift(1).fillna(0.0)
        wx["rain_lag2"] = wx["rain_1h_fill0"].shift(2).fillna(0.0)
        wx["rain_3h_sum"] = wx["rain_1h_fill0"] + wx["rain_lag1"] + wx["rain_lag2"]
        wx["log1p_Precip"] = np.log1p(wx["rain_1h_mm"])
        wx["log1p_rain_lag1"] = np.log1p(wx["rain_lag1"])
        wx["log1p_rain_lag2"] = np.log1p(wx["rain_lag2"])
        wx["log1p_rain_3h"] = np.log1p(wx["rain_3h_sum"])
        wx["log1p_D24R"] = np.log1p(wx["D24R"].clip(lower=0.0).fillna(0.0))
        wx["is_rain"] = (wx["rain_1h_mm"] > 0.0).astype(np.float32)
        wx["is_heavy_rain"] = (wx["rain_1h_mm"] >= 5.0).astype(np.float32)
        wx["is_trace_rain"] = wx["D24R_trace"].fillna(False).astype(np.float32)

        morn = wx[wx["hour_w"].isin([7, 8])].groupby("date_w")["rain_1h_fill0"].sum().rename("morn_rain_raw")
        wx = wx.merge(morn, on="date_w", how="left")
        wx["morning_rain_causal"] = np.where(wx["hour_w"] >= 9, np.log1p(wx["morn_rain_raw"].fillna(0.0)), 0.0)

        wx["temp_cold_hinge"] = np.maximum(0.0, 18.0 - wx["Temp"])
        wx["temp_hot_hinge"] = np.maximum(0.0, wx["Temp"] - 28.0)
        bsg_cold, bsg_hot = compute_temp_bsg_features(wx["Temp"])
        wx["temp_bsg_cold"] = bsg_cold
        wx["temp_bsg_hot"] = bsg_hot
        wx["temp_x_humd"] = (wx["Temp"] - 24.0) * (wx["HUMD"] - 0.75)

        wx_cols = [
            "dt_aligned", "Temp", "HUMD", "rain_1h_mm", "D24R",
            "log1p_Precip", "log1p_rain_lag1", "log1p_rain_lag2", "log1p_rain_3h",
            "log1p_D24R", "is_rain", "is_heavy_rain", "is_trace_rain",
            "morning_rain_causal", "temp_cold_hinge", "temp_hot_hinge",
            "temp_bsg_cold", "temp_bsg_hot", "temp_x_humd",
        ]
        panel = panel.merge(wx[wx_cols], left_on="datetime", right_on="dt_aligned", how="left", validate="many_to_one").drop(columns=["dt_aligned"])

        panel["date"] = panel["datetime"].str[:10]
        panel["hour"] = panel["datetime"].str[11:13].astype(np.int8)
        panel["y_rent"] = (panel["rent_total"] - panel["rent_invalid"]).astype(np.float64)
        panel["y_ret"] = (panel["return_total"] - panel["return_invalid"]).astype(np.float64)
        panel = panel.drop(columns=["rent_total", "rent_invalid", "return_total", "return_invalid"])

        cal_merge_cols = [c for c in cal.columns if c not in {"day_off_reason", "event_name", "cal_correction_note"}]
        panel = panel.merge(cal[cal_merge_cols], on="date", how="left", validate="many_to_one")

        hr_agg = panel.groupby(["year", "date", "datetime", "hour"], as_index=False).agg(
            tot_rent=("y_rent", "sum"),
            tot_upd=("n_updates", "sum"),
            n_usable=("trough_bikes", lambda s: int(s.notna().sum())),
            n_nodata=("trough_missing_reason", lambda s: int((s == "no-data day").sum())),
        )
        nodata_dates = set(hr_agg.loc[hr_agg["n_nodata"] > 50, "date"])
        blackout_mask_hr = (
            ((hr_agg["n_usable"] > 0) & (hr_agg["tot_upd"] == 0) & (hr_agg["tot_rent"] >= 50))
            | ((hr_agg["n_usable"] > 0) & (hr_agg["tot_rent"] >= 100) & (hr_agg["tot_upd"] / np.maximum(hr_agg["tot_rent"], 1) < 0.05))
            | ((hr_agg["date"].isin(nodata_dates)) & (hr_agg["hour"] == 0))
        )
        blackout_dts = set(hr_agg.loc[blackout_mask_hr, "datetime"])
        panel["is_tdx_blackout"] = panel["datetime"].isin(blackout_dts)
        panel["is_full_normal_op"] = panel["trough_bikes"].notna() & (panel["minutes_stopped"] == 0)
        panel["has_complete_weather"] = panel["Temp"].notna() & panel["HUMD"].notna() & panel["rain_1h_mm"].notna()

        base_ok = panel["is_full_normal_op"] & (~panel["is_tdx_blackout"]) & panel["has_complete_weather"]
        panel["valid_rent"] = base_ok & (panel["exp_rent_eff"] >= 1.0)
        panel["valid_ret"] = base_ok & (panel["exp_ret_eff"] >= 1.0)

        assert int(panel.loc[panel["year"] == 2024, "valid_rent"].sum()) == 753991
        assert int(panel.loc[panel["year"] == 2024, "valid_ret"].sum()) == 912531
        assert int(panel.loc[panel["year"] == 2025, "valid_rent"].sum()) == 750910
        assert int(panel.loc[panel["year"] == 2025, "valid_ret"].sum()) == 997627

        stn_df = build_continuous_station_features(stations_meta, panel[panel["year"] == 2024])
        stn_df.to_csv(stn_path, index=False)

        panel = panel.drop(columns=["trough_missing_reason"])
        panel.to_parquet(panel_path, index=False)
        info = {"rebuilt": True, "elapsed_sec": round(time.perf_counter() - t0, 2), "dssi_root": dssi_root.name}
    else:
        cal = pd.read_csv(cal_path)
        panel = pd.read_parquet(panel_path)
        if "temp_bsg_cold" not in panel.columns or "temp_bsg_hot" not in panel.columns:
            bsg_cold, bsg_hot = compute_temp_bsg_features(panel["Temp"])
            panel["temp_bsg_cold"] = bsg_cold
            panel["temp_bsg_hot"] = bsg_hot
            panel.to_parquet(panel_path, index=False)
        stn_df = pd.read_csv(stn_path, dtype={"station_id": str})
        info = {"rebuilt": False, "elapsed_sec": 0.0, "dssi_root": dssi_root.name}

    return panel, cal, stn_df, info


# Bounded Asymmetric Super-Gaussian (2b=4) temperature basis — calibrated strictly on 2024 data
TEMP_BSG_T0 = 24.0
TEMP_BSG_SIGMA_COLD = 13.0
TEMP_BSG_SIGMA_HOT = 11.0


def compute_temp_bsg_features(temp: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Bounded Asymmetric Super-Gaussian (2b=4) temperature response basis:
      b_cold(T) = 1 - exp(-((max(T0 - T, 0) / sigma_L)^4))  in [0, 1)
      b_hot(T)  = 1 - exp(-((max(T - T0, 0) / sigma_R)^4))  in [0, 1)
    Properties:
      - C^3 flat comfort plateau around T0 = 24.0C (f'(T0)=f''(T0)=f'''(T0)=0)
      - Steepest physical inflection at 11.9C (cold snap) and 34.2C (heatwave)
      - Strictly bounded in [0, 1): zero risk of unseen-extreme tail-cliff collapse!
    """
    t = np.asarray(temp, dtype=np.float64)
    bsg_cold = np.where(
        np.isfinite(t),
        1.0 - np.exp(-((np.maximum(0.0, TEMP_BSG_T0 - np.nan_to_num(t, nan=TEMP_BSG_T0)) / TEMP_BSG_SIGMA_COLD) ** 4)),
        np.nan,
    )
    bsg_hot = np.where(
        np.isfinite(t),
        1.0 - np.exp(-((np.maximum(0.0, np.nan_to_num(t, nan=TEMP_BSG_T0) - TEMP_BSG_T0) / TEMP_BSG_SIGMA_HOT) ** 4)),
        np.nan,
    )
    return bsg_cold, bsg_hot


def build_continuous_station_features(stations_meta: pd.DataFrame, train_2024: pd.DataFrame) -> pd.DataFrame:
    st = stations_meta.copy()
    lat = st["PositionLat"].to_numpy()
    lon = st["PositionLon"].to_numpy()

    for name, (la, lo) in LANDMARKS.items():
        st[f"dist_{name}_m"] = haversine_m(lat, lon, la, lo)
    st["dist_nearest_mrt_m"] = np.minimum.reduce([
        st["dist_mrt_gongguan_m"], st["dist_mrt_techbldg_m"], st["dist_mrt_taipower_m"]
    ])

    st["kern_venue_300"] = np.exp(-st["dist_sports_center_m"] / 300.0)
    st["kern_venue_600"] = np.exp(-st["dist_sports_center_m"] / 600.0)
    st["kern_campus_450"] = np.exp(-st["dist_campus_core_m"] / 450.0)
    st["kern_mrt_350"] = np.exp(-st["dist_nearest_mrt_m"] / 350.0)
    st["kern_dorm_400"] = np.exp(-st["dist_dorms_east_m"] / 400.0)

    tr_r = train_2024[train_2024["valid_rent"]]
    tr_f = train_2024[train_2024["valid_ret"]]
    all_sids = st["station_id"].tolist()
    active_sids = sorted(set(tr_r["station_id"].unique()) | set(tr_f["station_id"].unique()))
    active_mask = st["station_id"].isin(active_sids).to_numpy()

    piv_r = (
        tr_r[tr_r["weekday"] == 1].groupby(["station_id", "hour"]).apply(
            lambda g: g["y_rent"].sum() / max(g["exp_rent_eff"].sum(), 1.0), include_groups=False
        ).unstack(fill_value=0.0).reindex(index=active_sids, columns=range(24), fill_value=0.0)
    )
    piv_f = (
        tr_f[tr_f["weekday"] == 1].groupby(["station_id", "hour"]).apply(
            lambda g: g["y_ret"].sum() / max(g["exp_ret_eff"].sum(), 1.0), include_groups=False
        ).unstack(fill_value=0.0).reindex(index=active_sids, columns=range(24), fill_value=0.0)
    )

    mat_raw = np.hstack([piv_r.to_numpy(), piv_f.to_numpy()])
    row_sums = np.maximum(mat_raw.sum(axis=1, keepdims=True), 1e-6)
    mat_norm = mat_raw / row_sums
    mat_centered = mat_norm - mat_norm.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(mat_centered, full_matrices=False)

    act_df = pd.DataFrame({"station_id": active_sids})
    for k in range(4):
        v = U[:, k] * S[k]
        v = (v - v.mean()) / max(v.std(), 1e-8)
        act_df[f"svd_func_{k+1}"] = v

    net = piv_r.to_numpy() - piv_f.to_numpy()
    am_net = net[:, [7, 8, 9]].mean(axis=1)
    pm_net = net[:, [17, 18, 19]].mean(axis=1)
    tot_scale = np.maximum((piv_r.to_numpy() + piv_f.to_numpy()).mean(axis=1), 1e-4)
    tidal = (am_net - pm_net) / tot_scale
    act_df["tidal_index_2024"] = (tidal - tidal.mean()) / max(tidal.std(), 1e-8)

    wd_r = tr_r.groupby(["station_id", "weekday"]).apply(
        lambda g: g["y_rent"].sum() / max(g["exp_rent_eff"].sum(), 1.0), include_groups=False
    ).unstack(fill_value=1e-4).reindex(active_sids, fill_value=1e-4)
    log_wd_ratio = np.log(np.maximum(wd_r[1].to_numpy(), 1e-4) / np.maximum(wd_r[0].to_numpy(), 1e-4))
    act_df["log_wd_we_ratio_2024"] = (log_wd_ratio - log_wd_ratio.mean()) / max(log_wd_ratio.std(), 1e-8)

    vol_act = tr_r.groupby("station_id")["y_rent"].sum().reindex(active_sids, fill_value=0.0).to_numpy()
    lv_act = np.log1p(vol_act)
    act_df["log_vol_2024"] = lv_act
    act_df["log_vol_2024_z"] = (lv_act - lv_act.mean()) / max(lv_act.std(), 1e-8)

    st = st.merge(act_df, on="station_id", how="left")
    st["is_active_2024"] = active_mask.astype(np.int8)

    # Smooth geographic Gaussian-kernel interpolation for cold-start stations (zero hard zones!)
    if not np.all(active_mask):
        interp_cols = ["svd_func_1", "svd_func_2", "svd_func_3", "svd_func_4",
                       "tidal_index_2024", "log_wd_we_ratio_2024", "log_vol_2024", "log_vol_2024_z"]
        act_lat = st.loc[active_mask, "PositionLat"].to_numpy()
        act_lon = st.loc[active_mask, "PositionLon"].to_numpy()
        act_vals = st.loc[active_mask, interp_cols].to_numpy(dtype=np.float64)
        cold_idx = np.flatnonzero(~active_mask)
        c_lat = st.loc[~active_mask, "PositionLat"].to_numpy()
        c_lon = st.loc[~active_mask, "PositionLon"].to_numpy()
        d_mat = haversine_m(c_lat[:, None], c_lon[:, None], act_lat[None, :], act_lon[None, :])
        W_c = np.exp(-0.5 * (d_mat / 320.0) ** 2)
        W_c = W_c / np.maximum(W_c.sum(axis=1, keepdims=True), 1e-12)
        interp_vals = W_c @ act_vals
        for j, col_nm in enumerate(interp_cols):
            st.iloc[cold_idx, st.columns.get_loc(col_nm)] = interp_vals[:, j]

    return st


def build_spatial_laplacian(
    stn_df: pd.DataFrame, station_ids: list[str], length_scale_m: float = 320.0
) -> tuple[np.ndarray, np.ndarray]:
    """Geographic continuous Gaussian Graph Laplacian L_geo (for radial crowd / weather diffusion)."""
    sub = stn_df.set_index("station_id").loc[station_ids]
    lat = sub["PositionLat"].to_numpy()
    lon = sub["PositionLon"].to_numpy()
    dist_mat = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    W = np.exp(-0.5 * (dist_mat / length_scale_m) ** 2)
    np.fill_diagonal(W, 0.0)
    mean_deg = max(float(W.sum(axis=1).mean()), 1e-6)
    W_norm = W / mean_deg
    L = np.diag(W_norm.sum(axis=1)) - W_norm
    return W_norm, L


def build_functional_laplacian(
    stn_df: pd.DataFrame, station_ids: list[str], sigma: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Continuous Functional-Profile Graph Laplacian L_func (built purely from 2024 SVD + tidal axes,
    ZERO hard regional boundaries!). Ensures functionally similar stations (e.g., two portals or
    two teaching hubs) smoothly share tidal/semester slope strength without bleeding across
    adjacent opposite-polarity stations (e.g., dorm vs classroom 250m apart).
    """
    cols = ["svd_func_1", "svd_func_2", "svd_func_3", "tidal_index_2024", "log_wd_we_ratio_2024"]
    sub = stn_df.set_index("station_id").loc[station_ids, cols].to_numpy(dtype=np.float64)
    diff = sub[:, None, :] - sub[None, :, :]
    sq_dist = np.sum(diff ** 2, axis=-1)
    W = np.exp(-0.5 * sq_dist / (sigma ** 2))
    np.fill_diagonal(W, 0.0)
    mean_deg = max(float(W.sum(axis=1).mean()), 1e-6)
    W_norm = W / mean_deg
    L = np.diag(W_norm.sum(axis=1)) - W_norm
    return W_norm, L


# ---------------------------------------------------------------------------
# 3. Generalized Penalized & Weighted Schur-Complement NB2-FE / Poisson-FE Solver
# ---------------------------------------------------------------------------

def nb2_loglik(y: np.ndarray, mu: np.ndarray, alpha: float, obs_w: np.ndarray | None = None) -> float:
    mu_safe = np.maximum(mu, 1e-300)
    if alpha <= 0.0:
        ll_i = y * np.log(mu_safe) - mu_safe - special.gammaln(y + 1.0)
    else:
        r = 1.0 / alpha
        ll_i = (
            special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1.0)
            + r * np.log(r / (r + mu_safe))
            + y * np.log(mu_safe / (r + mu_safe))
        )
    if obs_w is not None:
        return float(np.sum(obs_w * ll_i))
    return float(np.sum(ll_i))


def fit_penalized_glm_fe_core(
    y: np.ndarray,
    X: sparse.csc_matrix,
    offset: np.ndarray,
    g_idx: np.ndarray,
    G: int,
    alpha: float,
    Z: sparse.csc_matrix,
    P_theta: np.ndarray | None = None,
    obs_w: np.ndarray | None = None,
    fe_shrink_lambda: float = 0.0,
    psi_prior: np.ndarray | None = None,
    theta_init: np.ndarray | None = None,
    psi_init: np.ndarray | None = None,
    max_iter: int = 35,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    N, K = X.shape
    w_obs = np.ones(N, dtype=np.float64) if obs_w is None else obs_w.astype(np.float64)

    if K > 0:
        col_norm = np.sqrt(np.asarray(X.power(2).mean(axis=0)).ravel())
        x_scale = np.where(col_norm > 1e-12, col_norm, 1.0)
        Xs = X @ sparse.diags(1.0 / x_scale)
        P_scaled = (
            (P_theta / x_scale[:, None]) / x_scale[None, :]
            if P_theta is not None
            else np.zeros((K, K), dtype=np.float64)
        )
        theta = np.zeros(K, dtype=np.float64) if theta_init is None else (theta_init * x_scale).copy()
    else:
        x_scale = np.empty(0, dtype=np.float64)
        Xs = X
        P_scaled = np.zeros((0, 0), dtype=np.float64)
        theta = np.empty(0, dtype=np.float64)

    if psi_init is None:
        sum_y = np.asarray(Z.T @ (w_obs * y)).ravel()
        sum_e = np.asarray(Z.T @ (w_obs * np.exp(offset))).ravel()
        psi = np.log(np.maximum(sum_y, 1e-6) / np.maximum(sum_e, 1e-6))
    else:
        psi = psi_init.copy()

    if psi_prior is None:
        psi_prior = np.zeros(G, dtype=np.float64)

    def eval_obj(mu_arr: np.ndarray, th_arr: np.ndarray, ps_arr: np.ndarray) -> float:
        val = nb2_loglik(y, mu_arr, alpha, w_obs)
        if K > 0 and P_theta is not None:
            val -= 0.5 * float(th_arr @ (P_scaled @ th_arr))
        if fe_shrink_lambda > 0.0:
            diff = ps_arr - psi_prior
            val -= 0.5 * fe_shrink_lambda * float(np.dot(diff, diff))
        return val

    eta = offset + psi[g_idx] + (Xs @ theta if K > 0 else 0.0)
    mu = np.exp(np.clip(eta, -25.0, 18.0))
    obj = eval_obj(mu, theta, psi)

    for it in range(max_iter):
        denom = 1.0 + alpha * mu if alpha > 0.0 else np.ones_like(mu)
        score = w_obs * (y - mu) / denom
        w = w_obs * mu / denom
        w_g = np.asarray(Z.T @ w).ravel() + fe_shrink_lambda
        inv_wg = 1.0 / np.maximum(w_g, 1e-12)

        sum_s_g = np.asarray(Z.T @ score).ravel()
        if fe_shrink_lambda > 0.0:
            sum_s_g = sum_s_g - fe_shrink_lambda * (psi - psi_prior)

        if K > 0:
            WXs = sparse.diags(w) @ Xs
            XtWX = (Xs.T @ WXs).toarray()
            sum_WX = (Z.T @ WXs).toarray()
            H = XtWX - (sum_WX.T * inv_wg) @ sum_WX + P_scaled
            H.flat[:: K + 1] += 1e-9

            grad_theta = np.asarray(Xs.T @ score).ravel() - sum_WX.T @ (sum_s_g * inv_wg)
            if P_theta is not None:
                grad_theta = grad_theta - P_scaled @ theta

            try:
                d_theta = np.linalg.solve(H, grad_theta)
            except np.linalg.LinAlgError:
                d_theta = np.linalg.lstsq(H, grad_theta, rcond=None)[0]
            d_psi = (sum_s_g - sum_WX @ d_theta) * inv_wg
        else:
            d_theta = np.empty(0, dtype=np.float64)
            d_psi = sum_s_g * inv_wg

        max_step = max(
            float(np.max(np.abs(d_theta))) if K > 0 else 0.0,
            float(np.max(np.abs(d_psi))),
        )
        step = min(1.0, 5.0 / max(max_step, 1e-12))
        accepted = False
        for _ in range(18):
            th_n = theta + step * d_theta
            ps_n = psi + step * d_psi
            eta_n = offset + ps_n[g_idx] + (Xs @ th_n if K > 0 else 0.0)
            mu_n = np.exp(np.clip(eta_n, -25.0, 18.0))
            obj_n = eval_obj(mu_n, th_n, ps_n)
            if np.isfinite(obj_n) and obj_n >= obj - 1e-8:
                accepted = True
                break
            step *= 0.5

        if not accepted:
            break

        max_d = max(
            float(np.max(np.abs(step * d_theta))) if K > 0 else 0.0,
            float(np.max(np.abs(step * d_psi))),
        )
        theta, psi, eta, mu, obj = th_n, ps_n, eta_n, mu_n, obj_n
        if max_d < tol:
            break

    theta_out = theta / x_scale if K > 0 else np.empty(0, dtype=np.float64)
    ll_unpen = nb2_loglik(y, mu, alpha, w_obs)
    return theta_out, psi, mu, ll_unpen, it + 1


# ---------------------------------------------------------------------------
# 4. Evaluation Metrics Suite
# ---------------------------------------------------------------------------

def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu_s = np.maximum(mu, 1e-12)
    term = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu_s) - (y - mu_s), mu_s)
    return float(2.0 * np.mean(term))


def nb2_deviance(y: np.ndarray, mu: np.ndarray, alpha: float = 0.20) -> float:
    mu_s = np.maximum(mu, 1e-12)
    r = 1.0 / alpha
    t1 = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu_s), 0.0)
    t2 = (y + r) * np.log((y + r) / (mu_s + r))
    return float(2.0 * np.mean(t1 - t2))


def evaluate_predictions(
    df: pd.DataFrame,
    y_col: str,
    exp_col: str,
    pred_rate: np.ndarray,
    model_name: str,
    side: str,
    split: str,
    alpha_ref: float = 0.20,
) -> dict[str, Any]:
    y = df[y_col].to_numpy(dtype=np.float64)
    exp = df[exp_col].to_numpy(dtype=np.float64)
    obs_rate = y / exp
    pr = np.clip(np.asarray(pred_rate, dtype=np.float64), 0.0, 500.0)
    pred_cnt = pr * exp

    w_sum = float(np.sum(exp))
    err_r = obs_rate - pr
    wmse = float(np.sum(exp * err_r ** 2) / w_sum)
    wrmse = math.sqrt(wmse)
    wmae = float(np.sum(exp * np.abs(err_r)) / w_sum)
    ybar = float(np.sum(exp * obs_rate) / w_sum)
    wvar = float(np.sum(exp * (obs_rate - ybar) ** 2) / w_sum)
    wr2 = 1.0 - wmse / max(wvar, 1e-12)

    rmse_cnt = math.sqrt(float(np.mean((y - pred_cnt) ** 2)))
    mae_cnt = float(np.mean(np.abs(y - pred_cnt)))
    r2_cnt = 1.0 - float(np.sum((y - pred_cnt) ** 2)) / max(float(np.sum((y - y.mean()) ** 2)), 1e-12)
    bias_pct = (float(np.sum(pred_cnt)) / max(float(np.sum(y)), 1e-12) - 1.0) * 100.0

    align_scale = float(np.sum(y)) / max(float(np.sum(pred_cnt)), 1e-12)
    pr_al = pr * align_scale
    wrmse_aligned = math.sqrt(float(np.sum(exp * (obs_rate - pr_al) ** 2) / w_sum))

    def slice_wrmse(mask: np.ndarray) -> float:
        if not np.any(mask):
            return np.nan
        ew = exp[mask]
        return math.sqrt(float(np.sum(ew * (obs_rate[mask] - pr[mask]) ** 2) / np.sum(ew)))

    peak_mask = (df["weekday"].to_numpy() == 1) & (df["hour"].isin([7, 8, 9, 17, 18, 19]).to_numpy())
    rain_mask = df["rain_1h_mm"].to_numpy() >= 1.0
    event_mask = df["event"].to_numpy() == 1
    exam_mask = df["exam"].to_numpy() == 1

    return {
        "split": split,
        "side": side,
        "model": model_name,
        "N": int(len(y)),
        "wRMSE_rate": wrmse,
        "wMAE_rate": wmae,
        "wR2_rate": wr2,
        "wRMSE_shape_aligned": wrmse_aligned,
        "RMSE_count": rmse_cnt,
        "MAE_count": mae_cnt,
        "R2_count": r2_cnt,
        "Pois_Dev": poisson_deviance(y, pred_cnt),
        "NB2_Dev": nb2_deviance(y, pred_cnt, alpha=alpha_ref),
        "Bias_pct": bias_pct,
        "Peak_wRMSE": slice_wrmse(peak_mask),
        "Rain_wRMSE": slice_wrmse(rain_mask),
        "Event_wRMSE": slice_wrmse(event_mask),
        "Exam_wRMSE": slice_wrmse(exam_mask),
    }


def apply_causal_trailing_level_calibration(
    df_test: pd.DataFrame,
    pred_rate_raw: np.ndarray,
    y_col: str,
    exp_col: str,
    window_days: int = 14,
    prior_days: float = 3.0,
) -> np.ndarray:
    """
    Strictly causal trailing operational level tracker (uses ONLY strictly past days < d),
    with Empirical-Bayes shrinkage toward 1.0 (prior_days) to prevent Day-2 single-holiday shocks.
    """
    dates = df_test["date"].to_numpy()
    uniq_dates = np.sort(np.unique(dates))
    y = df_test[y_col].to_numpy(dtype=np.float64)
    exp = df_test[exp_col].to_numpy(dtype=np.float64)
    pred_cnt = pred_rate_raw * exp

    day_df = pd.DataFrame({"date": dates, "y": y, "p": pred_cnt}).groupby("date", sort=True).sum()
    y_d = day_df["y"].to_numpy()
    p_d = day_df["p"].to_numpy()

    mult_by_date: dict[str, float] = {}
    for idx, d_str in enumerate(uniq_dates):
        if idx == 0:
            mult_by_date[d_str] = 1.0
        else:
            s0 = max(0, idx - window_days)
            n_hist = idx - s0
            sy = float(np.sum(y_d[s0:idx]))
            sp = float(np.sum(p_d[s0:idx]))
            mean_sp = sp / max(n_hist, 1)
            prior_mass = prior_days * mean_sp
            mult_by_date[d_str] = float(np.clip((sy + prior_mass) / max(sp + prior_mass, 1e-6), 0.5, 2.5))

    g_mult = pd.Series(dates).map(mult_by_date).to_numpy(dtype=np.float64)
    return pred_rate_raw * g_mult


# ---------------------------------------------------------------------------
# 5. Feature Matrix Builders
# ---------------------------------------------------------------------------

def _hour_dummies(hour: np.ndarray, mask: np.ndarray) -> sparse.csc_matrix:
    idx = np.flatnonzero(mask != 0)
    if len(idx) == 0:
        return sparse.csc_matrix((len(hour), 24), dtype=np.float64)
    return sparse.csc_matrix((mask[idx].astype(np.float64), (idx, hour[idx].astype(np.int32))), shape=(len(hour), 24), dtype=np.float64)


def _diurnal_block(hour: np.ndarray) -> np.ndarray:
    blk = np.zeros(len(hour), dtype=np.int8)
    blk[(hour >= 7) & (hour <= 11)] = 1
    blk[(hour >= 12) & (hour <= 16)] = 2
    blk[hour >= 17] = 3
    return blk


def _cold_start_geo_weights(
    stn_df: pd.DataFrame,
    cold_sids: list[str],
    train_station_order: list[str],
    length_scale_m: float = 320.0,
) -> np.ndarray:
    """Continuous Gaussian geographic interpolation weights from cold-start stations to training stations."""
    st_idx = stn_df.set_index("station_id")
    c_lat = st_idx.loc[cold_sids, "PositionLat"].to_numpy(dtype=np.float64)
    c_lon = st_idx.loc[cold_sids, "PositionLon"].to_numpy(dtype=np.float64)
    t_lat = st_idx.loc[train_station_order, "PositionLat"].to_numpy(dtype=np.float64)
    t_lon = st_idx.loc[train_station_order, "PositionLon"].to_numpy(dtype=np.float64)
    d_mat = haversine_m(c_lat[:, None], c_lon[:, None], t_lat[None, :], t_lon[None, :])
    W = np.exp(-0.5 * (d_mat / length_scale_m) ** 2)
    return W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)


def build_feature_matrix(
    df: pd.DataFrame,
    spec: str,
    stn_df: pd.DataFrame,
    station_order: list[str],
    L_geo: np.ndarray | None = None,
    L_func: np.ndarray | None = None,
    lam_ridge: float = 25.0,
    lam_lap: float = 80.0,
    policy_ramp_override: str | None = None,
    include_policy_step: bool | None = None,
) -> tuple[sparse.csc_matrix, list[str], np.ndarray | None]:
    N = len(df)
    hr = df["hour"].to_numpy(dtype=np.int32)
    blocks: list[sparse.csc_matrix] = []
    names: list[str] = []
    pen_blocks: list[np.ndarray] = []

    def add_dense_cols(cols: list[tuple[str, np.ndarray]], ridge: float = 1e-6) -> None:
        if not cols:
            return
        mat = np.column_stack([c[1].astype(np.float64) for c in cols])
        blocks.append(sparse.csc_matrix(mat))
        names.extend([c[0] for c in cols])
        pen_blocks.append(np.eye(len(cols)) * ridge)

    def add_hour_block(prefix: str, mask: np.ndarray, ridge: float = 1e-4) -> None:
        blocks.append(_hour_dummies(hr, mask))
        names.extend([f"{prefix}_h{h:02d}" for h in range(24)])
        pen_blocks.append(np.eye(24) * ridge)

    # ---- Stage 1 Single-Factor Specs ----
    if spec == "S0_BaseFE":
        return sparse.csc_matrix((N, 0), dtype=np.float64), [], None
    if spec == "SF_01_RainOnly":
        add_dense_cols([("log1p_Precip", df["log1p_Precip"].to_numpy())])
    elif spec == "SF_02_RainDynamics":
        add_dense_cols([
            ("log1p_Precip", df["log1p_Precip"].to_numpy()),
            ("is_rain", df["is_rain"].to_numpy()),
            ("is_trace_rain", df["is_trace_rain"].to_numpy()),
            ("log1p_rain_lag1", df["log1p_rain_lag1"].to_numpy()),
            ("log1p_rain_lag2", df["log1p_rain_lag2"].to_numpy()),
            ("log1p_D24R", df["log1p_D24R"].to_numpy()),
            ("morning_rain_causal", df["morning_rain_causal"].to_numpy()),
        ])
    elif spec == "SF_03_TempOnly":
        add_dense_cols([("Temp", df["Temp"].to_numpy())])
    elif spec == "SF_03b_TempSpline":
        add_dense_cols([
            ("Temp", df["Temp"].to_numpy()),
            ("temp_cold_hinge", df["temp_cold_hinge"].to_numpy()),
            ("temp_hot_hinge", df["temp_hot_hinge"].to_numpy()),
        ])
    elif spec == "SF_04_HUMDOnly":
        add_dense_cols([("HUMD", df["HUMD"].to_numpy())])
    elif spec == "SF_05_SemesterScalar":
        add_dense_cols([("Semester", df["semester"].to_numpy())])
    elif spec == "SF_06_Semester24h":
        add_hour_block("Sem", df["semester"].to_numpy())
    elif spec == "SF_07_AcademicFine":
        add_hour_block("SemRaw", df["semester_raw"].to_numpy())
        add_hour_block("Open", df["opening"].to_numpy())
        add_hour_block("Midterm", df["midterm"].to_numpy())
        add_hour_block("Final", df["final_exam"].to_numpy())
        add_dense_cols([
            ("sem_buffer_pre", df["sem_buffer_pre"].to_numpy()),
            ("sem_buffer_post", df["sem_buffer_post"].to_numpy()),
            ("sem_progress", df["sem_progress"].to_numpy()),
            ("is_summer_break", df["is_summer_break"].to_numpy()),
        ])
    elif spec == "SF_08_DayOfWeekHoliday":
        dow = df["day_of_week"].to_numpy()
        wd = df["weekday"].to_numpy()
        add_dense_cols([
            ("dow_tue", ((dow == 2) & (wd == 1)).astype(float)),
            ("dow_wed", ((dow == 3) & (wd == 1)).astype(float)),
            ("dow_thu", ((dow == 4) & (wd == 1)).astype(float)),
            ("dow_fri", ((dow == 5) & (wd == 1)).astype(float)),
            ("is_sunday_off", df["is_sunday_off"].to_numpy()),
            ("is_national_holiday", df["is_national_holiday"].to_numpy()),
            ("is_cny", df["is_cny"].to_numpy()),
            ("is_typhoon", df["is_typhoon"].to_numpy()),
            ("is_makeup_workday", df["is_makeup_workday"].to_numpy()),
            ("is_long_weekend", df["is_long_weekend"].to_numpy()),
        ])
        add_hour_block("FriWork", df["is_friday_work"].to_numpy(), ridge=1.0)
    elif spec == "SF_09_EventOnly":
        add_dense_cols([("Event", df["event"].to_numpy())])
    elif spec == "SF_09b_EventTierScope":
        # Full-rank decomposition (event_tier2 = event_campus + event_venue + event_both - event_tier1)
        add_dense_cols([
            ("event_tier1", df["event_tier1"].to_numpy()),
            ("event_campus", df["event_campus"].to_numpy()),
            ("event_venue", df["event_venue"].to_numpy()),
            ("event_both", df["event_both"].to_numpy()),
        ])
    elif spec == "SF_10_PolicyRegime":
        add_dense_cols([
            ("policy_ramp_60d", df["policy_ramp_60d"].to_numpy()),
        ])
    elif spec == "SF_11_SeasonFourier":
        add_dense_cols([
            ("sin_doy1", df["sin_doy1"].to_numpy()),
            ("cos_doy1", df["cos_doy1"].to_numpy()),
            ("sin_doy2", df["sin_doy2"].to_numpy()),
            ("cos_doy2", df["cos_doy2"].to_numpy()),
        ])

    # ---- Stage 3 Multi-Paradigm Full Model Specs ----
    elif spec in {"M02_Baseline_NB2_FE", "M03_Baseline_Poisson_FE"}:
        add_hour_block("Sem", df["semester"].to_numpy(), ridge=0.0)
        add_hour_block("Open", df["opening"].to_numpy(), ridge=0.0)
        add_hour_block("Exam", df["exam"].to_numpy(), ridge=0.0)
        add_dense_cols([
            ("Temp", df["Temp"].to_numpy()),
            ("log1p_Precip", df["log1p_Precip"].to_numpy()),
            ("HUMD", df["HUMD"].to_numpy()),
            ("Event", df["event"].to_numpy()),
        ], ridge=0.0)
        return sparse.hstack(blocks, format="csc"), names, None

    elif spec in {
        "M04_RichCalWeather",
        "M05_RichRegimePolicy",
        "M06_RichRecency",
        "M07_Poisson_RichRegime",
        "M08_FineDayRegimeFE",
        "M09_ContinuousKernel",
        "M10_Hierarchical_EB_Slopes",
        "M11_GraphLaplacian_Waterflow",
        "M12_Station_x_Sem_FE",
        "M16_CHAMPION_DualGraph_QMLE",
    }:
        sem_wd = (df["semester_raw"].to_numpy() == 1) & (df["weekday"].to_numpy() == 1)
        sem_we = (df["semester_raw"].to_numpy() == 1) & (df["weekday"].to_numpy() == 0)
        if spec != "M12_Station_x_Sem_FE":
            add_hour_block("SemWD", sem_wd.astype(np.float64), ridge=0.5)
            add_hour_block("SemWE", sem_we.astype(np.float64), ridge=0.5)
        add_hour_block("Open", df["opening"].to_numpy(), ridge=1.0)
        add_hour_block("Midterm", df["midterm"].to_numpy(), ridge=1.0)
        add_hour_block("Final", df["final_exam"].to_numpy(), ridge=1.0)
        if spec != "M08_FineDayRegimeFE":
            add_hour_block("FriWork", df["is_friday_work"].to_numpy(), ridge=1.0)
            add_hour_block("SunOff", df["is_sunday_off"].to_numpy(), ridge=1.0)

        dow = df["day_of_week"].to_numpy()
        wd = df["weekday"].to_numpy()
        core_cols = [
            ("sem_buffer_pre", df["sem_buffer_pre"].to_numpy()),
            ("sem_buffer_post", df["sem_buffer_post"].to_numpy()),
            ("sem_progress", df["sem_progress"].to_numpy()),
            ("is_summer_break", df["is_summer_break"].to_numpy()),
            ("dow_tue", ((dow == 2) & (wd == 1)).astype(float)),
            ("dow_wed", ((dow == 3) & (wd == 1)).astype(float)),
            ("dow_thu", ((dow == 4) & (wd == 1)).astype(float)),
            ("is_national_holiday", df["is_national_holiday"].to_numpy()),
            ("is_cny", df["is_cny"].to_numpy()),
            ("is_typhoon", df["is_typhoon"].to_numpy()),
            ("is_makeup_workday", df["is_makeup_workday"].to_numpy()),
            ("is_labor_day", df["is_labor_day"].to_numpy()),
            ("is_long_weekend", df["is_long_weekend"].to_numpy()),
            *(
                [
                    ("temp_bsg_cold", df["temp_bsg_cold"].to_numpy()),
                    ("temp_bsg_hot", df["temp_bsg_hot"].to_numpy()),
                ]
                if spec == "M16_CHAMPION_DualGraph_QMLE"
                else [
                    ("Temp", df["Temp"].to_numpy()),
                    ("temp_cold_hinge", df["temp_cold_hinge"].to_numpy()),
                    ("temp_hot_hinge", df["temp_hot_hinge"].to_numpy()),
                ]
            ),
            ("HUMD", df["HUMD"].to_numpy()),
            ("temp_x_humd", df["temp_x_humd"].to_numpy()),
            ("log1p_Precip", df["log1p_Precip"].to_numpy()),
            ("is_rain", df["is_rain"].to_numpy()),
            ("is_trace_rain", df["is_trace_rain"].to_numpy()),
            ("log1p_rain_lag1", df["log1p_rain_lag1"].to_numpy()),
            ("log1p_rain_lag2", df["log1p_rain_lag2"].to_numpy()),
            ("log1p_D24R", df["log1p_D24R"].to_numpy()),
            ("morning_rain_causal", df["morning_rain_causal"].to_numpy()),
            # Full-rank event tier x scope decomposition (no exact collinearity with event_tier2)
            ("event_tier1", df["event_tier1"].to_numpy()),
            ("event_campus", df["event_campus"].to_numpy()),
            ("event_venue", df["event_venue"].to_numpy()),
            ("event_both", df["event_both"].to_numpy()),
        ]
        # Policy regime terms:
        # - Stage 3 M05..M12 test slow 120d ramp (+ step) to diagnose extrapolation overshoot
        # - M16_CHAMPION uses a SINGLE clean monotonic 60d saturating adoption ramp (ex-ante
        #   6-month / 3*tau=180d saturation prior; removes perverse negative launch-day step!)
        ramp_col = None
        use_step = False
        if spec in {
            "M05_RichRegimePolicy", "M07_Poisson_RichRegime", "M08_FineDayRegimeFE",
            "M09_ContinuousKernel", "M10_Hierarchical_EB_Slopes", "M11_GraphLaplacian_Waterflow",
            "M12_Station_x_Sem_FE",
        }:
            ramp_col = policy_ramp_override or "policy_ramp_120d"
            use_step = True if include_policy_step is None else include_policy_step
        elif spec == "M16_CHAMPION_DualGraph_QMLE":
            ramp_col = policy_ramp_override or "policy_ramp_60d"
            use_step = False if include_policy_step is None else include_policy_step

        if ramp_col is not None:
            if use_step:
                core_cols.append(("is_free30_policy", df["is_free30_policy"].to_numpy()))
            core_cols.append((ramp_col, df[ramp_col].to_numpy()))

        add_dense_cols(core_cols, ridge=0.1)

        add_dense_cols([
            ("rain_x_weekday", df["log1p_Precip"].to_numpy() * wd),
            ("rain_x_semester", df["log1p_Precip"].to_numpy() * df["semester_raw"].to_numpy()),
            ("temp_x_weekday", (df["Temp"].to_numpy() - 24.0) * wd),
        ], ridge=0.5)

        if spec in {"M09_ContinuousKernel", "M11_GraphLaplacian_Waterflow", "M12_Station_x_Sem_FE", "M16_CHAMPION_DualGraph_QMLE"}:
            st_map = stn_df.set_index("station_id")
            sids = df["station_id"]
            kv300 = sids.map(st_map["kern_venue_300"]).to_numpy(dtype=np.float64)
            kv600 = sids.map(st_map["kern_venue_600"]).to_numpy(dtype=np.float64)
            kcamp = sids.map(st_map["kern_campus_450"]).to_numpy(dtype=np.float64)
            kmrt = sids.map(st_map["kern_mrt_350"]).to_numpy(dtype=np.float64)
            kdorm = sids.map(st_map["kern_dorm_400"]).to_numpy(dtype=np.float64)
            tidal = sids.map(st_map["tidal_index_2024"]).to_numpy(dtype=np.float64)
            wd_rat = sids.map(st_map["log_wd_we_ratio_2024"]).to_numpy(dtype=np.float64)
            svd1 = sids.map(st_map["svd_func_1"]).to_numpy(dtype=np.float64)
            svd2 = sids.map(st_map["svd_func_2"]).to_numpy(dtype=np.float64)
            svd3 = sids.map(st_map["svd_func_3"]).to_numpy(dtype=np.float64)

            ev_v = df["event_any_venue"].to_numpy(dtype=np.float64)
            ev_c = df["event_any_campus"].to_numpy(dtype=np.float64)
            sem_r = df["semester_raw"].to_numpy(dtype=np.float64)
            ex_r = df["exam"].to_numpy(dtype=np.float64)
            op_r = df["opening"].to_numpy(dtype=np.float64)
            rain_r = df["log1p_Precip"].to_numpy(dtype=np.float64)
            am_m = ((hr >= 7) & (hr <= 11)).astype(np.float64)
            pm_m = ((hr >= 16) & (hr <= 21)).astype(np.float64)

            spatial_cols = [
                ("ev_venue_x_kern300", ev_v * kv300),
                ("ev_venue_x_kern600", ev_v * kv600),
                ("ev_campus_x_kcamp", ev_c * kcamp),
                ("ev_campus_x_kmrt", ev_c * kmrt),
                ("sem_x_kcamp", sem_r * kcamp),
                ("sem_x_kmrt", sem_r * kmrt),
                ("sem_x_kdorm", sem_r * kdorm),
                ("sem_x_wd_rat", sem_r * wd_rat),
                ("sem_x_svd1", sem_r * svd1),
                ("sem_x_svd2", sem_r * svd2),
                ("sem_x_svd3", sem_r * svd3),
                ("sem_am_x_tidal", sem_r * am_m * tidal),
                ("sem_pm_x_tidal", sem_r * pm_m * tidal),
                ("exam_x_kcamp", ex_r * kcamp),
                ("exam_x_svd1", ex_r * svd1),
                ("open_x_kcamp", op_r * kcamp),
                ("rain_x_kcamp", rain_r * kcamp),
                ("rain_x_kmrt", rain_r * kmrt),
                ("rain_x_svd1", rain_r * svd1),
            ]
            add_dense_cols(spatial_cols, ridge=1.0)

        if spec in {"M10_Hierarchical_EB_Slopes", "M11_GraphLaplacian_Waterflow", "M16_CHAMPION_DualGraph_QMLE"}:
            sid_to_i = {s: i for i, s in enumerate(station_order)}
            s_idx = df["station_id"].map(sid_to_i).fillna(-1).to_numpy(dtype=np.int32)
            S_n = len(station_order)
            blk = _diurnal_block(hr)

            # Precompute smooth geographic interpolation weights for any cold-start stations (s_idx < 0)
            cold_mask_all = s_idx < 0
            cold_geo_W: dict[str, np.ndarray] = {}
            if np.any(cold_mask_all):
                cold_sids = sorted(df.loc[cold_mask_all, "station_id"].unique().tolist())
                W_c = _cold_start_geo_weights(stn_df, cold_sids, station_order, length_scale_m=320.0)
                cold_geo_W = {cs: W_c[i] for i, cs in enumerate(cold_sids)}

            sem_wd_f = ((df["semester_raw"].to_numpy() == 1) & (wd == 1)).astype(np.float64)
            sem_we_f = ((df["semester_raw"].to_numpy() == 1) & (wd == 0)).astype(np.float64)
            st_ramp_col = ramp_col or ("policy_ramp_60d" if spec == "M16_CHAMPION_DualGraph_QMLE" else "policy_ramp_120d")

            if spec == "M16_CHAMPION_DualGraph_QMLE":
                # Dual-Graph assignment (ZERO hard regional cuts!):
                #   - "func" Laplacian (L_func) for diurnal tidal & academic calendar polarity
                #   - "geo"  Laplacian (L_geo)  for radial physical crowd spillover & weather
                slope_specs = [
                    ("st_sem_wd_am", sem_wd_f * (blk == 1), 1.0, "func"),
                    ("st_sem_wd_mid", sem_wd_f * (blk == 2), 1.0, "func"),
                    ("st_sem_wd_pm", sem_wd_f * (blk == 3), 1.0, "func"),
                    ("st_sem_wd_nt", sem_wd_f * (blk == 0), 2.0, "func"),
                    ("st_sem_we", sem_we_f, 1.5, "func"),
                    ("st_fri_work", df["is_friday_work"].to_numpy(dtype=np.float64), 2.0, "func"),
                    ("st_summer", df["is_summer_break"].to_numpy(dtype=np.float64), 1.8, "func"),
                    ("st_opening", df["opening"].to_numpy(dtype=np.float64), 2.5, "func"),
                    ("st_rain", df["log1p_Precip"].to_numpy(dtype=np.float64), 2.0, "geo"),
                    ("st_ev_venue", df["event_any_venue"].to_numpy(dtype=np.float64), 1.5, "geo"),
                    ("st_ev_campus", df["event_any_campus"].to_numpy(dtype=np.float64), 2.0, "geo"),
                    ("st_policy_60d", df[st_ramp_col].to_numpy(dtype=np.float64), 2.5, "geo"),
                ]
            else:
                gtype = "geo" if spec == "M11_GraphLaplacian_Waterflow" else "none"
                slope_specs = [
                    ("st_sem_wd_am", sem_wd_f * (blk == 1), 1.0, gtype),
                    ("st_sem_wd_mid", sem_wd_f * (blk == 2), 1.0, gtype),
                    ("st_sem_wd_pm", sem_wd_f * (blk == 3), 1.0, gtype),
                    ("st_sem_wd_nt", sem_wd_f * (blk == 0), 2.0, gtype),
                    ("st_sem_we", sem_we_f, 1.5, gtype),
                    ("st_rain", df["log1p_Precip"].to_numpy(dtype=np.float64), 2.0, gtype),
                    ("st_ev_venue", df["event_any_venue"].to_numpy(dtype=np.float64), 1.5, gtype),
                    ("st_policy_ramp", df[st_ramp_col].to_numpy(dtype=np.float64), 2.5, gtype),
                ]

            rows_all = np.arange(N, dtype=np.int32)
            sid_vals = df["station_id"].to_numpy()
            for sname, vec, mult, gtype in slope_specs:
                nz_known = (vec != 0.0) & (s_idx >= 0)
                if not cold_geo_W:
                    sp_blk = sparse.csc_matrix(
                        (vec[nz_known], (rows_all[nz_known], s_idx[nz_known])), shape=(N, S_n), dtype=np.float64
                    )
                else:
                    # Known stations: exact 1-hot; cold-start stations: continuous Gaussian geo interpolation
                    r_parts = [rows_all[nz_known]]
                    c_parts = [s_idx[nz_known]]
                    v_parts = [vec[nz_known]]
                    if gtype == "geo":
                        nz_cold = np.flatnonzero((vec != 0.0) & cold_mask_all)
                        for ri in nz_cold:
                            w_row = cold_geo_W[sid_vals[ri]]
                            keep = w_row > 1e-4
                            cols_k = np.flatnonzero(keep).astype(np.int32)
                            r_parts.append(np.full(len(cols_k), ri, dtype=np.int32))
                            c_parts.append(cols_k)
                            v_parts.append(vec[ri] * (w_row[keep] / w_row[keep].sum()))
                    sp_blk = sparse.csc_matrix(
                        (np.concatenate(v_parts), (np.concatenate(r_parts), np.concatenate(c_parts))),
                        shape=(N, S_n),
                        dtype=np.float64,
                    )
                blocks.append(sp_blk)
                names.extend([f"{sname}_{sid}" for sid in station_order])
                P_blk = (lam_ridge * mult) * np.eye(S_n)
                if gtype == "geo" and L_geo is not None and lam_lap > 0.0:
                    P_blk = P_blk + (lam_lap * mult) * L_geo
                elif gtype == "func" and L_func is not None and lam_lap > 0.0:
                    P_blk = P_blk + (lam_lap * mult) * L_func
                pen_blocks.append(P_blk)
    else:
        raise ValueError(f"Unknown feature spec: {spec}")

    X_sp = sparse.hstack(blocks, format="csc")
    P_full = sparse.block_diag(pen_blocks, format="csc").toarray() if pen_blocks else None
    return X_sp, names, P_full


# ---------------------------------------------------------------------------
# 6. Unified GLM-FE Fit & Predict
# ---------------------------------------------------------------------------

def _make_fe_key(df: pd.DataFrame, fe_mode: str) -> pd.Series:
    h_str = df["hour"].astype(str).str.zfill(2)
    if fe_mode == "station_only":
        return df["station_id"]
    if fe_mode == "station_hour_wd":
        return df["station_id"] + "_" + h_str + "_" + df["weekday"].astype(str)
    if fe_mode == "station_hour_day4":
        return df["station_id"] + "_" + h_str + "_" + df["day_regime_4"].astype(str)
    if fe_mode == "station_hour_wd_sem":
        return df["station_id"] + "_" + h_str + "_" + df["weekday"].astype(str) + "_" + df["semester_raw"].astype(str)
    raise ValueError(f"Unknown fe_mode: {fe_mode}")


def fit_and_predict_glm_spec(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stn_df: pd.DataFrame,
    y_col: str,
    exp_col: str,
    spec: str,
    model_name: str,
    side: str,
    family: str = "nb2",
    fe_mode: str = "station_hour_wd",
    recency_halflife_days: float | None = None,
    lam_ridge: float = 25.0,
    lam_lap: float = 80.0,
    fe_shrink_lambda: float = 0.0,
    fixed_alpha: float | None = None,
    return_train_preds: bool = False,
    policy_ramp_override: str | None = None,
    include_policy_step: bool | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    tr = train_df.copy()
    station_order = sorted(tr["station_id"].unique().tolist())
    _, L_geo = build_spatial_laplacian(stn_df, station_order, length_scale_m=320.0)
    _, L_func = build_functional_laplacian(stn_df, station_order, sigma=1.5)

    tr["fe_key"] = _make_fe_key(tr, fe_mode)
    tr["parent_fe"] = _make_fe_key(tr, "station_hour_wd")

    fe_sum_y = tr.groupby("fe_key")[y_col].transform("sum")
    zero_mask = fe_sum_y == 0
    zero_fe_keys = set(tr.loc[zero_mask, "fe_key"].unique())

    est = tr[~zero_mask].reset_index(drop=True)
    fe_cats, g_idx = np.unique(est["fe_key"].to_numpy(), return_inverse=True)
    G = len(fe_cats)
    N = len(est)
    y = est[y_col].to_numpy(dtype=np.float64)
    exp_min = est[exp_col].to_numpy(dtype=np.float64)
    offset = np.log(exp_min)

    if recency_halflife_days is not None and recency_halflife_days > 0:
        d_max = float(est["days_from_2024_01_01"].max())
        age = d_max - est["days_from_2024_01_01"].to_numpy(dtype=np.float64)
        obs_w = np.exp(-np.log(2.0) * age / recency_halflife_days)
        obs_w = obs_w * (len(obs_w) / np.sum(obs_w))
    else:
        obs_w = None

    X_sp, feat_names, P_theta = build_feature_matrix(
        est, spec, stn_df, station_order, L_geo=L_geo, L_func=L_func, lam_ridge=lam_ridge, lam_lap=lam_lap,
        policy_ramp_override=policy_ramp_override, include_policy_step=include_policy_step,
    )
    K = X_sp.shape[1]
    Z = sparse.csc_matrix((np.ones(N, dtype=np.float64), (np.arange(N), g_idx)), shape=(N, G))

    psi_prior = None
    if fe_shrink_lambda > 0.0:
        w_eff = np.ones(N) if obs_w is None else obs_w
        p_agg = pd.DataFrame({
            "pfe": est["parent_fe"].to_numpy(),
            "wy": w_eff * y,
            "we": w_eff * exp_min,
        }).groupby("pfe").sum()
        p_rate = np.log(np.maximum(p_agg["wy"], 1e-4) / np.maximum(p_agg["we"], 1e-4)).to_dict()
        fe_to_parent = est.drop_duplicates("fe_key").set_index("fe_key")["parent_fe"].to_dict()
        psi_prior = np.array([p_rate.get(fe_to_parent[fk], -3.0) for fk in fe_cats], dtype=np.float64)

    if family == "poisson":
        alpha_hat = 0.0
        theta_hat, psi_hat, mu_hat, ll_train, n_iter = fit_penalized_glm_fe_core(
            y, X_sp, offset, g_idx, G, 0.0, Z=Z, P_theta=P_theta, obs_w=obs_w,
            fe_shrink_lambda=fe_shrink_lambda, psi_prior=psi_prior, max_iter=35, tol=1e-7,
        )
    elif fixed_alpha is not None:
        alpha_hat = float(fixed_alpha)
        theta_hat, psi_hat, mu_hat, ll_train, n_iter = fit_penalized_glm_fe_core(
            y, X_sp, offset, g_idx, G, alpha_hat, Z=Z, P_theta=P_theta, obs_w=obs_w,
            fe_shrink_lambda=fe_shrink_lambda, psi_prior=psi_prior, max_iter=35, tol=1e-7,
        )
    else:
        cache = {"theta": None, "psi": None}
        def neg_pll(log_a: float) -> float:
            av = float(np.exp(log_a))
            th, ps, _, llv, _ = fit_penalized_glm_fe_core(
                y, X_sp, offset, g_idx, G, av, Z=Z, P_theta=P_theta, obs_w=obs_w,
                fe_shrink_lambda=fe_shrink_lambda, psi_prior=psi_prior,
                theta_init=cache["theta"], psi_init=cache["psi"], max_iter=18, tol=1e-5,
            )
            cache["theta"], cache["psi"] = th, ps
            return -llv

        opt = optimize.minimize_scalar(
            neg_pll, bounds=(np.log(0.01), np.log(2.5)), method="bounded", options={"xatol": 1e-2}
        )
        alpha_hat = float(np.exp(opt.x))
        theta_hat, psi_hat, mu_hat, ll_train, n_iter = fit_penalized_glm_fe_core(
            y, X_sp, offset, g_idx, G, alpha_hat, Z=Z, P_theta=P_theta, obs_w=obs_w,
            fe_shrink_lambda=fe_shrink_lambda, psi_prior=psi_prior,
            theta_init=cache["theta"], psi_init=cache["psi"], max_iter=30, tol=1e-7,
        )

    psi_map = dict(zip(fe_cats, psi_hat))
    parent_fallback = (
        tr.groupby("parent_fe").apply(
            lambda g: np.log(max(g[y_col].sum(), 1e-6) / max(g[exp_col].sum(), 1e-6)), include_groups=False
        ).to_dict()
    )
    global_fallback = math.log(max(tr[y_col].sum(), 1e-6) / max(tr[exp_col].sum(), 1e-6))

    # Build (n_train_stations, 48) parent FE matrix for continuous geographic cold-start interpolation
    hw_mat = np.full((len(station_order), 48), global_fallback, dtype=np.float64)
    for si, sid in enumerate(station_order):
        for h in range(24):
            hs = f"{h:02d}"
            for wv in (0, 1):
                pk = f"{sid}_{hs}_{wv}"
                if pk in psi_map and fe_mode == "station_hour_wd":
                    hw_mat[si, h + 24 * wv] = psi_map[pk]
                elif pk in parent_fallback:
                    hw_mat[si, h + 24 * wv] = parent_fallback[pk]

    train_sid_set = set(station_order)

    def predict_df(d_in: pd.DataFrame) -> np.ndarray:
        fks = _make_fe_key(d_in, fe_mode)
        pfks = _make_fe_key(d_in, "station_hour_wd")
        X_te, _, _ = build_feature_matrix(
            d_in, spec, stn_df, station_order, L_geo=L_geo, L_func=L_func, lam_ridge=lam_ridge, lam_lap=lam_lap,
            policy_ramp_override=policy_ramp_override, include_policy_step=include_policy_step,
        )
        lin = np.asarray(X_te @ theta_hat, dtype=np.float64).ravel() if K > 0 else np.zeros(len(d_in), dtype=np.float64)

        is_zero = fks.isin(zero_fe_keys).to_numpy()
        psi_v = fks.map(psi_map).to_numpy(dtype=np.float64, copy=True)
        miss = np.isnan(psi_v) & (~is_zero)
        if np.any(miss):
            pf_v = pfks.iloc[np.flatnonzero(miss)].map(parent_fallback).to_numpy(dtype=np.float64)
            psi_v[miss] = pf_v

            # Smooth continuous geographic-kernel FE interpolation for unseen cold-start stations
            still_miss = np.isnan(psi_v) & (~is_zero)
            if np.any(still_miss):
                miss_idx = np.flatnonzero(still_miss)
                miss_sids = d_in["station_id"].iloc[miss_idx]
                cold_sids = sorted([s for s in miss_sids.unique().tolist() if s not in train_sid_set])
                if cold_sids:
                    W_c = _cold_start_geo_weights(stn_df, cold_sids, station_order, length_scale_m=320.0)
                    cold_hw = W_c @ hw_mat  # (n_cold, 48)
                    c_map = {cs: cold_hw[i] for i, cs in enumerate(cold_sids)}
                    hrs_m = d_in["hour"].iloc[miss_idx].to_numpy(dtype=np.int32)
                    wds_m = d_in["weekday"].iloc[miss_idx].to_numpy(dtype=np.int32)
                    for k_pos, ri in enumerate(miss_idx):
                        sid_r = d_in["station_id"].iat[ri]
                        if sid_r in c_map:
                            psi_v[ri] = c_map[sid_r][hrs_m[k_pos] + 24 * wds_m[k_pos]]
                        else:
                            psi_v[ri] = global_fallback
                else:
                    psi_v[still_miss] = global_fallback

        out = np.where(is_zero, 0.0, np.exp(np.clip(psi_v + lin, -20.0, 8.0)))
        return out

    pred_train = predict_df(train_df)
    pred_test = predict_df(test_df)

    m_tr = evaluate_predictions(train_df, y_col, exp_col, pred_train, model_name, side, "TRAIN_2024")
    m_te = evaluate_predictions(test_df, y_col, exp_col, pred_test, model_name, side, "TEST_2025_RAW")
    pred_test_causal = apply_causal_trailing_level_calibration(test_df, pred_test, y_col, exp_col, window_days=14)
    m_te_c = evaluate_predictions(test_df, y_col, exp_col, pred_test_causal, model_name, side, "TEST_2025_CAUSAL14D")

    res = {
        "model_name": model_name,
        "spec": spec,
        "side": side,
        "family": family,
        "alpha": alpha_hat,
        "K": K,
        "G": G,
        "ll_train": ll_train,
        "aic_train": 2.0 * (G + K + (1 if family == "nb2" else 0)) - 2.0 * ll_train,
        "n_iter": n_iter,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "metrics_train": m_tr,
        "metrics_test_raw": m_te,
        "metrics_test_causal": m_te_c,
        "pred_test_raw": pred_test,
        "pred_test_causal": pred_test_causal,
        "feat_names": feat_names,
        "theta_hat": theta_hat,
    }
    if return_train_preds:
        res["pred_train"] = pred_train
    return res


# ---------------------------------------------------------------------------
# 7. Historical Group Mean Baseline
# ---------------------------------------------------------------------------

def fit_predict_hist_baseline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, y_col: str, exp_col: str, side: str
) -> dict[str, Any]:
    t0 = time.perf_counter()
    grp4 = train_df.groupby(["station_id", "hour", "weekday", "semester"]).apply(
        lambda g: g[y_col].sum() / g[exp_col].sum(), include_groups=False
    )
    grp3 = train_df.groupby(["station_id", "hour", "weekday"]).apply(
        lambda g: g[y_col].sum() / g[exp_col].sum(), include_groups=False
    )
    glob = float(train_df[y_col].sum() / train_df[exp_col].sum())

    def pred(d: pd.DataFrame) -> np.ndarray:
        k4 = pd.MultiIndex.from_frame(d[["station_id", "hour", "weekday", "semester"]])
        k3 = pd.MultiIndex.from_frame(d[["station_id", "hour", "weekday"]])
        p = grp4.reindex(k4).to_numpy(dtype=np.float64, copy=True)
        miss = np.isnan(p)
        if miss.any():
            p[miss] = grp3.reindex(k3[miss]).to_numpy(dtype=np.float64)
        return np.where(np.isnan(p), glob, p)

    p_tr = pred(train_df)
    p_te = pred(test_df)
    p_te_c = apply_causal_trailing_level_calibration(test_df, p_te, y_col, exp_col, window_days=14)
    return {
        "model_name": "M01_HistGroupMean",
        "spec": "M01_HistGroupMean",
        "side": side,
        "family": "nonparametric",
        "alpha": np.nan,
        "K": 0,
        "G": len(grp4),
        "ll_train": np.nan,
        "aic_train": np.nan,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "metrics_train": evaluate_predictions(train_df, y_col, exp_col, p_tr, "M01_HistGroupMean", side, "TRAIN_2024"),
        "metrics_test_raw": evaluate_predictions(test_df, y_col, exp_col, p_te, "M01_HistGroupMean", side, "TEST_2025_RAW"),
        "metrics_test_causal": evaluate_predictions(test_df, y_col, exp_col, p_te_c, "M01_HistGroupMean", side, "TEST_2025_CAUSAL14D"),
        "pred_test_raw": p_te,
        "pred_test_causal": p_te_c,
    }


# ---------------------------------------------------------------------------
# 8. Exact Exposure-Offset Histogram Poisson/NB2 Residual GBDT (Pure NumPy!)
# ---------------------------------------------------------------------------

ML_FEATURE_COLS = [
    "hour", "day_of_week", "weekday", "semester_raw", "opening", "midterm", "final_exam",
    "sem_buffer_pre", "sem_buffer_post", "sem_progress", "is_summer_break", "is_winter_break",
    "is_national_holiday", "is_cny", "is_typhoon", "is_makeup_workday", "is_long_weekend",
    "is_friday_work", "is_sunday_off", "event", "event_tier1", "event_any_venue", "event_any_campus",
    "is_free30_policy", "policy_ramp_60d",
    "Temp", "temp_bsg_cold", "temp_bsg_hot", "HUMD", "log1p_Precip", "is_rain", "log1p_rain_lag1", "log1p_rain_lag2",
    "log1p_D24R", "morning_rain_causal", "temp_x_humd",
    "PositionLat", "PositionLon", "dist_sports_center_m", "dist_campus_core_m", "dist_nearest_mrt_m",
    "kern_venue_300", "kern_campus_450", "kern_mrt_350",
    "svd_func_1", "svd_func_2", "svd_func_3", "svd_func_4",
    "tidal_index_2024", "log_wd_we_ratio_2024", "log_vol_2024_z",
]


def _prepare_ml_matrix(df: pd.DataFrame, stn_df: pd.DataFrame) -> np.ndarray:
    st_cols = [
        "PositionLat", "PositionLon", "dist_sports_center_m", "dist_campus_core_m", "dist_nearest_mrt_m",
        "kern_venue_300", "kern_campus_450", "kern_mrt_350",
        "svd_func_1", "svd_func_2", "svd_func_3", "svd_func_4",
        "tidal_index_2024", "log_wd_we_ratio_2024", "log_vol_2024_z",
    ]
    st_idx = stn_df.set_index("station_id")[st_cols]
    joined = df.join(st_idx, on="station_id")
    return joined[ML_FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)


class _HistTreeNode:
    __slots__ = ("feat", "thresh_bin", "val", "left", "right")
    def __init__(self, val: float = 0.0):
        self.feat: int = -1
        self.thresh_bin: int = -1
        self.val: float = val
        self.left: _HistTreeNode | None = None
        self.right: _HistTreeNode | None = None


def fit_exposure_hist_gbdt(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stn_df: pd.DataFrame,
    y_col: str,
    exp_col: str,
    init_rate_train: np.ndarray,
    init_rate_test: np.ndarray,
    model_name: str,
    side: str,
    n_trees: int = 60,
    max_depth: int = 4,
    lr: float = 0.06,
    reg_lambda: float = 40.0,
    min_child_hess: float = 150.0,
    n_bins: int = 32,
    alpha_nb2: float = 0.0,
    recency_halflife_days: float | None = 120.0,
    subsample_frac: float = 0.65,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Fast vectorized Histogram GBDT optimizing exact Exposure-Offset Poisson / NB2 deviance
    initialized at an FE anchor. Zero hard regional boundaries.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    X_tr = _prepare_ml_matrix(train_df, stn_df)
    X_te = _prepare_ml_matrix(test_df, stn_df)
    N, F = X_tr.shape

    bin_edges: list[np.ndarray] = []
    B_tr = np.zeros((N, F), dtype=np.uint8)
    B_te = np.zeros((len(test_df), F), dtype=np.uint8)
    for j in range(F):
        col = X_tr[:, j]
        uniq = np.unique(col)
        if len(uniq) <= n_bins:
            edges = (uniq[:-1] + uniq[1:]) / 2.0 if len(uniq) > 1 else np.array([], dtype=np.float32)
        else:
            qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
            edges = np.unique(np.quantile(col, qs).astype(np.float32))
        bin_edges.append(edges)
        B_tr[:, j] = np.searchsorted(edges, col).astype(np.uint8)
        B_te[:, j] = np.searchsorted(edges, X_te[:, j]).astype(np.uint8)

    y = train_df[y_col].to_numpy(dtype=np.float64)
    exp_tr = train_df[exp_col].to_numpy(dtype=np.float64)
    offset_tr = np.log(exp_tr)

    if recency_halflife_days is not None and recency_halflife_days > 0:
        d_max = float(train_df["days_from_2024_01_01"].max())
        age = d_max - train_df["days_from_2024_01_01"].to_numpy(dtype=np.float64)
        obs_w = np.exp(-np.log(2.0) * age / recency_halflife_days)
        obs_w = obs_w * (len(obs_w) / np.sum(obs_w))
    else:
        obs_w = np.ones(N, dtype=np.float64)

    f_tr = np.log(np.maximum(init_rate_train, 1e-6))
    f_te = np.log(np.maximum(init_rate_test, 1e-6))
    feat_gain = np.zeros(F, dtype=np.float64)
    n_sub = int(N * subsample_frac)

    def build_node(idx: np.ndarray, g: np.ndarray, h: np.ndarray, depth: int) -> _HistTreeNode:
        G_tot = float(g[idx].sum())
        H_tot = float(h[idx].sum())
        leaf_val = -G_tot / (H_tot + reg_lambda)
        node = _HistTreeNode(val=float(np.clip(leaf_val, -1.0, 1.0)))
        if depth >= max_depth or H_tot < 2.0 * min_child_hess or len(idx) < 250:
            return node

        best_gain = 0.0
        best_f = -1
        best_b = -1
        parent_score = (G_tot * G_tot) / (H_tot + reg_lambda)

        g_sub = g[idx]
        h_sub = h[idx]
        feat_cand = rng.choice(F, size=max(1, int(F * 0.70)), replace=False)
        for j in feat_cand:
            nb = len(bin_edges[j]) + 1
            if nb <= 1:
                continue
            bins_j = B_tr[idx, j]
            gh_g = np.bincount(bins_j, weights=g_sub, minlength=nb)
            gh_h = np.bincount(bins_j, weights=h_sub, minlength=nb)
            cg = np.cumsum(gh_g)[:-1]
            ch = np.cumsum(gh_h)[:-1]
            rg = G_tot - cg
            rh = H_tot - ch
            valid = (ch >= min_child_hess) & (rh >= min_child_hess)
            if not np.any(valid):
                continue
            gains = (cg * cg) / (ch + reg_lambda) + (rg * rg) / (rh + reg_lambda) - parent_score
            gains = np.where(valid, gains, -1.0)
            b_idx = int(np.argmax(gains))
            if gains[b_idx] > best_gain:
                best_gain = float(gains[b_idx])
                best_f = int(j)
                best_b = b_idx

        if best_f < 0 or best_gain < 10.0:
            return node

        feat_gain[best_f] += best_gain
        node.feat = best_f
        node.thresh_bin = best_b
        mask_l = B_tr[idx, best_f] <= best_b
        node.left = build_node(idx[mask_l], g, h, depth + 1)
        node.right = build_node(idx[~mask_l], g, h, depth + 1)
        return node

    def pred_tree(node: _HistTreeNode, B: np.ndarray) -> np.ndarray:
        out = np.empty(B.shape[0], dtype=np.float64)
        stack = [(node, np.arange(B.shape[0], dtype=np.int32))]
        while stack:
            nd, ix = stack.pop()
            if nd.feat < 0:
                out[ix] = nd.val
            else:
                ml = B[ix, nd.feat] <= nd.thresh_bin
                if np.any(ml):
                    stack.append((nd.left, ix[ml]))
                if np.any(~ml):
                    stack.append((nd.right, ix[~ml]))
        return out

    active_tr = init_rate_train > 0.0
    active_te = init_rate_test > 0.0
    active_idx = np.flatnonzero(active_tr)

    for _ in range(n_trees):
        mu = np.exp(np.clip(offset_tr + f_tr, -20.0, 12.0))
        denom = 1.0 + alpha_nb2 * mu if alpha_nb2 > 0.0 else 1.0
        g = obs_w * (mu - y) / denom
        h = obs_w * mu / denom
        sub_ix = rng.choice(active_idx, size=min(n_sub, len(active_idx)), replace=False)
        tree = build_node(sub_ix, g, h, depth=0)
        f_tr[active_tr] += lr * pred_tree(tree, B_tr[active_tr])
        f_te[active_te] += lr * pred_tree(tree, B_te[active_te])

    pred_tr = np.where(active_tr, np.exp(np.clip(f_tr, -15.0, 6.0)), 0.0)
    pred_te = np.where(active_te, np.exp(np.clip(f_te, -15.0, 6.0)), 0.0)
    pred_te_c = apply_causal_trailing_level_calibration(test_df, pred_te, y_col, exp_col, window_days=14)
    top_feats = sorted(zip(ML_FEATURE_COLS, feat_gain.tolist()), key=lambda kv: kv[1], reverse=True)[:15]

    return {
        "model_name": model_name,
        "spec": model_name,
        "side": side,
        "family": "gbdt_residual",
        "alpha": alpha_nb2,
        "K": n_trees,
        "G": 0,
        "ll_train": nb2_loglik(y, pred_tr * exp_tr, alpha_nb2),
        "aic_train": np.nan,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "metrics_train": evaluate_predictions(train_df, y_col, exp_col, pred_tr, model_name, side, "TRAIN_2024"),
        "metrics_test_raw": evaluate_predictions(test_df, y_col, exp_col, pred_te, model_name, side, "TEST_2025_RAW"),
        "metrics_test_causal": evaluate_predictions(test_df, y_col, exp_col, pred_te_c, model_name, side, "TEST_2025_CAUSAL14D"),
        "pred_test_raw": pred_te,
        "pred_test_causal": pred_te_c,
        "top_features": top_feats,
    }


# ---------------------------------------------------------------------------
# 9. PyTorch Exposure-Offset NB2 / Poisson Neural Network with Continuous Embeddings
# ---------------------------------------------------------------------------

def fit_pytorch_nb2_net(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stn_df: pd.DataFrame,
    y_col: str,
    exp_col: str,
    model_name: str,
    side: str,
    init_rate_train: np.ndarray | None = None,
    init_rate_test: np.ndarray | None = None,
    loss_type: str = "nb2",          # "nb2" | "poisson"
    epochs: int = 12,
    batch_size: int = 4096,
    lr: float = 3e-3,
    weight_decay: float = 3e-4,
    alpha_init: float = 0.20,
    recency_halflife_days: float | None = 120.0,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)

    station_order = sorted(train_df["station_id"].unique().tolist())
    sid_to_i = {s: i for i, s in enumerate(station_order)}
    n_st = len(station_order)

    X_tr_raw = _prepare_ml_matrix(train_df, stn_df)
    X_te_raw = _prepare_ml_matrix(test_df, stn_df)
    mu_x = X_tr_raw.mean(axis=0, keepdims=True)
    sd_x = np.maximum(X_tr_raw.std(axis=0, keepdims=True), 1e-6)
    X_tr = np.ascontiguousarray(((X_tr_raw - mu_x) / sd_x).astype(np.float32))
    X_te = np.ascontiguousarray(((X_te_raw - mu_x) / sd_x).astype(np.float32))

    st_tr = train_df["station_id"].map(sid_to_i).fillna(-1).to_numpy(dtype=np.int64, copy=True)
    st_te_raw = test_df["station_id"].map(sid_to_i).fillna(-1).to_numpy(dtype=np.int64, copy=True)
    hr_tr = train_df["hour"].to_numpy(dtype=np.int64, copy=True)
    hr_te = test_df["hour"].to_numpy(dtype=np.int64, copy=True)
    dw_tr = (train_df["day_of_week"].to_numpy(dtype=np.int64, copy=True) - 1)
    dw_te = (test_df["day_of_week"].to_numpy(dtype=np.int64, copy=True) - 1)
    wd_tr = train_df["weekday"].to_numpy(dtype=np.int64, copy=True)
    wd_te = test_df["weekday"].to_numpy(dtype=np.int64, copy=True)

    # Cold-start unseen test stations: prepare geographic-kernel weights to interpolate embeddings cleanly
    cold_sids_te = sorted([s for s in test_df["station_id"].unique().tolist() if s not in sid_to_i])
    cold_sid_to_extra: dict[str, int] = {}
    cold_W_te: np.ndarray | None = None
    st_te = st_te_raw.copy()
    if cold_sids_te:
        cold_W_te = _cold_start_geo_weights(stn_df, cold_sids_te, station_order, length_scale_m=320.0)
        for ci, cs in enumerate(cold_sids_te):
            cold_sid_to_extra[cs] = n_st + ci
        cold_map_full = {**sid_to_i, **cold_sid_to_extra}
        st_te = test_df["station_id"].map(cold_map_full).to_numpy(dtype=np.int64, copy=True)

    y_tr = train_df[y_col].to_numpy(dtype=np.float32, copy=True)
    off_tr = np.log(train_df[exp_col].to_numpy(dtype=np.float32, copy=True))
    is_residual = init_rate_train is not None
    if is_residual:
        base_tr = np.log(np.maximum(init_rate_train, 1e-6)).astype(np.float32)
        base_te = np.log(np.maximum(init_rate_test, 1e-6)).astype(np.float32)
        active_tr = init_rate_train > 0.0
        active_te = init_rate_test > 0.0
    else:
        base_tr = np.zeros(len(train_df), dtype=np.float32)
        base_te = np.zeros(len(test_df), dtype=np.float32)
        active_tr = np.ones(len(train_df), dtype=bool)
        active_te = np.ones(len(test_df), dtype=bool)

    if recency_halflife_days is not None and recency_halflife_days > 0:
        d_max = float(train_df["days_from_2024_01_01"].max())
        age = d_max - train_df["days_from_2024_01_01"].to_numpy(dtype=np.float32, copy=True)
        w_tr = np.exp(-math.log(2.0) * age / recency_halflife_days).astype(np.float32)
        w_tr = w_tr * (len(w_tr) / float(w_tr.sum()))
    else:
        w_tr = np.ones(len(train_df), dtype=np.float32)

    class SpatioTemporalNet(nn.Module):
        def __init__(self, n_stations: int, n_cont: int, residual_mode: bool):
            super().__init__()
            self.residual_mode = residual_mode
            self.emb_st = nn.Embedding(n_stations, 16)
            self.emb_hr = nn.Embedding(24, 10)
            self.emb_dw = nn.Embedding(7, 4)
            self.st_hr_u = nn.Embedding(n_stations, 12)
            self.hr_wd_v = nn.Embedding(48, 12)
            in_dim = 16 + 10 + 4 + n_cont
            hidden = 96 if not residual_mode else 64
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Dropout(0.18 if residual_mode else 0.10),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Dropout(0.18 if residual_mode else 0.10),
                nn.Linear(hidden, 1),
            )
            if residual_mode:
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)
            self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init), dtype=torch.float32))

        def forward(self, st, hr, dw, wd, x):
            h_in = torch.cat([self.emb_st(st), self.emb_hr(hr), self.emb_dw(dw), x], dim=-1)
            out = self.net(h_in).squeeze(-1)
            if not self.residual_mode:
                hw = hr + 24 * wd
                out = out + (self.st_hr_u(st) * self.hr_wd_v(hw)).sum(dim=-1)
            else:
                out = 0.7 * torch.tanh(out)
            return out

    model = SpatioTemporalNet(n_st, X_tr.shape[1], is_residual)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    idx_act = np.flatnonzero(active_tr)
    t_st = torch.from_numpy(st_tr[idx_act].copy())
    t_hr = torch.from_numpy(hr_tr[idx_act].copy())
    t_dw = torch.from_numpy(dw_tr[idx_act].copy())
    t_wd = torch.from_numpy(wd_tr[idx_act].copy())
    t_x = torch.from_numpy(X_tr[idx_act].copy())
    t_y = torch.from_numpy(y_tr[idx_act].copy())
    t_off = torch.from_numpy((off_tr[idx_act] + base_tr[idx_act]).copy())
    t_w = torch.from_numpy(w_tr[idx_act].copy())
    N_act = len(idx_act)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(N_act)
        for s0 in range(0, N_act, batch_size):
            b = perm[s0 : s0 + batch_size]
            opt.zero_grad(set_to_none=True)
            eta_res = model(t_st[b], t_hr[b], t_dw[b], t_wd[b], t_x[b])
            eta = torch.clamp(t_off[b] + eta_res, -18.0, 12.0)
            mu = torch.exp(eta)
            yb = t_y[b]
            if loss_type == "poisson":
                nll = mu - yb * eta
            else:
                alpha = torch.clamp(torch.exp(model.log_alpha), 1e-3, 3.0)
                r = 1.0 / alpha
                nll = -(
                    torch.lgamma(yb + r) - torch.lgamma(r)
                    + r * (torch.log(r) - torch.log(r + mu))
                    + yb * (torch.log(mu + 1e-12) - torch.log(r + mu))
                )
            loss = (t_w[b] * nll).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

    model.eval()
    alpha_learned = 0.0 if loss_type == "poisson" else float(torch.clamp(torch.exp(model.log_alpha), 1e-3, 3.0).item())
    emb_np = model.emb_st.weight.detach().numpy().copy()

    # If unseen cold-start stations exist in test_df, extend embedding tables via Gaussian geo-interpolation
    if cold_sids_te and cold_W_te is not None:
        with torch.no_grad():
            W_t = torch.from_numpy(cold_W_te.astype(np.float32))
            for attr_name in ("emb_st", "st_hr_u"):
                old_emb = getattr(model, attr_name)
                w_train = old_emb.weight.data[:n_st]
                w_cold = W_t @ w_train
                new_emb = nn.Embedding(n_st + len(cold_sids_te), old_emb.embedding_dim)
                new_emb.weight.data[:n_st] = w_train
                new_emb.weight.data[n_st:] = w_cold
                setattr(model, attr_name, new_emb)

    def predict_torch(st_a, hr_a, dw_a, wd_a, x_a, base_a, act_a) -> np.ndarray:
        preds = np.zeros(len(st_a), dtype=np.float64)
        with torch.no_grad():
            for s0 in range(0, len(st_a), 16384):
                s1 = min(s0 + 16384, len(st_a))
                eta_r = model(
                    torch.from_numpy(st_a[s0:s1].copy()),
                    torch.from_numpy(hr_a[s0:s1].copy()),
                    torch.from_numpy(dw_a[s0:s1].copy()),
                    torch.from_numpy(wd_a[s0:s1].copy()),
                    torch.from_numpy(x_a[s0:s1].copy()),
                ).numpy()
                preds[s0:s1] = np.exp(np.clip(base_a[s0:s1] + eta_r, -15.0, 6.0))
        preds[~act_a] = 0.0
        return preds

    pred_tr = predict_torch(st_tr, hr_tr, dw_tr, wd_tr, X_tr, base_tr, active_tr)
    pred_te = predict_torch(st_te, hr_te, dw_te, wd_te, X_te, base_te, active_te)
    pred_te_c = apply_causal_trailing_level_calibration(test_df, pred_te, y_col, exp_col, window_days=14)

    return {
        "model_name": model_name,
        "spec": model_name,
        "side": side,
        "family": f"pytorch_{loss_type}_residual" if is_residual else f"pytorch_{loss_type}_embed",
        "alpha": alpha_learned,
        "K": int(sum(p.numel() for p in model.parameters())),
        "G": 0,
        "ll_train": nb2_loglik(train_df[y_col].to_numpy(), pred_tr * train_df[exp_col].to_numpy(), alpha_learned),
        "aic_train": np.nan,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "metrics_train": evaluate_predictions(train_df, y_col, exp_col, pred_tr, model_name, side, "TRAIN_2024"),
        "metrics_test_raw": evaluate_predictions(test_df, y_col, exp_col, pred_te, model_name, side, "TEST_2025_RAW"),
        "metrics_test_causal": evaluate_predictions(test_df, y_col, exp_col, pred_te_c, model_name, side, "TEST_2025_CAUSAL14D"),
        "pred_test_raw": pred_te,
        "pred_test_causal": pred_te_c,
        "station_embeddings": emb_np,
        "station_order": station_order,
    }


# ---------------------------------------------------------------------------
# 10. Stage 2 Global vs. Local Heterogeneity Forensics
# ---------------------------------------------------------------------------

def run_station_heterogeneity_forensics(
    train_df: pd.DataFrame,
    stn_df: pd.DataFrame,
    y_col: str,
    exp_col: str,
    side: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_order = sorted(train_df["station_id"].unique().tolist())
    W_geo, _ = build_spatial_laplacian(stn_df, station_order, length_scale_m=320.0)

    target_vars = [
        ("log1p_Precip", "降雨 log(1+Precip)"),
        ("Temp", "氣溫 Temp (°C)"),
        ("HUMD", "相對濕度 HUMD"),
        ("semester_raw", "學期期間 Semester (全域)"),
        ("sem_am", "學期 × 早尖峰 (07–11)"),
        ("sem_pm", "學期 × 晚尖峰 (16–20)"),
        ("opening", "開學週 Opening"),
        ("exam", "期中/期末考週 Exam"),
        ("event_any_venue", "體育館場館活動 Event_venue"),
        ("event_any_campus", "全校型活動 Event_campus"),
        ("is_friday_work", "週五上班日 Friday"),
        ("is_typhoon", "颱風假 Typhoon"),
        ("policy_ramp_60d", "前30分免費爬坡 PolicyRamp"),
    ]

    rows = []
    for sid, g in train_df.groupby("station_id", sort=True):
        if len(g) < 500 or g[y_col].sum() < 200:
            continue
        hr = g["hour"].to_numpy()
        wd = g["weekday"].to_numpy()
        g_key = hr * 2 + wd
        df_s = pd.DataFrame({"k": g_key, "y": g[y_col].to_numpy()})
        ok = df_s.groupby("k")["y"].transform("sum").to_numpy() > 0
        if ok.sum() < 300:
            continue
        gs = g.iloc[np.flatnonzero(ok)].reset_index(drop=True)
        hr_s = gs["hour"].to_numpy()
        wd_s = gs["weekday"].to_numpy()
        sem_s = gs["semester_raw"].to_numpy(dtype=np.float64)

        _, g_idx = np.unique(hr_s * 2 + wd_s, return_inverse=True)
        G = int(g_idx.max()) + 1
        N = len(gs)
        Z = sparse.csc_matrix((np.ones(N), (np.arange(N), g_idx)), shape=(N, G))
        y = gs[y_col].to_numpy(dtype=np.float64)
        off = np.log(gs[exp_col].to_numpy(dtype=np.float64))

        X_cols = np.column_stack([
            gs["log1p_Precip"].to_numpy(dtype=np.float64),
            gs["Temp"].to_numpy(dtype=np.float64),
            gs["HUMD"].to_numpy(dtype=np.float64),
            sem_s,
            gs["opening"].to_numpy(dtype=np.float64),
            gs["exam"].to_numpy(dtype=np.float64),
            gs["event_any_venue"].to_numpy(dtype=np.float64),
            gs["event_any_campus"].to_numpy(dtype=np.float64),
            gs["is_friday_work"].to_numpy(dtype=np.float64),
            gs["is_typhoon"].to_numpy(dtype=np.float64),
            gs["policy_ramp_60d"].to_numpy(dtype=np.float64),
        ])
        K = X_cols.shape[1]
        X_sp = sparse.csc_matrix(X_cols)
        P_tiny = np.eye(K) * 0.05
        th, _, mu, _, _ = fit_penalized_glm_fe_core(
            y, X_sp, off, g_idx, G, 0.20, Z=Z, P_theta=P_tiny, max_iter=20, tol=1e-6
        )
        w = mu / (1.0 + 0.20 * mu)
        w_g = np.asarray(Z.T @ w).ravel()
        WXs = sparse.diags(w) @ X_sp
        sum_WX = (Z.T @ WXs).toarray()
        H = (X_sp.T @ WXs).toarray() - (sum_WX.T / np.maximum(w_g, 1e-12)) @ sum_WX + P_tiny
        se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(H)), 1e-10))

        am_s = sem_s * ((hr_s >= 7) & (hr_s <= 11))
        pm_s = sem_s * ((hr_s >= 16) & (hr_s <= 20))
        ot_s = sem_s * (~((hr_s >= 7) & (hr_s <= 11)) & ~((hr_s >= 16) & (hr_s <= 20)))
        X_b = np.column_stack([X_cols[:, :3], am_s, pm_s, ot_s, X_cols[:, 4:]])
        Kb = X_b.shape[1]
        Xb_sp = sparse.csc_matrix(X_b)
        Pb = np.eye(Kb) * 0.05
        th_b, _, mu_b, _, _ = fit_penalized_glm_fe_core(
            y, Xb_sp, off, g_idx, G, 0.20, Z=Z, P_theta=Pb, max_iter=20, tol=1e-6
        )
        wb = mu_b / (1.0 + 0.20 * mu_b)
        wgb = np.asarray(Z.T @ wb).ravel()
        WXb = sparse.diags(wb) @ Xb_sp
        sWXb = (Z.T @ WXb).toarray()
        Hb = (Xb_sp.T @ WXb).toarray() - (sWXb.T / np.maximum(wgb, 1e-12)) @ sWXb + Pb
        se_b = np.sqrt(np.maximum(np.diag(np.linalg.pinv(Hb)), 1e-10))

        rec = {"station_id": sid, "side": side}
        map_a = [
            ("log1p_Precip", 0), ("Temp", 1), ("HUMD", 2), ("semester_raw", 3),
            ("opening", 4), ("exam", 5), ("event_any_venue", 6), ("event_any_campus", 7),
            ("is_friday_work", 8), ("is_typhoon", 9), ("policy_ramp_60d", 10),
        ]
        for nm, idx_c in map_a:
            rec[f"b_{nm}"] = float(th[idx_c])
            rec[f"se_{nm}"] = float(se[idx_c])
        rec["b_sem_am"] = float(th_b[3])
        rec["se_sem_am"] = float(se_b[3])
        rec["b_sem_pm"] = float(th_b[4])
        rec["se_sem_pm"] = float(se_b[4])
        rows.append(rec)

    stn_slopes = pd.DataFrame(rows).set_index("station_id").reindex(station_order).reset_index()

    summary_rows = []
    for var_key, var_label in target_vars:
        b = stn_slopes[f"b_{var_key}"].to_numpy(dtype=np.float64)
        s = stn_slopes[f"se_{var_key}"].to_numpy(dtype=np.float64)
        ok = np.isfinite(b) & np.isfinite(s) & (s > 0)
        b_ok, s_ok = b[ok], s[ok]
        w_iv = 1.0 / (s_ok ** 2)
        b_bar = float(np.sum(w_iv * b_ok) / np.sum(w_iv))
        Q = float(np.sum(w_iv * (b_ok - b_bar) ** 2))
        df_q = max(len(b_ok) - 1, 1)
        I2 = max(0.0, (Q - df_q) / max(Q, 1e-12)) * 100.0
        C_dl = float(np.sum(w_iv) - np.sum(w_iv ** 2) / np.sum(w_iv))
        tau = math.sqrt(max(0.0, (Q - df_q) / max(C_dl, 1e-12)))

        W_sub = W_geo[np.ix_(ok, ok)]
        b_c = b_ok - b_ok.mean()
        denom_m = float(np.dot(b_c, b_c))
        moran_i = float((len(b_ok) / W_sub.sum()) * (b_c @ W_sub @ b_c) / denom_m) if (denom_m > 1e-12 and W_sub.sum() > 1e-12) else 0.0

        sign_major = 1.0 if b_bar >= 0 else -1.0
        opp_sign_pct = float(np.mean((b_ok * sign_major) < 0) * 100.0)
        q10, q25, q50, q75, q90 = np.quantile(b_ok, [0.10, 0.25, 0.50, 0.75, 0.90])

        if tau < 0.08 and opp_sign_pct < 15.0:
            shock_type = "全域一致衝擊 (Global Uniform)"
        elif moran_i >= 0.18:
            shock_type = "空間連續擴散局部衝擊 (Spatially Diffused)"
        else:
            shock_type = "站點功能極化衝擊 (Function Polarized)"

        summary_rows.append({
            "side": side,
            "variable": var_key,
            "label": var_label,
            "mean_beta": b_bar,
            "median_beta": q50,
            "tau_hetero_sd": tau,
            "IQR_beta": q75 - q25,
            "spread_90_10": q90 - q10,
            "I2_pct": I2,
            "opp_sign_pct": opp_sign_pct,
            "moran_I_geo": moran_i,
            "classification": shock_type,
        })

    return pd.DataFrame(summary_rows), stn_slopes


# ---------------------------------------------------------------------------
# 11. Parallel Job Runner (macOS spawn-safe)
# ---------------------------------------------------------------------------

def _worker_run_job(job: dict[str, Any]) -> dict[str, Any]:
    test_dir = Path(job["test_dir"])
    panel = pd.read_parquet(test_dir / "derived_data" / "panel_minute_floored_2024_2025.parquet")
    stn_df = pd.read_csv(test_dir / "derived_data" / "station_continuous_features.csv", dtype={"station_id": str})

    side = job["side"]
    y_col = "y_rent" if side == "Rent Y+" else "y_ret"
    exp_col = "exp_rent_eff" if side == "Rent Y+" else "exp_ret_eff"
    v_col = "valid_rent" if side == "Rent Y+" else "valid_ret"

    if "train_end_date" in job:
        tr24 = panel[(panel["date"] <= job["train_end_date"]) & panel[v_col]].reset_index(drop=True)
        common_sids = set(tr24["station_id"].unique())
        te_mask = (panel["date"] >= job["val_start_date"]) & (panel["date"] <= job["val_end_date"]) & panel[v_col]
        if job.get("restrict_common_stations", True):
            te_mask = te_mask & panel["station_id"].isin(common_sids)
        te25 = panel[te_mask].reset_index(drop=True)
    else:
        tr24 = panel[(panel["year"] == 2024) & panel[v_col]].reset_index(drop=True)
        common_sids = set(tr24["station_id"].unique())
        te_mask = (panel["year"] == 2025) & panel[v_col]
        if job.get("restrict_common_stations", True):
            te_mask = te_mask & panel["station_id"].isin(common_sids)
        te25 = panel[te_mask].reset_index(drop=True)

    kind = job["kind"]
    if kind == "hist":
        out = fit_predict_hist_baseline(tr24, te25, y_col, exp_col, side)
    elif kind == "glm":
        out = fit_and_predict_glm_spec(
            tr24, te25, stn_df, y_col, exp_col,
            spec=job["spec"],
            model_name=job["model_name"],
            side=side,
            family=job.get("family", "nb2"),
            fe_mode=job.get("fe_mode", "station_hour_wd"),
            recency_halflife_days=job.get("recency_halflife_days", None),
            lam_ridge=job.get("lam_ridge", 25.0),
            lam_lap=job.get("lam_lap", 80.0),
            fe_shrink_lambda=job.get("fe_shrink_lambda", 0.0),
            fixed_alpha=job.get("fixed_alpha", None),
            return_train_preds=False,
            policy_ramp_override=job.get("policy_ramp_override", None),
            include_policy_step=job.get("include_policy_step", None),
        )
    elif kind == "gbdt_on_anchor":
        anchor = fit_and_predict_glm_spec(
            tr24, te25, stn_df, y_col, exp_col,
            spec=job["anchor_spec"], model_name="anchor", side=side,
            family=job.get("anchor_family", "poisson"),
            recency_halflife_days=job.get("recency_halflife_days", 120.0),
            lam_ridge=job.get("lam_ridge", 150.0), lam_lap=job.get("lam_lap", 600.0),
            return_train_preds=True,
            policy_ramp_override=job.get("policy_ramp_override", None),
            include_policy_step=job.get("include_policy_step", None),
        )
        out = fit_exposure_hist_gbdt(
            tr24, te25, stn_df, y_col, exp_col,
            init_rate_train=anchor["pred_train"], init_rate_test=anchor["pred_test_raw"],
            model_name=job["model_name"], side=side,
            n_trees=job.get("n_trees", 60), max_depth=job.get("max_depth", 4),
            lr=job.get("lr", 0.06), reg_lambda=job.get("reg_lambda", 40.0),
            alpha_nb2=job.get("alpha_nb2", 0.0),
            recency_halflife_days=job.get("recency_halflife_days", 120.0),
        )
    elif kind == "pytorch_standalone":
        out = fit_pytorch_nb2_net(
            tr24, te25, stn_df, y_col, exp_col,
            model_name=job["model_name"], side=side,
            loss_type=job.get("loss_type", "nb2"),
            epochs=job.get("epochs", 12),
            recency_halflife_days=job.get("recency_halflife_days", 120.0),
        )
    elif kind == "pytorch_residual":
        anchor = fit_and_predict_glm_spec(
            tr24, te25, stn_df, y_col, exp_col,
            spec=job["anchor_spec"], model_name="anchor", side=side,
            family=job.get("anchor_family", "poisson"),
            recency_halflife_days=job.get("recency_halflife_days", 120.0),
            lam_ridge=job.get("lam_ridge", 150.0), lam_lap=job.get("lam_lap", 600.0),
            return_train_preds=True,
            policy_ramp_override=job.get("policy_ramp_override", None),
            include_policy_step=job.get("include_policy_step", None),
        )
        out = fit_pytorch_nb2_net(
            tr24, te25, stn_df, y_col, exp_col,
            model_name=job["model_name"], side=side,
            init_rate_train=anchor["pred_train"], init_rate_test=anchor["pred_test_raw"],
            loss_type=job.get("loss_type", "poisson"),
            epochs=job.get("epochs", 8), lr=job.get("lr", 1.5e-3),
            weight_decay=job.get("weight_decay", 3e-3),
            recency_halflife_days=job.get("recency_halflife_days", 120.0),
        )
    else:
        raise ValueError(f"Unknown job kind: {kind}")

    if not job.get("return_preds", False):
        out.pop("pred_test_raw", None)
        out.pop("pred_test_causal", None)
    if not job.get("return_coefs", False):
        out.pop("theta_hat", None)
        out.pop("feat_names", None)
    out.pop("station_embeddings", None)
    return out


def run_parallel_jobs(jobs: list[dict[str, Any]], max_workers: int = 4) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_to_idx = {ex.submit(_worker_run_job, j): i for i, j in enumerate(jobs)}
        for fut in as_completed(fut_to_idx):
            results[fut_to_idx[fut]] = fut.result()
    return [r for r in results if r is not None]
