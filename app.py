# app.py
# VBO Lap Viewer + Compare (Auto everything, click-to-map cursor)
# Works with RaceLogic / RaceBox .vbo (whitespace data)
#
# Dependencies:
#   streamlit, pandas, numpy, plotly, streamlit-plotly-events
#
# Run:
#   cd ~/Desktop
#   source .venv/bin/activate
#   streamlit run app.py

from __future__ import annotations

import io
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events


# -----------------------------
# Utilities
# -----------------------------

def _sec_to_mmss(ss: float) -> str:
    if not np.isfinite(ss):
        return "—"
    m = int(ss // 60)
    s = ss - 60 * m
    return f"{m}:{s:06.3f}"


def _robust_float_array(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _is_hhmmss_series(t: np.ndarray) -> bool:
    t2 = t[np.isfinite(t)]
    if t2.size == 0:
        return False
    mx = float(np.nanmax(t2))
    # Typical HHMMSS.sss ranges in VBO: 000000..235959
    return 10000 < mx < 300000


def _hhmmss_to_seconds(v: float) -> float:
    # v like 125430.123 = 12:54:30.123
    hh = int(v // 10000)
    rem = v - hh * 10000
    mm = int(rem // 100)
    ss = rem - mm * 100
    return hh * 3600 + mm * 60 + ss


def _guess_latlon_units(series: pd.Series) -> str:
    x = _robust_float_array(series)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "deg"
    p99 = float(np.nanpercentile(np.abs(x), 99))
    # RaceLogic/RaceBox often store as minutes (e.g., 775.23) => >200
    return "min" if p99 > 200 else "deg"


def _convert_latlon_to_deg(lat_raw: pd.Series, lon_raw: pd.Series) -> Tuple[np.ndarray, np.ndarray, str, str]:
    lat_u = _guess_latlon_units(lat_raw)
    lon_u = _guess_latlon_units(lon_raw)
    lat = _robust_float_array(lat_raw)
    lon = _robust_float_array(lon_raw)
    if lat_u == "min":
        lat = lat / 60.0
    if lon_u == "min":
        lon = lon / 60.0
    return lat, lon, lat_u, lon_u


def _normalize_lon_sign(lon_deg: np.ndarray) -> np.ndarray:
    # Many VBOs store longitude minutes with negative sign but some are flipped.
    # We try to infer Thailand region (~100E). If median is near -100, flip.
    finite = lon_deg[np.isfinite(lon_deg)]
    if finite.size == 0:
        return lon_deg
    med = float(np.nanmedian(finite))
    # choose sign that makes it closer to +100 than -100
    if abs(med + 100) < abs(med - 100):
        return -lon_deg
    return lon_deg


def _convert_speed_to_kmh(speed_raw: pd.Series) -> np.ndarray:
    v = _robust_float_array(speed_raw)
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return v
    p95 = float(np.nanpercentile(vv, 95))
    vmax = float(np.nanmax(vv))
    # If it's m/s, typical vmax ~ 40-80 (<=100), p95 < 70
    if p95 < 70 and vmax < 100:
        return v * 3.6
    return v


def _equirect_xy(lat_deg: np.ndarray, lon_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    R = 6371000.0
    finite = np.isfinite(lat_deg) & np.isfinite(lon_deg)
    lat0 = float(np.nanmedian(lat_deg[finite])) if finite.any() else 0.0
    lon0 = float(np.nanmedian(lon_deg[finite])) if finite.any() else 0.0

    lat0r = np.deg2rad(lat0)
    lon0r = np.deg2rad(lon0)
    latr = np.deg2rad(lat_deg)
    lonr = np.deg2rad(lon_deg)

    x = (lonr - lon0r) * np.cos(lat0r) * R
    y = (latr - lat0r) * R
    return x, y, lat0, lon0


def _project_point(lat_deg: float, lon_deg: float, lat0: float, lon0: float) -> np.ndarray:
    R = 6371000.0
    lat0r = np.deg2rad(lat0)
    lon0r = np.deg2rad(lon0)
    x = (np.deg2rad(lon_deg) - lon0r) * np.cos(lat0r) * R
    y = (np.deg2rad(lat_deg) - np.deg2rad(lat0)) * R
    return np.array([x, y], dtype=float)


def _auto_gate_params(x: np.ndarray, y: np.ndarray, speed_kmh: np.ndarray) -> Tuple[float, float, float]:
    xr = float(np.nanmax(x) - np.nanmin(x))
    yr = float(np.nanmax(y) - np.nanmin(y))
    diag = float(np.hypot(xr, yr))
    gate_radius = float(np.clip(0.10 * diag, 60.0, 350.0))
    gate_width = float(np.clip(0.02 * diag, 8.0, 60.0))
    s = speed_kmh[np.isfinite(speed_kmh)]
    min_speed = float(np.clip(np.nanpercentile(s, 20), 30.0, 120.0)) if s.size else 30.0
    return gate_width, gate_radius, min_speed


def _score_sf_line(A: np.ndarray, B: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    P = np.stack([x, y], axis=1)
    d = B - A
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return 1e9
    u = d / L
    n = np.array([-u[1], u[0]])
    dist = np.abs((P - A) @ n)
    M = (A + B) / 2.0
    r = np.hypot(P[:, 0] - M[0], P[:, 1] - M[1])
    mask = r < 800
    if mask.sum() < 50:
        mask = np.ones(len(P), dtype=bool)
    return float(np.nanpercentile(dist[mask], 1))


def _detect_crossings(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    speed_kmh: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    gate_width_m: float,
    gate_radius_m: float,
    min_speed_kmh: float,
    direction_lock: bool,
) -> np.ndarray:
    P = np.stack([x, y], axis=1)
    d = (B - A).astype(float)
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return np.array([], dtype=float)

    u = d / (L + 1e-12)
    n = np.array([-u[1], u[0]])
    M = (A + B) / 2.0

    s = (P - A) @ n
    flip = (s[:-1] * s[1:]) < 0
    idx = np.where(flip)[0]
    if idx.size == 0:
        return np.array([], dtype=float)

    close = np.minimum(np.abs(s[idx]), np.abs(s[idx + 1])) <= gate_width_m
    idx = idx[close]
    if idx.size == 0:
        return np.array([], dtype=float)

    a = np.abs(s[idx])
    b = np.abs(s[idx + 1])
    w = a / (a + b + 1e-12)

    tc = t[idx] + w * (t[idx + 1] - t[idx])
    xc = x[idx] + w * (x[idx + 1] - x[idx])
    yc = y[idx] + w * (y[idx + 1] - y[idx])
    vc = speed_kmh[idx] + w * (speed_kmh[idx + 1] - speed_kmh[idx])

    dist_mid = np.hypot(xc - M[0], yc - M[1])
    keep = (dist_mid <= gate_radius_m) & np.isfinite(vc) & (vc >= min_speed_kmh)
    tc = tc[keep]
    idx = idx[keep]
    if tc.size == 0:
        return np.array([], dtype=float)

    if direction_lock:
        # keep only crossings in the dominant direction
        track_dir = np.array([-u[1], u[0]])
        motion = np.stack([x[idx + 1] - x[idx], y[idx + 1] - y[idx]], axis=1)
        mun = motion / (np.linalg.norm(motion, axis=1)[:, None] + 1e-12)
        if float(np.nanmean(mun @ track_dir)) < 0:
            track_dir = -track_dir
        tc = tc[(mun @ track_dir) > 0]

    return np.sort(tc)


def _dedupe_crossings(tc: np.ndarray) -> np.ndarray:
    if tc.size <= 1:
        return tc
    dt = np.diff(tc)
    med = float(np.median(dt)) if dt.size else 0.0
    min_dt = max(20.0, 0.55 * med) if med > 0 else 25.0
    keep = [0]
    for i in range(1, len(tc)):
        if tc[i] - tc[keep[-1]] >= min_dt:
            keep.append(i)
    return tc[np.array(keep, dtype=int)]


def _build_laps_table(df: pd.DataFrame, tc: np.ndarray) -> pd.DataFrame:
    if tc.size < 2:
        return pd.DataFrame()

    t = df["t_s"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    v = df["speed_kmh"].to_numpy(dtype=float)

    rows = []
    for i, (ta, tb) in enumerate(zip(tc[:-1], tc[1:]), start=1):
        m = (t >= ta) & (t < tb)
        if int(m.sum()) < 10:
            continue
        xs = x[m]
        ys = y[m]
        vs = v[m]

        steps = np.hypot(np.diff(xs), np.diff(ys))
        steps = np.clip(steps, 0.0, 30.0)
        dist = float(np.nansum(steps))
        lap_time = float(tb - ta)
        vmax = float(np.nanmax(vs)) if np.isfinite(vs).any() else float("nan")
        rows.append({"lap_idx": i, "lap_time_s": lap_time, "lap_dist_m": dist, "vmax_kmh": vmax, "t0": ta, "t1": tb})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Filter out out-laps / in-laps by distance + time clustering
    d = out["lap_dist_m"].to_numpy(float)
    tt = out["lap_time_s"].to_numpy(float)
    dmed = float(np.median(d))
    tmed = float(np.median(tt))
    keep = (d > 0.90 * dmed) & (d < 1.10 * dmed) & (tt > 0.80 * tmed) & (tt < 1.25 * tmed)
    out2 = out.loc[keep].copy()
    if out2.empty:
        out2 = out.copy()

    out2 = out2.reset_index(drop=True)
    out2["lap"] = np.arange(1, len(out2) + 1)

    best = float(out2["lap_time_s"].min())
    out2["delta_s"] = out2["lap_time_s"] - best
    out2["lap_time"] = out2["lap_time_s"].map(_sec_to_mmss)

    return out2[["lap", "lap_time", "lap_time_s", "delta_s", "vmax_kmh", "lap_dist_m", "t0", "t1"]]


def _resample_by_distance(seg: pd.DataFrame, n: int = 1200) -> Dict[str, np.ndarray]:
    t = seg["t_s"].to_numpy(dtype=float)
    x = seg["x"].to_numpy(dtype=float)
    y = seg["y"].to_numpy(dtype=float)
    v = seg["speed_kmh"].to_numpy(dtype=float)

    if t.size < 5:
        return {}

    steps = np.hypot(np.diff(x), np.diff(y))
    steps = np.clip(steps, 0.0, 30.0)
    s = np.concatenate([[0.0], np.cumsum(steps)])

    L = float(s[-1])
    if L <= 1e-6:
        return {}

    sn = s / L
    grid = np.linspace(0.0, 1.0, int(n))
    tr = t - float(t[0])

    # Ensure strictly increasing for interp
    order = np.argsort(sn)
    sn2 = sn[order].copy()
    tr2 = tr[order]
    v2 = v[order]
    x2 = x[order]
    y2 = y[order]
    eps = 1e-9
    for i in range(1, len(sn2)):
        if sn2[i] <= sn2[i - 1]:
            sn2[i] = sn2[i - 1] + eps

    out = {
        "grid": grid,
        "t": np.interp(grid, sn2, tr2),
        "v": np.interp(grid, sn2, v2),
        "x": np.interp(grid, sn2, x2),
        "y": np.interp(grid, sn2, y2),
        "lap_len_m": np.array([L], dtype=float),
    }
    return out


# -----------------------------
# VBO parsing
# -----------------------------

def _parse_vbo_sections(raw_text: str) -> Dict[str, str]:
    # sections like [header], [column names], [data], [laptiming]
    sections: Dict[str, str] = {}
    pattern = re.compile(r"\[([^\]]+)\]\s*(.*?)(?=\n\[[^\]]+\]|\Z)", re.S)
    for m in pattern.finditer(raw_text):
        name = m.group(1).strip().lower()
        body = m.group(2).strip()
        sections[name] = body
    return sections


def _parse_vbo_file(file_bytes: bytes) -> Tuple[Dict[str, str], pd.DataFrame]:
    txt = file_bytes.decode("utf-8", errors="ignore")
    sec = _parse_vbo_sections(txt)

    col_raw = sec.get("column names") or sec.get("columnnames") or ""
    cols = [c.strip() for c in col_raw.replace(",", " ").split() if c.strip()]
    cols = [c.lower() for c in cols]

    data_raw = sec.get("data", "")
    if not data_raw.strip():
        raise ValueError("ไม่พบ [data] ในไฟล์ .vbo")

    # Read whitespace separated data
    df = pd.read_csv(io.StringIO(data_raw), sep=r"\s+", names=cols if cols else None, engine="python")

    # If no headers found, create generic
    if not cols:
        df.columns = [f"c{i}" for i in range(df.shape[1])]

    return sec, df


def _pick_columns(df: pd.DataFrame) -> Dict[str, str]:
    # auto map synonyms
    cols = [c.lower() for c in df.columns.astype(str).tolist()]
    def first_match(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in cols:
                return cand
        # partial contains
        for c in cols:
            for cand in candidates:
                if cand in c:
                    return c
        return None

    time_col = first_match(["time", "timestamp", "t"])
    lat_col = first_match(["lat", "latitude"])
    lon_col = first_match(["lon", "lng", "long", "longitude"])
    speed_col = first_match(["velocity", "speed", "v"])

    return {
        "time": time_col or "",
        "lat": lat_col or "",
        "lon": lon_col or "",
        "speed": speed_col or "",
    }


def _sf_candidates_from_laptiming_text(laptiming: str) -> List[Tuple[float, float, float, float, str]]:
    """
    Returns candidates in degrees (lat1,lon1,lat2,lon2) plus a label.
    laptiming example:
        Start +00775.25926 -006060.46859 +00775.26416 -006060.47089 ¬ Start / Finish
    Those look like minutes. We'll convert minutes->deg and also try lon sign +/-
    """
    # capture 4 numbers after 'Start'
    m = re.search(r"Start\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)", laptiming)
    if not m:
        return []
    a, b, c, d = [float(x) for x in m.groups()]
    # treat as minutes unless obviously degrees
    # (these are ~775 and ~-6060 => minutes)
    # convert minutes to degrees
    cands = []
    for sgn in [1.0, -1.0]:
        lat1 = a / 60.0
        lon1 = sgn * (b / 60.0)
        lat2 = c / 60.0
        lon2 = sgn * (d / 60.0)
        cands.append((lat1, lon1, lat2, lon2, f"laptiming(latlon_sgn={int(sgn)})"))

    # also swapped interpretation (sometimes columns are swapped)
    for sgn in [1.0, -1.0]:
        lat1 = b / 60.0
        lon1 = sgn * (a / 60.0)
        lat2 = d / 60.0
        lon2 = sgn * (c / 60.0)
        cands.append((lat1, lon1, lat2, lon2, f"laptiming(swapped_sgn={int(sgn)})"))

    # filter valid lat range
    out = []
    for lat1, lon1, lat2, lon2, label in cands:
        if abs(lat1) <= 90 and abs(lat2) <= 90 and abs(lon1) <= 180 and abs(lon2) <= 180:
            out.append((lat1, lon1, lat2, lon2, label))
    return out


def _auto_sf_from_trace(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, float]]:
    """
    Fallback if no laptiming: choose a gate perpendicular to the longest straight.
    This is a simple heuristic (good enough for MVP).
    """
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    v = df["speed_kmh"].to_numpy(float)

    # use top-speed points cluster
    finite = np.isfinite(v)
    if finite.sum() < 50:
        raise ValueError("GPS/Speed ข้อมูลน้อยเกินไปสำหรับ auto start/finish")
    thr = float(np.nanpercentile(v[finite], 95))
    m = finite & (v >= thr)
    if m.sum() < 10:
        m = finite

    xm = float(np.nanmedian(x[m]))
    ym = float(np.nanmedian(y[m]))

    # local direction by PCA on neighborhood
    P = np.stack([x[m] - xm, y[m] - ym], axis=1)
    C = np.cov(P.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    dir_vec = eigvecs[:, np.argmax(eigvals)]
    # gate is perpendicular
    n = np.array([-dir_vec[1], dir_vec[0]])
    n = n / (np.linalg.norm(n) + 1e-12)

    gate_len = 40.0
    A = np.array([xm, ym], float) - n * gate_len
    B = np.array([xm, ym], float) + n * gate_len
    params = {"gate_len_m": gate_len, "speed_thr": thr}
    return A, B, "auto(fallback)", params


# -----------------------------
# Streamlit app
# -----------------------------

st.set_page_config(page_title="VBO Lap Viewer + Compare", layout="wide")

st.title("VBO Lap Viewer + Compare")
st.caption("Auto mapping • Auto Start/Finish • Click on graph to show position on map (Circuit-Tools style)")

uploaded = st.file_uploader("Upload .vbo (RaceLogic / RaceBox)", type=["vbo"], accept_multiple_files=False)

if "cursor_idx" not in st.session_state:
    st.session_state["cursor_idx"] = 0

if "last_file_sig" not in st.session_state:
    st.session_state["last_file_sig"] = ""

if not uploaded:
    st.info("อัปโหลดไฟล์ .vbo ก่อน แล้วระบบจะทำทุกอย่างอัตโนมัติ")
    st.stop()

file_bytes = uploaded.getvalue()
file_sig = f"{uploaded.name}-{len(file_bytes)}"

# reset cursor when file changes
if st.session_state["last_file_sig"] != file_sig:
    st.session_state["cursor_idx"] = 0
    st.session_state["last_file_sig"] = file_sig

try:
    sec, df_raw = _parse_vbo_file(file_bytes)
except Exception as e:
    st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")
    st.stop()

auto_map = _pick_columns(df_raw)

# Advanced mapping (optional)
with st.sidebar:
    st.subheader("Automatic mapping")
    st.write({k: v for k, v in auto_map.items()})

    adv = st.toggle("Advanced: override column mapping", value=False)
    if adv:
        cols = df_raw.columns.astype(str).tolist()
        time_col = st.selectbox("Time column", cols, index=cols.index(auto_map["time"]) if auto_map["time"] in cols else 0)
        lat_col = st.selectbox("Latitude column", cols, index=cols.index(auto_map["lat"]) if auto_map["lat"] in cols else 0)
        lon_col = st.selectbox("Longitude column", cols, index=cols.index(auto_map["lon"]) if auto_map["lon"] in cols else 0)
        spd_col = st.selectbox("Speed/Velocity column", cols, index=cols.index(auto_map["speed"]) if auto_map["speed"] in cols else 0)
        auto_map = {"time": time_col.lower(), "lat": lat_col.lower(), "lon": lon_col.lower(), "speed": spd_col.lower()}

    plot_n = st.selectbox("Plot resolution", [600, 900, 1200, 1600, 2000], index=2)

# Validate mapping
need = ["time", "lat", "lon", "speed"]
missing = [k for k in need if not auto_map.get(k)]
if missing:
    st.error(f"ไฟล์นี้หา column ไม่เจอ: {missing}  (เปิด Advanced mapping แล้วเลือกเองได้)")
    st.stop()

# Normalize df columns to lower
df_raw.columns = df_raw.columns.astype(str).str.lower()

time_raw = df_raw[auto_map["time"]]
lat_raw = df_raw[auto_map["lat"]]
lon_raw = df_raw[auto_map["lon"]]
speed_raw = df_raw[auto_map["speed"]]

lat_deg, lon_deg, lat_u, lon_u = _convert_latlon_to_deg(lat_raw, lon_raw)
lon_deg = _normalize_lon_sign(lon_deg)

t_arr = _robust_float_array(time_raw)
if _is_hhmmss_series(t_arr):
    t_s = np.array([_hhmmss_to_seconds(v) if np.isfinite(v) else np.nan for v in t_arr], dtype=float)
else:
    t_s = t_arr.copy()

# make time start at 0
finite_t = np.isfinite(t_s)
t0 = float(np.nanmin(t_s[finite_t])) if finite_t.any() else 0.0
t_s = t_s - t0

speed_kmh = _convert_speed_to_kmh(speed_raw)

# Basic validity mask
mask = np.isfinite(t_s) & np.isfinite(lat_deg) & np.isfinite(lon_deg) & np.isfinite(speed_kmh)
df = pd.DataFrame(
    {
        "t_s": t_s[mask],
        "lat_deg": lat_deg[mask],
        "lon_deg": lon_deg[mask],
        "speed_kmh": speed_kmh[mask],
    }
).reset_index(drop=True)

# XY projection
x, y, lat0, lon0 = _equirect_xy(df["lat_deg"].to_numpy(float), df["lon_deg"].to_numpy(float))
df["x"] = x
df["y"] = y

# Start/Finish selection
sf_source = "auto"
sf_params = {}

A = B = None
laptiming_txt = sec.get("laptiming", "")
if laptiming_txt:
    cands = _sf_candidates_from_laptiming_text(laptiming_txt)
    if cands:
        gate_width, gate_radius, min_speed = _auto_gate_params(x, y, df["speed_kmh"].to_numpy(float))
        best = None
        best_n = -1
        best_score = 1e18
        best_label = ""
        for lat1, lon1, lat2, lon2, label in cands:
            A0 = _project_point(lat1, lon1, lat0, lon0)
            B0 = _project_point(lat2, lon2, lat0, lon0)
            sc = _score_sf_line(A0, B0, x, y)
            tc0 = _dedupe_crossings(
                _detect_crossings(
                    df["t_s"].to_numpy(float),
                    x,
                    y,
                    df["speed_kmh"].to_numpy(float),
                    A0,
                    B0,
                    gate_width,
                    gate_radius,
                    min_speed,
                    direction_lock=True,
                )
            )
            n = int(tc0.size)
            if (n > best_n) or (n == best_n and sc < best_score):
                best = (A0, B0, tc0)
                best_n = n
                best_score = sc
                best_label = label
        if best and best_n >= 2:
            A, B, tc = best
            sf_source = best_label
            sf_params = {"gate_width_m": gate_width, "gate_radius_m": gate_radius, "min_speed_kmh": min_speed, "score": best_score, "crossings": best_n}
        else:
            A = B = None

if A is None or B is None:
    # fallback
    try:
        A, B, sf_source, sf_params = _auto_sf_from_trace(df)
    except Exception as e:
        st.error(f"Auto Start/Finish ไม่สำเร็จ: {e}")
        st.stop()

# Crossings & laps
gate_width, gate_radius, min_speed = _auto_gate_params(x, y, df["speed_kmh"].to_numpy(float))
tc = _dedupe_crossings(
    _detect_crossings(
        df["t_s"].to_numpy(float),
        x,
        y,
        df["speed_kmh"].to_numpy(float),
        A,
        B,
        gate_width,
        gate_radius,
        min_speed,
        direction_lock=True,
    )
)

laps = _build_laps_table(df, tc)

# UI Tabs
tab_overview, tab_compare, tab_raw = st.tabs(["Overview", "Compare Laps", "Raw Data"])

with tab_overview:
    c1, c2, c3 = st.columns([1.2, 1.2, 1.0], gap="large")

    with c1:
        st.subheader("Track trace + Start/Finish")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["x"].to_list(), y=df["y"].to_list(), mode="lines", name="Trace"))
        fig.add_trace(go.Scatter(x=[float(A[0]), float(B[0])], y=[float(A[1]), float(B[1])], mode="lines", name="S/F"))
        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            showlegend=True,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"S/F source: {sf_source} • params: {sf_params}")

    with c2:
        st.subheader("Lap table (Circuit-Tools style)")
        if laps.empty:
            st.warning("ยังตัดรอบไม่สำเร็จ (crossings น้อยเกินไป) — ลองไฟล์อื่น หรือ GPS/Speed อาจไม่ครบ")
        else:
            show = laps.copy()
            show["delta_s"] = show["delta_s"].map(lambda v: f"{v:+.2f}")
            show["vmax_kmh"] = show["vmax_kmh"].map(lambda v: f"{v:.1f}")
            show["lap_dist_m"] = show["lap_dist_m"].map(lambda v: f"{v:,.0f}")
            st.dataframe(show[["lap", "lap_time", "delta_s", "vmax_kmh", "lap_dist_m"]], use_container_width=True, hide_index=True)

            best_i = int(laps["lap_time_s"].idxmin())
            st.success(f"Best lap: Lap {int(laps.loc[best_i,'lap'])} — {laps.loc[best_i,'lap_time']}")

    with c3:
        st.subheader("Quick checks")
        st.write(f"Rows: {len(df):,}")
        st.write(f"Lat/Lon units: lat={lat_u}, lon={lon_u}")
        st.write(f"Speed range (km/h): {float(np.nanmin(df['speed_kmh'])):.1f} → {float(np.nanmax(df['speed_kmh'])):.1f}")
        st.write(f"Crossings detected: {int(tc.size)}")
        if not laps.empty:
            st.write(f"Laps (valid): {len(laps)}")
        else:
            st.write("Laps (valid): 0")

with tab_compare:
    if laps.empty or len(laps) < 2:
        st.warning("ต้องมีอย่างน้อย 2 lap ที่ตัดรอบได้ก่อน ถึงจะ Compare ได้")
        st.stop()

    # Build labels like "Lap 3 — 1:13.360"
    lap_labels = [f"Lap {int(r.lap)} — {r.lap_time}" for r in laps.itertuples(index=False)]

    # Default: best lap as reference, second best as compare
    best_order = laps.sort_values("lap_time_s").reset_index(drop=True)
    default_ref = int(best_order.loc[0, "lap"]) - 1
    default_cmp = int(best_order.loc[1, "lap"]) - 1

    colA, colB, colC = st.columns([1.1, 1.1, 0.9], gap="large")
    with colA:
        ref_idx = st.selectbox("Reference lap", list(range(len(laps))), index=default_ref, format_func=lambda i: lap_labels[i], key="ref_sel")
    with colB:
        cmp_idx = st.selectbox("Compare lap", list(range(len(laps))), index=default_cmp, format_func=lambda i: lap_labels[i], key="cmp_sel")
    with colC:
        st.write(" ")
        st.write(" ")
        st.metric("Gap (Compare − Ref)", f"{(laps.loc[cmp_idx,'lap_time_s']-laps.loc[ref_idx,'lap_time_s']):+.3f} s")

    ref = laps.iloc[ref_idx]
    cmp_ = laps.iloc[cmp_idx]

    seg_ref = df[(df["t_s"] >= float(ref.t0)) & (df["t_s"] < float(ref.t1))].copy()
    seg_cmp = df[(df["t_s"] >= float(cmp_.t0)) & (df["t_s"] < float(cmp_.t1))].copy()

    r = _resample_by_distance(seg_ref, int(plot_n))
    c = _resample_by_distance(seg_cmp, int(plot_n))

    if not r or not c:
        st.error("resample ไม่สำเร็จ (lap สั้น/ข้อมูลไม่พอ)")
        st.stop()

    grid = r["grid"]
    v_ref = r["v"]
    v_cmp = c["v"]
    dt = c["t"] - r["t"]  # Compare - Ref

    # Cursor controls
    # slider maps to index in [0..n-1]
    n = len(grid)
    st.session_state["cursor_idx"] = int(np.clip(st.session_state["cursor_idx"], 0, n - 1))

    cursor_idx = st.slider(
        "Cursor position (ลากเพื่อเลื่อนตำแหน่ง หรือคลิกบนกราฟด้านล่าง)",
        0,
        n - 1,
        value=int(st.session_state["cursor_idx"]),
        key="cursor_slider",
    )
    st.session_state["cursor_idx"] = cursor_idx

    x_cursor = float(grid[cursor_idx])

    # ---- Speed overlay plot (with click)
    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(x=grid.tolist(), y=v_ref.tolist(), mode="lines", name="Reference"))
    fig_speed.add_trace(go.Scatter(x=grid.tolist(), y=v_cmp.tolist(), mode="lines", name="Compare"))
    fig_speed.add_vline(x=x_cursor, line_width=2)
    fig_speed.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Speed overlay (km/h) — normalized lap distance (คลิกเพื่อเลือกตำแหน่ง)",
        xaxis_title="Lap distance (normalized 0→1)",
        yaxis_title="Speed (km/h)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ---- Delta time plot (with click)
    fig_dt = go.Figure()
    fig_dt.add_trace(go.Scatter(x=grid.tolist(), y=dt.tolist(), mode="lines", name="Delta t"))
    fig_dt.add_hline(y=0.0, line_width=1)
    fig_dt.add_vline(x=x_cursor, line_width=2, line_dash="dot")
    fig_dt.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Delta time (Compare − Reference) (คลิกเพื่อเลือกตำแหน่ง)",
        xaxis_title="Lap distance (normalized 0→1)",
        yaxis_title="Delta t (s)",
        showlegend=False,
    )

    # Render with click events (IMPORTANT: unique keys)
    st.markdown("### Graphs (click-to-cursor)")
    ev1 = plotly_events(fig_speed, click_event=True, hover_event=False, select_event=False, override_height=360, key="ev_speed")
    ev2 = plotly_events(fig_dt, click_event=True, hover_event=False, select_event=False, override_height=300, key="ev_dt")

    # Update cursor from click (prefer speed click if present)
    picked = None
    if ev1 and isinstance(ev1, list) and len(ev1) > 0 and "pointIndex" in ev1[0]:
        picked = int(ev1[0]["pointIndex"])
    elif ev2 and isinstance(ev2, list) and len(ev2) > 0 and "pointIndex" in ev2[0]:
        picked = int(ev2[0]["pointIndex"])

    if picked is not None and 0 <= picked < n and picked != st.session_state["cursor_idx"]:
        st.session_state["cursor_idx"] = picked
        # rerun to move vlines + map markers immediately
        st.rerun()

    # ---- Map with cursor markers
    st.markdown("### Map (cursor marker)")
    fig_map = go.Figure()
    fig_map.add_trace(go.Scatter(x=r["x"].tolist(), y=r["y"].tolist(), mode="lines", name="Ref lap"))
    fig_map.add_trace(go.Scatter(x=c["x"].tolist(), y=c["y"].tolist(), mode="lines", name="Compare lap"))

    # cursor markers
    fig_map.add_trace(go.Scatter(
        x=[float(r["x"][st.session_state["cursor_idx"]])],
        y=[float(r["y"][st.session_state["cursor_idx"]])],
        mode="markers",
        name="Cursor (Ref)",
        marker=dict(size=10),
    ))
    fig_map.add_trace(go.Scatter(
        x=[float(c["x"][st.session_state["cursor_idx"]])],
        y=[float(c["y"][st.session_state["cursor_idx"]])],
        mode="markers",
        name="Cursor (Compare)",
        marker=dict(size=10),
    ))

    # SF line
    fig_map.add_trace(go.Scatter(x=[float(A[0]), float(B[0])], y=[float(A[1]), float(B[1])], mode="lines", name="S/F"))

    fig_map.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_map.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig_map, use_container_width=True)

    # quick numeric readout at cursor
    st.markdown("### Cursor readout")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Distance (norm)", f"{float(grid[st.session_state['cursor_idx']]):.4f}")
    col2.metric("Speed Ref", f"{float(v_ref[st.session_state['cursor_idx']]):.1f} km/h")
    col3.metric("Speed Cmp", f"{float(v_cmp[st.session_state['cursor_idx']]):.1f} km/h")
    col4.metric("Delta t", f"{float(dt[st.session_state['cursor_idx']]):+.3f} s")

    # Safety / sanity (helps debugging)
    if float(np.nanmax(v_ref)) < 10 and float(np.nanmax(v_cmp)) < 10:
        st.warning("Speed ดูต่ำผิดปกติ (อาจ mapping speed ผิดคอลัมน์) — เปิด Advanced mapping แล้วลองเลือก speed/velocity ใหม่")

with tab_raw:
    st.subheader("Raw data preview")
    st.caption("แสดง sample เพื่อเช็คว่าคอลัมน์ map ถูกต้องหรือไม่")
    st.write("Detected columns:", ", ".join(df_raw.columns.astype(str).tolist()))
    st.dataframe(df_raw.head(200), use_container_width=True)
