"""Core logic for the EURUSD Level 1 macro panel: data loading, metrics, scoring.

No Streamlit imports here so it can be tested offline.
Convention (same as the Pine drivers panel): every metric gets an impact in [-1, +1]
on EURUSD = sign * clip(z, -2, 2) / 2. Block score = weighted average impact * 100.
Total score = weighted average of block scores using the block weights.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import requests

START = "2015-01-01"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
ECB_URL = "https://data-api.ecb.europa.eu/service/data/{flow}/{key}"

FRED_IDS = ["DFF", "DGS2", "DGS10", "T10Y2Y", "CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "PAYEMS", "UNRATE", "GDPC1"]
ECB_IDS = {
    "ECB_DFR": ("FM", "D.U2.EUR.4F.KR.DFR.LEV"),
    "EA_2Y": ("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y"),
    "EA_10Y": ("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y"),
    "EA_HICP": ("ICP", "M.U2.N.000000.4.ANR"),
    "EA_HICP_CORE": ("ICP", "M.U2.N.XEF000.4.ANR"),
    "EA_UNEMP": ("LFSI", "M.I9.S.UNEHRT.TOTAL0.15_74.T"),
    "EA_GDP": ("MNA", "Q.Y.I9.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY"),
    "EA_GDP_QQ": ("MNA", "Q.Y.I9.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.G1"),
}

BLOCKS = ["Rates & differentials", "Yield curve (info)", "Inflation", "Labor & growth", "Central bank tone"]
DEFAULT_BLOCK_WEIGHTS = {
    "Rates & differentials": 30,
    "Yield curve (info)": 0,
    "Inflation": 20,
    "Labor & growth": 20,
    "Central bank tone": 30,
}
TONE_LABELS = ["Very dovish", "Dovish", "Neutral", "Hawkish", "Very hawkish"]
TONE_VALUES = dict(zip(TONE_LABELS, [-2, -1, 0, 1, 2]))
DURATION_FACTORS = {"Data-dependent (short)": 0.5, "Sustained (long)": 1.0}


# ───────────────────────── data loading ─────────────────────────
def fetch_fred(series_id: str, api_key: str) -> pd.Series:
    r = requests.get(
        FRED_URL,
        params=dict(series_id=series_id, api_key=api_key, file_type="json", observation_start=START),
        timeout=25,
    )
    r.raise_for_status()
    obs = r.json()["observations"]
    s = pd.Series(
        [pd.to_numeric(o["value"], errors="coerce") for o in obs],
        index=pd.to_datetime([o["date"] for o in obs]),
        dtype="float64",
    )
    return s.dropna().sort_index()


def _parse_ecb_time(values) -> pd.DatetimeIndex:
    vals = [str(v) for v in values]
    if any("Q" in v for v in vals):
        return pd.PeriodIndex([v.replace("-", "") for v in vals], freq="Q").to_timestamp()
    return pd.to_datetime(vals)


def fetch_ecb(flow: str, key: str) -> pd.Series:
    r = requests.get(
        ECB_URL.format(flow=flow, key=key),
        params=dict(format="csvdata", startPeriod=START),
        headers={"Accept": "text/csv"},
        timeout=25,
    )
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values, index=_parse_ecb_time(df["TIME_PERIOD"]))
    return s.dropna().sort_index()


def load_all(api_key: str) -> tuple[dict, dict]:
    """Return (raw series dict, errors dict). A failing series never breaks the rest."""
    raw: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}
    for sid in FRED_IDS:
        try:
            raw[sid] = fetch_fred(sid, api_key)
        except Exception as e:  # noqa: BLE001
            errors[sid] = f"{type(e).__name__}: {e}"
    for name, (flow, key) in ECB_IDS.items():
        try:
            raw[name] = fetch_ecb(flow, key)
        except Exception as e:  # noqa: BLE001
            errors[name] = f"{type(e).__name__}: {e}"
    return raw, errors


# ───────────────────────── series helpers ─────────────────────────
def diff_daily(a: pd.Series, b: pd.Series) -> pd.Series:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).sort_index().ffill().dropna()
    return df["a"] - df["b"]


def _monthly(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = s.index.to_period("M")
    return s[~s.index.duplicated(keep="last")]


def diff_monthly(a: pd.Series, b: pd.Series) -> pd.Series:
    df = pd.concat([_monthly(a).rename("a"), _monthly(b).rename("b")], axis=1).dropna()
    out = df["a"] - df["b"]
    out.index = out.index.to_timestamp()
    return out


def yoy(s: pd.Series, periods: int) -> pd.Series:
    return (s.pct_change(periods) * 100).dropna()


# ───────────────────────── metrics ─────────────────────────
@dataclass
class Metric:
    key: str
    block: str
    label: str
    needs: list
    build: Callable[[dict], pd.Series]
    sign: int          # +1: if the metric rises, EURUSD rises; -1: opposite; 0: info only
    mode: str          # "change" (z of the change over `lookback` obs) or "level" (z of the level)
    lookback: int
    weight: float
    unit: str
    note: str = ""


METRICS: list[Metric] = [
    Metric("policy_gap", "Rates & differentials", "Fed funds − ECB deposit rate", ["DFF", "ECB_DFR"],
           lambda r: diff_daily(r["DFF"], r["ECB_DFR"]), -1, "change", 63, 1.0, "pp",
           "Policy-rate gap, 3-month change. Widening favors USD."),
    Metric("gap_2y", "Rates & differentials", "US 2Y − Euro area 2Y", ["DGS2", "EA_2Y"],
           lambda r: diff_daily(r["DGS2"], r["EA_2Y"]), -1, "change", 21, 1.5, "pp",
           "Short-end spread, 1-month change. The main rate-differential driver."),
    Metric("gap_10y", "Rates & differentials", "US 10Y − Euro area 10Y", ["DGS10", "EA_10Y"],
           lambda r: diff_daily(r["DGS10"], r["EA_10Y"]), -1, "change", 21, 1.0, "pp",
           "Long-end spread, 1-month change. Euro area = AAA-rated curve (Germany-like)."),
    Metric("us_curve", "Yield curve (info)", "US curve 10Y − 2Y", ["T10Y2Y"],
           lambda r: r["T10Y2Y"], 0, "change", 21, 1.0, "pp", "No thesis defined yet: information only."),
    Metric("ea_curve", "Yield curve (info)", "Euro area curve 10Y − 2Y", ["EA_2Y", "EA_10Y"],
           lambda r: diff_daily(r["EA_10Y"], r["EA_2Y"]), 0, "change", 21, 1.0, "pp",
           "No thesis defined yet: information only."),
    Metric("infl_gap", "Inflation", "US core PCE y/y − EA HICP y/y", ["PCEPILFE", "EA_HICP"],
           lambda r: diff_monthly(yoy(r["PCEPILFE"], 12), r["EA_HICP"]), -1, "change", 3, 1.0, "pp",
           "Inflation gap, 3-month change. Higher US relative inflation = hawkish Fed = USD up."),
    Metric("us_nfp", "Labor & growth", "US nonfarm payrolls (3M avg)", ["PAYEMS"],
           lambda r: r["PAYEMS"].diff().rolling(3).mean().dropna(), -1, "level", 1, 1.0, "k jobs",
           "Strong job creation vs its own history = USD up."),
    Metric("us_unemp", "Labor & growth", "US unemployment rate", ["UNRATE"],
           lambda r: r["UNRATE"], +1, "change", 3, 0.5, "pp", "3-month change. Rising unemployment = USD down."),
    Metric("ea_unemp", "Labor & growth", "Euro area unemployment rate", ["EA_UNEMP"],
           lambda r: r["EA_UNEMP"], -1, "change", 3, 0.5, "pp", "3-month change. Rising unemployment = EUR down."),
    Metric("us_gdp", "Labor & growth", "US real GDP y/y", ["GDPC1"],
           lambda r: yoy(r["GDPC1"], 4), -1, "change", 1, 0.5, "pp", "1-quarter change in y/y growth."),
    Metric("ea_gdp", "Labor & growth", "Euro area GDP y/y", ["EA_GDP"],
           lambda r: r["EA_GDP"], +1, "change", 1, 0.5, "pp", "1-quarter change in y/y growth."),
]


def evaluate(m: Metric, s: pd.Series) -> dict | None:
    s = s.dropna()
    if m.mode == "change":
        ch = s.diff(m.lookback).dropna()
        if len(ch) < 24:
            return None
        cur, sd = ch.iloc[-1], ch.std()
        change = cur
    else:
        if len(s) < 24:
            return None
        sd = s.std()
        cur = s.iloc[-1] - s.mean()
        change = s.iloc[-1] - s.iloc[-2]
    if not sd > 0:
        return None
    z = float(cur / sd)
    impact = float(m.sign * np.clip(z, -2, 2) / 2)
    return dict(latest=float(s.iloc[-1]), change=float(change), z=z, impact=impact, asof=s.index[-1])


def tone_impact(fed_label: str, fed_dur: str, ecb_label: str, ecb_dur: str) -> float:
    fed = TONE_VALUES[fed_label] * DURATION_FACTORS[fed_dur]
    ecb = TONE_VALUES[ecb_label] * DURATION_FACTORS[ecb_dur]
    return float(np.clip((ecb - fed) / 4, -1, 1))


def compute(raw: dict, block_weights: dict, tone: float) -> dict:
    rows = []
    for m in METRICS:
        if any(k not in raw for k in m.needs):
            rows.append(dict(metric=m, res=None, reason="data unavailable"))
            continue
        try:
            res = evaluate(m, m.build(raw))
        except Exception as e:  # noqa: BLE001
            res, reason = None, f"{type(e).__name__}: {e}"
        else:
            reason = "not enough history" if res is None else ""
        rows.append(dict(metric=m, res=res, reason=reason))

    # block scores (average impact, -1..1)
    blocks = {}
    for b in BLOCKS:
        if b == "Central bank tone":
            blocks[b] = dict(score=tone, items=[("Central bank tone (manual)", 1.0, tone)])
            continue
        items = [(r["metric"].label, r["metric"].weight, r["res"]["impact"]) for r in rows
                 if r["metric"].block == b and r["res"] is not None]
        wsum = sum(w for _, w, _ in items)
        info_only = all(r["metric"].sign == 0 for r in rows if r["metric"].block == b)
        score = None if (wsum == 0 or info_only) else sum(w * i for _, w, i in items) / wsum
        blocks[b] = dict(score=score, items=items)

    used = {b: block_weights.get(b, 0) for b in BLOCKS if blocks[b]["score"] is not None and block_weights.get(b, 0) > 0}
    bw_sum = sum(used.values())
    total = sum(w * blocks[b]["score"] for b, w in used.items()) / bw_sum * 100 if bw_sum > 0 else None

    # contribution of every item to the total, in score points
    contribs = []
    if bw_sum > 0:
        for b, bw in used.items():
            items = blocks[b]["items"]
            wsum = sum(w for _, w, _ in items)
            for label, w, imp in items:
                contribs.append((label, 100 * (bw / bw_sum) * (w / wsum) * imp))
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    return dict(rows=rows, blocks=blocks, total=total, contribs=contribs, used_weights=used)


def label_for(score: float | None) -> str:
    if score is None or np.isnan(score):
        return "N/A"
    if score >= 35: return "BULLISH EURUSD"
    if score >= 12: return "BULLISH BIAS"
    if score <= -35: return "BEARISH EURUSD"
    if score <= -12: return "BEARISH BIAS"
    return "NEUTRAL"


# ───────────────────────── data releases (charts + actual/previous/forecast tables) ─────────────────────────
@dataclass
class Release:
    name: str        # e.g. "US CPI y/y"
    group: str       # tab: "US inflation", "Euro area inflation", "US labor", "GDP"
    series: pd.Series
    unit: str        # "%", "k", "pp"
    decimals: int
    freq: str        # "M" or "Q"
    eur_sign: int    # effect on EURUSD if the release comes ABOVE forecast (+1 up, -1 down)


def _qoq_annualized(s: pd.Series) -> pd.Series:
    return (((s.pct_change() + 1) ** 4 - 1) * 100).dropna()


def build_releases(raw: dict) -> list[Release]:
    out: list[Release] = []

    def add(name, group, build, needs, unit, dec, freq, sign):
        if any(k not in raw for k in needs):
            return
        try:
            s = build().dropna().round(dec)
        except Exception:  # noqa: BLE001
            return
        if len(s):
            out.append(Release(name, group, s, unit, dec, freq, sign))

    add("US CPI y/y", "US inflation", lambda: yoy(raw["CPIAUCSL"], 12), ["CPIAUCSL"], "%", 1, "M", -1)
    add("US core CPI y/y", "US inflation", lambda: yoy(raw["CPILFESL"], 12), ["CPILFESL"], "%", 1, "M", -1)
    add("US PCE y/y", "US inflation", lambda: yoy(raw["PCEPI"], 12), ["PCEPI"], "%", 1, "M", -1)
    add("US core PCE y/y", "US inflation", lambda: yoy(raw["PCEPILFE"], 12), ["PCEPILFE"], "%", 1, "M", -1)
    add("Euro area HICP y/y", "Euro area inflation", lambda: raw["EA_HICP"], ["EA_HICP"], "%", 1, "M", +1)
    add("Euro area core HICP y/y", "Euro area inflation", lambda: raw["EA_HICP_CORE"], ["EA_HICP_CORE"], "%", 1, "M", +1)
    add("US nonfarm payrolls (monthly change)", "US labor", lambda: raw["PAYEMS"].diff(), ["PAYEMS"], "k", 0, "M", -1)
    add("US unemployment rate", "US labor", lambda: raw["UNRATE"], ["UNRATE"], "%", 1, "M", +1)
    add("US GDP q/q annualized", "GDP", lambda: _qoq_annualized(raw["GDPC1"]), ["GDPC1"], "%", 1, "Q", -1)
    add("US GDP y/y", "GDP", lambda: yoy(raw["GDPC1"], 4), ["GDPC1"], "%", 1, "Q", -1)
    add("Euro area GDP q/q", "GDP", lambda: raw["EA_GDP_QQ"], ["EA_GDP_QQ"], "%", 1, "Q", +1)
    add("Euro area GDP y/y", "GDP", lambda: raw["EA_GDP"], ["EA_GDP"], "%", 1, "Q", +1)
    return out


def period_label(ts: pd.Timestamp, freq: str) -> str:
    p = pd.Period(ts, freq=freq)
    return f"{p.year}-Q{p.quarter}" if freq == "Q" else f"{p.year}-{p.month:02d}"


def release_table(rel: Release, n: int, forecasts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Last n releases: Period, Actual, Previous, Forecast (from forecasts.csv when present)."""
    s = rel.series
    df = pd.DataFrame({"Actual": s, "Previous": s.shift(1)}).tail(n)
    df.insert(0, "Period", [period_label(i, rel.freq) for i in df.index])
    df["Forecast"] = np.nan
    if forecasts is not None and len(forecasts):
        f = forecasts[forecasts["indicator"] == rel.name].set_index("period")["forecast"]
        df["Forecast"] = df["Period"].map(f).astype("float64")
    return df.iloc[::-1].reset_index(drop=True)  # newest first


def surprise_table(rel: Release, table: pd.DataFrame) -> pd.DataFrame:
    """Rows that have a forecast: surprise = Actual − Forecast and its typical EURUSD effect."""
    t = table.dropna(subset=["Forecast"]).copy()
    if t.empty:
        return t
    tol = 10 ** -rel.decimals / 2
    t["Surprise"] = (t["Actual"] - t["Forecast"]).round(rel.decimals)

    def result(x):
        if abs(x) < tol:
            return "In line", "● neutral"
        above = x > 0
        up = (above and rel.eur_sign > 0) or ((not above) and rel.eur_sign < 0)
        return ("Above forecast" if above else "Below forecast"), ("▲ EURUSD" if up else "▼ EURUSD")

    res = t["Surprise"].apply(result)
    t["Result"] = [r[0] for r in res]
    t["Typical EURUSD effect"] = [r[1] for r in res]
    return t[["Period", "Actual", "Forecast", "Surprise", "Result", "Typical EURUSD effect"]]
