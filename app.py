"""EURUSD · Level 1 — Macro direction panel (Streamlit)."""
import os

import numpy as np
import pandas as pd
import streamlit as st

import macro_core as core

st.set_page_config(page_title="EURUSD · Level 1 macro", layout="wide")

NAVY, UP, DN, NEU = "#1c2b45", "#089981", "#f23645", "#3b5b8f"
FORECASTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecasts.csv")


@st.cache_data(ttl=3600, show_spinner="Loading macro data…")
def get_data(api_key: str):
    return core.load_all(api_key)


def load_forecasts() -> pd.DataFrame:
    try:
        df = pd.read_csv(FORECASTS_FILE, dtype={"indicator": str, "period": str})
        df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")
        return df.dropna(subset=["forecast"])
    except Exception:
        return pd.DataFrame(columns=["indicator", "period", "forecast"])


def fmt(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def reading(sign, impact):
    if sign == 0:
        return "info"
    return "▲ EURUSD up" if impact > 0.25 else "▼ EURUSD down" if impact < -0.25 else "● neutral"


# ───────────── sidebar ─────────────
qp = st.query_params
st.sidebar.header("Settings")

api_key = ""
try:
    api_key = st.secrets.get("FRED_API_KEY", "")
except Exception:  # no secrets file
    pass
if not api_key:
    api_key = st.sidebar.text_input("FRED API key", type="password", help="Free at fred.stlouisfed.org → My Account → API Keys")

window = st.sidebar.radio("Chart window", [12, 24], index=1, format_func=lambda m: f"Last {m} months")

st.sidebar.subheader("Central bank tone (manual)")
st.sidebar.caption("Update after each meeting or speech. Bookmark the URL to keep your settings.")


def qget(name, default, options):
    v = qp.get(name, default)
    return v if v in options else default


dur_opts = list(core.DURATION_FACTORS)
fed_tone = st.sidebar.select_slider("Fed tone", core.TONE_LABELS, value=qget("fed", "Neutral", core.TONE_LABELS))
fed_dur = st.sidebar.radio("Fed duration", dur_opts, index=dur_opts.index(qget("fed_d", dur_opts[0], dur_opts)))
ecb_tone = st.sidebar.select_slider("ECB tone", core.TONE_LABELS, value=qget("ecb", "Neutral", core.TONE_LABELS))
ecb_dur = st.sidebar.radio("ECB duration", dur_opts, index=dur_opts.index(qget("ecb_d", dur_opts[0], dur_opts)))
try:
    qp["fed"], qp["fed_d"], qp["ecb"], qp["ecb_d"] = fed_tone, fed_dur, ecb_tone, ecb_dur
except Exception:
    pass
st.sidebar.caption("Duration: a sustained tone counts 2× a data-dependent one. Tone impact = (ECB − Fed) ÷ 4.")

st.sidebar.subheader("Block weights (%)")
bw = {b: st.sidebar.number_input(b, 0, 100, core.DEFAULT_BLOCK_WEIGHTS[b], 5) for b in core.BLOCKS}
st.sidebar.caption("Normalised automatically. The yield curve has no thesis yet, so it is information only.")

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

# ───────────── data ─────────────
st.title("EURUSD · Level 1 — Macro Direction Indicator")

if not api_key:
    st.info("Enter your free FRED API key in the sidebar (or set FRED_API_KEY in Streamlit secrets) to load US data. Euro area data comes from the ECB and needs no key.")
    st.stop()

raw, errors = get_data(api_key)
tone = core.tone_impact(fed_tone, fed_dur, ecb_tone, ecb_dur)
out = core.compute(raw, bw, tone)
total = out["total"]

# ───────────── banner ─────────────
lab = core.label_for(total)
color = UP if (total or 0) >= 12 else DN if (total or 0) <= -12 else NEU
asof = max([s.index[-1] for s in raw.values()], default=None)
st.markdown(
    f"""<div style="background:{color};color:#fff;padding:18px 24px;border-radius:10px;text-align:center">
    <div style="font-size:14px;opacity:.85">WEEKLY MACRO BIAS · EURUSD</div>
    <div style="font-size:34px;font-weight:700;letter-spacing:.5px">{fmt(total, 0, True)} &nbsp;·&nbsp; {lab}</div>
    <div style="font-size:12px;opacity:.85">Latest data point: {asof.date() if asof is not None else '–'}</div></div>""",
    unsafe_allow_html=True,
)

st.subheader("Why")
if out["contribs"]:
    for label, pts in out["contribs"][:3]:
        st.markdown(f"- **{label}**: {'▲' if pts > 0 else '▼'} EURUSD ({pts:+.1f} pts)")
else:
    st.write("Not enough data to rank the drivers.")

# ───────────── blocks ─────────────
st.subheader("Blocks")
cols = st.columns(len(core.BLOCKS))
for c, b in zip(cols, core.BLOCKS):
    sc = out["blocks"][b]["score"]
    c.metric(b, "info" if sc is None else f"{sc * 100:+.0f}", f"weight {bw[b]}%", delta_color="off")

# ───────────── data releases ─────────────
st.subheader("Data Releases")

releases = core.build_releases(raw)
forecasts = load_forecasts()
since = pd.Timestamp.today().normalize() - pd.DateOffset(months=window)

groups = ["US inflation", "Euro area inflation", "US labor", "GDP"]
tabs = st.tabs(groups)


def series_frame(rels):
    df = pd.concat({r.name: r.series for r in rels}, axis=1).sort_index()
    return df[df.index >= since]


for tab, g in zip(tabs, groups):
    with tab:
        rels = [r for r in releases if r.group == g]
        if not rels:
            st.info("No data available for this group (see the data problems list at the bottom).")
            continue

        # charts
        if g in ("US inflation", "Euro area inflation"):
            st.markdown("**Year-over-year, %**")
            st.line_chart(series_frame(rels))
        elif g == "US labor":
            nfp = [r for r in rels if "payrolls" in r.name]
            unemp = [r for r in rels if "unemployment" in r.name]
            c1, c2 = st.columns(2)
            if nfp:
                c1.markdown("**Nonfarm payrolls, monthly change (thousands)**")
                c1.bar_chart(series_frame(nfp))
            if unemp:
                c2.markdown("**Unemployment rate, %**")
                c2.line_chart(series_frame(unemp))
        else:
            yy = [r for r in rels if r.name.endswith("y/y")]
            qq = [r for r in rels if r.name.endswith("q/q") or "annualized" in r.name]
            c1, c2 = st.columns(2)
            if yy:
                c1.markdown("**GDP growth, y/y %**")
                c1.line_chart(series_frame(yy))
            if qq:
                c2.markdown("**GDP growth, q/q % (US annualized)**")
                c2.bar_chart(series_frame(qq))

                     # one editable table per indicator: Period, Actual, Forecast, Result, EURUSD effect
        def build_view(rel, base, fc):
            t = base.drop(columns=["Previous"]).copy()
            t["Forecast"] = t["Period"].map(fc).astype("float64")
            s = core.surprise_table(rel, t)
            if len(s):
                s = s.set_index("Period")
                t["Surprise"] = [(f"{s.loc[p, 'Surprise']:+.{rel.decimals}f}" if s.loc[p, "Surprise"] != 0 else f"{0:.{rel.decimals}f}") if p in s.index else "" for p in t["Period"]]
                t["Result"] = [s.loc[p, "Result"] if p in s.index else "" for p in t["Period"]]
                t["EURUSD effect"] = [s.loc[p, "Typical EURUSD effect"] if p in s.index else "" for p in t["Period"]]
            else:
                t["Surprise"], t["Result"], t["EURUSD effect"] = "", "", ""
            return t

        def on_edit(name):
            view = st.session_state[f"view_{name}"]
            edits = st.session_state[f"ed_{name}"].get("edited_rows", {})
            fc = st.session_state[f"fc_{name}"]
            for i, ch in edits.items():
                if "Forecast" in ch:
                    p = view["Period"].iloc[int(i)]
                    v = ch["Forecast"]
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        fc.pop(p, None)
                    else:
                        fc[p] = float(v)

        for rel in rels:
            n = window if rel.freq == "M" else window // 3
            base = core.release_table(rel, n, forecasts)
            name = rel.name
            if f"fc_{name}" not in st.session_state:
                st.session_state[f"fc_{name}"] = {r.Period: r.Forecast for r in base.itertuples() if pd.notna(r.Forecast)}
            view = build_view(rel, base, st.session_state[f"fc_{name}"])
            st.session_state[f"view_{name}"] = view
            st.markdown(f"**{name}** ({rel.unit})")
            st.data_editor(
                view, hide_index=True, width="stretch", key=f"ed_{name}",
                on_change=on_edit, args=(name,),
                column_order=["Period", "Actual", "Forecast", "Result", "EURUSD effect"],
                disabled=["Period", "Actual", "Surprise", "Result", "EURUSD effect"],
                column_config={
                    "Period": st.column_config.TextColumn(alignment="center"),
                    "Actual": st.column_config.NumberColumn(format=f"%.{rel.decimals}f", alignment="center"),
                    "Forecast": st.column_config.NumberColumn(format=f"%.{rel.decimals}f", alignment="center", help="Type the consensus forecast"),
                    "Surprise": st.column_config.TextColumn(alignment="center"),
                    "Result": st.column_config.TextColumn(alignment="center"),
                    "EURUSD effect": st.column_config.TextColumn(alignment="center"),
                },
            )

# ───────────── details of the score ─────────────
st.subheader("Score details")
for b in core.BLOCKS:
    st.markdown(f"**{b}**")
    if b == "Central bank tone":
        st.dataframe(pd.DataFrame([
            dict(Bank="Fed", Tone=fed_tone, Duration=fed_dur),
            dict(Bank="ECB", Tone=ecb_tone, Duration=ecb_dur),
            dict(Bank="Net impact on EURUSD", Tone=f"{tone * 100:+.0f}", Duration=reading(1, tone)),
        ]), hide_index=True, width="stretch")
        continue
    rows = []
    for r in out["rows"]:
        m, res = r["metric"], r["res"]
        if m.block != b:
            continue
        if res is None:
            rows.append({"Metric": m.label, "Latest": "–", "Change": "–", "z": "–", "Reading": r["reason"] or "n/a", "As of": "–", "Thesis": m.note})
        else:
            rows.append({"Metric": m.label, "Latest": f"{fmt(res['latest'])} {m.unit}", "Change": f"{fmt(res['change'], 2, True)} {m.unit}",
                         "z": fmt(res["z"], 1, True), "Reading": reading(m.sign, res["impact"]), "As of": str(res["asof"].date()), "Thesis": m.note})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ───────────── spreads ─────────────
st.subheader("Spreads")
c1, c2 = st.columns(2)
try:
    g = core.diff_daily(raw["DGS2"], raw["EA_2Y"])
    c1.markdown("US 2Y − Euro area 2Y (pp)")
    c1.line_chart(g[g.index >= since])
except Exception:
    c1.info("2Y spread unavailable")
try:
    g = core.diff_daily(raw["DFF"], raw["ECB_DFR"])
    c2.markdown("Fed funds − ECB deposit rate (pp)")
    c2.line_chart(g[g.index >= since])
except Exception:
    c2.info("Policy-rate gap unavailable")

if errors:
    with st.expander(f"Data problems ({len(errors)} series failed)"):
        for k, v in errors.items():
            st.write(f"`{k}` — {v}")
        st.caption("A failed series only disables the metrics that need it. Send me this list and I will fix the series IDs.")

st.caption("Sources: FRED (Federal Reserve Bank of St. Louis) and ECB Data Portal. Impact = sign × clip(z, ±2) ÷ 2; block score = weighted average × 100; total = weighted average of blocks.")
