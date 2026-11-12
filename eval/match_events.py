#!/usr/bin/env python3
"""System-test matching: app event log  vs  scripted session CSV.

Session CSV (Vanessa, tests/scripts/S01.csv) — exact header:
    session_id,video,subject,condition,event_id,start_s,end_s,event_type,notes

  * start_s / end_s are used exactly as written (seconds from video start).
  * event_type=none marks plain normal driving. Any alert inside a `none`
    stretch is a false alarm, and the total `none` time is the denominator
    for "false alarms per 10 min of normal driving".
  * event_type=mirror_check must NOT alert; an alert there is a false alarm.
  * notes may carry tags like `occlusion=hand; mount=B` — the report is also
    broken down by every tag key found.
  * video file names may start with the session id (S01_mahitha_day-none.mp4).

App event log (exported from the app's Event Log, one file per session):
    t_s,event_type,level,source,mode,model_a_class,model_a_conf,
    gemini_event,gemini_confirmed,severity,latency_ms,fps
  t_s is seconds since monitoring started. Start the app and the video
  recording together; if they are off, pass --offset (app_t + offset = video_t).

A scripted event is DETECTED when an app alert of the same event_type starts
inside [start_s - pre, end_s + post]. Detection delay = alert_t - start_s.

  python eval/match_events.py --script tests/scripts/S01.csv --log Test_sessions/S01_log.csv
  python eval/match_events.py --script-dir tests/scripts --log-dir Test_sessions --out results/system
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveaware_data.labels import EVENT_TYPES, NON_ALERT_EVENT_TYPES  # noqa: E402

SCRIPT_COLS = ["session_id", "video", "subject", "condition", "event_id",
               "start_s", "end_s", "event_type", "notes"]
ALERTABLE = [e for e in EVENT_TYPES if e not in NON_ALERT_EVENT_TYPES]


def parse_tags(notes) -> dict[str, str]:
    if not isinstance(notes, str):
        return {}
    return {k.strip(): v.strip() for k, v in re.findall(r"([A-Za-z_]+)\s*=\s*([^;,]+)", notes)}


def load_script(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in SCRIPT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    bad = set(df["event_type"]) - set(EVENT_TYPES)
    if bad:
        raise ValueError(f"{path.name}: unknown event_type {sorted(bad)}. Allowed: {EVENT_TYPES}")
    if (df["end_s"] < df["start_s"]).any():
        raise ValueError(f"{path.name}: end_s before start_s in event_id "
                         f"{df.loc[df.end_s < df.start_s, 'event_id'].tolist()}")
    return df


MEASURED = {"eye_closure_long", "perclos_high", "nod"}


def collapse_gemini_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per alert the driver actually got.

    In full mode the app logs each detection twice: the local row (source rules /
    model_a) and, a moment later, the Gemini verdict (source gemini:...) with the same
    t_s and event_type. Counting both would turn every verified alert into a false
    alarm. Gemini rows are dropped; when Gemini dismissed a non-measured event as
    false_alarm, the app cleared that alert, so its local row is dropped too.
    """
    if "source" not in df.columns:
        return df
    src = df["source"].fillna("").astype(str)
    gem = df[src.str.startswith("gemini")]
    local = df[~src.str.startswith("gemini")]
    if "gemini_event" in df.columns and len(gem):
        dismissed = gem[(gem["gemini_event"] == "false_alarm") & (~gem["event_type"].isin(MEASURED))]
        drop = set()
        for _, g in dismissed.iterrows():
            cand = local[(local.event_type == g.event_type) & ((local.t_s - g.t_s).abs() < 0.05)
                         & (~local.index.isin(drop))]
            if len(cand):
                drop.add(cand.index[0])
        local = local.drop(index=list(drop))
    return local


def load_log(path: Path, offset: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("t_s", "event_type"):
        if c not in df.columns:
            raise ValueError(f"{path.name}: app log needs column {c!r}")
    df = collapse_gemini_rows(df.copy())
    df["t_s"] = df["t_s"].astype(float) + offset
    if "level" in df.columns:
        df = df[df["level"].fillna(1).astype(int) >= 1]
    return df.sort_values("t_s").reset_index(drop=True)


def match_session(script: pd.DataFrame, log: pd.DataFrame, pre: float, post: float):
    used = set()
    events = []
    for _, ev in script.iterrows():
        rec = ev.to_dict()
        rec.update(parse_tags(ev["notes"]))
        rec["detected"] = None
        rec["delay_s"] = None
        if ev["event_type"] in ALERTABLE:
            lo, hi = ev["start_s"] - pre, ev["end_s"] + post
            cand = log[(log.event_type == ev["event_type"]) & (log.t_s >= lo) & (log.t_s <= hi)
                       & (~log.index.isin(used))]
            if len(cand):
                hit = cand.iloc[0]
                used.add(cand.index[0])
                rec["detected"] = True
                rec["delay_s"] = round(max(0.0, hit.t_s - ev["start_s"]), 2)
            else:
                rec["detected"] = False
        events.append(rec)
    ev_df = pd.DataFrame(events)

    # classify every unmatched alert
    alerts = log.copy()
    alerts["matched"] = alerts.index.isin(used)
    def where(t):
        inside = script[(script.start_s <= t) & (script.end_s >= t)]
        return inside.iloc[0]["event_type"] if len(inside) else "unscripted"
    alerts["during"] = alerts["t_s"].map(where)
    return ev_df, alerts


def summarise(ev: pd.DataFrame, alerts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    rows = []
    for et in ALERTABLE:
        sub = ev[ev.event_type == et]
        n_alerts = int((alerts.event_type == et).sum())
        tp = int(sub["detected"].fillna(False).astype(bool).sum())
        rows.append({
            "behavior": et, "scripted": len(sub), "detected": tp,
            "recall": round(tp / len(sub), 3) if len(sub) else None,
            "app_alerts": n_alerts,
            "precision": round(tp / n_alerts, 3) if n_alerts else None,
            "median_delay_s": sub["delay_s"].dropna().median() if tp else None,
        })
    out["per_behavior"] = pd.DataFrame(rows)

    normal = ev[ev.event_type == "none"]
    normal_s = float((normal.end_s - normal.start_s).sum())
    fa_normal = int((~alerts.matched & (alerts.during == "none")).sum())
    fa_mirror = int((~alerts.matched & (alerts.during == "mirror_check")).sum())
    fa_all = int((~alerts.matched).sum())
    summary = {
        "normal_driving_min": round(normal_s / 60, 2),
        "false_alarms_in_normal": fa_normal,
        "false_alarms_per_10min_normal": round(fa_normal / normal_s * 600, 2) if normal_s else None,
        "alerts_on_mirror_checks": fa_mirror,
        "mirror_checks": int((ev.event_type == "mirror_check").sum()),
        "unmatched_alerts_total": fa_all,
    }
    for col in ("latency_ms", "fps"):
        if col in alerts.columns and alerts[col].notna().any():
            summary[f"{col}_median"] = float(alerts[col].median())
            summary[f"{col}_p95"] = float(alerts[col].quantile(0.95))
    out["summary"] = pd.DataFrame([summary])

    # recall by condition and by any note tag
    alertable = ev[ev.event_type.isin(ALERTABLE)].copy()
    alertable["detected"] = alertable["detected"].astype(bool)
    by = [alertable.groupby("condition")["detected"].agg(["count", "mean"])
          .rename(columns={"mean": "recall"}).reset_index().assign(group="condition")
          .rename(columns={"condition": "value"})]
    for tag in [c for c in alertable.columns if c not in SCRIPT_COLS + ["detected", "delay_s"]]:
        g = alertable.dropna(subset=[tag]).groupby(tag)["detected"].agg(["count", "mean"])
        by.append(g.rename(columns={"mean": "recall"}).reset_index()
                  .assign(group=tag).rename(columns={tag: "value"}))
    out["by_condition"] = pd.concat(by, ignore_index=True)[["group", "value", "count", "recall"]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script"); ap.add_argument("--log")
    ap.add_argument("--script-dir"); ap.add_argument("--log-dir")
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--pre", type=float, default=1.0, help="s before start_s that still counts")
    ap.add_argument("--post", type=float, default=3.0, help="s after end_s that still counts")
    ap.add_argument("--out", default="results/system")
    a = ap.parse_args(argv)

    pairs = []
    if a.script and a.log:
        pairs.append((Path(a.script), Path(a.log)))
    elif a.script_dir and a.log_dir:
        for s in sorted(Path(a.script_dir).glob("S*.csv")):
            logs = sorted(Path(a.log_dir).glob(f"{s.stem}*log*.csv"))
            if logs:
                pairs.append((s, logs[0]))
            else:
                print(f"! no log for {s.name} (expected {s.stem}_log.csv)")
    else:
        ap.error("give --script + --log, or --script-dir + --log-dir")

    all_ev, all_alerts = [], []
    for s, l in pairs:
        ev, al = match_session(load_script(s), load_log(l, a.offset), a.pre, a.post)
        al["session_id"] = ev["session_id"].iloc[0] if len(ev) else s.stem
        all_ev.append(ev); all_alerts.append(al)
    ev = pd.concat(all_ev, ignore_index=True)
    alerts = pd.concat(all_alerts, ignore_index=True)
    res = summarise(ev, alerts)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ev.to_csv(out / "matched_events.csv", index=False)
    alerts.to_csv(out / "classified_alerts.csv", index=False)
    for k, df in res.items():
        df.to_csv(out / f"{k}.csv", index=False)
        print(f"\n== {k}\n{df.to_string(index=False)}")
    print(f"\nwritten to {out}/  (fill a `failed_stage` column in matched_events.csv for misses:"
          " face_not_found / wrong_class / rule_threshold / wrong_gemini_verdict)")


if __name__ == "__main__":
    main()
