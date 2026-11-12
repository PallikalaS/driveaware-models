import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import match_events  # noqa: E402

HEADER = ("t_s,event_type,level,source,mode,model_a_class,model_a_conf,"
          "gemini_event,gemini_confirmed,severity,latency_ms,fps")


def test_full_mode_log_counts_each_alert_once(tmp_path):
    log = tmp_path / "S01_log.csv"
    log.write_text("\n".join([
        HEADER,
        "10.00,eye_closure_long,3,rules,full,,,,,,,30",
        "10.00,eye_closure_long,3,gemini:authority_override,full,,,drowsy,yes,high,1100,30",
        "40.00,phone_use,2,model_a,full,phone_use,0.95,,,,,30",
        "40.00,phone_use,2,gemini:gemini,full,phone_use,0.95,false_alarm,no,low,900,30",   # dismissed
        "70.00,gaze_away,1,rules,full,,,,,,,30",                                             # offline, no verdict
        "90.00,nod,2,rules,full,,,,,,,30",
        "90.00,nod,2,gemini:gemini,full,,,false_alarm,no,low,900,30",   # measured: can't be dismissed
    ]) + "\n")
    df = match_events.load_log(log, 0.0)
    assert list(df.event_type) == ["eye_closure_long", "gaze_away", "nod"]
    assert set(df.source) == {"rules"}


def test_rules_only_log_unchanged(tmp_path):
    log = tmp_path / "S02_log.csv"
    log.write_text(HEADER + "\n5.00,yawn,1,rules,rules,,,,,,,30\n")
    assert len(match_events.load_log(log, 0.0)) == 1
