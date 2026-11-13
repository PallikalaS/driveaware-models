import json
import sys
import types as pytypes
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gemini_tuning as gt  # noqa: E402

B = "gs://bucket/gemini"


def write_jsonl(d: Path, split: str, n: int, prefix=B + "/keyframes"):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{split}.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"contents": [{"role": "user", "parts": [
                {"fileData": {"mimeType": "image/jpeg", "fileUri": f"{prefix}/r{i}.jpg"}}, {"text": "x"}]},
                {"role": "model", "parts": [{"text": "{}"}]}]}) + "\n")


def test_upload_checks_and_dry_run(tmp_path, capsys):
    j = tmp_path / "jsonl"; write_jsonl(j, "train", 120); write_jsonl(j, "val", 20)
    assert gt.main(["--bucket-dir", B, "upload", "--keyframes", str(tmp_path / "kf"), "--jsonl", str(j), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "train 120 rows, val 20 rows" in out and "gcloud storage rsync" in out


def test_upload_rejects_wrong_image_prefix_and_empty_val(tmp_path):
    j = tmp_path / "jsonl"; write_jsonl(j, "train", 5, prefix="gs://other/kf"); write_jsonl(j, "val", 1)
    with pytest.raises(SystemExit, match="not under"):
        gt.main(["--bucket-dir", B, "upload", "--keyframes", "k", "--jsonl", str(j), "--dry-run"])
    j2 = tmp_path / "j2"; write_jsonl(j2, "train", 5); write_jsonl(j2, "val", 0)
    with pytest.raises(SystemExit, match="val.jsonl is empty"):
        gt.main(["--bucket-dir", B, "upload", "--keyframes", "k", "--jsonl", str(j2), "--dry-run"])


class FakeTunings:
    def __init__(self): self.calls = []
    def tune(self, **kw):
        self.calls.append(kw); return pytypes.SimpleNamespace(name="projects/1/locations/us-central1/tuningJobs/9", state="JOB_STATE_PENDING")
    def get(self, name):
        return pytypes.SimpleNamespace(state="JobState.JOB_STATE_SUCCEEDED",
                                       tuned_model=pytypes.SimpleNamespace(endpoint="projects/1/locations/us-central1/endpoints/42"))


def test_tune_and_status(tmp_path, monkeypatch, capsys):
    # stand-in for google.genai.types (not installed in CI)
    fake = pytypes.ModuleType("google.genai.types")
    fake.CreateTuningJobConfig = lambda **kw: kw
    fake.TuningDataset = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "google.genai", pytypes.ModuleType("google.genai"))
    monkeypatch.setitem(sys.modules, "google.genai.types", fake)
    sys.modules["google.genai"].types = fake
    monkeypatch.chdir(tmp_path)
    tun = FakeTunings()
    mk = lambda loc: pytypes.SimpleNamespace(tunings=tun)  # noqa: E731
    a = gt.argparse.Namespace(bucket_dir=B, location="us", base_model="gemini-3.5-flash", name="v1", epochs=0)
    assert gt.tune(a, make_client=mk) == 0
    kw = tun.calls[0]
    assert kw["base_model"] == "gemini-3.5-flash"
    assert kw["training_dataset"] == {"gcs_uri": B + "/train.jsonl"}
    assert kw["config"]["validation_dataset"] == {"gcs_uri": B + "/val.jsonl"} and "epoch_count" not in kw["config"]
    assert (tmp_path / "tuning_job.txt").read_text().strip().endswith("tuningJobs/9")
    s = gt.argparse.Namespace(job="", wait=False, location="us")
    assert gt.status(s, make_client=mk) == 0
    out = capsys.readouterr().out
    assert "VERTEX_LOCATION=us GEMINI_MODEL=projects/1/locations/us-central1/endpoints/42 bash deploy.sh" in out
