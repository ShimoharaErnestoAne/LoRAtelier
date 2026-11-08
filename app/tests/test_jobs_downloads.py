"""ジョブのページのダウンロード: the model downloads (設定) and the link downloads / dropped files are listed newest
first, each with the way to stop it; a dropped file being received cannot be stopped from there."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_downloads_listed_for_the_job_page(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import addons, remote, server
    from lorastudio.models import CATALOG
    cid = CATALOG[0]["id"]
    monkeypatch.setitem(addons._state, cid, {"state": "running", "done_bytes": 5, "total_bytes": 10, "message": "ダウンロード中: a", "error": None, "started": 3})
    monkeypatch.setitem(remote._state, "k1", {"id": "k1", "name": "x.safetensors", "state": "failed", "done_bytes": 0, "total_bytes": 0, "message": "失敗", "error": "404", "started": 5})
    monkeypatch.setitem(remote._state, "k2", {"id": "k2", "name": "y.safetensors", "state": "running", "done_bytes": 1, "total_bytes": 0, "message": "受け取り中", "error": None, "started": 1, "src": "upload"})
    v = server._downloads_view()
    assert [d["key"] for d in v] == ["link:k1", f"model:{cid}", "link:k2"]
    assert v[1]["name"] == CATALOG[0]["name"] and v[1]["cancel"] == f"/api/downloads/{cid}/cancel" and v[1]["done_bytes"] == 5
    assert v[0]["cancel"] == "/api/import/download/k1/cancel" and v[0]["error"] == "404"
    assert v[2]["cancel"] is None and v[2]["what"] == "ファイルの受け取り"
    assert server.status()["downloads"] == 2
