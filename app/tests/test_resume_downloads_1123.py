"""A download whose connection drops part way is continued from its .part by itself (2026-11-23).  Risks: a 20-50GB
download that stops for good at the first dropped connection (the user has to notice and press again), one that is
tried again for ever, one tried again after a failure that cannot get better (no room, a missing file, a login, a file
replaced at the source), a wait the cancel button cannot stop, and a catalog download that counts its bytes twice
after a continuation.  Made-up answers only: nothing goes out to the network, nothing real is read."""
from __future__ import annotations

import errno
import io
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config, remote
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    waits = []
    monkeypatch.setattr(remote, "_pause", lambda s: waits.append(s))   # no real waiting
    monkeypatch.setattr(remote.time, "monotonic", _Clock(waits))
    return tmp_path, waits


class _Clock:
    """time.monotonic for the waits: it moves on by what _pause was asked to wait."""
    def __init__(self, waits):
        self.waits = waits

    def __call__(self):
        return 1000.0 + sum(self.waits)


class _Body(io.BytesIO):
    status = 200
    headers: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _link(remote, monkeypatch, answers, st=None):
    """A link download of x.gguf answered by answers(headers, try_no); returns (state, placed bytes, headers asked)."""
    monkeypatch.setattr(remote, "probe", lambda *a, **k: {"ok": True})
    placed, asked = [], []
    monkeypatch.setattr(remote, "_place", lambda st, tmp, name, hint: placed.append(tmp.read_bytes()))

    def open_(url, h, timeout=60):
        asked.append(dict(h))
        return answers(h, len(asked))
    monkeypatch.setattr(remote, "_open", open_)
    st = st or {"id": "abc", "name": "x.gguf", "state": "running", "done_bytes": 0, "total_bytes": 0, "error": None}
    remote._download(st, "https://huggingface.co/a/b/resolve/main/x.gguf", "x.gguf", {}, None)
    return st, placed, asked


def _cut(data, total):
    class Cut(_Body):   # announces the whole rest, ends early
        headers = {"Content-Length": str(total), "ETag": '"v1"'}
    return Cut(data)


def _rest(data, start, total):
    class Rest(_Body):
        status = 206
        headers = {"Content-Range": f"bytes {start}-{total - 1}/{total}", "ETag": '"v1"'}
    return Rest(data)


def test_a_dropped_connection_is_continued_from_the_part(env, monkeypatch):
    tmp, waits = env
    from lorastudio import remote

    def answers(h, n):
        if n == 1:
            return _cut(b"a" * 400, 1000)                     # drops after 400 of 1000 bytes
        if n == 2:
            raise ConnectionResetError(10054, "接続が切断されました")   # the continuation fails once more
        return _rest(b"b" * 600, 400, 1000)
    st, placed, asked = _link(remote, monkeypatch, answers)
    assert st["error"] is None and placed == [b"a" * 400 + b"b" * 600]
    assert "Range" not in asked[0] and asked[1]["Range"] == "bytes=400-" and asked[2]["Range"] == "bytes=400-"
    assert asked[2]["If-Range"] == '"v1"'                      # still the same file at the source
    assert sum(waits) == remote.retry_wait(1) + remote.retry_wait(2) == 5 + 15


def test_the_waits_grow_and_stop_at_the_limit(env, monkeypatch):
    tmp, waits = env
    from lorastudio import remote
    assert [remote.retry_wait(n) for n in range(1, 7)] == [5, 15, 45, 120, 120, 120]
    msgs = []

    def answers(h, n):
        raise TimeoutError("timed out")
    st = {"id": "abc", "name": "x.gguf", "state": "running", "done_bytes": 0, "total_bytes": 0, "error": None}
    monkeypatch.setattr(remote, "_pause", lambda s: (msgs.append(st["message"]), waits.append(s)))
    st, placed, asked = _link(remote, monkeypatch, answers, st)
    assert st["state"] == "failed" and not placed
    assert len(asked) == 1 + remote.RETRY_TRIES                # tried again 5 times, not for ever
    assert "5 回とも" in st["error"] and "続きから" in st["error"]
    assert "通信が切れたため、5 秒後に続きから取り直します（1 回目 / 5 回）" in msgs
    assert any("（5 回目 / 5 回）" in m for m in msgs)


@pytest.mark.parametrize("failure, says", [
    (urllib.error.HTTPError("u", 404, "Not Found", {}, None), "404"),
    (OSError(errno.ENOSPC, "No space left on device"), "空き容量"),
])
def test_a_failure_that_cannot_get_better_is_not_tried_again(env, monkeypatch, failure, says):
    tmp, waits = env
    from lorastudio import remote

    def answers(h, n):
        raise failure
    st, placed, asked = _link(remote, monkeypatch, answers)
    assert st["state"] == "failed" and says in st["error"] and len(asked) == 1 and waits == []


def test_a_busy_server_is_waited_for(env, monkeypatch):
    tmp, waits = env
    from lorastudio import remote

    def answers(h, n):
        if n == 1:
            raise urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)
        class Whole(_Body):
            headers = {"Content-Length": "10"}
        return Whole(b"c" * 10)
    st, placed, asked = _link(remote, monkeypatch, answers)
    assert st["error"] is None and placed == [b"c" * 10] and len(asked) == 2


def test_a_file_replaced_at_the_source_is_not_tried_again(env, monkeypatch):
    tmp, waits = env
    from lorastudio import remote
    part = tmp / "models" / "_incoming" / "abc_x.gguf.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"a" * 500)
    remote._etag_file(part).write_text('"v1"', encoding="utf-8")

    def answers(h, n):
        class Other(_Body):   # ignores If-Range and continues another file
            status = 206
            headers = {"Content-Range": "bytes 500-999/1000", "ETag": '"v9"'}
        return Other(b"z" * 500)
    st, placed, asked = _link(remote, monkeypatch, answers)
    assert st["state"] == "failed" and "差し替わ" in st["error"] and len(asked) == 1 and waits == []


def test_cancel_during_the_wait_stops_at_once(env, monkeypatch):
    tmp, waits = env
    from lorastudio import remote
    st = {"id": "abc", "name": "x.gguf", "state": "running", "done_bytes": 0, "total_bytes": 0, "error": None}

    def pause(s):
        waits.append(s)
        st["cancel"] = True          # the user pressed cancel while it waits
    monkeypatch.setattr(remote, "_pause", pause)

    def answers(h, n):
        return _cut(b"a" * 100, 1000)
    st, placed, asked = _link(remote, monkeypatch, answers, st)
    assert st["state"] == "cancelled" and len(asked) == 1 and len(waits) == 1   # not the whole 5 s, no second try
    assert not (tmp / "models" / "_incoming" / "abc_x.gguf.part").exists()


def test_a_catalog_download_is_continued_and_counts_its_bytes_once(env, monkeypatch):
    tmp, waits = env
    from lorastudio import addons
    dest = tmp / "models" / "diffusion_models" / "big.gguf"
    dest.parent.mkdir(parents=True)
    tries = []

    def answer(req, timeout=60):
        h = {k.lower(): v for k, v in req.header_items()}
        tries.append(h)
        if len(tries) == 1:
            return _cut(b"a" * 300, 1000)
        return _rest(b"b" * 700, 300, 1000)
    monkeypatch.setattr(addons._opener, "open", answer)
    st = {"done_bytes": 50}                                    # 50 bytes of the entry's earlier files
    assert addons._download("a/b", "big.gguf", dest, None, st) == dest
    assert dest.read_bytes() == b"a" * 300 + b"b" * 700 and tries[1]["range"] == "bytes=300-"
    assert st["done_bytes"] == 50 + 1000                       # the first 300 bytes are not counted twice


def test_a_catalog_download_that_needs_a_login_is_not_tried_again(env, monkeypatch):
    tmp, waits = env
    from lorastudio import addons
    dest = tmp / "models" / "x.safetensors"
    dest.parent.mkdir(parents=True)
    n = []

    def answer(req, timeout=60):
        n.append(1)
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
    monkeypatch.setattr(addons._opener, "open", answer)
    with pytest.raises(addons.NeedLogin):
        addons._download("a/b", "x.safetensors", dest, None, {"done_bytes": 0})
    assert len(n) == 1 and waits == []
