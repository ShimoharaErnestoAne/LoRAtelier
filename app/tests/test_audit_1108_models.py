"""2026-11-08 review of the model / download code: tokens do not follow redirects to other hosts, settings.json is
never replaced by defaults after a failed read, catalog downloads never take or overwrite a foreign file, model
records are changed on a copy, and hashes are reused across starts."""
from __future__ import annotations

import json
import struct
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "data dir"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config
    config._settings = None
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    return tmp_path


def _safetensors(data: bytes = b"\0" * 8) -> bytes:
    hdr = json.dumps({"w": {"dtype": "F16", "shape": [4], "data_offsets": [0, len(data)]}}).encode()
    return struct.pack("<Q", len(hdr)) + hdr + data


def test_token_is_not_sent_to_the_host_a_download_redirects_to(env):
    from lorastudio import addons, remote
    h = remote.NoAuthOffHost()
    req = urllib.request.Request("https://huggingface.co/a/b/resolve/main/m.safetensors",
                                 headers={"Authorization": "Bearer t"})
    off = h.redirect_request(req, None, 302, "Found", {}, "https://cdn-lfs.hf.co/x?sig=1")
    same = h.redirect_request(req, None, 302, "Found", {}, "https://huggingface.co/other")
    assert not off.has_header("Authorization") and same.get_header("Authorization") == "Bearer t"
    # links pasted on the settings page and catalog downloads both go through it
    assert any(isinstance(x, remote.NoAuthOffHost) for x in remote._opener.handlers)
    assert any(isinstance(x, remote.NoAuthOffHost) for x in addons._opener.handlers)


def test_unreadable_settings_stop_the_start_instead_of_running_on_defaults(env, monkeypatch):
    from lorastudio import config
    path = env / "settings.json"
    path.write_text(json.dumps({"models": [{"id": "keep"}]}), encoding="utf-8")
    real = Path.read_text
    calls = []

    def locked(self, *a, **k):
        if self == path:
            calls.append(1)
            raise PermissionError("in use")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", locked)
    monkeypatch.setattr(config.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="読めませんでした"):
        config.Settings(path)
    assert len(calls) == 5 and json.loads(real(path, encoding="utf-8"))["models"] == [{"id": "keep"}]


def test_a_read_that_fails_once_is_retried(env, monkeypatch):
    from lorastudio import config
    path = env / "settings.json"
    path.write_text(json.dumps({"models": [{"id": "keep"}]}), encoding="utf-8")
    real, n = Path.read_text, []

    def once(self, *a, **k):
        if self == path and not n:
            n.append(1)
            raise PermissionError("in use")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", once)
    monkeypatch.setattr(config.time, "sleep", lambda s: None)
    assert config.Settings(path).get("models") == [{"id": "keep"}]


def test_broken_json_is_still_moved_aside(env):
    from lorastudio import config
    path = env / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    s = config.Settings(path)
    assert s.get("models") == [] and not path.exists() and list(env.glob("settings.broken-*.json"))


class _Body:
    def __init__(self, data: bytes):
        self.data, self.status = data, 200
        self.headers = {"Content-Length": str(len(data))}

    def read(self, size=-1):
        out, self.data = self.data, b""
        return out

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_catalog_download_never_takes_or_overwrites_a_foreign_file(env, monkeypatch):
    from lorastudio import addons
    dest = env / "models" / "vae" / "ae.safetensors"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"someone else's cut-off copy")
    good = _safetensors()
    monkeypatch.setattr(addons._opener, "open", lambda *a, **k: _Body(good))
    st = {"done_bytes": 0}
    got = addons._download("r/x", "ae.safetensors", dest, None, st, gb=len(good) / 1e9)
    assert got == dest.with_name("ae_2.safetensors") and got.read_bytes() == good
    assert dest.read_bytes() == b"someone else's cut-off copy"
    # a whole copy already there (fetched by another entry that shares the file) is used as it is
    monkeypatch.setattr(addons._opener, "open", lambda *a, **k: pytest.fail("downloaded again"))
    assert addons._download("r/x", "ae.safetensors", got, None, {"done_bytes": 0}, gb=len(good) / 1e9) == got


def test_model_records_are_changed_on_a_copy(env):
    from lorastudio import config
    from lorastudio.models import registry
    config.settings().set("models", [{"id": "a", "name": "A", "files": {}}])
    live = config.settings().get("models")
    registry.update("a", name="B")
    assert live[0]["name"] == "A"                       # the list a concurrent save may be writing is not touched
    assert config.settings().get("models")[0]["name"] == "B"


def test_link_check_reuses_the_hash_kept_in_settings(env):
    from lorastudio import config, linking
    f = env / "m.safetensors"
    f.write_bytes(b"model")
    st = f.stat()
    config.settings().set("hash_cache", {f"{f}|{st.st_size}|{int(st.st_mtime)}": "kept"})
    assert linking._sha256(f) == "kept"
    import hashlib
    real = hashlib.sha256(b"model").hexdigest()
    assert linking._sha256(f, fresh=True) == real        # right before a replace the file is read again
    assert config.settings().get("hash_cache")[f"{f}|{st.st_size}|{int(st.st_mtime)}"] == real


def test_dropped_file_names_are_made_safe(env):
    from lorastudio import remote
    st, tmp = remote.begin_upload("a:b.safetensors")
    assert st["name"] == "a_b.safetensors" and ":" not in tmp.name
