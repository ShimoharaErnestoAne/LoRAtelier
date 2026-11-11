"""Translation of the result chips into Japanese (画像からプロンプト's 「日本語」, the image page's dictionary), on
fake data: a fake Ollama and scratch folders.  Failure risks: a word the model once answered nothing for is never
asked about again (the button then does nothing, with nothing said), the app's own page refused as a foreign one,
a translation job that ends without its result reaching the page."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAMO = "hf.co/pfnet/plamo-3-translate-2b-GGUF:Q4_K_M"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORASTUDIO_DATA", str(tmp_path / "データ 空白"))
    for m in [m for m in sys.modules if m.startswith("lorastudio")]:
        del sys.modules[m]
    from lorastudio import config
    config._settings = None
    return tmp_path


def _worker(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["w", str(tmp_path), str(tmp_path / "d"), str(tmp_path / "m")])
    spec = importlib.util.spec_from_file_location("i2t_worker_1111", ROOT / "worker" / "i2t_worker.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    (tmp_path / "d").mkdir(exist_ok=True)
    return w


def _fake_i2t(w, monkeypatch, answers: list):
    """An i2t whose translation model gives the next of `answers` ({word: ja}) for each job."""
    class VLM:
        def __init__(self, host, model, timeout=0):
            self.host, self.model = host, model

        def unload(self):
            pass

    class Gate:
        def wait(self, *a):
            pass
    saved = {}
    w.i2t = SimpleNamespace(
        chip_core=lambda t: (t, ""), lookup_ja=lambda table, core: table.get(core.lower(), ""),
        dedup=lambda xs: list(dict.fromkeys(xs)), OllamaVLM=VLM, VLM_TIMEOUT=60, DEFAULT_CONFIG={"translate_model": PLAMO},
        load_config=lambda: {"translate_model": PLAMO, "ollama_host": "http://h", "low_vram": True},
        load_translations=lambda: dict(saved), TRANSLATIONS_PATH=Path(sys.argv[2]) / "translations.json",
        translate_chips_ja=lambda vlm, words, keep_alive=None, checkpoint=None: answers.pop(0),
        OperationCancelled=type("Cancelled", (Exception,), {}), describe_error=str)
    monkeypatch.setattr(w, "chip_table", lambda: dict(saved))
    monkeypatch.setattr(w, "_save_translations", lambda table: saved.update(table) or True)
    monkeypatch.setattr(w, "_progress", lambda rid: lambda m, f=-1.0: None)
    monkeypatch.setattr(w, "_text_llm", lambda cfg, cancel: VLM("http://h", "reader"))
    monkeypatch.setattr(w, "_store_readings", lambda *a: {})
    w._job.update(rid=1, cancel=threading.Event(), gate=Gate(), vlm=None)
    return saved


def test_a_word_the_model_once_answered_nothing_for_is_asked_about_again(env, monkeypatch):
    """Ollama short of memory answers nothing once: the next 「日本語」 must ask again, not fall silent."""
    w = _worker(monkeypatch, env)
    words = ["wearing a blue raincoat"]
    _fake_i2t(w, monkeypatch, [{}, {"wearing a blue raincoat": "青いレインコートを着ている"}])
    out = w._translate(1, {"words": words}, threading.Event())
    assert out["translated"] == {} and out["untranslated"] == words          # the page can say so
    assert w.missing_ja(words, []) == words                                    # asked again
    out = w._translate(1, {"words": w.missing_ja(words, [])}, threading.Event())
    assert out["translated"] and out["untranslated"] == [] and w.missing_ja(words, []) == []
    assert not w._skip_words[PLAMO]                                            # translated: forgotten


def test_a_word_that_keeps_coming_back_is_left_out_for_a_while_only(env, monkeypatch):
    w = _worker(monkeypatch, env)
    words = ["1girl", "long hair"]
    _fake_i2t(w, monkeypatch, [{"long hair": "長い髪"}, {}, {}])
    w._translate(1, {"words": words}, threading.Event())
    assert w.missing_ja(words, []) == ["1girl"]                 # copied once: may be a hiccup
    w._translate(1, {"words": ["1girl"]}, threading.Event())
    assert w.missing_ja(words, []) == []                         # twice: not after every analysis
    assert w.missing_ja(words, [], True) == ["1girl"]            # the user pressed 「日本語」 themselves
    now = time.time()
    monkeypatch.setattr(w.time, "time", lambda: now + w._SKIP_SECONDS + 1)
    assert w.missing_ja(words, []) == ["1girl"]                 # ...and once more after a while
    w._translate(1, {"words": ["1girl"]}, threading.Event())
    assert w.missing_ja(words, []) == []


def test_the_apps_own_page_may_send_the_translation_requests(env, monkeypatch):
    """Sec-Fetch-Site is sent to 127.0.0.1 / localhost only; a page opened by the PC's LAN address
    (start_intranet.bat) sends Origin alone."""
    from lorastudio import server
    refusal = server.OwnPagesOnly.refusal
    for listen, host in (("127.0.0.1", "127.0.0.1:7865"), ("127.0.0.1", "localhost:7865"), ("127.0.0.1", "[::1]:7865"),
                         ("0.0.0.0", "192.168.1.20:7865"), ("0.0.0.0", "my-pc:7865"), ("0.0.0.0", "127.0.0.1:7865")):
        monkeypatch.setattr(server.config, "LISTEN", listen)
        for method in ("POST", "PUT", "GET"):    # /api/i2t/job, /api/i2t/call; /api/i2t/config; /api/events
            assert refusal(method, {"host": host, "origin": f"http://{host}", "sec-fetch-site": "same-origin"}) is None
            assert refusal(method, {"host": host, "origin": f"http://{host.upper()}"}) is None     # no Sec-Fetch-Site
            assert refusal(method, {"host": host}) is None
        assert refusal("POST", {"host": host, "origin": "https://evil.example"})
    monkeypatch.setattr(server.config, "LISTEN", "127.0.0.1")
    assert refusal("GET", {"host": "192.168.1.20:7865"})        # this PC only: its LAN address is not answered


def test_the_translation_requests_reach_the_worker(env, monkeypatch):
    from fastapi.testclient import TestClient
    from lorastudio import i2tlink, server
    asked, started = [], []
    monkeypatch.setattr(i2tlink.client, "call", lambda fn, *a, **k: asked.append((fn, a)) or ["a word"])
    monkeypatch.setattr(i2tlink.client, "start_job", lambda op, **kw: started.append((op, kw)) or {"rid": 3, "op": op})
    c = TestClient(server.app)
    own = {"Sec-Fetch-Site": "same-origin", "Origin": "http://testserver"}
    r = c.post("/api/i2t/call", json={"fn": "missing_ja", "args": [["a word"], [], True]}, headers=own)
    assert r.status_code == 200 and r.json() == {"value": ["a word"]} and asked == [("missing_ja", (["a word"], [], True))]
    r = c.post("/api/i2t/job", json={"op": "translate", "words": ["a word"]}, headers=own)
    assert r.status_code == 200 and r.json()["op"] == "translate" and started == [("translate", {"words": ["a word"]})]


def test_a_finished_translation_job_reaches_the_page_and_frees_the_slot(env, monkeypatch):
    from lorastudio import i2tlink
    events = []
    monkeypatch.setattr(i2tlink.bus, "publish", lambda type_, **k: events.append((type_, k)))
    c = i2tlink.I2tClient()
    sent = []
    monkeypatch.setattr(c, "_send", sent.append)
    job = c.start_job("translate", words=["a word"])
    assert sent == [{"rid": job["rid"], "op": "translate", "words": ["a word"]}] and not job["waiting_gpu"]
    kept = [{"host": "http://h", "model": PLAMO, "low_vram": False}]
    lines = [{"event": "progress", "rid": job["rid"], "message": "日本語に訳しています", "fraction": -1.0},
             {"rid": job["rid"], "ok": True, "value": {"translated": {"a word": "語"}, "model": PLAMO, "untranslated": []},
              "vlm_kept": kept}]
    proc = SimpleNamespace(stdout=[json.dumps(x, ensure_ascii=False).encode("utf-8") + b"\n" for x in lines])
    c.proc = proc
    c._reader(proc)
    done = [k for t, k in events if t == "i2t" and k.get("event") == "done"]
    assert len(done) == 1 and done[0]["op"] == "translate" and done[0]["ok"] and done[0]["rid"] == job["rid"]
    assert done[0]["value"]["translated"] == {"a word": "語"}
    assert c.job is None and c.last_done["value"]["model"] == PLAMO and c.kept_vlm == kept
    assert c.start_job("translate", words=["next"])["rid"] != job["rid"]      # the next one starts


# ------------------------------------------------------------------ the whole way, with a real worker process
class _Ollama(BaseHTTPRequestHandler):
    """Answers as Ollama does for the translation jobs: PLaMo (raw /api/generate), JSON lists (/api/chat)."""
    empty = False
    seen: list = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        p = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        type(self).seen.append((self.path, p.get("model"), p.get("keep_alive")))
        if self.path == "/api/show":
            return self._send({"capabilities": ["completion"]})
        if self.path == "/api/generate":
            if "prompt" not in p:
                return self._send({"done": True})                       # unload
            if type(self).empty:
                return self._send({"response": ""})
            text = p["prompt"].split("input<|plamo:msg|>")[1].split("<|plamo:tag|>")[0]
            if p["prompt"].endswith("<|plamo:val|>en<|plamo:msg|>"):     # the image page: Japanese into English
                return self._send({"response": "Made-up tag."})
            return self._send({"response": "\n".join("訳" + str(len(t)) for t in text.split("\n"))})
        props = (p.get("format") or {}).get("properties") or {"out": {"minItems": 1}}
        key = next(iter(props))
        self._send({"message": {"content": json.dumps({key: ["よみ"] * props[key]["minItems"]})}})


def test_japanese_through_the_server_and_a_real_worker(env, monkeypatch):
    """missing_ja -> the translate job -> chips_ja as the page calls them, with the bundled i2t in a worker process."""
    from fastapi.testclient import TestClient
    from lorastudio import config, i2tlink, server
    if not i2tlink.client.available():
        pytest.skip("the analysis environment is not installed")
    config.settings().set("i2t_models_dir", str(env / "models"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Ollama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Ollama.empty, _Ollama.seen = False, []
    c = TestClient(server.app)
    own = {"Sec-Fetch-Site": "same-origin", "Origin": "http://testserver"}

    def call(fn, *args):
        r = c.post("/api/i2t/call", json={"fn": fn, "args": list(args)}, headers=own)
        assert r.status_code == 200, r.text
        return r.json()["value"]

    def translate(words, op="translate"):
        r = c.post("/api/i2t/job", json={"op": op, "words": words}, headers=own)
        assert r.status_code == 200, r.text
        end = time.time() + 60
        while c.get("/api/i2t/job").json()["job"] and time.time() < end:
            time.sleep(0.05)
        state = c.get("/api/i2t/job").json()
        assert state["job"] is None and state["last_done"]["rid"] == r.json()["rid"]
        return state["last_done"]
    try:
        cfg = c.get("/api/i2t/state").json()["config"]
        cfg.update(ollama_host=f"http://127.0.0.1:{srv.server_port}", translate_model=PLAMO, low_vram=True)
        assert c.put("/api/i2t/config", json=cfg, headers=own).status_code == 200
        sentence = "A woman in a red coat stands on a rainy street at night."
        words = ["long hair", sentence, "sitting on a made-up bench"]
        missing = call("missing_ja", words, [])
        assert missing == words[1:]                                    # "long hair" is in the bundled dictionary
        done = translate(missing)
        assert done["ok"] and done["op"] == "translate" and len(done["value"]["translated"]) == 2
        labels = call("chips_ja", words)
        assert labels[0] != "long hair" and labels[1] == f"訳{len(sentence)}" and labels[2].startswith("訳")
        assert call("missing_ja", words, []) == []
        assert (PLAMO, 0) in [(m, k) for _p, m, k in _Ollama.seen]     # 省 VRAM: unloaded when the job ends
        # Ollama answers nothing once: said, and asked again the next time
        _Ollama.empty = True
        other = ["wearing a made-up raincoat"]
        done = translate(call("missing_ja", other, []))
        assert done["ok"] and done["value"]["translated"] == {} and done["value"]["untranslated"] == other
        _Ollama.empty = False
        assert call("missing_ja", other, []) == other
        assert translate(other)["value"]["untranslated"] == [] and call("chips_ja", other)[0].startswith("訳")
        # the image page: a Japanese word the dictionary does not know, into English, with its reading
        done = translate(["作り物の語"], "translate_en")
        assert done["ok"] and done["value"]["translated"] == {"作り物の語": "made-up tag"} and done["value"]["readings"] == 1
        table = call("ja_table")
        assert table["user"]["made-up tag"] == "作り物の語" and table["readings"]["made-up tag"] == "よみ"
        assert call("register_ja", "made-up tag", "") == {"made-up tag": ""} and "made-up tag" not in call("ja_table")["user"]
    finally:
        proc = i2tlink.client.proc
        i2tlink.client.stop()
        if proc is not None:
            proc.wait(10)
        srv.shutdown()
        srv.server_close()
