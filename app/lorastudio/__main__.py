"""Entry point: python -m lorastudio  (bound to 127.0.0.1 only, unless LORASTUDIO_LISTEN says otherwise)."""
from __future__ import annotations

import argparse
import threading
import time

import uvicorn

from . import config


def _probe(port: int, timeout: float = 3) -> str:
    """'free', 'lorastudio' (our server answers), or 'other' (port taken by something else)."""
    import json
    import socket
    import urllib.request
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex((config.HOST, port)) != 0:
            return "free"
    try:
        with urllib.request.urlopen(f"http://{config.HOST}:{port}/api/status", timeout=timeout) as r:
            return "lorastudio" if "worker" in json.loads(r.read()) else "other"
    except (OSError, ValueError):
        return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--port", type=int, default=config.PORT)
    args = ap.parse_args()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    from .procjob import bind_children_to_this_process, exit_with_console, hold_instance_lock, open_browser

    # Already running (e.g. start.bat double-clicked twice)?  Open the existing one instead of starting a
    # second server on the same data: that one would mark the first one's training as interrupted.  A
    # busy server (training, analysis) can answer slowly, so the lock decides, not the port probe.
    if not hold_instance_lock(config.DATA_DIR / "app.lock"):
        for port in range(args.port, args.port + 20):
            if _probe(port, timeout=10) == "lorastudio":
                url = f"http://{config.HOST}:{port}/"
                print(f"LoRAtelier はすでに起動しています: {url}")
                if not args.no_browser:
                    open_browser(url)
                return
        print("LoRAtelier はすでに起動しています（起動の途中か、応答に時間がかかっています）。"
              "しばらく待ってから、ブラウザーで開いている画面を再読み込みしてください。")
        raise SystemExit(1)   # start.bat pauses so the message can be read
    port = args.port
    for _ in range(20):
        if _probe(port) == "free":
            break
        port += 1   # another program uses this port: try the next one
    args.port = port
    # before anything is started: ComfyUI, workers and training end when this process ends
    bind_children_to_this_process()
    exit_with_console()
    from . import server

    # restore queues after the server is already answering (startup never waits for this)
    threading.Thread(target=server.resume_pending_on_startup, daemon=True).start()
    url = f"http://{config.HOST}:{args.port}/"
    if not args.no_browser:
        # a tab left open from the last start reconnects by itself (EventSource retries every 2 s): a new tab
        # each start piles up tabs, and their live connections can use up the browser's 6 per server
        def open_unless_reconnected():
            time.sleep(3.5)
            if not server.bus.listeners():
                open_browser(url)
        threading.Thread(target=open_unless_reconnected, daemon=True).start()
    print(f"LoRAtelier: {url}  (このウィンドウを閉じると終了します)")
    if config.LISTEN != config.HOST:
        print(_lan_message(args.port))
    uvicorn.run(server.app, host=config.LISTEN, port=args.port, log_level="warning", access_log=False)


def _lan_message(port: int) -> str:
    import socket
    try:
        ips = sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)} - {"127.0.0.1"})
    except OSError:
        ips = []
    urls = "\n".join(f"  http://{ip}:{port}/" for ip in ips) or f"  http://<この PC の IP アドレス>:{port}/"
    return ("同じネットワークの他の PC からは、次のアドレスで開けます（パスワードはありません。"
            "ネットワーク内の誰でも操作できます）。\n" + urls + "\n"
            "開けない場合は、Windows のファイアウォールで python.exe の通信（プライベート ネットワーク）を許可してください。\n"
            "LoRA の書き出し先やモデルの置き場所はこの PC のフォルダーで、「フォルダーを開く」もこの PC の画面に開きます。")


if __name__ == "__main__":
    main()
