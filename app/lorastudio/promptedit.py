"""Prompt rewriting by instruction (画像生成の「指示でプロンプトを変える」) with a local LLM served by Ollama
(the same Ollama 「画像からプロンプト」 uses).  Only text is sent, and only to the local Ollama server.

Editing a prompt word by word (tag in, tag out) is what made the edits miss: the model read the prompt as
a list of strings, not as a picture.  So the work is done in Japanese, on the meaning:
  1. 理解と編集: the prompt is read as a picture and written down as a Japanese description; the user's
     instruction is applied to that description, giving the Japanese description of the picture after it
     (and, in Japanese, what must not appear).
  2. 書き戻し: the Japanese description is written back as a prompt in the image model's format (Danbooru
     tags or English sentences), reusing the original wording wherever the picture did not change.
The two Japanese descriptions are shown to the user, so what the model understood can be seen and corrected.

The model is unloaded from VRAM shortly after use, and at once when training or generation takes the GPU.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config
from .gpu import gpu

KEY = "prompt_llm"
DEFAULT_HOST = "http://127.0.0.1:11434"
KEEP_ALIVE = "2m"
TIMEOUT_S = 180
MAX_HISTORY = 8
_SKIP = ("embed", "translate")   # models that cannot follow an editing instruction well
# Ollama model families, best first (qwen3-vl keeps thinking and returns nothing, so it is last); models
# over MAX_AUTO_BYTES are not picked automatically (VRAM).
FAMILY_PREF = {"qwen35": 3, "qwen3": 2, "qwen2": 1}
MAX_AUTO_BYTES = 10e9

_last_model = ""   # the model used last (unloaded when the GPU is taken)


class LlmError(Exception):
    pass


def get_conf() -> dict:
    """host, and the model the user picked on the page ("chosen"; the older "model" key was also written
    when a model was merely used, so it is not read: the best installed model is picked instead)."""
    c = config.settings().get(KEY, {}) or {}
    return {"host": c.get("host") or DEFAULT_HOST, "model": c.get("chosen") or ""}


def set_conf(host: str | None = None, model: str | None = None) -> dict:
    c = dict(config.settings().get(KEY, {}) or {})
    if host is not None:
        c["host"] = host.strip().rstrip("/") or DEFAULT_HOST
    if model is not None:
        c["chosen"] = model.strip()
    config.settings().set(KEY, c)
    return get_conf()


def _req(host: str, path: str, body: dict | None = None, timeout=10) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(host + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("error") or detail
        except json.JSONDecodeError:
            pass
        raise LlmError(detail) from None
    except (urllib.error.URLError, OSError) as e:
        raise LlmError(f"Ollama に接続できません（{host}）。Ollama が起動しているか確認してください。"
                       "「画像からプロンプト」の設定から導入・起動もできます。") from e


def models() -> dict:
    """Installed Ollama models and the one to use: the user's choice, else the best family that fits."""
    c = get_conf()
    tags = [m for m in _req(c["host"], "/api/tags").get("models") or [] if "embedding" not in (m.get("capabilities") or [])]
    names = [m["name"] for m in tags]

    def rank(m):
        fam = (m.get("details") or {}).get("family") or ""
        return (any(s in m["name"].lower() for s in _SKIP), fam == "qwen3vl", (m.get("size") or 0) > MAX_AUTO_BYTES,
                -FAMILY_PREF.get(fam, 0))
    model = c["model"] if c["model"] in names else (min(tags, key=rank)["name"] if tags else "")
    return {"host": c["host"], "models": names, "model": model}


def _chat(host, model, system, shots, user, schema) -> dict:
    body = {"model": model, "stream": False, "format": schema, "keep_alive": KEEP_ALIVE, "think": False,
            "options": {"temperature": 0.2, "num_predict": 2048},
            "messages": [{"role": "system", "content": system}, *shots, {"role": "user", "content": user}]}
    try:
        r = _req(host, "/api/chat", body, timeout=TIMEOUT_S)
    except LlmError as e:
        if "think" not in str(e).lower():
            raise
        body.pop("think")   # a model without the thinking switch
        r = _req(host, "/api/chat", body, timeout=TIMEOUT_S)
    text = ((r.get("message") or {}).get("content") or "").strip()
    if not text and (r.get("message") or {}).get("thinking"):
        raise LlmError(f"「{model}」は考える途中で止まり、答えを返しませんでした。考える機能のないモデル（qwen2.5 など）を選んでください。")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise LlmError("LLM の返事を読み取れませんでした。もう一度試すか、別のモデルを選んでください。") from None


def _shots(examples) -> list[dict]:
    out = []
    for q, a in examples:
        out += [{"role": "user", "content": json.dumps(q, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]
    return out


# ---------------------------------------------------------------- step 1: understand and edit, in Japanese
EDIT_SYSTEM = """あなたは、画像生成のプロンプトを、ユーザーの日本語の指示どおりに直す手伝いをします。
プロンプトは文字の並びとしてではなく、「どんな絵か」という意味で読んでください。

1. "今の絵": 今のプロンプトが表している絵を、日本語の文章で具体的に書く（人物、髪、目、表情、服、持ち物、ポーズ、構図、場所、時間や光、画風や画質。重みで強調・弱めているものはそれも）。
2. "変更後の絵": 指示を受けた後の絵を、日本語の文章で全部書く。指示で言われた所だけでなく、それに合わせて自然に変わる所も考える。
   指示に関係のない所は、今の絵のまま書く。強調・弱めの指示は「〜を強調（1.3）」のように強さも書く。
3. "出てほしくないもの": 生成で出てほしくないものを日本語で書く（今のネガティブプロンプトの内容に、指示で増えた・減ったものを反映）。
   変更後の絵で欲しいものは、ここに入れない。
4. "変えたこと": 何をどう変えたかを日本語で 1 文。
次の語は LoRA の呼び出し語なので、指示で名指しされない限り、変更後の絵にもそのまま残す: {keep}"""

EDIT_SCHEMA = {"type": "object", "properties": {"今の絵": {"type": "string"}, "変更後の絵": {"type": "string"},
                                                 "出てほしくないもの": {"type": "string"}, "変えたこと": {"type": "string"}},
               "required": ["今の絵", "変更後の絵", "出てほしくないもの", "変えたこと"]}

EDIT_EXAMPLES = [
    ({"プロンプト": "1girl, solo, long hair, black hair, twin braids, school uniform, red necktie, standing, classroom, (smile:1.1), anime coloring, masterpiece",
      "ネガティブプロンプト": "lowres, realistic, 3d",
      "指示": "髪を銀色のショートにして、アニメから実写に。眼鏡はなしで"},
     {"今の絵": "女の子が 1 人。黒髪のロングを三つ編みのおさげにしている。制服に赤いネクタイ。教室で立っていて、少し強調した笑顔。アニメ塗りの高品質な絵。",
      "変更後の絵": "女の子が 1 人。銀髪のショートヘア（短いのでおさげはない）。制服に赤いネクタイ。教室で立っていて、少し強調した笑顔（1.1）。実写の写真で高品質。",
      "出てほしくないもの": "低解像度、眼鏡（実写にするので、実写っぽさや 3D を避ける指定はやめる）",
      "変えたこと": "髪を銀髪のショートに、画風を実写に変え、眼鏡を避けるようにしました。"}),
]

NO_NEGATIVE = ("\nこの画像生成モデルはネガティブプロンプトを使いません（CFG 1 で生成するので効きません）。"
               "\"出てほしくないもの\" は空にし、出てほしくないものは変更後の絵に書かないことで表す。")
HISTORY = ("\nこの会話でのこれまでの指示と、そのとき LLM がどう直したかを古い順に渡します。今のプロンプトには、もうその変更が入っています。"
           "新しい指示は、前の直し方への訂正や不満（「違う」「そこは触らないで」「さっき〜と言った」など）のことがあります。"
           "そのときは前の直し方を改め、ユーザーが前に言った決まりはこの後も守ること。")


# ---------------------------------------------------------------- step 2: write back as a prompt
WRITE_TAGS = """あなたは、日本語で書かれた絵の説明を、画像生成のプロンプトに書き直します。
プロンプトは英語の Danbooru タグを ", " で区切ったものです。
- 元のプロンプトと元の絵の説明も渡します。変わっていない部分は、元のプロンプトのタグを、重みや順番も含めてそのまま使う。
- 変わった部分だけ、新しい絵の説明に合うタグに置き換える。元の説明にあって新しい説明にないものは入れない。
- 1 つの要素に 1 タグ（"silver hair", "short hair"）。人数は 1girl, 1boy, 2girls のように数え、2 人以上なら solo は入れない。
- 強調・弱めは "(タグ:1.3)" の形。説明に強さが書いてあればその値。
- "ネガティブ" は、出てほしくないものを同じ形式のタグで。元のネガティブで変わっていないものは、そのままのタグを使う。
JSON だけを返す。"""

WRITE_SENTENCES = """あなたは、日本語で書かれた絵の説明を、画像生成のプロンプト（絵を説明する英語の文章）に書き直します。
- 元のプロンプトと元の絵の説明も渡します。変わっていない部分は、元のプロンプトの言い回しをそのまま使う。
- 変わった部分だけ、新しい絵の説明に合わせて書き直す。元の説明にあって新しい説明にないものは書かない。タグの羅列にはしない。
- "ネガティブ" は、出てほしくないものを、元のネガティブと同じ書き方で。
JSON だけを返す。"""

WRITE_SCHEMA = {"type": "object", "properties": {"プロンプト": {"type": "string"}, "ネガティブ": {"type": "string"}},
                "required": ["プロンプト", "ネガティブ"]}

WRITE_EXAMPLES = {
    "tags": [({"元のプロンプト": EDIT_EXAMPLES[0][0]["プロンプト"], "元のネガティブ": EDIT_EXAMPLES[0][0]["ネガティブプロンプト"],
               "元の絵": EDIT_EXAMPLES[0][1]["今の絵"], "新しい絵": EDIT_EXAMPLES[0][1]["変更後の絵"],
               "出てほしくないもの": EDIT_EXAMPLES[0][1]["出てほしくないもの"]},
              {"プロンプト": "1girl, solo, short hair, silver hair, school uniform, red necktie, standing, classroom, (smile:1.1), photorealistic, masterpiece",
               "ネガティブ": "lowres, glasses"})],
    "natural": [({"元のプロンプト": "A girl with long black hair in twin braids stands in a classroom, wearing a school uniform. Anime style.",
                  "元のネガティブ": "blurry, realistic",
                  "元の絵": "黒髪のロングを三つ編みにした女の子が、制服で教室に立っている。アニメ調。",
                  "新しい絵": "銀髪のショートの女の子が、制服で教室に立っている。実写の写真。",
                  "出てほしくないもの": "ぼやけ、帽子"},
                 {"プロンプト": "A photo of a girl with short silver hair standing in a classroom, wearing a school uniform.",
                  "ネガティブ": "blurry, hat"})],
}


# ---------------------------------------------------------------- entry
def rewrite(positive: str, negative: str, instruction: str, style: str, keep: list[str], model: str | None = None,
            negative_used: bool = True, history: list | None = None) -> dict:
    """history: earlier turns of this conversation [{instruction, changes}], oldest first.  negative_used:
    False for models that generate at CFG 1 - the negative prompt is then never changed."""
    instruction = (instruction or "").strip()
    if not instruction:
        raise ValueError("どう変えたいかを入力してください。")
    c = get_conf()
    model = model or models()["model"]
    if not model:
        raise LlmError("Ollama にモデルが入っていません。「画像からプロンプト」の設定でモデルを取得してください。")
    global _last_model
    _last_model = model
    keep_text = "、".join(keep) if keep else "（なし）"

    # 1. understand and edit, in Japanese
    system = EDIT_SYSTEM.format(keep=keep_text) + ("" if negative_used else NO_NEGATIVE) + (HISTORY if history else "")
    ask = {"プロンプト": positive, "指示": instruction}
    if negative_used:
        ask["ネガティブプロンプト"] = negative
    if history:
        ask = {"これまでの指示": _history(history), **ask}
    edit = _chat(c["host"], model, system, _shots(EDIT_EXAMPLES), json.dumps(ask, ensure_ascii=False), EDIT_SCHEMA)
    before, after = str(edit.get("今の絵") or "").strip(), str(edit.get("変更後の絵") or "").strip()
    avoid = str(edit.get("出てほしくないもの") or "").strip() if negative_used else ""
    if not after:
        raise LlmError("LLM が変更後の絵を書きませんでした。指示を言い換えるか、別のモデルを選んでください。")

    # 2. write the Japanese description back as a prompt in the image model's format
    kind = "natural" if style == "natural" else "tags"
    src = {"元のプロンプト": positive, "元のネガティブ": negative if negative_used else "", "元の絵": before,
           "新しい絵": after, "出てほしくないもの": avoid}
    out = _chat(c["host"], model, WRITE_SENTENCES if kind == "natural" else WRITE_TAGS, _shots(WRITE_EXAMPLES[kind]),
                json.dumps(src, ensure_ascii=False), WRITE_SCHEMA)
    new_pos = str(out.get("プロンプト") or "").strip()
    new_neg = str(out.get("ネガティブ") or "").strip() if negative_used else negative.strip()
    if not new_pos:
        raise LlmError("LLM が空のプロンプトを返しました。指示を言い換えるか、別のモデルを選んでください。")
    r = {"positive": new_pos, "negative": new_neg, "note": str(edit.get("変えたこと") or "").strip(),
         "before": before, "after": after, "model": model,
         "lost_words": [w for w in keep if w.lower() not in new_pos.lower()]}
    r["changes"] = [r["note"]] if r["note"] else []
    if (new_pos, new_neg) == (positive.strip(), negative.strip()):
        r["note"] = "LLM は何も変えませんでした。指示を言い換えるか、別のモデルを選んでください。"
        r["unchanged"] = True
    return r


def _history(history) -> list[dict]:
    out = []
    for h in (history or [])[-MAX_HISTORY:]:
        ins = str((h or {}).get("instruction") or "").strip()
        if ins:
            out.append({"指示": ins[:300], "直し方": [str(c)[:200] for c in (h.get("changes") or [])][:6]})
    return out


def _unload():
    """GPU hook: training / generation takes the GPU - free the LLM's VRAM (no-op when nothing is loaded)."""
    global _last_model
    if not _last_model:
        return
    try:
        _req(get_conf()["host"], "/api/generate", {"model": _last_model, "keep_alive": 0}, timeout=5)
    except LlmError:
        pass
    _last_model = ""   # unloaded (or Ollama is gone): not asked again on every job start


gpu.on_exclusive(_unload)
