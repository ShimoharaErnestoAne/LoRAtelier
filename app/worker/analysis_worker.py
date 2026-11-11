"""Resident AI analysis worker.  Runs in envs/analysis (torch + imgutils; kept separate from the engines' pinned deps) as a child process
of the UI server and talks JSON lines over stdin/stdout.  Models are loaded lazily on first use
and kept resident (no reload per image); `release` frees VRAM before training starts.

Models (licenses checked, see LICENSES.md):
  * deepghs imgutils (MIT): WD14 EVA02-Large tagger v3 (Apache-2.0), anime person/face/head
    detection (MIT), CCIP character similarity (OpenRAIL), anime/real classification
  * torchvision Faster R-CNN v2 (BSD-3, COCO) for people in photographs
  * masks for masked training: skytnt anime-seg ISNet (Apache-2.0, through imgutils) for illustrations,
    torchvision Mask R-CNN v2 (BSD-3, COCO) for photographs
  * SigLIP2 so400m (Apache-2.0) image embeddings
  * Qwen3-VL-4B-Instruct (Apache-2.0) natural-language captions
"""
from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import traceback

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

ANALYZER_VERSION = "ai-4"
_out_lock = threading.Lock()


def emit(obj):
    with _out_lock:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_image(path, crop=None):
    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        # a full-size RGBA copy of a large photo is costly: only images with transparency are put on white
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
        else:
            bg = im.convert("RGB")
    if crop:
        bg = bg.crop(tuple(int(v) for v in crop))
    return bg


def _features(out):
    """transformers 5 returns a model output object; 4.x returned the tensor."""
    return out if hasattr(out, "norm") else out.pooler_output


class Models:
    def __init__(self):
        self.torch = None
        self.device = "cpu"
        self.siglip = None
        self.frcnn = None
        self.maskrcnn = None
        self.qwen = None
        self.gpu_error = None

    def _torch(self):
        if self.torch is None:
            import torch
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                # is_available() only asks the driver how many GPUs there are; CUDA itself starts on the first
                # tensor, and a GPU left broken (after sleep, a driver reset) fails only there.  Start it now: on
                # failure this process analyses on the CPU (the ONNX models too) instead of failing every image.
                try:
                    torch.zeros(1, device="cuda")
                except Exception as e:
                    log(traceback.format_exc())
                    self.device, self.gpu_error = "cpu", f"{type(e).__name__}: {e}"[:300]
                    os.environ["ONNX_MODE"] = "cpu"
        return self.torch

    def loading(self, name):
        emit({"event": "loading", "model": name})

    # -- SigLIP2
    def get_siglip(self):
        if self.siglip is None:
            torch = self._torch()
            from transformers import AutoModel, AutoProcessor
            self.loading("SigLIP2")
            name = "google/siglip2-so400m-patch16-384"
            proc = AutoProcessor.from_pretrained(name)
            model = AutoModel.from_pretrained(name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
            self.siglip = (proc, model.to(self.device).eval())
        return self.siglip

    def embed_images(self, images):
        torch = self._torch()
        proc, model = self.get_siglip()
        with torch.no_grad():
            inp = proc(images=images, return_tensors="pt").to(self.device)
            inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
            f = _features(model.get_image_features(**inp))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    def embed_texts(self, texts):
        torch = self._torch()
        proc, model = self.get_siglip()
        with torch.no_grad():
            inp = proc(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
            f = _features(model.get_text_features(**inp))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    # -- photo person detector
    def get_frcnn(self):
        if self.frcnn is None:
            self._torch()
            import torchvision
            self.loading("Faster R-CNN")
            w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=0.4)
            self.frcnn = m.to(self.device).eval()
        return self.frcnn

    def detect_people_photo(self, img):
        torch = self._torch()
        import torchvision.transforms.functional as TF
        m = self.get_frcnn()
        # the detector scales its input to at most 1333 px itself: a 24 MP photo sent whole only fills the VRAM
        small = img
        if max(img.size) > 1333:
            small = img.copy()
            small.thumbnail((1333, 1333))
        sx, sy = img.size[0] / small.size[0], img.size[1] / small.size[1]
        x = TF.to_tensor(small).to(self.device)
        with torch.no_grad():
            r = m([x])[0]
        out = []
        for box, label, score in zip(r["boxes"].cpu().numpy(), r["labels"].cpu().numpy(), r["scores"].cpu().numpy()):
            if int(label) == 1:  # COCO person
                out.append({"bbox": [float(box[0]) * sx, float(box[1]) * sy, float(box[2]) * sx, float(box[3]) * sy],
                            "score": float(score)})
        return out

    # -- photo person masks (masked training)
    def get_maskrcnn(self):
        if self.maskrcnn is None:
            self._torch()
            import torchvision
            self.loading("Mask R-CNN")
            w = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            m = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=0.5)
            self.maskrcnn = m.to(self.device).eval()
        return self.maskrcnn

    def person_masks_photo(self, img):
        """Soft masks (H, W, 0..1) of the people in a photograph, at the image's size."""
        torch = self._torch()
        import torchvision.transforms.functional as TF
        small = img.copy()
        small.thumbnail((1024, 1024))
        with torch.no_grad():
            r = self.get_maskrcnn()([TF.to_tensor(small).to(self.device)])[0]
        out = []
        for m, label in zip(r["masks"].cpu().numpy(), r["labels"].cpu().numpy()):
            if int(label) == 1:  # COCO person
                out.append(np.asarray(Image.fromarray((m[0] * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR),
                                      np.float32) / 255.0)
        return out

    # -- Qwen3-VL captioner
    def get_qwen(self):
        if self.qwen is None:
            torch = self._torch()
            if self.gpu_error:   # a 4B VLM on the CPU takes minutes per image: say so instead of crawling
                raise RuntimeError("GPU が使えないため、説明文の生成を止めました。しばらくして GPU が使えるようになると"
                                   "自動で GPU に戻ります。続くときは PC を再起動してください。")
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.loading("Qwen3-VL-4B")
            name = "Qwen/Qwen3-VL-4B-Instruct"
            proc = AutoProcessor.from_pretrained(name)
            model = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=torch.bfloat16, device_map=self.device)
            self.qwen = (proc, model.eval())
        return self.qwen

    def caption(self, img, instruction):
        return self.captions([img], instruction)[0]

    def captions(self, imgs, instruction):
        """Several images in one generate call (left padding): the GPU is mostly idle when captioning one
        image at a time.  Out of memory: the batch is split in halves and retried."""
        torch = self._torch()
        proc, model = self.get_qwen()
        small = []
        for img in imgs:
            img = img.copy()
            img.thumbnail((896, 896))
            small.append(img)
        convs = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": instruction}]}]
                 for img in small]
        try:
            proc.tokenizer.padding_side = "left"
            inputs = proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        except Exception as e:
            if "out of memory" not in str(e).lower() or len(imgs) == 1:
                raise
            del e
            torch.cuda.empty_cache()
            h = len(imgs) // 2
            return self.captions(imgs[:h], instruction) + self.captions(imgs[h:], instruction)
        texts = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return [t.strip() for t in texts]

    def release(self):
        self.siglip = None
        self.frcnn = None
        self.maskrcnn = None
        self.qwen = None
        try:
            from imgutils.utils import ts_lru_cache  # noqa: F401
            import imgutils.tagging.wd14 as wd
            for fn in ("_get_wd14_model", "_get_wd14_labels"):
                f = getattr(wd, fn, None)
                if f is not None and hasattr(f, "cache_clear"):
                    f.cache_clear()
        except Exception:
            pass
        # every ONNX session imgutils keeps (CCIP, ISNet, YOLO person / head, classifiers) is held by a cached
        # loader function: clear them all, not only the tagger's
        for name, mod in list(sys.modules.items()):
            if not name.startswith("imgutils.") or mod is None:
                continue
            for f in list(vars(mod).values()):
                if callable(f) and hasattr(f, "cache_clear"):
                    try:
                        f.cache_clear()
                    except Exception:
                        pass
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


M = Models()


def _norm_box(b, w, h):
    return [b[0] / w, b[1] / h, b[2] / w, b[3] / h]


def analyze(path, crop):
    from imgutils.detect import detect_heads, detect_person
    from imgutils.tagging import get_wd14_tags
    from imgutils.validate import anime_real_score

    img = load_image(path, crop)
    w, h = img.size
    res = {"version": ANALYZER_VERSION, "size": [w, h]}
    t0 = time.time()

    real = anime_real_score(img)  # {'anime': p, 'real': p}
    p_real = float(real.get("real", 0.0))
    # The anime/real classifier alone calls scanned prints and paintings "real" (verified on ukiyo-e scans:
    # p_real up to 0.96).  A photo needs both the classifier and SigLIP zero-shot (photo vs artwork) to agree.
    emb_full = M.embed_images([img])[0]
    zs_photo, zs_art = domain_zero_shot(emb_full)
    is_photo = p_real >= 0.5 and zs_photo > zs_art
    label = "photo" if is_photo else "illustration"
    agree = (p_real >= 0.5) == (zs_photo > zs_art)
    margin = abs(p_real - 0.5)
    conf = ("高" if margin > 0.4 else "中") if agree else "低"
    res["domain"] = {"label": label, "p_real": round(p_real, 3), "zs_photo": round(float(zs_photo), 4),
                     "zs_art": round(float(zs_art), 4), "confidence": conf}

    rating, general, chars = get_wd14_tags(img, model_name="EVA02_Large", general_threshold=0.2, character_threshold=0.5)
    res["tags"] = {"general": {k: round(float(v), 3) for k, v in general.items()},
                   "character": {k: round(float(v), 3) for k, v in chars.items()},
                   "rating": {k: round(float(v), 3) for k, v in rating.items()}}

    if label == "photo":
        persons = M.detect_people_photo(img)
        res["heads"] = []
    else:
        persons = [{"bbox": list(map(float, b)), "score": float(s)} for b, _l, s in detect_person(img)]
        res["heads"] = [{"bbox": list(map(float, b)), "score": float(s)} for b, _l, s in detect_heads(img)]
    res["persons"] = persons
    main = max(persons, key=lambda p: (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]), default=None)
    if main:
        x0, y0, x1, y1 = main["bbox"]
        area = (x1 - x0) * (y1 - y0) / (w * h)
        edges = [n for n, hit in (("left", x0 < 2), ("top", y0 < 2), ("right", x1 > w - 2), ("bottom", y1 > h - 2)) if hit]
        res["subject"] = {"bbox": [x0, y0, x1, y1], "area_ratio": round(area, 4),
                          "eff_res": int(min(x1 - x0, y1 - y0)), "touches": edges}
        subj_img = img.crop((int(x0), int(y0), int(x1), int(y1)))
    else:
        subj_img = None

    if label != "photo" and subj_img is not None:
        from imgutils.metrics import ccip_extract_feature
        res["ccip"] = [round(float(v), 6) for v in np.asarray(ccip_extract_feature(subj_img)).ravel()]

    res["embed"] = [round(float(v), 5) for v in emb_full]
    # read only where there is no CCIP (the identity check of photographs): not computed and stored for illustrations
    if subj_img is not None and "ccip" not in res:
        res["subject_embed"] = [round(float(v), 5) for v in M.embed_images([subj_img])[0]]
    if label == "photo":
        res["zs"] = zero_shot(emb_full)
    res["elapsed"] = round(time.time() - t0, 2)
    return res


ZS_PROMPTS = {
    "portrait": "a close-up photo of a person's face", "upper body": "an upper body photo of a person",
    "full body": "a full body photo of a person standing", "from side": "a photo of a person seen from the side, in profile",
    "from behind": "a photo of a person seen from behind", "simple background": "a photo of a person in front of a plain studio background",
}
_zs_cache = {}


def zero_shot(img_emb):
    if "t" not in _zs_cache:
        _zs_cache["t"] = M.embed_texts(list(ZS_PROMPTS.values()))
    sims = _zs_cache["t"] @ np.asarray(img_emb)
    return {k: round(float(s), 4) for k, s in zip(ZS_PROMPTS, sims)}


DOMAIN_PROMPTS = (["a photograph", "a photo taken with a camera"],
                  ["a painting", "an illustration", "a woodblock print", "a drawing", "an anime illustration"])


def domain_zero_shot(img_emb):
    if "d" not in _zs_cache:
        _zs_cache["d"] = (M.embed_texts(DOMAIN_PROMPTS[0]), M.embed_texts(DOMAIN_PROMPTS[1]))
    ph, art = _zs_cache["d"]
    e = np.asarray(img_emb)
    return float((ph @ e).max()), float((art @ e).max())


def make_mask(path, mode, domain, out, crop=None):
    """Mask for masked training, written to `out` as an 8-bit grey PNG at the image's own size (white = learnt).
    person: the main person (the largest one inside the crop, which is what is trained; other people count as
    background).  background: everything but the people.  Edges are widened a little so hair and outlines are not
    cut off.  Returns facts for the UI (people and coverage inside the crop)."""
    import cv2
    img = load_image(path)
    w, h = img.size
    if crop and mode == "person":
        cx0, cy0, cx1, cy1 = (max(0, min(int(v), lim)) for v, lim in zip(crop, (w, h, w, h)))
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            cx0, cy0, cx1, cy1 = 0, 0, w, h
    else:
        cx0, cy0, cx1, cy1 = 0, 0, w, h

    def inside(b):   # area of a box inside the crop
        return max(0.0, min(b[2], cx1) - max(b[0], cx0)) * max(0.0, min(b[3], cy1) - max(b[1], cy0))

    if domain == "photo":
        method = "maskrcnn"
        people = M.person_masks_photo(img)
        n = sum(1 for x in people if float(x[cy0:cy1, cx0:cx1].sum()) > 0.05 * float(x.sum()))
        if mode == "person":
            m = max(people, key=lambda x: float(x[cy0:cy1, cx0:cx1].sum()), default=None)
        else:
            m = np.maximum.reduce(people) if people else np.zeros((h, w), np.float32)
    else:
        from imgutils.detect import detect_person
        from imgutils.segment import get_isnetis_mask
        method = "isnetis"
        seg = np.clip(np.asarray(get_isnetis_mask(img), np.float32), 0, 1)
        boxes = [b for b, _l, _s in detect_person(img)]
        shown = [b for b in boxes if inside(b) > 0.05 * max(1.0, (b[2] - b[0]) * (b[3] - b[1]))]   # in the crop
        n = len(shown) if mode == "person" else len(boxes)
        m = seg
        if mode == "person":
            if not shown and seg[cy0:cy1, cx0:cx1].mean() < 0.01:
                m = None
            elif len(boxes) > 1 and shown:   # keep the largest person in the crop: the segmenter does not tell people apart
                x0, y0, x1, y1 = max(boxes, key=inside)
                px, py = 0.04 * (x1 - x0), 0.04 * (y1 - y0)
                keep = np.zeros_like(seg)
                keep[max(0, int(y0 - py)):int(y1 + py), max(0, int(x0 - px)):int(x1 + px)] = 1
                m = seg * keep
    if m is None or (mode == "person" and float(m[cy0:cy1, cx0:cx1].mean()) < 0.005):
        return {"status": "none", "people": n, "method": method, "coverage": 0.0}
    k = max(3, int(min(w, h) * 0.012)) | 1
    m8 = cv2.dilate((np.clip(m, 0, 1) * 255).astype(np.uint8), np.ones((k, k), np.uint8))
    m8 = cv2.GaussianBlur(m8, (k, k), 0)
    if mode == "background":
        m8 = 255 - m8
    tmp = out + ".part.png"
    Image.fromarray(m8, "L").save(tmp)
    os.replace(tmp, out)
    return {"status": "done", "people": n, "method": method, "coverage": round(float(m8[cy0:cy1, cx0:cx1].mean()) / 255, 4),
            "size": [w, h]}


def ccip_diff(feats, refs=None):
    """Differences of each feature to each reference; without refs, of all features to each other (computed once,
    not as a part of the matrix of both lists)."""
    from imgutils.metrics import ccip_batch_differences
    if not refs:
        d = ccip_batch_differences([np.asarray(f, dtype=np.float32) for f in feats])
        return [[float(v) for v in row] for row in d]
    allf = [np.asarray(f, dtype=np.float32) for f in refs + feats]
    d = ccip_batch_differences(allf)
    nref = len(refs)
    return [[float(d[i][j]) for j in range(nref)] for i in range(nref, len(allf))]


def main():
    # torch before any ONNX model: onnxruntime-gpu finds cuDNN / cuBLAS only through the DLLs torch loads, and a
    # model opened before that silently runs on the CPU for the whole session (imgutils happens to import torch today)
    M._torch()
    emit({"event": "ready", "version": ANALYZER_VERSION, "pid": os.getpid(), "gpu_error": M.gpu_error})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        op, rid = req.get("op"), req.get("rid")
        try:
            if op == "analyze":
                emit({"rid": rid, "ok": True, "result": analyze(req["path"], req.get("crop"))})
            elif op == "caption":
                img = load_image(req["path"], req.get("crop"))
                emit({"rid": rid, "ok": True, "text": M.caption(img, req["instruction"])})
            elif op == "captions":
                imgs = []
                for p, c in zip(req["paths"], req["crops"]):
                    try:
                        imgs.append(load_image(p, c))
                    except Exception:   # one unreadable image must not fail the others of the batch
                        imgs.append(None)
                texts = iter(M.captions([im for im in imgs if im is not None], req["instruction"])
                             if any(im is not None for im in imgs) else [])
                emit({"rid": rid, "ok": True, "texts": [None if im is None else next(texts, None) for im in imgs]})
            elif op == "ccip_diff":
                emit({"rid": rid, "ok": True, "matrix": ccip_diff(req["feats"], req.get("refs"))})
            elif op == "embed_images":
                imgs = [load_image(p) for p in req["paths"]]
                emit({"rid": rid, "ok": True, "embeds": M.embed_images(imgs).round(5).tolist()})
            elif op == "embed_texts":
                emit({"rid": rid, "ok": True, "embeds": M.embed_texts(req["texts"]).round(5).tolist()})
            elif op == "ccip_features":
                from imgutils.metrics import ccip_extract_feature
                emit({"rid": rid, "ok": True, "feats": [np.asarray(ccip_extract_feature(load_image(p))).ravel().round(6).tolist()
                                                        for p in req["paths"]]})
            elif op == "tag_images":
                from imgutils.tagging import get_wd14_tags
                out = []
                for p in req["paths"]:
                    _r, general, _c = get_wd14_tags(load_image(p), model_name="EVA02_Large", general_threshold=0.05)
                    out.append({k: round(float(v), 3) for k, v in general.items()})
                emit({"rid": rid, "ok": True, "tags": out})
            elif op == "mask":
                emit({"rid": rid, "ok": True, "result": make_mask(req["path"], req["mode"], req.get("domain"), req["out"],
                                                                   req.get("crop"))})
            elif op == "release":
                M.release()
                emit({"rid": rid, "ok": True})
            elif op == "quit":
                emit({"rid": rid, "ok": True})
                break
            else:
                emit({"rid": rid, "ok": False, "error": f"unknown op {op}"})
        except Exception as e:  # report, never die silently
            log(traceback.format_exc())
            msg = f"{type(e).__name__}: {e}"
            if "out of memory" in str(e).lower():
                M.release()
                msg = "GPU メモリ不足: " + msg
            elif "cuda error" in str(e).lower():
                # torch keeps a failed CUDA state for the life of the process: every later request would fail the
                # same way.  End the process; the next request starts a new worker (at most 3 times per 10 min).
                emit({"rid": rid, "ok": False, "error": ("GPU が使えない状態になったため、解析を止めました。もう一度実行すると"
                                                         "解析をやり直します。続けて起きるときは PC を再起動してください。 "
                                                         + msg)[:500]})
                break
            emit({"rid": rid, "ok": False, "error": msg[:500]})


if __name__ == "__main__":
    main()
