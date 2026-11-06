/* 「画像からプロンプト」: the i2t window as a page of LoRAtelier.  The analysis is i2t's own code
   (bundled copy, run by worker/i2t_worker.py); this file rebuilds the window: images and the 8 slots,
   構図と強さ, 出力, results with chips and Japanese labels, settings, pose editor, history, generation info,
   Ollama setup, model downloads, keyboard shortcuts.  Feature list: docs/I2T_PORT.md. */
"use strict";

const I = {
  st: null, c: null, cfg: null, slots: {}, result: null, histPath: null, chips: { pos: null, neg: null },
  undo: { pos: [], neg: [] }, sel: { pos: new Set(), neg: new Set() }, anchor: { pos: null, neg: null }, focusBox: null,
  ja: false, labels: { pos: null, neg: null }, job: null, poseControl: null, look: "", lookParts: {}, camera: {}, strength: {},
  ollama: { ok: false, models: [], vision: null, error: null }, hover: null, est: null, paused: false, tab: null,
};
const SLOT_ORDER = ["person", "pose", "outfit", "background", "props", "body", "style", "expression"];

function iApi(fn, ...args) { return api("/api/i2t/call", { body: { fn, args } }).then((r) => r.value); }
let iCfgSeq = 0;
const iSaveCfg = saveDebounce(async () => {
  const seq = iCfgSeq;
  try {
    const r = await api("/api/i2t/config", { method: "PUT", body: I.cfg });
    if (seq === iCfgSeq) I.cfg = r;   // a setting changed meanwhile is kept (and sent by the next save)
  } catch (e) { fail(e); }
}, 600);
function iSet(key, value) { I.cfg[key] = value; iCfgSeq++; iSaveCfg(); }
function multi() { return I.cfg.mode === "multi"; }
function iStatus(text) { const el = $("#i-status") || $("#s-i2t-status"); if (el) el.textContent = text; }

// ------------------------------------------------------------------ page
async function renderI2t() {
  S.project = null;
  $("#crumbs").innerHTML = `<span>›</span><span>画像からプロンプト</span>`;
  if (!$("#i-root")) main.innerHTML = `<div class="muted">読み込み中…（初回は i2t の準備に数秒かかります）</div>`;
  let st;
  try { st = await api("/api/i2t/state"); } catch (e) { main.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
  if (!st.available) { main.innerHTML = `<div class="err">${esc(st.message)}</div>`; return; }
  I.st = st; I.c = st.consts; I.cfg = st.config; I.job = st.job; I.paused = !!(st.job && st.job.paused);
  main.innerHTML = `<div id="i-root" class="i2t">
    <div class="i-top row">
      <b>${esc(I.c.app_name)} v${esc(I.c.version)}</b>
      <span class="small">モード</span>${Object.entries(I.c.modes).map(([k, l]) => `<label class="small"><input type="radio" name="i-mode" value="${k}" ${I.cfg.mode === k ? "checked" : ""}>${esc(l)}</label>`).join("")}
      <button id="i-pick">画像を選択...</button><input type="file" id="i-file" accept="image/*" hidden multiple>
      <button id="i-clear">✕ 画像をクリア</button>
      <div class="spacer"></div>
      <button id="i-settings">⚙ 設定...</button><button id="i-help">？ ヘルプ</button>
    </div>
    <div class="i-path small muted" id="i-path"></div>
    <div class="i-body">
      <div class="i-left">
        <div id="i-preview" class="i-preview"></div>
        <div class="muted small" id="i-preview-hint"></div>
        <div class="panel i-box" id="i-controls"></div>
        <div class="panel i-box" id="i-output"></div>
      </div>
      <div class="i-right">
        <div class="i-res">
          <div class="row"><b id="i-pos-title">ポジティブプロンプト</b><div class="spacer"></div>
            <button class="small" id="i-lang" hidden>日本語</button><button class="small" data-copy="pos">コピー</button>
            <button class="primary small" id="i-to-gen" disabled title="ポジティブとネガティブを画像生成の入力欄に入れて、画像生成の画面に移ります">画像生成で使う →</button>
            <button class="small" id="i-to-i2i" disabled title="メインの画像を元画像（I2I）にし、ポジティブとネガティブを入れて画像生成の画面に移ります。ほかの欄の画像から読んだポーズ・服装・背景などは、プロンプトとしてメインの画像に重ねます">メインの画像から生成（I2I）→</button></div>
          <div id="i-pos" class="i-text" tabindex="0"></div>
          <div class="row" style="margin-top:8px"><b id="i-neg-title">ネガティブプロンプト</b><div class="spacer"></div><button class="small" data-copy="neg">コピー</button></div>
          <div id="i-neg" class="i-text small-h" tabindex="0"></div>
          <nav class="tabs i-tabs" id="i-tabs">
            <a data-tab="extras">追加プロンプト</a><a data-tab="persons">人物ごと</a><a data-tab="tags">検出タグ (信頼度)</a>
            <a data-tab="analysis">分析メモ</a><a data-tab="notes">メモ (推奨設定・コツ)</a></nav>
          <div id="i-tab" class="i-tabbody"></div>
          <div class="row" style="margin-top:8px">
            <button id="i-save-txt">TXT に保存...</button><button id="i-save-json">JSON に保存...</button><button id="i-copy-both">両方コピー</button>
            <button id="i-history">履歴...</button><button id="i-geninfo" disabled>生成情報...</button>
          </div>
        </div>
      </div>
    </div>
    <div class="i-bottom panel">
      <div class="row">
        <button class="primary big" id="i-run">▶ 分析する</button><button id="i-stop" disabled>■ 中止</button><button id="i-pause" disabled>⏸ 一時停止</button>
        <div class="i-prog"><div id="i-bar"></div></div><span id="i-pct" class="small"></span>
        <span id="i-ollama" class="small muted"></span>
      </div>
      <div id="i-status" class="small i-statustext">画像を入れてください（ドロップ / 「画像を選択...」/ Ctrl+V）</div>
    </div>
    <div id="i-modal"></div>
    <div id="i-menu"></div>
  </div>`;
  iBindTop();
  iDrawControls();
  iDrawOutput();
  iRenderPreview();
  const done = st.last_done;   // an analysis that finished while this page was not open
  if (!I.job && done && done.op === "analyze" && done.ok && done.rid !== I.shownRid) { I.shownRid = done.rid; await iShowResult(done.result, done.history); }
  else await iShowResult(I.result, I.histPath, true);
  iUpdateButtons();
  if (I.job) iOnJob(I.job);
  iProbeOllama();
  if (!I.c.tagger_models[I.cfg.tagger_model]?.ready) setTimeout(() => iTaggerDialog("first"), 400);
}

function iBindTop() {
  $$("input[name=i-mode]").forEach((r) => r.onchange = () => { iSet("mode", r.value); iRenderPreview(); iUpdateButtons(); });
  $("#i-pick").onclick = () => { I.pickSlot = null; $("#i-file").click(); };
  $("#i-file").onchange = (e) => { const fs = [...e.target.files]; e.target.value = ""; iAddFiles(fs, I.pickSlot); };
  $("#i-clear").onclick = () => { if (I.job) return; I.slots = {}; iRenderPreview(); iUpdateButtons(); iStatus("画像を外しました"); };
  $("#i-settings").onclick = () => { I.gotoSettings = true; location.hash = "#/settings"; };
  $("#i-help").onclick = () => iHelpDialog();
  $("#i-run").onclick = iAnalyze;
  $("#i-stop").onclick = iCancel;
  $("#i-pause").onclick = () => api(`/api/i2t/${I.paused ? "resume" : "pause"}`, { body: {} }).catch(fail);   // the "paused" event updates the button
  $$("[data-copy]").forEach((b) => b.onclick = () => iCopy(iJoin(b.dataset.copy)));
  $("#i-copy-both").onclick = () => { const p = iJoin("pos"), n = iJoin("neg"); iCopy(n ? `Positive:\n${p}\n\nNegative:\n${n}` : p); };
  $("#i-save-txt").onclick = () => iSave("txt");
  $("#i-save-json").onclick = () => iSave("json");
  $("#i-history").onclick = iHistoryDialog;
  $("#i-geninfo").onclick = iGenInfoDialog;
  $("#i-to-gen").onclick = iToImageGen;
  $("#i-to-i2i").onclick = iToImageGenI2I;
  $("#i-lang").onclick = () => iSetLang(!I.ja);
  $$("#i-tabs a").forEach((a) => a.onclick = () => { I.tab = a.dataset.tab; iDrawTab(); });
  for (const box of ["pos", "neg"]) {
    const el = $(`#i-${box}`);
    el.onclick = (e) => iChipClick(box, e);
    el.onfocus = () => { I.focusBox = box; };
  }
}

// ------------------------------------------------------------------ images
function iSlotLabel(k) { return I.c.ref_slots[k]; }
function iFirstEmpty() { return SLOT_ORDER.find((k) => !I.slots[k]); }

async function iUpload(file) {
  const r = await fetch(`/api/i2t/upload?name=${encodeURIComponent(file.name || "paste.png")}`, { method: "POST", body: file });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
  return d;
}

async function iAddFiles(files, slot) {
  if (I.job) { iStatus("処理の最中は画像を入れられません"); return; }
  files = files.filter((f) => f.type.startsWith("image/") || /\.(png|jpe?g|webp|gif|bmp|tiff?)$/i.test(f.name));
  if (!files.length) { iStatus("画像ファイルではありません"); return; }
  if (!multi()) {
    if (files.length > 1) toast("1 枚モードでは先頭の 1 枚だけを読み込みます（8 枚モードなら複数まとめて入れられます）");
    files = files.slice(0, 1); slot = "person";
  } else if (files.length > 1) {   // several files: fill ①→⑧ from the start, the rest is dropped
    I.slots = {}; files = files.slice(0, 8);
  }
  try {
    for (const [i, f] of files.entries()) {
      const k = files.length > 1 ? SLOT_ORDER[i] : (slot || iFirstEmpty());
      if (!k) { iStatus("空いている枠がありません。差し替えたい枠にドロップしてください"); return; }
      iStatus(`読み込み中: ${f.name || "貼り付けた画像"}`);
      I.slots[k] = await iUpload(f);
    }
  } catch (e) { fail(e); }
  iAfterLoad();
}

function iAfterLoad() {
  iRenderPreview(); iUpdateButtons();
  const empty = multi() ? SLOT_ORDER.slice(1).filter((k) => !I.slots[k]) : [];
  let msg = I.slots.person ? "「▶ 分析する」を押してください" : "① メインの画像を入れてください";
  if (multi() && I.slots.person && empty.length) {   // which image an empty slot takes its part from
    const fromOutfit = empty.filter((k) => k === "props" && I.slots.outfit), fromFirst = empty.filter((k) => !fromOutfit.includes(k));
    msg += `（${[fromFirst.length ? `${fromFirst.map((k) => iSlotLabel(k).slice(2)).join("・")} は 1 枚目から` : "", fromOutfit.length ? "小物 は ③ から" : ""].filter(Boolean).join("、")}）`;
  }
  const gi = SLOT_ORDER.filter((k) => I.slots[k] && I.slots[k].geninfo);
  if (gi.length) msg += `　生成情報あり (${I.slots[gi[0]].geninfo.source}。「生成情報...」で表示)`;
  iStatus(msg);
}

function iRenderPreview() {
  const el = $("#i-preview");
  if (!el) return;
  const mos = I.cfg.preview_mosaic;
  if (!multi()) {
    const s = I.slots.person;
    el.className = "i-preview single";
    el.innerHTML = s ? `<div class="i-imgwrap"><img src="/api/i2t/file?thumb=1${mos ? "&mosaic=8" : ""}&path=${encodeURIComponent(s.path)}" data-slot="person" alt=""><svg class="i-boxes" id="i-boxes"></svg></div>`
      : `<div class="i-empty" data-slot="person">ここに画像をドロップ<br>または「画像を選択...」</div>`;
    $("#i-preview-hint").textContent = "右クリックで画像を外す";
    const img = $("img", el);
    if (img) img.onload = iDrawBoxes;
  } else {
    el.className = "i-preview multi";
    el.innerHTML = SLOT_ORDER.map((k, i) => {
      const s = I.slots[k];
      const same = s && SLOT_ORDER.slice(0, i).find((o) => I.slots[o] && I.slots[o].path === s.path);
      const need = k === "person" ? "必須" : k === "props" ? "空なら ③ か 1 枚目" : "空なら 1 枚目";
      return `<div class="i-slot" data-slot="${k}" title="${esc(I.c.ref_what_ja[k])}"><div class="i-slothead">${esc(iSlotLabel(k))}</div>
        ${s ? `<img src="/api/i2t/file?thumb=1${mos ? "&mosaic=12" : ""}&path=${encodeURIComponent(s.path)}" alt="">${same ? `<div class="i-same">= ${esc(iSlotLabel(same).slice(0, 1))} と同じ</div>` : ""}`
          : `<div class="i-slotempty">${need}<br>ドロップ / クリック</div>`}</div>`;
    }).join("");
    $("#i-preview-hint").textContent = "クリックで選択、右クリックで外す / ほかの枠にも使う";
  }
  $$("[data-slot]", el).forEach((c) => {
    c.onclick = () => { if (!multi() && I.slots.person) return; I.pickSlot = c.dataset.slot; $("#i-file").click(); };
    c.oncontextmenu = (e) => { e.preventDefault(); iSlotMenu(c.dataset.slot, e); };
    c.onmouseenter = () => { I.hover = c.dataset.slot; };
    c.onmouseleave = () => { I.hover = null; };
  });
  el.ondragover = (e) => { e.preventDefault(); el.classList.add("drag"); };
  el.ondragleave = () => el.classList.remove("drag");
  el.ondrop = (e) => {
    e.preventDefault(); e.stopPropagation(); el.classList.remove("drag");
    const t = e.target.closest("[data-slot]");
    iAddFiles([...e.dataTransfer.files], multi() && e.dataTransfer.files.length === 1 && t ? t.dataset.slot : null);
  };
  iUpdatePath();
}

function iUpdatePath() {
  const el = $("#i-path");
  if (!el) return;
  if (!multi()) { const s = I.slots.person; el.textContent = s ? `${s.path} (${s.size[0]}×${s.size[1]})` : ""; return; }
  el.textContent = SLOT_ORDER.map((k) => { const s = I.slots[k]; return `${iSlotLabel(k).slice(0, 1)} ${s ? `${s.name} (${s.size[0]}×${s.size[1]})` : "(1 枚目)"}`; }).join("　");
}

function iDrawBoxes() {
  const svg = $("#i-boxes"), img = $("#i-preview img"), r = I.result, s = I.slots.person;
  if (!svg || !img) return;
  svg.innerHTML = "";
  if (!r || !s || r.image !== s.path) return;
  const k = img.clientWidth / s.size[0];
  svg.setAttribute("width", img.clientWidth); svg.setAttribute("height", img.clientHeight);
  const rect = (b, color, label, dash) => `<rect x="${b[0] * k}" y="${b[1] * k}" width="${(b[2] - b[0]) * k}" height="${(b[3] - b[1]) * k}" fill="none" stroke="${color}" stroke-width="${dash ? 1 : 2}" ${dash ? 'stroke-dasharray="3 3"' : ""}/>`
    + (label ? `<text x="${b[0] * k + 3}" y="${b[1] * k + 13}" fill="${color}" font-weight="bold" font-size="12">${label}</text>` : "");
  const colors = ["#ff4d4d", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#14b8a6"];
  let out = (r.background_people || []).map((b) => rect(b, "#9ca3af", "", true)).join("");
  if (r.main_person && r.main_person.box) out += rect(r.main_person.box, "#ff4d4d", "");
  (r.persons || []).forEach((p, i) => { if (p.box) out += rect(p.box, colors[i % colors.length], String(i + 1)); });
  svg.innerHTML = out;
}

function iSlotMenu(slot, e) {
  if (I.job) return;
  const m = $("#i-menu");
  if (!multi()) {
    if (!I.slots.person) return;
    m.innerHTML = `<div class="ctxmenu" style="left:${e.clientX}px;top:${e.clientY}px"><button data-rm>画像を外す</button></div>`;
  } else {
    const s = I.slots[slot];
    if (!s) return;
    m.innerHTML = `<div class="ctxmenu" style="left:${e.clientX}px;top:${e.clientY}px"><button data-rm>${esc(iSlotLabel(slot))} を外す</button><hr>
      ${SLOT_ORDER.filter((k) => k !== slot).map((k) => { const o = I.slots[k]; const same = o && o.path === s.path;
        return `<button data-use="${k}" ${same ? "disabled" : ""}>この画像を ${esc(iSlotLabel(k))} にも使う${same ? " (同じ画像)" : o ? " (差し替え)" : ""}</button>`; }).join("")}</div>`;
  }
  const close = () => { m.innerHTML = ""; document.removeEventListener("click", close); };
  setTimeout(() => document.addEventListener("click", close), 0);
  $("[data-rm]", m).onclick = () => { delete I.slots[slot]; close(); iRenderPreview(); iUpdateButtons(); };
  $$("[data-use]", m).forEach((b) => b.onclick = () => { I.slots[b.dataset.use] = { ...I.slots[slot] }; close(); iRenderPreview(); });
}

// paste (Ctrl+V): bitmaps and copied files; text fields keep normal text paste
document.addEventListener("paste", (e) => {
  if (S.route[0] !== "i2t" || !$("#i-root")) return;
  if ($("#i-modal").innerHTML) return;
  const files = [...(e.clipboardData?.files || [])].filter((f) => f.type.startsWith("image/"));
  if (!files.length) return;
  e.preventDefault();
  const named = files.map((f) => f.name && f.name !== "image.png" ? f : new File([f], `paste-${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 17)}.png`, { type: f.type }));
  const slot = multi() ? (I.hover || iFirstEmpty() || "person") : "person";
  iAddFiles(named.slice(0, 1), slot).then(() => iStatus(`貼り付けました (${named[0].name} として保存)`));
});

// ------------------------------------------------------------------ 構図と強さ / 出力
function iDrawControls() {
  const el = $("#i-controls"), c = I.c;
  const looks = I.cfg.camera_looks || [];
  el.innerHTML = `<div class="small"><b>構図と強さ</b> <span class="muted">(LLM・GPU 不要。指定しなければ画像のまま)</span></div>
    <div class="i-grid">
      <span>ポーズ</span><div class="row"><button class="small" id="i-bones">ボーン編集...</button><button class="small" id="i-bones-clear">解除</button>
        <span class="small" id="i-pose-label" title="">${esc(iPoseText())}</span></div>
      <span>撮影</span><div><select id="i-look"><option value="">${esc(c.look_none)}</option>${looks.map((l) => `<option ${I.look === l.name ? "selected" : ""}>${esc(l.name)}</option>`).join("")}
          <option value="${LOOK_PARTS}" ${I.look === LOOK_PARTS ? "selected" : ""}>${esc(c.look_parts_menu || "組み合わせ...")}</option></select>
        <div id="i-lookparts" ${I.look === LOOK_PARTS ? "" : "hidden"}></div></div>
      <span>カメラ</span><div class="row">${Object.entries(c.camera_options).map(([dim, opts]) => `<label class="small">${esc(c.camera_labels_ja[dim])}
        <select data-cam="${dim}">${opts.map(([v, l]) => `<option value="${v}" ${(I.camera[dim] || "") === v ? "selected" : ""}>${esc(l)}</option>`).join("")}</select></label>`).join("")}</div>
      <span>強さ</span><div class="row">${c.strength_groups.map(([g, l]) => `<label class="small">${esc(l)}
        <select data-str="${g}">${[["weak", "弱い"], ["", "普通"], ["strong", "強い"]].map(([v, t]) => `<option value="${v}" ${(I.strength[g] || "") === v ? "selected" : ""}>${t}</option>`).join("")}</select></label>`).join("")}</div>
    </div>`;
  $("#i-bones").onclick = iPoseEditor;
  $("#i-bones-clear").onclick = () => { I.poseControl = null; iDrawControls(); };
  $("#i-look").onchange = (e) => { I.look = e.target.value; $("#i-lookparts").hidden = I.look !== LOOK_PARTS; iDrawLookParts(); };
  iDrawLookParts();
  $$("[data-cam]", el).forEach((s) => s.onchange = () => { I.camera[s.dataset.cam] = s.value; });
  $$("[data-str]", el).forEach((s) => s.onchange = () => { I.strength[s.dataset.str] = s.value; });
}

/* 撮影の組み合わせ: 写り・カメラ・レンズ・質感・光・フィルム・粒子・被写界深度を項目ごとに選び、選んだ英語の語をこの順に
   つないでプロンプトに入れる (i2t.py の LOOK_PARTS / compose_look_parts) */
const LOOK_PARTS = "__parts__";

function iDrawLookParts() {
  const el = $("#i-lookparts"), c = I.c;
  if (!el || el.hidden) return;
  const parts = c.look_parts || {};
  const text = () => Object.keys(parts).map((k) => I.lookParts[k]).filter(Boolean).join(", ");
  el.innerHTML = `<div class="row" style="margin-top:4px">${Object.entries(parts).map(([k, opts]) => `<label class="small">${esc(c.look_part_labels_ja[k])}
      <select data-lp="${k}" style="max-width:15em"><option value="">${esc(c.look_none)}</option>${opts.map(([v, l]) =>
        `<option value="${esc(v)}" title="${esc(v)}" ${I.lookParts[k] === v ? "selected" : ""}>${esc(l)}</option>`).join("")}</select></label>`).join("")}</div>
    <div class="small muted" id="i-lookparts-text" style="margin-top:2px"></div>`;
  const show = () => { $("#i-lookparts-text").textContent = text() ? `入る語: ${text()}` : "項目ごとに選ぶと、選んだ語をこの順にプロンプトに入れます。"; };
  $$("[data-lp]", el).forEach((s) => s.onchange = () => { if (s.value) I.lookParts[s.dataset.lp] = s.value; else delete I.lookParts[s.dataset.lp]; show(); });
  show();
}

function iPoseText() {
  if (!I.poseControl) return "指定なし";
  const t = I.poseControl.tags.map((x) => x.replace(/_/g, " ")).join(", ") || "(タグにできる姿勢なし)";
  return t.length <= 26 ? t : t.slice(0, 25) + "…";
}

function iControls() {
  const out = {};
  if (I.poseControl) out.pose = { tags: [...I.poseControl.tags] };
  const cam = Object.fromEntries(Object.entries(I.camera).filter(([, v]) => v));
  if (Object.keys(cam).length) out.camera = cam;
  const look = (I.cfg.camera_looks || []).find((l) => l.name === I.look);
  if (look) out.look = { ...look };
  else if (I.look === LOOK_PARTS && Object.keys(I.lookParts).length) out.look_parts = { ...I.lookParts };
  const str = Object.fromEntries(Object.entries(I.strength).filter(([, v]) => v));
  if (Object.keys(str).length) out.strength = str;
  return out;
}

function iDrawOutput() {
  const el = $("#i-output"), c = I.c, f = I.cfg;
  const chk = (k, label) => `<label class="small"><input type="checkbox" data-ck="${k}" ${f[k] ? "checked" : ""}>${label}</label>`;
  el.innerHTML = `<div class="small"><b>出力</b></div>
    <div class="i-grid">
      <span>エンジン</span><select id="i-engine">${Object.entries(c.engines).map(([k, l]) => `<option value="${k}" ${f.engine === k ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>
      <span>出力先</span><select id="i-target">${Object.entries(c.targets).map(([k, t]) => `<option value="${k}" ${f.target === k ? "selected" : ""}>${esc(t.label)}</option>`).join("")}</select>
      <span></span><div class="row">${chk("mosaic", "モザイク指定を追加")}<label class="small">強さ <select id="i-mosaic">${Object.entries(c.mosaic_levels).map(([k, v]) => `<option value="${k}" ${f.mosaic_strength === k ? "selected" : ""}>${esc(v.label)}</option>`).join("")}</select></label></div>
      <span></span><div class="row">${chk("avoid_duplicates", "分身を防ぐ")}
        <label class="small">性別をそろえる <select id="i-gender">${[["none", "そろえない"], ["female", "全員女性"], ["male", "全員男性"]].map(([v, l]) => `<option value="${v}" ${f.gender === v ? "selected" : ""}>${l}</option>`).join("")}</select></label></div>
      <span></span><div class="row">${chk("remove_monochrome", "白黒をカラーで出す")}${chk("person_details", "人物の特徴を詳しく (顔立ち・髪型・体格。VLM)")}</div>
      <span></span><div class="row">${chk("turnaround", "三面図にする")}</div>
      <span></span><div class="row"><span class="small">LoRA 用に入れない:</span>${chk("no_identity", "人物の特徴")}${chk("no_style", "作風の特徴")}</div>
    </div>`;
  $("#i-engine").onchange = (e) => { iSet("engine", e.target.value); if (e.target.value === "hybrid") iProbeOllama(true); };
  $("#i-target").onchange = (e) => iSet("target", e.target.value);
  $("#i-mosaic").onchange = (e) => iSet("mosaic_strength", e.target.value);
  $("#i-gender").onchange = (e) => iSet("gender", e.target.value);
  $$("[data-ck]", el).forEach((b) => b.onchange = async () => {
    iSet(b.dataset.ck, b.checked);
    if (b.dataset.ck === "person_details" && b.checked) {
      const missing = await iApi("missing_face_models").catch(() => []);
      if (missing.length) iConfirm("顔の切り出しに使うモデルを取得します（初回のみ。取得しなくても動きます）:",
        missing.map((m) => "・" + m).join("\n") + `\n\n保存先: ${I.c.models_dir}`, "取得する", () => iStartJob("face_models", {}, "顔用のモデルを取得中..."));
    }
  });
}

// ------------------------------------------------------------------ jobs (analysis and the other long work)
async function iStartJob(op, body, message) {
  try {
    const j = await api("/api/i2t/job", { body: { op, ...body } });
    I.job = j; I.paused = false;
    iStatus(message || j.message);
    iUpdateButtons();
    iProgress(op === "analyze" || op === "tagger_download" ? 0 : null);
    return j;
  } catch (e) { fail(e); iStatus("エラー: " + e.message); return null; }
}

async function iAnalyze() {
  if (I.job || !I.slots.person) return;
  if (!I.c.tagger_models[I.cfg.tagger_model]?.ready) { iTaggerDialog("run"); return; }
  if (I.cfg.target === "mossimage" && !I.c.moss_catalog_ready && !I.mossAsked) { iMossCatalogDialog(); return; }
  I.result = null; I.chips = { pos: null, neg: null }; iShowResult(null);
  const refs = multi() ? Object.fromEntries(SLOT_ORDER.slice(1).filter((k) => I.slots[k]).map((k) => [k, I.slots[k].path])) : {};
  I.analyzedCfg = { ...I.cfg };
  I.analyzedMain = { ...I.slots.person, refs: Object.keys(refs).length };   // the base image of 「メインの画像から生成」
  await iStartJob("analyze", { path: I.slots.person.path, refs, controls: iControls(), cfg: I.cfg, mode: I.cfg.mode }, "準備中...");
}

function iCancel() { if (I.job && I.job.cancellable !== false) api("/api/i2t/cancel", { body: {} }).catch(fail); }

function iUpdateButtons() {
  if (!$("#i-run")) return;
  const busy = !!I.job;
  $("#i-run").disabled = busy || !I.slots.person;
  $("#i-stop").disabled = !busy || I.job.cancellable === false;
  const pausable = busy && (I.job.op === "analyze" || I.job.op === "translate");
  $("#i-pause").disabled = !pausable;
  $("#i-pause").textContent = I.paused ? "▶ 再開" : "⏸ 一時停止";
  $("#i-geninfo").disabled = !SLOT_ORDER.some((k) => I.slots[k] && I.slots[k].geninfo);
  $("#i-to-gen").disabled = !I.chips.pos;
  $("#i-to-i2i").disabled = !I.chips.pos || !(I.analyzedMain || I.slots.person);
}

function iProgress(frac) {
  const bar = $("#i-bar"), pct = $("#i-pct");
  if (!bar) return;
  if (frac == null) { bar.className = "indet"; bar.style.width = ""; pct.textContent = ""; return; }
  bar.className = ""; bar.style.width = `${Math.round(frac * 100)}%`; pct.textContent = `${Math.round(frac * 100)}%`;
}

// VLM steps report an estimate (seconds, from, to): move the bar by time, as the i2t window does
setInterval(() => {
  if (!I.est || !I.job || I.paused) return;
  const [secs, a, b, t0] = I.est;
  const f = Math.min(1, (Date.now() - t0) / 1000 / Math.max(1, secs));
  iProgress(a + (b - a) * f * 0.97);
}, 250);

function iOnJob(m) { if (m.message) iStatus(m.message); }

function onI2tEvent(m) {
  if (!$("#i-root") && !$("#s-i2t")) return;
  if ($("#s-stop")) $("#s-stop").hidden = m.event === "done";
  if (m.event === "progress") {
    if (I.job && m.rid === I.job.rid) { iStatus(m.message); if (m.fraction >= 0) iProgress(m.fraction); }
  } else if (m.event === "fraction") {
    iProgress(m.value);
    I.est = m.est ? [m.est[0], m.est[1], m.est[2], Date.now()] : null;
  } else if (m.event === "stop_state") {
    if (I.job) I.job.cancellable = m.cancellable;
    iUpdateButtons();
  } else if (m.event === "paused") {   // also when pressed on the job page
    if (I.job && m.rid === I.job.rid) { I.paused = m.paused; iUpdateButtons(); }
  } else if (m.event === "done") {
    const op = m.op;
    I.job = null; I.est = null; I.paused = false;
    iUpdateButtons();
    if (!m.ok) {
      iProgress(0);
      iStatus(m.cancelled ? m.error : "エラー: " + m.error);
      if (!m.cancelled) toast(m.error, "err");
      if (op === "tagger_download") toast("「▶ 分析する」を押すと、もう一度ダウンロードを確かめます");
      if (op === "moss_catalog") { I.afterDownload = null; if (!m.cancelled) I.mossAsked = false; }   // failed: asked again next time
      return;
    }
    iProgress(1);
    if (op === "analyze") { I.shownRid = m.rid; iShowResult(m.result, m.history); }
    else if (op === "tagger_download") {
      iStatus(`ダウンロード完了: ${m.value.name} を使用します`);
      I.c.tagger_models[m.value.name].ready = true; iSet("tagger_model", m.value.name);
      if ($("#s-i2t")) drawI2tSettings();
      if (I.afterDownload) { const f = I.afterDownload; I.afterDownload = null; setTimeout(f, 50); }
    } else if (op === "moss_catalog") {
      I.c.moss_catalog_ready = !!m.value.ready;
      iStatus("MOSS-Image のキャラ一覧を取得しました");
      if (I.afterDownload) { const f = I.afterDownload; I.afterDownload = null; setTimeout(f, 50); }
    } else if (op === "face_models") iStatus("顔用のモデルを取得しました");
    else if (op === "pull" && I.pullTranslate === m.value.name) {   // 訳のモデルとして取得: VLM の欄は変えない
      I.pullTranslate = null;
      iStatus(`${m.value.name} を取得しました（訳のモデル）`);
      iSet("translate_model", m.value.name);
      iProbeOllama();
      if ($("#s-i2t")) drawI2tSettings();
    }
    else if (op === "pull") { iStatus(`${m.value.name} を取得しました`); iAfterPull(m.value.name); }
    else if (op === "translate") { iStatus(`日本語に訳しました (${Object.keys(m.value.translated).length} 語)`); iRefreshLabels(); }
    else if (op === "pose_estimate") { iStatus("骨格を推定しました"); if (I.onPose) I.onPose(m.value); }
    else if (op === "ollama_setup") { iStatus("Ollama の準備ができました"); iAfterSetup(); }
  }
}

// ------------------------------------------------------------------ results
async function iShowResult(r, histPath, keepEdits) {
  if (keepEdits && r && I.chips.pos) {   // back on the page: show it as it was left (deleted chips stay deleted)
    I.tab = I.tab || "extras";
    iDrawBoxes(); iDrawChips(); iDrawTab(); iUpdateButtons();
    return;
  }
  I.result = r; I.histPath = histPath || null;
  I.undo = { pos: [], neg: [] }; I.sel = { pos: new Set(), neg: new Set() };
  if (!r) {
    I.chips = { pos: null, neg: null }; I.texts = null;
    $("#i-pos").innerHTML = ""; $("#i-neg").innerHTML = ""; $("#i-tab").innerHTML = ""; $("#i-lang").hidden = true;
    iUpdateButtons(); return;
  }
  const [pos, neg, texts] = await Promise.all([iApi("split_prompt_chips", r.positive_prompt), iApi("split_prompt_chips", r.negative_prompt), iApi("texts", r)]);
  I.chips = { pos, neg }; I.texts = texts;
  const tgt = I.c.targets[r.target] || {}, neg_ = tgt.negative;
  $("#i-neg-title").textContent = tgt.neg_label ? tgt.neg_label : neg_ === false ? "ネガティブプロンプト (この出力先は非対応のため空)" : neg_ === "optional" ? "ネガティブプロンプト (CFG 1 では無視されます)" : "ネガティブプロンプト";
  I.tab = (r.persons || []).length ? "persons" : "extras";
  iDrawBoxes(); iDrawChips(); iDrawTab(); iUpdateButtons();
  let st = `完了 (${r.elapsed_sec.toFixed(1)} 秒)  medium=${r.detected_medium}  rating=${r.rating}`;
  if ((r.persons || []).length) st += `  人物=${r.persons.length}`;
  if (r.vlm_model) st += `  vlm=${r.vlm_model}`;
  if (r.nsfw) st += "  NSFW" + (r.mosaic_prompt ? " (モザイク指定を追加)" : "");
  if ((r.warnings || []).length) st += "\n注意: " + r.warnings.join(" / ");
  iStatus(st);
  if (I.ja) iSetLang(true);
}

function iJoin(box) { const l = I.chips[box]; return l ? l.map((line) => line.join(", ")).join("\n") : ""; }

function iDrawChips() {
  const useChips = I.cfg.result_chips;
  $("#i-lang").hidden = !useChips || !I.chips.pos;
  $("#i-lang").textContent = I.ja ? "English" : "日本語";
  for (const box of ["pos", "neg"]) {
    const el = $(`#i-${box}`), lines = I.chips[box];
    if (!lines) { el.innerHTML = ""; continue; }
    if (!useChips) {   // plain text box: editable, used for copy / save
      el.innerHTML = `<textarea class="i-plain" data-box="${box}">${esc(iJoin(box))}</textarea>`;
      $("textarea", el).oninput = (e) => { I.chips[box] = e.target.value.split("\n").map((l) => l.split(/,\s*/).filter((x, i, a) => x || a.length === 1)); };
      continue;
    }
    let n = 0;
    const labels = I.ja ? I.labels[box] : null;
    el.innerHTML = lines.map((line) => `<div class="i-chipline">${line.map((chip) => { const i = n++;
      return `<span class="i-chip ${I.sel[box].has(i) ? "sel" : ""}" data-i="${i}" title="${esc(chip)}">${esc(labels ? labels[i] : chip)}</span>`; }).join("")}</div>`).join("");
  }
}

function iFlat(box) { return I.chips[box].flat(); }

function iChipClick(box, e) {
  const t = e.target.closest(".i-chip");
  I.focusBox = box;
  $(`#i-${box}`).focus();
  if (!t) { I.sel[box].clear(); iDrawChips(); return; }
  const i = +t.dataset.i, s = I.sel[box];
  if (e.shiftKey && I.anchor[box] != null) { const [a, b] = [Math.min(I.anchor[box], i), Math.max(I.anchor[box], i)]; for (let k = a; k <= b; k++) s.add(k); }
  else if (e.ctrlKey || e.metaKey) { s.has(i) ? s.delete(i) : s.add(i); I.anchor[box] = i; }
  else { s.clear(); s.add(i); I.anchor[box] = i; }
  iDrawChips();
}

function iDeleteSelected(box) {
  const s = I.sel[box];
  if (!s.size || !I.chips[box]) return;
  I.undo[box].push(JSON.stringify({ lines: I.chips[box], labels: I.labels[box] }));
  let n = 0;
  const keep = [];
  const lines = I.chips[box].map((line) => line.filter(() => { const k = !s.has(n); keep.push(k); n++; return k; }));
  I.chips[box] = lines.filter((l, i) => l.length || !I.chips[box][i].length);   // emptied lines go, blank lines stay
  if (I.labels[box]) I.labels[box] = I.labels[box].filter((_, i) => keep[i]);
  s.clear(); I.anchor[box] = null;
  iDrawChips(); iChipsChanged();
}

function iUndo(box) {
  const u = I.undo[box].pop();
  if (!u) return;
  const d = JSON.parse(u);
  I.chips[box] = d.lines; I.labels[box] = d.labels; I.sel[box].clear();
  iDrawChips(); iChipsChanged();
}

function iChipsChanged() {   // deleted chunks are also written to the history entry (as in i2t)
  if (I.histPath) iApi("history_update_prompts", I.histPath, iJoin("pos"), iJoin("neg")).catch(() => {});
}

document.addEventListener("keydown", (e) => {
  if (S.route[0] === "settings" && e.key === "Escape" && $("#i-modal") && $("#i-modal").innerHTML) { iCloseModal(); return; }
  if (S.route[0] !== "i2t" || !$("#i-root")) return;
  if (e.key === "F1") { e.preventDefault(); iHelpDialog(); return; }
  const modal = $("#i-modal").innerHTML;
  if (e.key === "Escape") { if (modal) { iCloseModal(); return; } iCancel(); return; }
  if (modal) return;
  const box = I.focusBox, inBox = box && document.activeElement === $(`#i-${box}`);
  if (inBox && (e.key === "Delete" || e.key === "Backspace")) { e.preventDefault(); iDeleteSelected(box); }
  if (inBox && e.key.toLowerCase() === "z" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); iUndo(box); }
});

async function iSetLang(ja) {
  I.ja = ja;
  if (!ja || !I.chips.pos) { iDrawChips(); return; }
  await iRefreshLabels();
  const chars = (I.result.tags || []).filter((t) => t.kind === "character").map((t) => String(t.tag || ""));
  const missing = await iApi("missing_ja", [...iFlat("pos"), ...iFlat("neg")], chars).catch(() => []);
  if (!missing.length) return;
  if (!I.ollama.ok) { iStatus(`辞書に無い ${missing.length} 語は英語のままです (訳すには Ollama と訳のモデルが必要です)`); return; }
  if (I.job) { iStatus("ほかの処理の最中は訳せません"); return; }
  iStartJob("translate", { words: missing });
}

async function iRefreshLabels() {
  if (!I.chips.pos) return;
  const [p, n] = await Promise.all([iApi("chips_ja", iFlat("pos")), iApi("chips_ja", iFlat("neg"))]);
  I.labels = { pos: p, neg: n };
  iDrawChips();
}

function iDrawTab() {
  const el = $("#i-tab"), r = I.result, t = I.texts;
  $$("#i-tabs a").forEach((a) => a.classList.toggle("on", a.dataset.tab === I.tab));
  if (!r || !t) { el.innerHTML = ""; return; }
  if (I.tab === "extras") {
    el.innerHTML = I.c.extra_button_rows.map((row) => `<div class="row">${row.map((k) => `<button class="small" data-extra="${k}">${esc(I.c.extra_short_ja[k])}</button>`).join("")}</div>`).join("")
      + `<textarea readonly class="i-tabtext">${esc(t.extras_text)}</textarea>`;
    $$("[data-extra]", el).forEach((b) => b.onclick = () => {
      const p = (r.extras[b.dataset.extra] || {}).prompt;
      if (p) iCopy(p); else iStatus(`${I.c.extra_kinds[b.dataset.extra]}: まだありません`);
    });
  } else if (I.tab === "persons") {
    const txt = t.person_lines || (r.references && Object.keys(r.references).length ? "(複数枚モードでは人物ごとの振り分けは行いません)" : "(複数人物は検出されませんでした)");
    el.innerHTML = `<div class="row"><div class="spacer"></div><button class="small" id="i-tabcopy">コピー</button></div><textarea readonly class="i-tabtext">${esc(txt)}</textarea>`;
    $("#i-tabcopy").onclick = () => iCopy(txt);
  } else if (I.tab === "tags") {
    el.innerHTML = `<div class="row"><div class="spacer"></div><button class="small" id="i-tabcopy">コピー</button></div><textarea readonly class="i-tabtext">${esc(t.tag_lines)}</textarea>`;
    $("#i-tabcopy").onclick = () => iCopy(t.tag_lines);
  } else {
    let txt = I.tab === "analysis" ? r.analysis : r.notes;
    if (I.tab === "analysis" && r.mosaic_prompt) {
      const lv = (I.analyzedCfg || I.cfg).mosaic_strength;
      txt += `\n\n[NSFW 判定] モザイク指定 (強さ: ${(I.c.mosaic_levels[lv] || I.c.mosaic_levels.strong).label}) をポジティブに追加し、ネガティブに uncensored を追加しました`;
    }
    el.innerHTML = `<textarea class="i-tabtext" data-edit="${I.tab}">${esc(txt)}</textarea>`;
    $("textarea", el).oninput = (e) => { r[I.tab === "analysis" ? "analysis" : "notes"] = e.target.value; };   // edited text is saved as shown
  }
}

function iCopy(text) {
  navigator.clipboard.writeText(text).then(() => toast("コピーしました")).catch(() => toast("コピーできませんでした", "err"));
}

async function iCurrentResult() {   // what is shown (deleted chips / edited memo) is what is saved
  return { ...I.result, positive_prompt: iJoin("pos"), negative_prompt: iJoin("neg") };
}

async function iSave(kind) {
  if (!I.result) return;
  const r = await iCurrentResult();
  const stem = (I.slots.person ? I.slots.person.name : "image").replace(/\.[^.]+$/, "").replace(/^[0-9a-f]{12}_/, "");
  const text = kind === "json" ? JSON.stringify(r, null, 2) + "\n" : (await iApi("texts", r)).to_text;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: kind === "json" ? "application/json" : "text/plain" }));
  a.download = `${stem}_prompt.${kind}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function iToImageGen() {
  try {
    const g = await api("/api/imagegen");
    await api("/api/imagegen/form", { method: "PUT", body: { ...g.form, positive: iJoin("pos"), negative: iJoin("neg") } });
    S.ig = null;
    location.hash = "#/gen";
    toast("ポジティブとネガティブを画像生成の入力欄に入れました");
  } catch (e) { fail(e); }
}

/* i2i from the analysis: the main image is the source image (the base), the prompt carries what the other
   slots' images gave (pose, outfit, background ...), laid over it.  With such elements the strength starts
   higher, so they can show (a low strength keeps the main image's pose and clothes). */
async function iToImageGenI2I() {
  const main = I.analyzedMain || I.slots.person;
  if (!main) return;
  try {
    const src = await api("/api/imagegen/init_from", { body: { path: main.path } });
    const g = await api("/api/imagegen");
    const [w, h] = src.size, n = g.native || 1024, k = Math.sqrt((n * n) / (w * h));
    const fit = (v) => Math.max(512, Math.min(2048, Math.round((v * k) / 16) * 16));
    const layered = (main.refs || 0) > 0;
    const denoise = layered ? 0.7 : (g.form.denoise ?? 0.6);
    await api("/api/imagegen/form", { method: "PUT", body: { ...g.form, positive: iJoin("pos"), negative: iJoin("neg"),
      init_image: src.name, denoise, width: fit(w), height: fit(h) } });
    S.ig = null;
    location.hash = "#/gen";
    toast(layered ? `メインの画像を元画像にしました。ほかの画像から読んだ要素はプロンプトで重ねます（変化の強さ ${denoise}。ポーズなどが変わらないときは強くしてください）`
      : `メインの画像を元画像にして、プロンプトを入れました（変化の強さ ${denoise}）`);
  } catch (e) { fail(e); }
}

// ------------------------------------------------------------------ Ollama
function iShowOllama() {   // the 「Ollama: …」 line, on the i2t page and on the settings page
  const el = $("#i-ollama") || $("#s-i2t-ollama"), o = I.ollama;
  if (!el) return;
  if (o.checking) { el.textContent = "Ollama: 確認中..."; return; }
  if (!o.ok) {   // not running or not installed yet: the way to install / start it is right here, not only in the settings
    el.innerHTML = `Ollama: 未接続（「タガーのみ」なら Ollama なしで使えます） <button class="small" data-osetup>Ollama を導入・起動...</button>`;
    el.title = o.error || "";
    $("[data-osetup]", el).onclick = iOllamaSetupDialog;
    return;
  }
  el.title = "";
  const has = o.models.includes(I.cfg.ollama_model);
  el.textContent = `Ollama: 接続OK (画像対応 ${o.vision.length} / 全 ${o.models.length})` + (I.cfg.engine === "hybrid" && !has ? `　選んだモデル ${I.cfg.ollama_model} は未取得です（設定の「追加 (pull)」で取得できます）` : "");
}

async function iProbeOllama(askSetup) {
  if (!$("#i-ollama") && !$("#s-i2t-ollama")) return;
  I.ollama = { ...I.ollama, checking: true };
  iShowOllama();
  try {
    const v = await api("/api/i2t/call", { body: { fn: "probe_ollama", args: [I.cfg.ollama_host, I.cfg.ollama_model], timeout: 180 } }).then((r) => r.value);
    I.ollama = { ok: true, models: v.models, vision: v.vision, error: null };
    iShowOllama();
    if ($("#s-vlm-list")) {
      $("#s-vlm-list").innerHTML = [...new Set([...v.vision, ...(I.cfg.ollama_models || []), I.cfg.ollama_model].filter(Boolean))].map((m) => `<option value="${esc(m)}">`).join("");
      $("#s-tr-list").innerHTML = [...new Set([...v.models, I.cfg.translate_model, ...(I.c.translate_choices || [])].filter(Boolean))].map((m) => `<option value="${esc(m)}">`).join("");
    }
  } catch (e) {
    I.ollama = { ok: false, models: [], vision: null, error: e.message };
    iShowOllama();
    if ($("#i-root") && (askSetup || I.cfg.engine === "hybrid") && !I.cfg.ollama_setup_asked) {
      iSet("ollama_setup_asked", true);
      iConfirm("Ollama に接続できません", "VLM（タガー + VLM）を使うには Ollama が必要です。導入サポートを開きますか？\n（タガーのみなら Ollama は要りません）", "導入サポートを開く", iOllamaSetupDialog);
    }
  }
}

function iAfterPull(name) {
  const list = I.cfg.ollama_models || [];
  if (!list.includes(name)) iSet("ollama_models", [...list, name]);
  iSet("ollama_model", name);
  iProbeOllama();
  if ($("#s-i2t")) drawI2tSettings();
}

async function iPull(name) {
  name = (name || "").trim();
  if (!name) return;
  try {
    const hf = await iApi("parse_hf_model", name);
    if (hf) {
      iStatus("Hugging Face のリポジトリを確かめています...");
      const info = await api("/api/i2t/call", { body: { fn: "check_hf_model", args: [name], timeout: 60 } }).then((r) => r.value);
      const size = info.size ? `${(info.size / 1e9).toFixed(1)} GB` : "サイズ不明";
      iConfirm("Hugging Face から VLM を追加します", `リポジトリ: ${info.repo}\n取得するファイル: ${info.file || "(Ollama が選ぶ既定の量子化)"} (${size})\nOllama での名前: ${info.name}${info.gated ? "\n\n注意: 利用条件への同意（ログイン）が必要なリポジトリです。取得できない場合は Hugging Face のページで同意してください。" : ""}`,
        "取得する", () => iStartJob("pull", { name: info.name, host: I.cfg.ollama_host }, `${info.name} を取得中...`));
      return;
    }
  } catch (e) { fail(e); return; }
  iConfirm("モデルを取得します", `${name} を Ollama に取得します（ollama pull）。中止しても、次回は続きから取得します。`, "取得する",
    () => iStartJob("pull", { name, host: I.cfg.ollama_host }, `${name} を取得中...`));
}

// ------------------------------------------------------------------ modals
function iModal(html, cls = "") {
  $("#i-modal").innerHTML = `<div class="i-veil"><div class="panel i-dialog ${cls}">${html}</div></div>`;
  return $("#i-modal .i-dialog");
}
function iCloseModal() { $("#i-modal").innerHTML = ""; if (I.onClose) { const f = I.onClose; I.onClose = null; f(); } }

function iConfirm(title, text, ok, fn) {
  const d = iModal(`<h3>${esc(title)}</h3><div class="small" style="white-space:pre-wrap">${esc(text)}</div>
    <div class="row" style="margin-top:12px"><button class="primary" data-ok>${esc(ok)}</button><button data-cancel>やめる</button></div>`);
  $("[data-ok]", d).onclick = () => { iCloseModal(); fn(); };
  $("[data-cancel]", d).onclick = iCloseModal;
}

/* The first analysis for MOSS-Image: ask before fetching the character list of hide1's catalog (i2t writes the names
   as "name from series", the catalog's wording).  Declining analyses without it and is not asked again until reload. */
function iMossCatalogDialog() {
  I.mossAsked = true;
  const d = iModal(`<h3>MOSS-Image のキャラ一覧の取得</h3>
    <div class="small">MOSS-Image では、キャラ名を一覧の書き方「名前 from 作品名」（例: dawn (pokemon) from pokemon）で書きます。そのために MOSS-Image のキャラ一覧（約 ${I.c.moss_catalog_mb}MB、hide1 の MOSS-Image Character &amp; Style Catalog）を取得します（初回のみ）。</div>
    <div class="small muted">取得元: ${esc(I.c.moss_catalog_page)}<br>保存先: ${esc(I.c.moss_catalog_dir)}</div>
    <div class="small muted">取得しない場合は、キャラ名をそのまま書いて分析します。</div>
    <div class="row" style="margin-top:12px"><button class="primary" data-ok>取得して分析する</button><button data-cancel>取得せずに分析する</button></div>`);
  $("[data-ok]", d).onclick = () => { iCloseModal(); I.afterDownload = iAnalyze; iStartJob("moss_catalog", {}, "MOSS-Image のキャラ一覧を取得中..."); };
  $("[data-cancel]", d).onclick = () => { iCloseModal(); iAnalyze(); };
}

function iTaggerDialog(why) {
  const names = Object.entries(I.c.tagger_models);
  const cur = I.cfg.tagger_model;
  const d = iModal(`<h3>タガーモデルの取得</h3>
    <div class="small">${why === "first" ? "はじめまして。画像を解析するタガーモデルを取得します（初回のみ）。" : `設定されているタガーモデル (${esc(cur)}) がまだ取得されていません。`}</div>
    <div class="small muted">保存先: ${esc(I.c.models_dir)}</div>
    <div style="margin:8px 0">${names.map(([k, v]) => `<label class="small" style="display:block"><input type="radio" name="i-tg" value="${k}" ${k === cur ? "checked" : ""}> ${esc(k)}　${esc(v.label)}　[${v.ready ? "取得済み" : "未取得"}]</label>`).join("")}</div>
    <div class="small muted">迷ったら eva02-large（最高精度）を選んでください。</div>
    <div class="row" style="margin-top:12px"><button class="primary" data-ok>ダウンロードして使う</button><button data-cancel>あとで</button></div>`);
  const go = () => {
    const name = $("input[name=i-tg]:checked", d).value;
    iCloseModal();
    if (I.c.tagger_models[name].ready) { iSet("tagger_model", name); iStatus(`${name} を使用します`); if (why === "run") iAnalyze(); if ($("#s-i2t")) drawI2tSettings(); return; }
    I.afterDownload = why === "run" ? iAnalyze : null;
    iStartJob("tagger_download", { name }, `${name} をダウンロード中...`);
  };
  $("[data-ok]", d).onclick = go;
  $("[data-cancel]", d).onclick = iCloseModal;
  d.onkeydown = (e) => { if (e.key === "Enter") go(); };
}

/* i2t's settings live on LoRAtelier's settings page (#/settings), one panel per group of the i2t
   settings window.  Changes are saved at once (i2t's config.json in data/i2t). */
async function drawI2tSettings() {
  const host = $("#s-i2t");
  if (!host) return;
  if (!I.cfg || !I.c) {
    try {
      const st = await api("/api/i2t/state");
      if (!st.available) { host.innerHTML = `<div class="panel"><h2>画像からプロンプト（i2t）</h2><div class="err small">${esc(st.message)}</div></div>`; return; }
      I.st = st; I.c = st.consts; I.cfg = st.config; I.job = st.job;
    } catch (e) { host.innerHTML = `<div class="panel"><h2>画像からプロンプト（i2t）</h2><div class="err small">${esc(e.message)}</div></div>`; return; }
  }
  const f = I.cfg, c = I.c;
  const vlmChoices = [...new Set([...(I.ollama.vision || []), ...(f.ollama_models || []), f.ollama_model].filter(Boolean))];
  const allModels = [...new Set([...(I.ollama.models || []), f.translate_model, ...(c.translate_choices || [])].filter(Boolean))];
  const chk = (k, label) => `<label class="small" style="display:block"><input type="checkbox" data-sk="${k}" ${f[k] ? "checked" : ""}> ${label}</label>`;
  const num = (k, label) => `<label class="small">${label} <input type="number" step="0.05" min="0.01" max="0.99" data-sn="${k}" value="${f[k]}" style="width:70px"></label>`;
  const panel = (title, body) => `<div class="panel" style="margin-bottom:10px"><h3 style="margin:0 0 6px">${title}</h3>${body}</div>`;
  host.innerHTML = `<h2 style="margin-top:20px">画像からプロンプト（i2t v${esc(c.version)}）</h2>
    <div class="muted small" style="margin-bottom:8px">「画像からプロンプト」ページの設定です。変えた内容はすぐ反映され、保存されます（保存先: data\\i2t\\config.json）。</div>
    <div class="small i-statustext" id="s-i2t-status"></div><div class="small muted" id="s-i2t-ollama"></div>
    ${panel("タガー", `<label class="small">モデル <select id="s-tagger">${Object.entries(c.tagger_models).map(([k, v]) => `<option value="${k}" ${f.tagger_model === k ? "selected" : ""}>${esc(k)} ${esc(v.label)}${v.ready ? " / 取得済み" : ""}</option>`).join("")}</select></label>
      <button class="small" id="s-tagger-get">取得...</button>
      <div class="row" style="margin-top:6px">しきい値 ${num("general_threshold", "一般")}${num("character_threshold", "キャラクター")}${num("style_threshold", "画風")}</div>`)}
    ${panel("VLM (Ollama)", `<div class="row"><label class="small">モデル <input type="text" list="s-vlm-list" id="s-vlm" value="${esc(f.ollama_model)}" style="width:280px"></label><datalist id="s-vlm-list">${vlmChoices.map((m) => `<option value="${esc(m)}">`).join("")}</datalist>
        <button class="small" id="s-pull">追加 (pull)</button><button class="small" id="s-stop" ${I.job ? "" : "hidden"}>■ 中止</button></div>
      <div class="small muted">Ollama のモデル名か、Hugging Face の URL（GGUF のリポジトリ）を入れて「追加 (pull)」で取得します。</div>
      <div class="row"><label class="small">訳のモデル <input type="text" list="s-tr-list" id="s-tr" value="${esc(f.translate_model)}" style="width:280px"></label><datalist id="s-tr-list">${allModels.map((m) => `<option value="${esc(m)}">`).join("")}</datalist><button class="small" id="s-tr-pull">取得</button><span class="small muted">結果の日本語訳用（PLaMo 3 Translate の 2B・8B も選べます）</span></div>
      <div class="row"><label class="small">Ollama URL <input type="text" id="s-host" value="${esc(f.ollama_host)}" style="width:220px"></label><button class="small" id="s-reconnect">再接続</button><button class="small" id="s-setup">導入...</button></div>`)}
    ${panel("プロンプトの書き方", `${chk("quality_tags", "品質タグ (masterpiece, best quality ...) を先頭に付ける")}
      ${chk("use_break", "複数人物は BREAK で人物ごとに区切る (SDXL 系。Regional Prompter などで使う)")}
      ${chk("split_extras", "カメラ・照明・画風のタグを本文から分離する (追加プロンプトのタブに出す)")}
      ${chk("split_background", "背景のタグを本文から分離する (本文を被写体だけにする)")}
      ${chk("era_tags", "年代スタイル (1990s / 2010s など) を判定して画風タグに足す (初回にモデル 17MB を取得)")}
      ${chk("anatomy_guard", "人物の破綻を防ぐ (bad anatomy などをネガティブに足す)")}
      ${chk("avoid_same_face", "同じ顔を避ける (2 人以上なら clone, twins などをネガティブに足す)")}`)}
    ${panel("人物", `${chk("main_person_only", "人物はメインの 1 人だけ (いちばん大きく写っている人。初回に 44MB 取得)")}
      ${chk("split_persons", "2 人以上なら人物ごとに分ける")}`)}
    ${panel("動作・表示", `${chk("low_vram", "GPU / VRAM の使用を抑える (タガーと年代判定を CPU で動かす。VLM は続けて分析できるよう 2 分だけ残し、学習・画像生成・素材の解析が始まるとすぐ VRAM から降ろす)")}
      ${chk("preview_mosaic", "プレビューをモザイク表示 (NSFW 対策。表示だけで、出力には影響しません)")}
      ${chk("result_chips", "結果をタグごとの塊で表示する (オフにすると文字の欄で直接書き換えられます)")}`)}
    ${panel("モデルの置き場所", `<div class="row"><input type="text" id="s-mdir" value="${esc(I.st.models_dir)}" style="flex:1;min-width:300px"><button class="small" id="s-mdir-save">変える</button></div>
      <div class="small muted">タガー・人物検出・骨格などのモデルの置き場所です（初期値: ${esc(I.st.default_models_dir)}）。i2t の models フォルダー（例: D:\\i2t\\models）を指定すると、同じファイルを二重に持たずに使えます。</div>`)}
    <div id="i-modal"></div>`;
  $("#s-tagger").onchange = (e) => iSet("tagger_model", e.target.value);
  $("#s-tagger-get").onclick = () => iTaggerDialog("settings");
  $$("[data-sk]", host).forEach((b) => b.onchange = () => iSet(b.dataset.sk, b.checked));
  $$("[data-sn]", host).forEach((b) => b.onchange = () => {
    const v = Math.min(0.99, Math.max(0.01, Math.round(+b.value * 100) / 100));
    b.value = v; iSet(b.dataset.sn, v);
  });
  $("#s-vlm").onchange = (e) => iSet("ollama_model", e.target.value.trim());
  $("#s-tr").onchange = (e) => iSet("translate_model", e.target.value.trim());
  $("#s-tr-pull").onclick = () => {
    const name = $("#s-tr").value.trim();
    if (!name) return;
    iConfirm("訳のモデルを取得します", `${name} を Ollama に取得します（ollama pull）。数 GB のダウンロードになることがあります。中止しても、次回は続きから取得します。`, "取得する",
      () => { I.pullTranslate = name; iStartJob("pull", { name, host: I.cfg.ollama_host }, `${name} を取得中...`); });
  };
  $("#s-host").onchange = (e) => { let h = e.target.value.trim(); if (h && !/^https?:\/\//.test(h)) h = "http://" + h; iSet("ollama_host", h.replace(/\/+$/, "")); e.target.value = I.cfg.ollama_host; };
  $("#s-pull").onclick = () => iPull($("#s-vlm").value);
  $("#s-stop").onclick = iCancel;
  $("#s-reconnect").onclick = async () => { await api("/api/i2t/config", { method: "PUT", body: I.cfg }).catch(fail); iProbeOllama(); };
  $("#s-setup").onclick = iOllamaSetupDialog;
  $("#s-mdir-save").onclick = async () => {
    try {
      const r = await api("/api/i2t/models_dir", { method: "PUT", body: { path: $("#s-mdir").value } });
      toast(`モデルの置き場所を ${r.models_dir} にしました`);
      I.cfg = null; I.c = null; drawI2tSettings();
    } catch (e) { fail(e); }
  };
  if (!I.ollama.models.length) iProbeOllama();
  else iShowOllama();
  if (I.gotoSettings) { I.gotoSettings = false; host.scrollIntoView({ block: "start" }); }
}

async function iOllamaSetupDialog() {
  if (I.job) return;
  iStatus("Ollama の状態と PC のスペックを確認中...");
  let info, fit;
  try {
    [info, fit] = await Promise.all([api("/api/i2t/call", { body: { fn: "gather_ollama_info", args: [I.cfg.ollama_host, I.cfg.ollama_model], timeout: 120 } }).then((r) => r.value), iApi("system_fit")]);
  } catch (e) { fail(e); return; }
  iStatus("");
  const state = info.reachable ? "connect" : info.exe ? "start" : "install";
  const s = fit.specs;
  const have = new Set(info.models || []);
  const pick = have.has(I.cfg.ollama_model) ? I.cfg.ollama_model : fit.recommend;
  const d = iModal(`<h3>Ollama 導入サポート</h3>
    <div class="small">状態: <b>${{ connect: "接続済み", start: "導入済み・未起動", install: "未導入" }[state]}</b>${info.version ? `（${esc(info.version)}）` : ""}</div>
    <div class="small">PC: ${esc(s.gpu || "GPU 不明")} / VRAM ${s.vram_gb != null ? s.vram_gb + " GB" : "不明"} / メモリ ${s.ram_gb} GB</div>
    ${state === "install" ? `<div class="row"><label class="small">インストール先 <input type="text" id="o-dir" value="${esc(I.c.default_install_dir)}" style="width:360px"></label></div>` : ""}
    <div class="row"><label class="small">モデル保存先 <input type="text" id="o-mdir" value="${esc(info.models_dir || "")}" style="width:360px"></label></div>
    <div class="small muted" id="o-space"></div>
    <div class="small" style="margin-top:8px"><b>使うモデル (PC のスペックから判定)</b></div>
    ${I.c.vlm_catalog.map(([name, size, need, desc]) => `<label class="small" style="display:block"><input type="radio" name="o-model" value="${esc(name)}" ${name === pick ? "checked" : ""}>
      ${esc(name)}　${have.has(name) ? "取得済み" : size + " GB"}　VRAM ${need}GB〜　[${name === fit.recommend ? "推奨" : esc(fit.fit[name])}]　<span class="muted">${esc(desc)}</span></label>`).join("")}
    <label class="small" style="display:block"><input type="radio" name="o-model" value="__other" ${I.c.vlm_catalog.some(([n]) => n === pick) ? "" : "checked"}> その他 (名前を入力) <input type="text" id="o-other" value="${I.c.vlm_catalog.some(([n]) => n === pick) ? "" : esc(pick)}" style="width:240px"></label>
    <label class="small" style="display:block;margin-top:6px"><input type="checkbox" id="o-pull" checked> 続けて選択したモデルを取得する (未取得のとき)</label>
    <div class="row" style="margin-top:12px"><button class="primary" data-ok>${{ install: "ダウンロードしてインストール", start: "Ollama を起動して使う", connect: "この設定で使う" }[state]}</button><button data-cancel>閉じる</button>
      <span class="small muted">管理者権限は不要です。</span></div>`);
  const space = async () => {
    const need = (state === "install" ? I.c.installer_gb + I.c.app_gb : 0);
    try {
      const freeM = await iApi("free_gb", $("#o-mdir").value);
      const freeI = state === "install" ? await iApi("free_gb", $("#o-dir").value) : null;
      $("#o-space").textContent = (state === "install" ? `インストール先の空き ${freeI.toFixed(1)} GB（本体 約 ${I.c.app_gb} GB + インストーラ 約 ${I.c.installer_gb} GB）　` : "") + `モデル保存先の空き ${freeM.toFixed(1)} GB` + (need ? "" : "");
    } catch { $("#o-space").textContent = ""; }
  };
  space();
  $$("#o-mdir, #o-dir", d).forEach((x) => x.onchange = space);
  $("[data-cancel]", d).onclick = () => { iCloseModal(); iStatus("Ollama の導入サポートを閉じました"); };
  $("[data-ok]", d).onclick = () => {
    const r = $("input[name=o-model]:checked", d).value;
    const model = r === "__other" ? $("#o-other").value.trim() : r;
    if (!model) { toast("モデルの名前を入れてください", "err"); return; }
    iSet("ollama_model", model);
    I.setupPull = $("#o-pull").checked && !have.has(model) ? model : null;
    iCloseModal();
    iStartJob("ollama_setup", { action: state, host: I.cfg.ollama_host, exe: info.exe, install_dir: state === "install" ? $("#o-dir").value : null,
      models_dir: $("#o-mdir").value, current_models_dir: info.models_dir || $("#o-mdir").value }, "Ollama の準備を始めます...");
  };
}

function iAfterSetup() {
  iProbeOllama();
  if (I.setupPull) { const m = I.setupPull; I.setupPull = null; setTimeout(() => iStartJob("pull", { name: m, host: I.cfg.ollama_host }, `${m} を取得中...`), 300); }
}

function iHelpDialog(idx = 0) {
  const secs = I.c.help_sections;
  const body = (t) => esc(t).split("\n").map((l) => l.startsWith("■") ? `<b>${l}</b>` : l).join("<br>");
  const d = iModal(`<div class="i-help"><div class="i-helptoc">${secs.map(([t], i) => `<a data-h="${i}" class="${i === idx ? "on" : ""}">${esc(t)}</a>`).join("")}</div>
    <div class="i-helpbody"><h3>${esc(secs[idx][0])}</h3><div class="small">${body(secs[idx][1])}</div>
    <div class="small muted" style="margin-top:12px">この画面では、ドロップ・Ctrl+V・右クリック・Del・Ctrl+Z・Esc・F1 は i2t と同じように使えます。「ウィンドウ」は LoRAtelier のこのページのことです。</div></div></div>
    <div class="row"><button data-cancel>閉じる</button></div>`, "wide");
  $$("[data-h]", d).forEach((a) => a.onclick = () => iHelpDialog(+a.dataset.h));
  $("[data-cancel]", d).onclick = iCloseModal;
}

async function iHistoryDialog() {
  let list;
  try { list = await iApi("history_entries"); } catch (e) { fail(e); return; }
  const mos = I.cfg.preview_mosaic;
  const d = iModal(`<h3>履歴</h3>
    <div class="i-histlist">${list.length ? list.map((h) => `<div class="i-hist" data-p="${esc(h.path)}" tabindex="0">
      ${h.thumb ? `<img src="/api/i2t/file?${mos ? "mosaic=8&" : ""}path=${encodeURIComponent(h.thumb)}" alt="">` : `<div class="i-histnothumb"></div>`}
      <div class="small"><div>${esc(h.saved_at || "")}　${esc(String(h.image || "").split(/[\\/]/).pop())}${Object.keys(h.references || {}).length ? `　ほか ${Object.keys(h.references).length} 枚` : ""}　<span class="muted">${esc((I.c.targets[h.target] || {}).label || h.target || "")}</span></div>
      <div class="muted">${esc(String(h.positive || "").slice(0, 200))}</div></div></div>`).join("") : `<div class="muted small">まだありません。</div>`}</div>
    <div class="small muted">新しい順に ${I.c.history_limit} 件まで残します。保存先: ${esc(I.c.history_dir)}。クリックで選択（Ctrl / Shift で複数）、ダブルクリックか Enter で開きます。</div>
    <div class="row" style="margin-top:8px"><button class="primary" id="h-open">開く</button><button id="h-del">削除</button><button id="h-clear">すべて削除...</button><div class="spacer"></div><button data-cancel>閉じる</button></div>`, "wide");
  const sel = new Set();
  let anchor = null;
  const items = $$(".i-hist", d);
  const draw = () => items.forEach((x, i) => x.classList.toggle("sel", sel.has(i)));
  items.forEach((x, i) => {
    x.onclick = (e) => {
      if (e.shiftKey && anchor != null) { for (let k = Math.min(anchor, i); k <= Math.max(anchor, i); k++) sel.add(k); }
      else if (e.ctrlKey) { sel.has(i) ? sel.delete(i) : sel.add(i); anchor = i; } else { sel.clear(); sel.add(i); anchor = i; }
      draw();
    };
    x.ondblclick = () => iOpenHistory(list[i]);
    x.onkeydown = (e) => { if (e.key === "Enter") iOpenHistory(list[i]); };
  });
  $("#h-open").onclick = () => { const i = [...sel][0]; if (i != null) iOpenHistory(list[i]); };
  $("#h-del").onclick = async () => { for (const i of sel) await iApi("history_delete", list[i].id).catch(fail); iHistoryDialog(); };
  $("#h-clear").onclick = () => iConfirm("履歴をすべて削除します", `${list.length} 件の履歴（結果とサムネイル）を削除します。元の画像は消えません。`, "すべて削除",
    async () => { for (const h of list) await iApi("history_delete", h.id).catch(fail); toast("履歴を削除しました"); });
  $("[data-cancel]", d).onclick = iCloseModal;
}

async function iOpenHistory(h) {
  if (I.job) { toast("処理の最中は開けません"); return; }
  let r;
  try { r = await iApi("history_result", h.path); } catch (e) { fail(e); return; }
  iCloseModal();
  const res = r.result, extra = r.extra || {};
  if (extra.mode && extra.mode !== I.cfg.mode) iSet("mode", extra.mode);
  $$("input[name=i-mode]").forEach((x) => { x.checked = x.value === I.cfg.mode; });
  I.slots = {};
  const missing = [];
  const paths = { person: res.image, ...(res.references || {}) };
  for (const [k, p] of Object.entries(paths)) {
    try { I.slots[k] = await api("/api/i2t/use_file", { body: { path: p } }); } catch { missing.push(String(p).split(/[\\/]/).pop()); }
  }
  I.analyzedCfg = { ...I.cfg, mosaic_strength: extra.mosaic_strength || I.cfg.mosaic_strength };
  I.analyzedMain = I.slots.person ? { ...I.slots.person, refs: Object.keys(res.references || {}).filter((k) => I.slots[k]).length } : null;
  iRenderPreview();
  await iShowResult(res, h.path);
  if (missing.length) iStatus(`履歴を開きました（元の画像が見つかりません: ${missing.join("、")}）`);
}

function iGenInfoDialog() {
  const seen = new Map();
  for (const k of SLOT_ORDER) { const s = I.slots[k]; if (s && s.geninfo && !seen.has(s.path)) seen.set(s.path, { label: multi() ? iSlotLabel(k) : s.name, s }); }
  const tabs = [...seen.values()];
  if (!tabs.length) return;
  const show = (i) => {
    const { s } = tabs[i], g = s.geninfo;
    const field = (label, v) => `<div class="row"><b class="small">${label}</b><div class="spacer"></div><button class="small" data-gc="${esc(v || "")}">コピー</button></div><textarea readonly class="i-tabtext short">${esc(v || "(なし)")}</textarea>`;
    const d = iModal(`<h3>画像に入っていた生成情報</h3><div class="small muted">画像を作ったときに保存されたプロンプトなどです（i2t の推定ではありません）。</div>
      <nav class="tabs">${tabs.map((t, k) => `<a data-gt="${k}" class="${k === i ? "on" : ""}">${esc(t.label)}</a>`).join("")}</nav>
      <div class="small">${esc(s.name)}（${esc(g.source)}）</div>
      ${field("ポジティブ", g.positive)}${field("ネガティブ", g.negative)}${field("設定", g.settings)}${field("元のデータ", g.raw)}
      <div class="row" style="margin-top:8px"><button data-cancel>閉じる</button></div>`, "wide");
    $$("[data-gt]", d).forEach((a) => a.onclick = () => show(+a.dataset.gt));
    $$("[data-gc]", d).forEach((b) => b.onclick = () => iCopy(b.dataset.gc));
    $("[data-cancel]", d).onclick = iCloseModal;
  };
  show(0);
}

// ------------------------------------------------------------------ pose editor
async function iPoseEditor() {
  if (I.job) return;
  const src = (multi() && I.slots.pose) || I.slots.person;
  const size = src ? src.size : I.c.pose_template_size;
  const VIEW_H = 560, k = VIEW_H / size[1], vw = Math.max(240, Math.round(size[0] * k));
  const template = () => { const [tw, th] = I.c.pose_template_size; const s = Math.min(size[0] / tw, size[1] / th);
    const ox = (size[0] - tw * s) / 2, oy = (size[1] - th * s) / 2; return I.c.pose_template.map(([x, y]) => [ox + x * s, oy + y * s, 1.0]); };
  let points = I.poseControl && I.poseControl.size[0] === size[0] && I.poseControl.size[1] === size[1] ? I.poseControl.points.map((p) => [...p]) : template();
  const d = iModal(`<h3>ボーンでポーズを編集</h3><div class="row" style="align-items:flex-start">
    <div style="position:relative;width:${vw}px;height:${VIEW_H}px;background:#202020">
      ${src ? `<img src="/api/i2t/file?path=${encodeURIComponent(src.path)}" style="width:${vw}px;height:${VIEW_H}px;filter:brightness(.55);position:absolute">` : ""}
      <svg id="pe-svg" width="${vw}" height="${VIEW_H}" style="position:absolute;left:0;top:0"></svg></div>
    <div style="width:240px" class="small">
      <button id="pe-auto" ${src ? "" : "disabled"} style="width:100%">画像から骨格を取る</button>
      <button id="pe-reset" style="width:100%;margin-top:4px">立ち姿に戻す</button>
      <button id="pe-save" style="width:100%;margin-top:12px">OpenPose 画像を保存...</button>
      <div class="muted">ControlNet (OpenPose) 用。<br>使わなければ保存不要</div>
      <div style="margin-top:12px">プロンプトに入るポーズ:</div><div id="pe-tags" style="color:#0a5"></div>
      <div class="muted" style="margin-top:12px">点をドラッグで移動、右クリックで表示 / 非表示。<br>向き (正面 / 後ろ) は骨格では決まらないので、<br>「カメラ」の向きで指定してください。</div>
      <div id="pe-status" style="color:#a60;margin-top:8px"></div>
      <div class="row" style="margin-top:16px"><button class="primary" id="pe-apply">適用</button><button data-cancel>キャンセル</button></div>
    </div></div>`, "wide");
  const svg = $("#pe-svg", d), MIN = I.c.pose_min_conf;
  const tagsSoon = debounce(async () => { try { const t = await iApi("pose_tags_from_points", points); $("#pe-tags").textContent = t.map((x) => x.replace(/_/g, " ")).join(", ") || "(タグにできる姿勢なし)"; } catch {} }, 150);
  const draw = () => {
    svg.innerHTML = I.c.pose_bones.map(([a, b]) => points[a][2] >= MIN && points[b][2] >= MIN ? `<line x1="${points[a][0] * k}" y1="${points[a][1] * k}" x2="${points[b][0] * k}" y2="${points[b][1] * k}" stroke="#ff5050" stroke-width="3"/>` : "").join("")
      + points.map(([x, y, c], i) => `<circle data-p="${i}" cx="${x * k}" cy="${y * k}" r="6" stroke="${c >= MIN ? "#fff" : "#888"}" fill="${c >= MIN ? "#3c3" : "none"}" stroke-width="2"><title>${esc(I.c.keypoints[i][1])}</title></circle>`).join("");
    tagsSoon();
  };
  const nearest = (e) => { const r = svg.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    let best = null, bd = 14; points.forEach((p, i) => { const dd = Math.hypot(p[0] * k - x, p[1] * k - y); if (dd <= bd) { bd = dd; best = i; } }); return best; };
  let drag = null;
  svg.onmousedown = (e) => { if (e.button === 0) drag = nearest(e); };
  svg.onmousemove = (e) => { if (drag == null) return; const r = svg.getBoundingClientRect();
    points[drag] = [Math.max(0, Math.min(size[0], (e.clientX - r.left) / k)), Math.max(0, Math.min(size[1], (e.clientY - r.top) / k)), 1.0]; draw(); };
  svg.onmouseup = svg.onmouseleave = () => { drag = null; };
  svg.oncontextmenu = (e) => { e.preventDefault(); const i = nearest(e); if (i == null) return; points[i][2] = points[i][2] >= MIN ? 0 : 1.0; draw(); };
  $("#pe-reset", d).onclick = () => { points = template(); $("#pe-status").textContent = ""; draw(); };
  $("#pe-auto", d).onclick = async () => {
    const ready = await iApi("pose_model_ready").catch(() => false), face = await iApi("missing_face_models").catch(() => []);
    const run = () => {
      I.onPose = (v) => { I.onPose = null; points = v.points; $("#pe-status").textContent = v.hidden.length ? "見えていない関節 (灰色、タグに使わない): " + v.hidden.join("、") : ""; draw(); };
      iStartJob("pose_estimate", { path: src.path }, "骨格を推定中...");
      $("#pe-status").textContent = "骨格を推定中...";
    };
    if (!ready || face.length) {   // asked inside the editor so the edited points are kept
      $("#pe-status").innerHTML = `画像から骨格を取るには、次のモデルを取得します (初回のみ。CPU で動き、GPU は使いません):<br>${face.map((m) => "・" + esc(m)).join("<br>")}<br>保存先: ${esc(I.c.models_dir)}<br><button class="small" id="pe-get">取得する</button>`;
      $("#pe-get").onclick = run;
    } else run();
  };
  $("#pe-save", d).onclick = async () => {
    try {
      const r = await fetch("/api/i2t_openpose", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ points, size }) });
      if (!r.ok) throw new Error((await r.json()).detail);
      const a = document.createElement("a"); a.href = URL.createObjectURL(await r.blob()); a.download = "openpose.png"; a.click();
    } catch (e) { fail(e); }
  };
  $("#pe-apply", d).onclick = async () => {
    const tags = await iApi("pose_tags_from_points", points).catch(() => []);
    I.poseControl = { points, size, tags }; iCloseModal(); iDrawControls();
  };
  $("[data-cancel]", d).onclick = iCloseModal;
  draw();
}

// leaving the page while working: the job keeps running on the server and is shown again on return
window.addEventListener("beforeunload", (e) => { if (S.route[0] === "i2t" && I.job && I.job.op === "ollama_setup") { e.preventDefault(); e.returnValue = ""; } });
