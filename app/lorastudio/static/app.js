/* LoRAtelier UI — vanilla JS, no build step.  Every user action gets immediate visual
   feedback; network work happens in the background and failures are shown, never hidden. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const main = $("#main");

async function api(path, opts = {}) {
  const o = { method: opts.method || (opts.body ? "POST" : "GET"), headers: {} };
  if (S.unloading) o.keepalive = true;   // pagehide: the request outlives the page
  if (opts.body instanceof FormData) o.body = opts.body;
  else if (opts.body !== undefined) { o.body = JSON.stringify(opts.body); o.headers["Content-Type"] = "application/json"; }
  const r = await fetch(path, o);
  let data = null;
  try { data = await r.json(); } catch { /* empty */ }
  if (!r.ok) throw new Error((data && data.detail) || `HTTP ${r.status}`);
  return data;
}

function toast(msg, kind = "", action) {
  const t = document.createElement("div");
  t.className = "toast " + kind;
  t.innerHTML = esc(msg);
  if (action) {
    const b = document.createElement("button");
    b.textContent = action.label;
    b.onclick = () => { action.fn(); t.remove(); };
    t.appendChild(b);
  }
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), kind === "err" ? 9000 : 4000);
}
const fail = (e) => toast(e.message || String(e), "err");
const fmtDur = (s) => s == null ? "計測中" : s < 60 ? `${s}秒` : s < 3600 ? `${Math.floor(s / 60)}分${s % 60}秒` : `${Math.floor(s / 3600)}時間${Math.floor((s % 3600) / 60)}分`;
const debounce = (fn, ms) => {
  let t = null, args;
  const d = (...a) => { clearTimeout(t); args = a; t = setTimeout(() => { t = null; fn(...args); }, ms); };
  d.flush = () => { if (t) { clearTimeout(t); t = null; fn(...args); } };
  return d;
};
/* Autosaves that wait for typing to pause: sent at once when the page changes (route) or the tab is closed
   (pagehide), so an edit made just before is not lost. */
const PENDING_SAVES = new Set();
function saveDebounce(fn, ms) {
  const d = debounce((...a) => { PENDING_SAVES.delete(call); return fn(...a); }, ms);
  const call = (...a) => { PENDING_SAVES.add(call); d(...a); };
  call.flush = () => d.flush();
  return call;
}
function flushSaves() { [...PENDING_SAVES].forEach((d) => d.flush()); }
// one pending re-check per box: redrawing a box again replaces its timer instead of starting another chain
const _polls = {};
function pollLater(key, ms, fn) { clearTimeout(_polls[key]); _polls[key] = setTimeout(() => { delete _polls[key]; fn(); }, ms); }

/* Inline input bar instead of window.prompt (no blocking browser dialogs).
   Resolves with the entered values, or null when cancelled (Esc / 取消). */
function ask(fields, okLabel = "実行", anchor = "#m-bulk") {
  return new Promise((resolve) => {
    $$(".askbar").forEach((x) => x.remove());
    const bar = document.createElement("div");
    bar.className = "askbar row panel";
    bar.style.margin = "6px 0";
    bar.innerHTML = fields.map((f, i) => `<label class="row small">${esc(f.label)}<input type="text" data-i="${i}" value="${esc(f.value || "")}"
      placeholder="${esc(f.placeholder || "")}" style="width:${f.width || 260}px"></label>`).join("") +
      `<button class="primary" data-ok>${esc(okLabel)}</button><button data-cancel>取消</button>`;
    const host = $(anchor) || main;
    host.after(bar);
    const inputs = $$("input", bar);
    const done = (v) => { bar.remove(); resolve(v); };
    $("[data-ok]", bar).onclick = () => done(inputs.map((x) => x.value));
    $("[data-cancel]", bar).onclick = () => done(null);
    inputs.forEach((x) => x.onkeydown = (e) => { if (e.key === "Enter") done(inputs.map((y) => y.value)); if (e.key === "Escape") done(null); });
    inputs[0].focus();
  });
}

// ------------------------------------------------------------------ state
const S = {
  route: [], status: null, project: null, images: [], report: null, history: {}, refs: [],
  filter: "attention", search: "", sel: new Set(), focus: null, lastClick: null, detail: null,
  plan: null, runs: [], resultsRun: null, picked: null,
};

// ------------------------------------------------------------------ routing
window.addEventListener("hashchange", route);
function route() {
  flushSaves();   // the old page's inputs are still there
  const h = location.hash.replace(/^#\/?/, "");
  S.route = h ? h.split("/") : [];
  const [a, pid, tab] = S.route;
  if (a === "p" && pid) openProject(pid, tab || "material");
  else if (a === "settings") renderSettings();
  else if (a === "help") renderHelp(pid);
  else if (a === "gen") renderImageGen();
  else if (a === "presets") renderPresetList();
  else if (a === "loras") renderLoraSettings();
  else if (a === "dict") renderDict();
  else if (a === "i2t") renderI2t();
  else if (a === "jobs") renderJobs();
  else renderHome();
}

// ------------------------------------------------------------------ events (SSE)
/* One live connection per tab, and a browser allows only 6 per server: a tab left in the background for
   5 minutes closes its connection (otherwise a few left-open tabs leave no room and every request of the
   visible tab waits forever).  Not at once: Chrome also reports a minimized or covered window as hidden,
   and generation results must keep arriving while the user looks elsewhere.  Coming back, the tab
   reconnects and reloads what may have changed meanwhile. */
let eventSrc = null;
function connectEvents() {
  if (eventSrc || document.hidden) return;
  const es = eventSrc = new EventSource("/api/events");
  es.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "queue" || (m.type === "run" && m.status)) refreshQueueSoon();
    if (S.route[0] === "jobs" && ["queue", "run", "gen", "imagegen", "eval", "pipeline", "i2t"].includes(m.type)) refreshJobsSoon();
    if (m.type === "images" && S.project && m.project === S.project.id) refreshImagesBg();
    else if (m.type === "identity" && S.project && m.project === S.project.id) refreshImagesBg();
    else if (m.type === "run" && S.project && m.project === S.project.id) onRunEvent(m);
    else if (m.type === "eval" && S.project && m.project === S.project.id) onEvalEvent(m);
    else if (m.type === "checkpoints" && S.project && m.project === S.project.id) refreshRunsSoon();
    else if (m.type === "worker" || m.type === "analysis") refreshStatusSoon();
    // a finished download changes the model lists (whole page); otherwise only the download boxes, so fields being typed in stay
    else if (m.type === "download") { if (S.route[0] === "settings") { if (m.state === "done") renderSettings(); else { drawCatalog(); drawImportState(); } } }
    else if (m.type === "gen" && S.project && m.project === S.project.id) onGenEvent(m);
    else if (m.type === "imagegen" && S.route[0] === "gen") { if (m.progress) igOnProgress(m); else igReloadSoon(); }
    else if (m.type === "imagegen" && S.route[0] === "presets" && !m.progress) plReloadSoon();
    else if (m.type === "i2t" && m.event === "done" && m.op === "moss_catalog" && S.project && !["i2t", "settings"].includes(S.route[0])) {
      if (m.ok) refreshImagesBg(); else if (!m.cancelled) toast(m.error, "err");
    }
    else if (m.type === "i2t" && (S.route[0] === "i2t" || S.route[0] === "settings")) onI2tEvent(m);
    else if (m.type === "import" && (m.failed || []).length) importFailed(m.failed);
  };
}
let hiddenTimer = null;
document.addEventListener("visibilitychange", () => {
  clearTimeout(hiddenTimer);
  if (document.hidden) {
    hiddenTimer = setTimeout(() => { if (document.hidden && eventSrc) { eventSrc.close(); eventSrc = null; } }, 5 * 60 * 1000);
    return;
  }
  if (eventSrc) return;
  connectEvents();
  refreshStatusSoon();
  if (S.project) { refreshImages(); refreshRunsSoon(); }
  if (S.route[0] === "gen") igReloadSoon();
  if (S.route[0] === "presets") plReloadSoon();
});
const refreshStatusSoon = debounce(refreshStatus, 300);

/* Files that could not be imported (locked, no permission ...): the others were imported. */
function importFailed(failed) {
  if (!failed || !failed.length) return;
  toast(`${failed.length} 枚は取り込めませんでした: ${failed.slice(0, 3).join(" ／ ")}${failed.length > 3 ? " ほか" : ""}`, "err");
}
async function refreshStatus() {
  try { S.status = await api("/api/status"); } catch { return; }
  const st = S.status, jp = $("#jobpill"), ap = $("#aipill");
  document.documentElement.classList.toggle("blurimg", st.blur_images === true);
  if (st.job) {
    const p = st.job.progress || {};
    jp.hidden = false;
    jp.textContent = (st.job.kind === "train" && p.total ? `${p.message || "実行中"} ${p.step || 0}/${p.total}` : (st.job.message || p.message || "実行中")) +
      (st.queued && st.queued.length ? `（ほか ${st.queued.length} 件待ち${st.queue_paused ? "・一時停止中" : ""}）` : "");
    jp.title = "ジョブの一覧を開きます";
    jp.onclick = () => { location.hash = "#/jobs"; };
  } else if (st.queue_paused && st.queued && st.queued.length) {
    jp.hidden = false;
    jp.textContent = `順番待ち ${st.queued.length} 件（一時停止中）`;
    jp.title = "ジョブの一覧を開きます";
    jp.onclick = () => { location.hash = "#/jobs"; };
  } else jp.hidden = true;
  const w = st.worker || "";
  if (st.gpu_owner) { ap.hidden = false; ap.textContent = `AI解析: ${st.gpu_owner}のため待機`; }
  else if (w.startsWith("loading")) { ap.hidden = false; ap.textContent = `AI準備中: ${w.split(":")[1]}`; }
  else if (st.analysis_queue > 0) { ap.hidden = false; ap.textContent = `AI解析${w === "ready_cpu" ? "（GPU が使えないため CPU）" : ""} 残り${st.analysis_queue}`; }
  else ap.hidden = true;
  ap.title = w === "ready_cpu" ? workerLabel(w) : "";
}
setInterval(() => { if (!document.hidden) refreshStatus(); }, 4000);

// ------------------------------------------------------------------ home
function projectSort() {
  try { return localStorage.getItem("lorastudio.psort") === "created" ? "created" : "updated"; } catch { return "updated"; }
}

/* Project cards, newest first by the chosen date (更新 / 作成); the date shown is the one sorted by. */
function drawProjectList(list) {
  const key = projectSort(), when = (p) => p[key] || p.updated || 0;
  const items = [...list].sort((a, b) => when(b) - when(a));
  $(".plist", main).innerHTML = items.length ? items.map((p) => `
      <a class="pcard" href="#/p/${esc(p.id)}" data-pid="${esc(p.id)}" data-pname="${esc(p.name)}">
        <div class="row"><b>${esc(p.name)}</b><span class="kind">${kindShort(p)}</span>
          <button class="pmore" data-pmore title="書き出し・削除など" aria-label="操作メニュー">⋮</button></div>
        <div class="muted small">${p.images ?? 0} 枚 ・ ${key === "created" ? "作成" : "更新"} ${when(p) ? new Date(when(p) * 1000).toLocaleString() : "不明"}</div>
        ${p.error ? `<div class="err small">${esc(p.error)}</div>` : ""}
      </a>`).join("") : `<div class="muted">まだありません。</div>`;
}

async function renderHome() {
  S.project = null;
  $("#crumbs").innerHTML = "";
  main.innerHTML = `<div class="muted">読み込み中…</div>`;
  let list = [], models = { models: [] };
  try { [list, models] = await Promise.all([api("/api/projects"), api("/api/models")]); } catch (e) { fail(e); }
  const noModel = !models.models.length;
  main.innerHTML = `
    ${misplacedHtml(models.misplaced)}
    ${noModel ? `<div class="panel" style="margin-bottom:14px" id="firstrun"><b>はじめに：ベースモデルを用意してください</b>
      <div class="muted">学習と画像生成の土台にするモデルです（初回だけ）。素材の取り込みと確認は、用意する前でもできます。</div>
      <div id="fr-body" class="muted small" style="margin-top:8px">読み込み中…</div></div>` : ""}
    <h1>新しく作る</h1>
    <div class="newbox">
      <button class="newcard" data-new="character"><b>キャラクター LoRA</b><span class="muted">特定のキャラクターや人物を、別の構図・ポーズ・背景でも描けるようにします。</span></button>
      <button class="newcard" data-new="character" data-src="multi"><b>複数キャラ LoRA</b><span class="muted">作ったキャラクター LoRA のプロジェクトをいくつか選んで、1 つの LoRA にまとめます。キャラごとの呼び出し語で出し分けます。</span></button>
      <button class="newcard" data-new="style"><b>スタイル LoRA</b><span class="muted">線・塗り・色使いなどの画風を、別の被写体にも適用できるようにします。</span></button>
      <button class="newcard" data-new="concept"><b>体型・特徴 LoRA</b><span class="muted">筋肉質・体格・手の形など、体の特徴を、人物を問わず加えられるようにします。</span></button>
      <button class="newcard" data-new="concept" data-target="outfit"><b>衣装・物・場所・ポーズ・表情 LoRA</b><span class="muted">特定の衣装・小物・場所・ポーズ・表情を、呼び出し語ひとつで、どの人物や場面にも出せるようにします。</span></button>
      <button class="newcard" data-new="enhance"><b>質感・補強 LoRA</b><span class="muted">肌の質感・細部の描き込み・陰影・色の鮮やかさなどを、何を描いても強められるようにします（試験的）。</span></button>
      <button class="newcard" data-new="enhance" data-target="slider"><b>スライダー LoRA</b><span class="muted">年齢・体格・明るさなど、決めた2つの方向の間を、LoRA の強さをプラス・マイナスに動かして調整できるようにします（試験的・画像不要）。</span></button>
      <button class="newcard" data-new="character" data-src="generate"><b>AI 生成画像からキャラ LoRA</b><span class="muted">プロンプトとモデル・LoRA で作ったキャラを、いろいろなアングルで生成して素材にし、呼び出し語ひとつで出せるようにします。</span></button>
      <button class="newcard" data-new="style" data-src="generate"><b>AI 生成画像からスタイル LoRA</b><span class="muted">プロンプトとモデル・LoRA で出せる画風を、人物・動物・風景・物などいろいろな題材で生成して素材にし、呼び出し語ひとつで出せるようにします。</span></button>
    </div>
    <div id="newform"></div>
    <div id="queue"></div>
    <div class="row" style="justify-content:space-between"><h1>プロジェクト</h1>
      <span class="row"><select id="p-sort" title="並べ方"><option value="updated">更新が新しい順</option><option value="created">作成が新しい順</option></select>
        <span class="muted small">各プロジェクトの ⋮ から書き出し・削除</span>
        <button id="p-import">zip から取り込む</button><input type="file" id="p-import-file" accept=".zip" hidden></span></div>
    <div class="plist"></div>`;
  drawProjectList(list);
  drawQueue();
  $("#p-sort").value = projectSort();
  $("#p-sort").onchange = (e) => { try { localStorage.setItem("lorastudio.psort", e.target.value); } catch { /* not kept */ } drawProjectList(list); };
  bindMisplaced(main, renderHome);
  $$("[data-new]").forEach((b) => b.onclick = () => newForm(b.dataset.new, models, b.dataset.src, b.dataset.target));
  $(".plist").addEventListener("contextmenu", onProjectMenu);
  $(".plist").addEventListener("click", (e) => {
    const b = e.target.closest("[data-pmore]");
    if (!b) return;
    e.preventDefault(); e.stopPropagation();   // the card itself is a link that opens the project
    if ($("#ctxmenu")?.dataset.pid === b.closest(".pcard").dataset.pid) { closeMenu(); return; }
    const r = b.getBoundingClientRect();
    onProjectMenu({ target: b, clientX: r.right, clientY: r.bottom + 4, preventDefault() {}, alignRight: true });
  });
  $("#p-import").onclick = () => $("#p-import-file").click();
  $("#p-import-file").onchange = (e) => importProjects([...e.target.files]);
  if (noModel) drawFirstRun();
}

const fmtSize = (n) => n >= 1e9 ? `${(n / 1e9).toFixed(1)}GB` : `${Math.max(1, Math.round(n / 1e6))}MB`;

function onProjectMenu(e) {
  const card = e.target.closest(".pcard");
  if (!card) return;
  e.preventDefault();
  closeMenu();
  const pid = card.dataset.pid;
  const m = document.createElement("div");
  m.id = "ctxmenu"; m.className = "ctxmenu"; m.dataset.pid = pid;
  m.innerHTML = `<button data-m="open">開く</button><hr>
    <button data-m="export">zip に書き出す（素材・説明文・設定）<span class="key" data-sz="base"></span></button>
    <button data-m="export_runs">zip に書き出す（学習結果も含める）<span class="key" data-sz="with_runs"></span></button><hr>
    <button data-m="delete" class="danger">削除</button>`;
  document.body.append(m);
  const r = m.getBoundingClientRect();
  m.style.left = Math.max(8, Math.min(e.alignRight ? e.clientX - r.width : e.clientX, innerWidth - r.width - 8)) + "px";
  m.style.top = Math.min(e.clientY, innerHeight - r.height - 8) + "px";
  api(`/api/projects/${pid}/export_size`).then((sz) => $$("[data-sz]", m).forEach((x) => x.textContent = `約 ${fmtSize(sz[x.dataset.sz])}`)).catch(() => {});
  m.onclick = async (ev) => {
    const b = ev.target.closest("[data-m]");
    if (!b) return;
    const op = b.dataset.m;
    if (op === "delete" && !b.dataset.armed) {
      // second click deletes: no blocking dialog, but no one-click loss of a whole project either
      b.dataset.armed = "1";
      b.innerHTML = `もう一度押すと削除（元に戻せません）`;
      const note = document.createElement("div");
      note.className = "muted small"; note.style.padding = "2px 10px 6px"; note.style.maxWidth = "300px";
      note.textContent = `「${card.dataset.pname}」の素材・説明文・学習結果をすべて削除します。書き出し済みの LoRA（models\loras）は残ります。`;
      m.append(note);
      return;
    }
    closeMenu();
    if (op === "open") location.hash = `#/p/${pid}`;
    else if (op === "export" || op === "export_runs") exportProject(pid, op === "export_runs");
    else if (op === "delete") {
      card.style.opacity = 0.4;
      try { await api(`/api/projects/${pid}`, { method: "DELETE" }); card.remove(); toast(`「${card.dataset.pname}」を削除しました`); }
      catch (err) { card.style.opacity = ""; fail(err); }
    }
  };
}

async function exportProject(pid, withRuns) {
  if (withRuns) {
    // refuse before the browser starts a download that can only fail
    try { const sz = await api(`/api/projects/${pid}/export_size`); if (sz.busy) return toast(sz.busy, "err"); } catch (e) { return fail(e); }
  }
  const a = document.createElement("a");
  a.href = `/api/projects/${pid}/export.zip?runs=${withRuns ? 1 : 0}`;
  a.download = "";
  document.body.append(a); a.click(); a.remove();
  toast("書き出しを始めました。進み具合はブラウザーのダウンロード表示で確認できます。");
}

/* Project zips written by "zip に書き出す": from the button on the home page, or dropped anywhere (one after another). */
/* While zips are being imported, a box stays on screen (not a toast that fades) until the last one is done.
   Zips dropped meanwhile wait their turn. */
const _importQ = [];
let _importing = false;
function drawImportBusy(cur) {
  let el = $("#importbusy");
  if (!cur) { if (el) el.remove(); return; }
  if (!el) { el = document.createElement("div"); el.id = "importbusy"; document.body.appendChild(el); }
  el.innerHTML = `<span class="spin"></span><span>「${esc(cur.name)}」を取り込み中…${_importQ.length ? `（このあと ${_importQ.length} 件）` : ""}<br>
    <span class="small">終わるまでこのページを閉じないでください</span></span>`;
}
window.addEventListener("beforeunload", (e) => { if (_importing) { e.preventDefault(); e.returnValue = ""; } });
async function importProjects(files) {
  _importQ.push(...files.filter(Boolean));
  if (_importing || !_importQ.length) { if (_importing) drawImportBusy(_importing); return; }
  const busy = (on) => { const b = $("#p-import"); if (b) { b.disabled = on; b.textContent = on ? "取り込み中…" : "zip から取り込む"; } };
  let done = 0;
  while (_importQ.length) {
    const file = _importQ.shift();
    _importing = file;
    busy(true);
    drawImportBusy(file);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const p = await api("/api/projects/import", { body: fd });
      done++;
      const lo = p.loras || { added: [], failed: [] };
      toast(`「${p.name}」をプロジェクトとして取り込みました` + (lo.added.length ? `。LoRA ${lo.added.length} 個（${lo.added.join("、")}）を LoRA の一覧に加えました` : ""),
        "", S.route[0] === "p" || !$("#p-import") ? { label: "開く", fn: () => { location.hash = `#/p/${p.id}/material`; } } : undefined);
      if (lo.failed.length) toast(`LoRA の一覧に加えられなかったものがあります: ${lo.failed.join(" / ")}`, "err");
    } catch (e) { fail(new Error(`「${file.name}」: ${e.message}`)); }
  }
  _importing = false;
  drawImportBusy(null);
  busy(false);
  if ($("#p-import-file")) $("#p-import-file").value = "";
  if (done && $("#p-import")) renderHome();
}

/* First run: the setup downloads no model.  The user ticks only the standard models they want to make LoRAs
   for (nothing is ticked in advance), or drops their own files into models\checkpoints. */
const FR_HINT = { illustrious: "アニメ・イラスト向け", pony: "アニメ・イラスト向け（Pony 系のタグで生成）", sdxl_base: "実写・汎用" };
const frPick = new Set();
async function drawFirstRun() {
  const el = $("#fr-body");
  if (!el) return;
  let cat;
  try { cat = await api("/api/catalog"); } catch (e) { el.innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  const defs = cat.items.filter((c) => c.default);
  const running = defs.some((c) => c.dl && c.dl.state === "running");
  const busy = (c) => c.installed_model || (c.dl && c.dl.state === "running");
  defs.forEach((c) => { if (busy(c)) frPick.delete(c.id); });
  const gb = defs.filter((c) => frPick.has(c.id)).reduce((a, c) => a + c.size_gb, 0).toFixed(1);
  el.innerHTML = `<div style="margin-top:4px">LoRA を作りたいモデルを選んでください。あとから設定画面で追加することもできます。</div>
    ${defs.map((c) => `<label class="row" style="margin-top:4px">
      <input type="checkbox" data-frpick="${esc(c.id)}" ${frPick.has(c.id) ? "checked" : ""} ${busy(c) ? "disabled" : ""}>
      <b>${esc(c.name)}</b><span>${esc(FR_HINT[c.variant] || "")} ・ 約 ${c.size_gb}GB</span>
      ${c.installed_model ? `<span class="muted">用意済み</span>` : c.dl ? `<span>${c.dl.state === "running" ? `ダウンロード中 ${(c.dl.done_bytes / 1e9).toFixed(1)} / ${(c.dl.total_bytes / 1e9).toFixed(1)} GB
          ${c.dl.cancel ? "（キャンセル中）" : `<button class="small" data-dlcancel="${esc(c.id)}">キャンセル</button>`}` :
        c.dl.state === "done" ? "完了" : c.dl.state === "cancelled" ? `<span class="muted">キャンセルしました</span>` : `<span class="err">${esc(c.dl.error || "失敗")}</span>`}</span>` : ""}</label>`).join("")}
    <div class="row" style="margin-top:6px"><button class="primary" id="fr-dl" ${frPick.size ? "" : "disabled"}>${frPick.size ? `選んだ ${frPick.size} 種類をダウンロード（約 ${gb}GB）` : "ダウンロードするモデルを選んでください"}</button></div>
    <div style="margin-top:8px">手元にモデルファイルがあれば、ダウンロードせずに使えます。
      <b>${esc(cat.models_dir)}\\checkpoints</b> にファイルを置くと、自動で一覧に出ます（コピーはしません）。
      <button class="small" data-opendir="checkpoints">フォルダーを開く</button></div>
    <div>Anima・Qwen-Image などのほかのモデルや LoRA は、<a href="#/settings">設定画面</a>から、Hugging Face・Civitai のリンクやダウンロードしたファイルを使ってあとで追加できます。</div>`;
  $$("[data-frpick]", el).forEach((x) => x.onchange = () => { x.checked ? frPick.add(x.dataset.frpick) : frPick.delete(x.dataset.frpick); drawFirstRun(); });
  bindDlCancel(el, drawFirstRun);
  $("#fr-dl").onclick = async () => {
    $("#fr-dl").disabled = true;
    try { for (const c of defs) if (frPick.has(c.id)) await api(`/api/downloads/${c.id}`, { body: {} }); } catch (e) { fail(e); }
    frPick.clear();
    drawFirstRun();
  };
  bindOpenDir(el);
  if (running) pollLater("firstrun", 3000, () => { if (!S.route[0]) drawFirstRun(); });
  // once a chosen model is ready, the home screen no longer needs this box
  else if (defs.some((c) => c.dl && c.dl.state === "done" && c.installed_model)) renderHome();
}

// Model files left in the folders an older version saved downloads to (e.g. models\anima): say which file to move
// from where to where.  Moved files are found again by name, so registered models and projects keep working.
function misplacedHtml(list) {
  if (!list || !list.length) return "";
  const groups = {};
  for (const x of list) (groups[`${x.from}|${x.to}`] ||= { ...x, files: [] }).files.push(x);
  return `<div class="panel warnbox" style="margin-bottom:14px"><b>古い置き場所にモデルが残っています</b>
    <div class="small">以前の版はモデルを種類ごとのフォルダーに保存していました。今の版で使うには、次のファイルを移動元から移動先のフォルダーに移してください。移したあとも、登録済みのモデルとプロジェクトはそのまま使えます。</div>
    ${Object.values(groups).map((g) => `<div class="small" style="margin-top:8px">
      ${g.files.map((f) => `<div>・${esc(f.what)}（${esc(f.name)}）</div>`).join("")}
      <div>移動元: <b>${esc(g.from)}</b> <button class="small" data-opendir="${esc(g.old_dir)}">開く</button></div>
      <div>移動先: <b>${esc(g.to)}</b> <button class="small" data-opendir="${esc(g.sub)}">開く</button></div></div>`).join("")}
    <div class="row" style="margin-top:8px"><button class="small" data-recheck>移したので確認し直す</button></div></div>`;
}

function bindMisplaced(root, redraw) {
  bindOpenDir(root);
  $$("[data-recheck]", root).forEach((b) => b.onclick = redraw);
}

function bindOpenDir(root) {
  $$("[data-opendir]", root).forEach((b) => b.onclick = async () => {
    try { await api("/api/models_dir/open", { body: { sub: b.dataset.opendir } }); } catch (e) { fail(e); }
  });
}

const KIND_LABEL = { character: "キャラクター", style: "スタイル", concept: "体型・特徴", enhance: "質感・補強" };
const KIND_SHORT = { character: "キャラ", style: "スタイル", concept: "体型・特徴", enhance: "質感・補強" };
// what a concept LoRA learns (captions.CONCEPT_TARGETS on the server); "body" is the original 体型・特徴 LoRA
const CONCEPT_UI = {
  body: { label: "体型・特徴", noun: "体の特徴", example: "muscular, abs", desc: "",
    caption: "人物・服・ポーズ・背景などはそのまま書きます。" },
  outfit: { label: "衣装", noun: "衣装", example: "sailor uniform, red neckerchief",
    desc: "特定の服装を、別の人物にも着せられるようにします。いろいろな人物がその衣装を着た素材を使います。",
    caption: "衣装以外の、人物・髪・ポーズ・背景などはそのまま書きます。" },
  object: { label: "物・小物", noun: "物", example: "sword, ornate hilt",
    desc: "武器・乗り物・アクセサリーなど、特定の物を描けるようにします。いろいろな場所や角度で写した素材を使います。",
    caption: "物そのものの形や色は書かず、置かれた場所・持っている人物・背景などを書きます。" },
  place: { label: "背景・場所", noun: "場所", example: "old library, bookshelves",
    desc: "特定の建物・部屋・風景を描けるようにします。人物がいないか、人物が小さく写った素材が向いています。",
    caption: "場所そのものは書かず、人物・時間帯・天気・光などを書きます。" },
  pose: { label: "ポーズ・動作・構図", noun: "ポーズ・構図", example: "jumping, arms up, from below",
    desc: "特定のポーズ・動き・カメラアングルを出せるようにします。いろいろな人物が同じポーズをとった素材を使います。",
    caption: "ポーズやアングルは書かず、人物・服・背景などを書きます。" },
  expression: { label: "表情", noun: "表情", example: "smug, half-closed eyes",
    desc: "特定の表情を、人物を問わず出せるようにします。顔がはっきり写った、いろいろな人物の素材を使います。",
    caption: "表情は書かず、人物・髪・服・背景などを書きます。" },
};
const conceptUI = (p) => CONCEPT_UI[(p && p.concept_target) || "body"] || CONCEPT_UI.body;
// a 質感・補強 LoRA made from words with its own two directions is what the スライダー card creates
const isSlider = (p) => p.kind === "enhance" && (p.enhance || {}).method === "words" && (p.enhance || {}).effect === "custom";
const kindLabel = (p) => p.kind === "concept" ? conceptUI(p).label : isSlider(p) ? "スライダー" : p.source === "multi" ? "複数キャラ" : KIND_LABEL[p.kind] || "スタイル";
const kindShort = (p) => p.kind === "concept" ? conceptUI(p).label : isSlider(p) ? "スライダー" : p.source === "multi" ? "複数キャラ" : KIND_SHORT[p.kind] || "スタイル";

function newForm(kind, models, source, target) {
  if (source === "generate") return newGenForm(kind);
  if (source === "multi") return newMultiForm(models);
  if (kind === "enhance") return newEnhanceForm(models, target === "slider");
  const ms = models.models.filter((m) => !m.missing.length);
  $("#newform").innerHTML = `<div class="panel" style="margin-bottom:20px">
    <h2>${kind === "concept" && target ? "衣装・物・場所・ポーズ・表情" : KIND_LABEL[kind]} LoRA を作成</h2>
    <div class="form">
      ${kind === "concept" && target ? `<label>覚えさせるもの</label><div class="radio">${Object.entries(CONCEPT_UI).filter(([k]) => k !== "body").map(([k, u]) =>
        `<label><input type="radio" name="nf-target" value="${k}" ${k === target ? "checked" : ""}> ${esc(u.label)} <span class="muted small">${esc(u.desc)}</span></label>`).join("")}</div>` : ""}
      <label>ベースモデル</label>
      <select id="nf-model">${ms.map((m) => `<option value="${m.id}">${esc(m.name)}（${esc(m.variant_info.label || m.variant)}）</option>`).join("")}
        <option value="">あとで選ぶ</option></select>
      ${kind === "character" ? `<label>衣装</label><div class="radio">
        <label><input type="radio" name="nf-outfit" value="include" checked> 衣装も含めて覚える <span class="muted small">呼び出し語だけでいつもの衣装になります</span></label>
        <label><input type="radio" name="nf-outfit" value="change"> 衣装を変えて使いたい <span class="muted small">衣装はプロンプトで指定します</span></label></div>` : ""}
      ${kind === "concept" ? `<label id="nf-concept-l">覚えさせたい${esc(CONCEPT_UI[target || "body"].noun)}</label><div><input type="text" id="nf-concept" placeholder="例: ${esc(CONCEPT_UI[target || "body"].example)}">
        <div class="muted small">英語のタグをカンマで区切って入れます。空欄でも作成でき、素材を入れたあとに候補から選べます。</div></div>` : ""}
      <label>プロジェクト名</label><input type="text" id="nf-name" placeholder="空欄なら自動で付けます">
    </div>
    <div class="row" style="margin-top:12px"><button class="primary big" id="nf-go">作成して素材を追加</button>
      <span class="muted small">呼び出し語は他の単語と重なりにくいものを自動で付けます（あとで変更できます）。</span></div></div>`;
  $$("input[name=nf-target]").forEach((r) => r.onchange = () => {
    const u = CONCEPT_UI[r.value];
    $("#nf-concept-l").textContent = `覚えさせたい${u.noun}`;
    $("#nf-concept").placeholder = `例: ${u.example}`;
  });
  $("#nf-go").onclick = async () => {
    const b = $("#nf-go"); b.disabled = true;
    try {
      const p = await api("/api/projects", { body: { kind, model_id: $("#nf-model").value || null, name: $("#nf-name").value,
        outfit_mode: ($("input[name=nf-outfit]:checked") || {}).value || "include",
        concept_words: $("#nf-concept") ? $("#nf-concept").value : undefined,
        concept_target: kind === "concept" ? (($("input[name=nf-target]:checked") || {}).value || "body") : undefined } });
      location.hash = `#/p/${p.id}/material`;
    } catch (e) { fail(e); b.disabled = false; }
  };
  $("#nf-name").focus();
}

/* Texture / enhancement LoRA: what to strengthen, and whether it is learned from image pairs or from words. */
async function enhanceCatalog() {
  if (!S.enhCat) S.enhCat = await api("/api/enhance/effects");
  return S.enhCat;
}

/* Multi-character LoRA: pick character projects; their material, captions and triggers are used as they are. */
async function newMultiForm(models) {
  let list;
  try { list = (await api("/api/projects")).filter((p) => p.kind === "character" && p.source !== "multi" && !p.error); } catch (e) { return fail(e); }
  const ms = models.models.filter((m) => !m.missing.length);
  $("#newform").innerHTML = `<div class="panel" style="margin-bottom:20px">
    <h2>複数キャラ LoRA を作成</h2>
    <div class="muted small">選んだキャラクター LoRA のプロジェクトの採用画像・説明文・呼び出し語を、そのまま使って 1 つの LoRA にします。
      素材や説明文を直すときは、それぞれのプロジェクトで直します。似たキャラどうしは混ざることがあるので、比較画像で確かめてください。</div>
    <div class="form" style="margin-top:10px">
      <label>まとめるキャラ</label><div>${list.length ? list.map((p) => `<label class="row"><input type="checkbox" class="nm-mem" value="${esc(p.id)}"> ${esc(p.name)}
        <span class="muted small">${p.images ?? 0} 枚</span></label>`).join("") : `<span class="muted">キャラクター LoRA のプロジェクトがまだありません。先にキャラごとに作ってください。</span>`}
        <div class="muted small">2 つ以上選びます（あとで足したり外したりできます）。</div></div>
      <label>ベースモデル</label>
      <select id="nf-model"><option value="">1 つ目のキャラと同じ</option>${ms.map((m) => `<option value="${m.id}">${esc(m.name)}（${esc(m.variant_info.label || m.variant)}）</option>`).join("")}</select>
      <label>プロジェクト名</label><input type="text" id="nf-name" placeholder="空欄なら自動で付けます">
    </div>
    <div class="row" style="margin-top:12px"><button class="primary big" id="nf-go">作成</button></div></div>`;
  $("#nf-go").onclick = async () => {
    const members = $$(".nm-mem:checked").map((c) => c.value);
    if (members.length < 2) return toast("キャラを 2 つ以上選んでください", "err");
    const b = $("#nf-go"); b.disabled = true;
    try {
      const p = await api("/api/projects", { body: { kind: "character", source: "multi", members, model_id: $("#nf-model").value || null, name: $("#nf-name").value } });
      location.hash = `#/p/${p.id}/material`;
    } catch (e) { fail(e); b.disabled = false; }
  };
}

/* The キャラ tab of a multi-character project: the member projects, each opened and edited on its own. */
async function renderMembers() {
  const p = S.project;
  let list = [];
  try { list = (await api("/api/projects")).filter((x) => x.kind === "character" && x.source !== "multi" && !x.error && !(p.members || []).includes(x.id)); } catch { /* the add list stays empty */ }
  const mem = p.member_info || [];
  $("#tab").innerHTML = `<div class="panel">
    <h2>まとめるキャラ</h2>
    <div class="muted small">各キャラの採用画像・説明文・呼び出し語を、学習を始めた時点の内容でそのまま使います。素材や呼び出し語はそれぞれのプロジェクトで直してください。</div>
    <table class="help-table"><thead><tr><th>キャラ</th><th>呼び出し語</th><th>採用画像</th><th></th></tr></thead><tbody>
    ${mem.map((m) => `<tr><td>${esc(m.name)}${m.missing ? ` <span class="err small">見つかりません</span>` : ""}</td><td>${esc(m.trigger || "")}</td>
      <td>${m.missing ? "" : `${m.n_used} 枚${m.n_used < 3 ? ` <span class="warntext small">3 枚以上必要</span>` : ""}`}</td>
      <td class="row">${m.missing ? "" : `<a href="#/p/${esc(m.pid)}/material"><button>開く</button></a>`}<button data-mrm="${esc(m.pid)}">外す</button></td></tr>`).join("")}
    </tbody></table>
    <div class="row" style="margin-top:10px">${list.length ? `<select id="mm-add">${list.map((x) => `<option value="${esc(x.id)}">${esc(x.name)}</option>`).join("")}</select>
      <button id="mm-addgo">追加</button>` : `<span class="muted small">追加できるキャラクター LoRA のプロジェクトはありません。</span>`}</div>
  </div>
  <div class="row" style="margin-top:12px"><a href="#/p/${esc(p.id)}/train"><button class="primary big">学習へ</button></a></div>`;
  const save = async (members, msg) => {
    try { S.project = await api(`/api/projects/${p.id}`, { method: "PATCH", body: { members } }); toast(msg); renderMembers(); } catch (e) { fail(e); }
  };
  $$("[data-mrm]").forEach((b) => b.onclick = () => save((S.project.members || []).filter((x) => x !== b.dataset.mrm), "キャラを外しました（次の学習から反映）"));
  if ($("#mm-addgo")) $("#mm-addgo").onclick = () => save([...(S.project.members || []), $("#mm-add").value], "キャラを追加しました（次の学習から反映）");
}

/* slider: the same "words" method with "自分で決める", opened with the two directions of the slider to fill in. */
async function newEnhanceForm(models, slider = false) {
  let cat;
  try { cat = await enhanceCatalog(); } catch (e) { return fail(e); }
  const ms = models.models.filter((m) => !m.missing.length && cat.families.includes(m.family));
  const effOpts = (method) => Object.entries(cat.effects).filter(([k]) => method === "words" || k !== "custom")
    .map(([k, e]) => `<option value="${k}">${esc(e.label)}</option>`).join("");
  $("#newform").innerHTML = `<div class="panel" style="margin-bottom:20px">
    <h2>${slider ? "スライダー" : "質感・補強"} LoRA を作成</h2>
    <div class="muted small">${slider ? "画像を使わず、言葉だけで作ります。強さをプラスにすると「プラス側」に、マイナスにすると「マイナス側」に寄ります。"
      + "モデルがもともと描けるものの間でしか動かせません。" : "描く内容を問わず、選んだ質感を強める LoRA です。"}試験的な機能で、初期設定はまだこのツールで検証していません。今は SDXL 系のモデルだけで作れます。</div>
    <div class="form" style="margin-top:10px">
      <label>作り方</label><div class="radio">
        <label><input type="radio" name="ne-method" value="pair" checked> ${esc(cat.methods.pair)} <span class="muted small">質感がよく写った画像を入れると、質感を弱めた画像をツールが作り、その違いだけを覚えます</span></label>
        <label><input type="radio" name="ne-method" value="words"> ${esc(cat.methods.words)} <span class="muted small">画像は使いません。モデルがもともと知っている質感を、強めたり弱めたりできるスライダーになります</span></label></div>
      <label>強めたいもの</label><div><select id="ne-effect">${effOpts("pair")}</select><div class="muted small" id="ne-desc"></div></div>
      <div id="ne-words" style="display:none">
        <label>${slider ? "プラス側（英語）" : "強めたいもの（英語）"}</label><input type="text" id="ne-pos" ${slider ? `placeholder="例: old, wrinkles, grey hair"` : ""}>
        <label>${slider ? "マイナス側（英語）" : "弱めたいもの（英語）"}</label><input type="text" id="ne-neg" ${slider ? `placeholder="例: young, smooth skin"` : ""}>
        <label>対象（英語・任意）</label><div><input type="text" id="ne-subj"><div class="muted small">例: person, portrait。空欄なら何にでも効くように学習します。</div></div>
      </div>
      <label>ベースモデル</label>
      <select id="nf-model">${ms.map((m) => `<option value="${m.id}">${esc(m.name)}（${esc(m.variant_info.label || m.variant)}）</option>`).join("")}
        <option value="">あとで選ぶ</option></select>
      <label>プロジェクト名</label><input type="text" id="nf-name" placeholder="空欄なら自動で付けます">
    </div>
    <div class="row" style="margin-top:12px"><button class="primary big" id="nf-go">作成</button></div></div>`;
  const method = () => $("input[name=ne-method]:checked").value;
  const fill = () => {
    const e = cat.effects[$("#ne-effect").value];
    $("#ne-desc").textContent = e.desc;
    $("#ne-pos").value = e.positive; $("#ne-neg").value = e.negative; $("#ne-subj").value = e.subject;
  };
  $$("input[name=ne-method]").forEach((r) => r.onchange = () => {
    const keep = $("#ne-effect").value;
    $("#ne-effect").innerHTML = effOpts(method());
    if (cat.effects[keep] && (method() === "words" || keep !== "custom")) $("#ne-effect").value = keep;
    $("#ne-words").style.display = method() === "words" ? "contents" : "none";
    $("#nf-go").textContent = method() === "words" ? "作成" : "作成して素材を追加";
    fill();
  });
  $("#ne-effect").onchange = fill;
  fill();
  $("#nf-go").textContent = "作成して素材を追加";
  if (slider) {
    // words + custom, and only the slider fields: the method and effect choices belong to 質感・補強
    $("input[name=ne-method][value=words]").checked = true;
    $("input[name=ne-method][value=words]").dispatchEvent(new Event("change"));
    $("#ne-effect").value = "custom"; fill();
    $$(".form > label", $("#newform")).slice(0, 2).forEach((l) => { l.style.display = "none"; l.nextElementSibling.style.display = "none"; });
    $("#ne-pos").focus();
  }
  $("#nf-go").onclick = async () => {
    const b = $("#nf-go"); b.disabled = true;
    const m = method();
    const enhance = { method: m, effect: $("#ne-effect").value };
    if (m === "words") Object.assign(enhance, { positive: $("#ne-pos").value, negative: $("#ne-neg").value, subject: $("#ne-subj").value });
    try {
      const p = await api("/api/projects", { body: { kind: "enhance", model_id: $("#nf-model").value || null, name: $("#nf-name").value, enhance } });
      location.hash = `#/p/${p.id}/${m === "words" ? "train" : "material"}`;
    } catch (e) { fail(e); b.disabled = false; }
  };
}

function newGenForm(kind = "character") {
  $("#newform").innerHTML = `<div class="panel" style="margin-bottom:20px">
    <h2>AI 生成画像から${kind === "style" ? "スタイル" : "キャラ"} LoRA を作成</h2>
    <div class="muted small">次の画面で、生成に使うモデル・LoRA・プロンプトを入れて素材を生成します。</div>
    <div class="form" style="margin-top:10px"><label>プロジェクト名</label><input type="text" id="nf-name" placeholder="空欄なら自動で付けます"></div>
    <div class="row" style="margin-top:12px"><button class="primary big" id="nf-go">作成</button></div></div>`;
  $("#nf-go").onclick = async () => {
    const b = $("#nf-go"); b.disabled = true;
    try {
      const p = await api("/api/projects", { body: { kind, source: "generate", name: $("#nf-name").value } });
      location.hash = `#/p/${p.id}/material`;
    } catch (e) { fail(e); b.disabled = false; }
  };
  $("#nf-name").focus();
}

// ------------------------------------------------------------------ project shell
async function openProject(pid, tab) {
  if (!S.project || S.project.id !== pid) {
    try { S.project = await api(`/api/projects/${pid}`); } catch (e) { fail(e); location.hash = "#/"; return; }
    S.images = []; S.sel.clear(); S.focus = null; S.detail = null; S.resultsRun = null; S.resultsEv = null; S.picked = null;
    _imgReq++; _imgBusy = false; _imgAgain = false;   // a list still on its way belongs to the previous project
  }
  const p = S.project;
  $("#crumbs").innerHTML = `<span>›</span><span>${esc(p.name)}</span>`;
  main.innerHTML = `
    <div class="phead">
      <input class="name" id="ph-name" value="${esc(p.name)}" title="プロジェクト名（クリックで編集）">
      <span class="kind">${esc(kindLabel(p))}</span>
      ${p.kind === "enhance" ? `<span class="muted small">作り方</span><select id="ph-emethod"></select>
        <span class="muted small">強めたいもの</span><select id="ph-effect"></select>` :
        p.source === "multi" ? `<span class="muted small">呼び出し語はキャラごと（${esc((p.member_info || []).map((m) => m.trigger).filter(Boolean).join(" / "))}）</span>` :
        `<span class="muted small">呼び出し語</span><input type="text" class="trig" id="ph-trig" value="${esc(p.trigger)}">`}
      ${p.kind === "character" && p.source !== "multi" ? `<select id="ph-outfit"><option value="include" ${p.outfit_mode !== "change" ? "selected" : ""}>衣装も含めて覚える</option>
        <option value="change" ${p.outfit_mode === "change" ? "selected" : ""}>衣装を変えて使いたい</option></select>` : ""}
      ${p.kind === "concept" ? `<span class="muted small">覚えさせたい${esc(conceptUI(p).noun)}</span><input type="text" class="trig" id="ph-concept" style="min-width:200px"
        value="${esc((p.concept_words || []).join(", "))}" placeholder="例: ${esc(conceptUI(p).example)}" title="英語のタグをカンマ区切りで入れます。説明文からこのタグを外し、呼び出し語に覚えさせます">` : ""}
      <label class="muted small">LoRA を作るモデル <select id="ph-model"><option value="">${p.model ? esc(p.model.name) : "選んでください"}</option></select></label>
    </div>
    <nav class="tabs">
      <a href="#/p/${pid}/material" class="${tab === "material" ? "on" : ""}">${p.kind === "enhance" && (p.enhance || {}).method === "words" ? "言葉" : p.source === "multi" ? "キャラ" : "素材"}</a>
      <a href="#/p/${pid}/train" class="${tab === "train" ? "on" : ""}">学習</a>
      <a href="#/p/${pid}/results" class="${tab === "results" ? "on" : ""}">結果</a>
    </nav>
    <div id="tab"></div>`;
  const saveMeta = async (body, okmsg) => {
    try {
      S.project = await api(`/api/projects/${pid}`, { method: "PATCH", body }); if (okmsg) toast(okmsg);
      // a tab clicked while saving redrew the header from the old values
      const put = (id, v) => { const x = $(id); if (x && x !== document.activeElement && S.project.id === pid) x.value = v; };
      put("#ph-name", S.project.name); put("#ph-trig", S.project.trigger || "");
      put("#ph-concept", (S.project.concept_words || []).join(", "));
    } catch (e) { fail(e); }
  };
  $("#ph-name").onchange = (e) => saveMeta({ name: e.target.value }, "名前を保存しました");
  if ($("#ph-trig")) $("#ph-trig").onchange = (e) => saveMeta({ trigger: e.target.value }, "呼び出し語を変更しました（説明文を更新します）");
  if (p.kind === "enhance") drawEnhanceHeader(pid, saveMeta);
  if ($("#ph-concept")) $("#ph-concept").onchange = (e) => saveMeta({ concept_words: e.target.value }, `覚えさせたい${conceptUI(p).noun}を変更しました（自動説明文を作り直します）`);
  if ($("#ph-outfit")) $("#ph-outfit").onchange = (e) => saveMeta({ outfit_mode: e.target.value }, "衣装の扱いを変更しました（自動説明文を作り直します）");
  // the base model the LoRA is trained on (also for projects created with 「あとで選ぶ」); used from the next training
  api("/api/models").then((d) => {
    const sel = $("#ph-model");
    if (!sel || S.project.id !== pid) return;
    const usable = d.models.filter((m) => !m.missing.length && m.variant_info.train !== false && m.variant_info.available !== false);
    sel.innerHTML = (S.project.model_id ? "" : `<option value="">選んでください</option>`) + usable.map((m) =>
      `<option value="${m.id}" ${m.id === S.project.model_id ? "selected" : ""}>${esc(m.name)}（${esc(m.variant_info.label || m.family)}）</option>`).join("");
    sel.onchange = async () => {
      await saveMeta({ model_id: sel.value || null }, "LoRA を作るモデルを変えました（次の学習から使われます）");
      if ((S.project.reset_settings || []).length) toast(`前のモデル向けの設定（${S.project.reset_settings.join("、")}）を初期値に戻しました`);
      if ((S.route[2] || "material") === "train") renderTrain();
    };
  }).catch(() => {});
  if (tab === "material") (p.kind === "enhance" && (p.enhance || {}).method === "words" ? renderEnhanceWords() : p.source === "multi" ? renderMembers() : renderMaterial());
  else if (tab === "train") renderTrain();
  else renderResults();
}

async function drawEnhanceHeader(pid, saveMeta) {
  let cat;
  try { cat = await enhanceCatalog(); } catch (e) { return fail(e); }
  const ms = $("#ph-emethod"), es = $("#ph-effect");
  if (!ms || S.project.id !== pid) return;
  const e = S.project.enhance || {};
  ms.innerHTML = Object.entries(cat.methods).map(([k, l]) => `<option value="${k}" ${k === e.method ? "selected" : ""}>${esc(l)}</option>`).join("");
  es.innerHTML = Object.entries(cat.effects).filter(([k]) => e.method === "words" || k !== "custom")
    .map(([k, x]) => `<option value="${k}" ${k === e.effect ? "selected" : ""}>${esc(x.label)}</option>`).join("");
  const reopen = () => openProject(pid, S.route[2] || "material");
  ms.onchange = async () => {
    await saveMeta({ enhance: { method: ms.value } }, "作り方を変えました（学習の詳細設定は初期値に戻りました）");
    reopen();
  };
  es.onchange = async () => {
    const x = cat.effects[es.value];
    const body = { effect: es.value };
    if (e.method === "words") Object.assign(body, { positive: x.positive, negative: x.negative, subject: x.subject });
    await saveMeta({ enhance: body }, e.method === "words" ? "強めたいものを変えました（言葉も入れ替えました）" : "強めたいものを変えました（次の学習から使われます）");
    reopen();
  };
}

/* Words method: no material images; the prompts are the whole input. */
function renderEnhanceWords() {
  const e = S.project.enhance || {};
  $("#tab").innerHTML = `<div class="panel" style="max-width:820px">
    <h2>言葉で決める</h2>
    <div class="muted small">この作り方では画像を使いません。下の言葉の違いを、モデル自身の描き方から学習します。モデルがもともと知らない質感は強められません。言葉は英語で、カンマで区切って入れます。</div>
    <div class="form" style="margin-top:10px">
      <label>強めたいもの</label><input type="text" id="ew-pos" value="${esc(e.positive)}">
      <label>弱めたいもの</label><input type="text" id="ew-neg" value="${esc(e.negative)}">
      <label>対象（任意）</label><div><input type="text" id="ew-subj" value="${esc(e.subject)}">
        <div class="muted small">例: person, portrait。空欄なら、人物・風景・物など何にでも効くように学習します。</div></div>
    </div>
    <div class="muted small" style="margin-top:8px">できた LoRA は、強さをプラスにすると「強めたいもの」の方向に、マイナスにすると「弱めたいもの」の方向に働きます。変えた内容は次の学習から使われます。</div>
    <div class="row" style="margin-top:10px"><a href="#/p/${S.project.id}/train"><button class="primary">学習の画面へ</button></a></div></div>`;
  const save = async () => {
    try {
      S.project = await api(`/api/projects/${S.project.id}`, { method: "PATCH", body: { enhance: {
        positive: $("#ew-pos").value, negative: $("#ew-neg").value, subject: $("#ew-subj").value } } });
      toast("保存しました（次の学習から使われます）");
    } catch (err) { fail(err); }
  };
  ["#ew-pos", "#ew-neg", "#ew-subj"].forEach((id) => $(id).onchange = save);
}

// ------------------------------------------------------------------ material
const VORDER = { exclude: 0, review: 1, hold: 2, unusable: 3, pending: 4, ok: 5 };
const FILTERS = [
  ["attention", "要確認を優先"], ["all", "すべて"], ["used", "採用"], ["unused", "不採用"],
  ["review", "要確認"], ["exclude", "除外候補"], ["unusable", "使用不能"], ["pending", "解析中"],
];

// natural-language models only: the VLM writes each caption (slow); the faster ways trade detail for time
function captionModeHtml() {
  const p = S.project;
  if ((p.caption_style !== "natural" && !p.caption_natural_ok) || p.kind === "enhance") return "";
  const mode = p.caption_mode || p.caption_mode_default || "ai";
  const opts = [["ai", "説明 AI が 1 枚ずつ文章で書く（詳しいが時間がかかる）"]];
  if (["character", "style"].includes(p.kind)) opts.push(["gen", "AI 生成の素材は生成時の指定から文章を作る（ほかは説明 AI）"]);
  opts.push(["tags", p.caption_natural_ok ? "タグを並べる（すぐ終わる）" : "解析のタグから作る（すぐ終わるが単調）"]);
  return `<label class="muted small" title="「生成時の指定から作る」は、指定したポーズや表情をそのまま書きます。モデルが指定どおりに描かなかった画像では、説明文と絵が食い違います。「解析のタグから作る」は、タグを並べた説明文になります。手で直した説明文は、どの方法でも上書きしません。">説明文の作り方
    <select id="m-capmode">${opts.map(([k, l]) => `<option value="${k}" ${mode === k ? "selected" : ""}>${l}</option>`).join("")}</select></label>`;
}

function renderMaterial() {
  $("#tab").innerHTML = `
    <div class="mat">
      <div>
        <div class="toolbar">
          <label class="row"><button class="primary" id="m-add">画像を追加</button><input type="file" id="m-file" multiple accept="image/*" hidden></label>
          <button id="m-folder">フォルダーから</button>
          <button id="m-tags-open" title="素材全体で、どのタグが何枚に付いているかを一覧にします">タグの一覧</button>
          ${["character", "style"].includes(S.project.kind) ? `<button id="m-gen-open">AI で生成して追加</button>` : ""}
          ${captionModeHtml()}
          <span class="muted small">またはドラッグ＆ドロップ、Ctrl+V で貼り付け</span>
          <div class="spacer"></div>
          <button id="m-undo" title="Ctrl+Z">元に戻す</button><button id="m-redo" title="Ctrl+Y">やり直す</button>
        </div>
        <div id="m-gen"></div>
        <div id="m-report"></div>
        <div id="m-tags"></div>
        <div class="toolbar"><div class="chips" id="m-filters"></div><div class="spacer"></div>
          <input type="text" id="m-search" placeholder="説明文・タグで絞り込み" style="width:180px"></div>
        <div class="toolbar" id="m-bulk"></div>
        <div class="grid" id="m-grid"></div>
      </div>
      <div class="detail panel" id="m-detail">${(S.detail = null, DETAIL_EMPTY)}</div>
    </div>
    <div class="startbar" id="m-start"></div>`;
  $("#m-add").onclick = () => $("#m-file").click();
  $("#m-file").onchange = (e) => { uploadFiles([...e.target.files]); e.target.value = ""; };
  $("#m-folder").onclick = folderImport;
  $("#m-tags-open").onclick = () => { S.tagsOpen = !S.tagsOpen; drawTagList(); };
  if ($("#m-capmode")) $("#m-capmode").onchange = async (e) => {
    try {
      S.project = await api(`/api/projects/${S.project.id}`, { method: "PATCH", body: { caption_mode: e.target.value } });
      toast("説明文の作り方を変えました。手で直した説明文はそのまま残し、ほかの自動説明文を作り直します");
      refreshImages();
    } catch (err) { fail(err); }
  };
  if ($("#m-gen-open")) $("#m-gen-open").onclick = () => { S.genOpen = !S.genOpen; renderGen(); };
  if (S.genFor !== S.project.id) { S.genFor = S.project.id; S.gen = null; S.genOpen = S.project.source === "generate"; }
  renderGen();
  $("#m-undo").onclick = () => undoRedo("undo");
  $("#m-redo").onclick = () => undoRedo("redo");
  $("#m-search").oninput = debounce((e) => { S.search = e.target.value.trim().toLowerCase(); drawGrid(); }, 150);
  $("#m-grid").addEventListener("click", onGridClick);
  $("#m-grid").addEventListener("contextmenu", onGridMenu);
  S.cards = new Map();
  refreshImages();
  refreshStartBar();
}

/* Source files moved away: the folder they are in now (that folder only; files are matched by content). */
async function relinkFolder() {
  const v = await ask([{ label: "見つからない元画像が今あるフォルダーのパス（このフォルダーだけを探します）", width: 380, placeholder: "例: D:\\images\\character" }],
    "探す", ".toolbar");
  const path = v && v[0].trim().replace(/^"|"$/g, "");
  if (!path) return;
  try {
    const r = await api(`/api/projects/${S.project.id}/relink`, { body: { path } });
    toast(r.still_missing ? `${r.fixed} 枚を戻しました。まだ ${r.still_missing} 枚が見つかりません` : `${r.fixed} 枚を戻しました`, r.fixed ? "" : "err");
    refreshImages();
  } catch (e) { fail(e); }
}

/* innerHTML only when the markup changed: a redraw that changes nothing keeps the elements, so a click
   that is half done (pressed, not released yet) is not lost. */
function setHTML(el, html) {
  if (!el || el._html === html) return false;
  el.innerHTML = html; el._html = html;
  return true;
}

async function folderImport() {
  const v = await ask([{ label: "フォルダーのパス（このフォルダーの画像だけを読み取ります）", width: 380, placeholder: "例: D:\\images\\character" }],
    "取り込む", ".toolbar");
  const path = v && v[0].trim().replace(/^"|"$/g, "");
  if (!path) return;
  try { const r = await api(`/api/projects/${S.project.id}/import_folder`, { body: { path } }); toast(`${r.queued} 枚を取り込みます`); }
  catch (e) { fail(e); }
}

async function uploadFiles(files) {
  if (S.project && S.project.source === "multi") return toast("複数キャラ LoRA には画像を直接入れられません。キャラごとのプロジェクトに入れてください", "err");
  files = files.filter((f) => /^image\//.test(f.type) || /\.(jpe?g|png|webp|bmp|tiff?|gif)$/i.test(f.name));
  if (!files.length) return toast("画像ファイルがありません", "err");
  toast(`${files.length} 枚を取り込み中…`);
  const pid = S.project.id;   // the rest still goes to this project when another one is opened meanwhile
  // send in small batches so the grid fills progressively
  for (let i = 0; i < files.length; i += 8) {
    const fd = new FormData();
    files.slice(i, i + 8).forEach((f) => fd.append("files", f, f.name));
    try { const r = await api(`/api/projects/${pid}/upload`, { body: fd }); if (S.project && S.project.id === pid) refreshImages(); importFailed(r.failed); }
    catch (e) { fail(e); break; }
  }
}

let _imgReq = 0, _imgBusy = false, _imgAgain = false, _imgLast = 0, _imgTimer = null;
/* Updates pushed while analysis runs (several per image): the whole list is rebuilt on the server for each
   request, so these come at most about once a second, and never while a request is still running. */
function refreshImagesBg() {
  if (_imgTimer) return;
  _imgTimer = setTimeout(() => { _imgTimer = null; refreshImages(); }, Math.max(120, 1000 - (Date.now() - _imgLast)));
}
let _imgFailed = false;   // one toast per failure streak (analysis asks about once a second)
const refreshImages = debounce(async () => {
  if (!S.project) return;
  if (_imgBusy) { _imgAgain = true; return; }   // asked again meanwhile: one more request when this one ends
  _imgBusy = true;
  const my = ++_imgReq, pid = S.project.id;
  let d;
  try { d = await api(`/api/projects/${pid}/images`); }
  catch (e) { if (my === _imgReq && !_imgFailed) { _imgFailed = true; fail(e); } return; }
  finally {
    if (my === _imgReq) _imgBusy = false;   // a project switch already freed it
    _imgLast = Date.now();
    if (_imgAgain && !_imgBusy) { _imgAgain = false; refreshImages(); }
  }
  _imgFailed = false;
  if (my !== _imgReq || !S.project || S.project.id !== pid) return;
  S.images = d.images; S.report = d.report; S.history = d.history; S.refs = d.reference_ids || [];
  if (!$("#m-grid")) return;
  drawReport(); drawGrid(); drawBulk(); drawUndo(); drawTagList();
  if (S.focus != null) {
    const im = S.images.find((x) => x.id === S.focus);
    if (im && (!S.detail || detailSig(im) !== S.detailSig)) loadDetail(S.focus, true);
  }
  refreshStartBar();
}, 120);

function drawUndo() {
  const u = $("#m-undo"), r = $("#m-redo");
  if (!u) return;
  u.disabled = !S.history.undo; r.disabled = !S.history.redo;
  u.title = S.history.undo ? `元に戻す: ${S.history.undo} (Ctrl+Z)` : "元に戻す操作はありません";
  r.title = S.history.redo ? `やり直す: ${S.history.redo} (Ctrl+Y)` : "";
}

/* MOSS-Image character LoRA: hide1's catalog (MOSS-Image SFT drawing names without a LoRA, not curated) may list the
   character the tagger finds on most images.  Information only; the list is fetched by the i2t worker after the user
   presses the button (the same file the i2t page uses). */
function mossCatalogNote(h) {
  if (!h) return "";
  const who = `${esc(h.tag)}（素材 ${h.of} 枚中 ${h.count} 枚）`;
  if (h.state === "missing") return `<div class="notes"><div class="note info">素材のキャラ ${who} が MOSS-Image のキャラ一覧（hide1 の MOSS-Image Character &amp; Style Catalog。LoRA なしで名前を指定した生成例）に載っているかを確かめられます。
      <button class="small" id="m-mosscat">キャラ一覧を取得して確かめる（約 ${h.size_mb}MB）</button></div></div>`;
  if (h.state === "listed") return `<div class="notes"><div class="note info">素材のキャラ ${who} は MOSS-Image のキャラ一覧に載っています（作品名 ${esc(h.series || "なし")}）。
      LoRA を使わず、プロンプトに「${esc(h.name)}${h.series ? " from " + esc(h.series) : ""}」と書くだけで描ける可能性があります。一覧は選別されていないため、載っていても描けるとは限りません。
      <a href="${esc(h.image)}" target="_blank" rel="noopener">一覧の生成例を見る</a></div></div>`;
  return "";
}

async function mossCatalogFetch(ev) {
  ev.target.disabled = true;
  try {
    await api("/api/i2t/job", { body: { op: "moss_catalog" } });
    toast("MOSS-Image のキャラ一覧を取得しています");
  } catch (e) { ev.target.disabled = false; fail(e); }
}

function drawReport() {
  const rep = S.report || { counts: {}, notes: [] };
  const c = rep.counts || {};
  const used = S.images.filter((i) => i.use).length;
  setHTML($("#m-filters"), FILTERS.map(([k, label]) => {
    const n = k === "attention" || k === "all" ? S.images.length : k === "used" ? used : k === "unused" ? S.images.length - used : (c[k] || 0);
    if (!["attention", "all", "used", "unused"].includes(k) && !n) return "";
    return `<span class="chip ${S.filter === k ? "on" : ""}" data-f="${k}">${label} ${n}</span>`;
  }).join(""));
  $$("#m-filters .chip").forEach((ch) => ch.onclick = () => { S.filter = ch.dataset.f; drawReport(); drawGrid(); });
  const sug = (rep.coverage || {}).concept_suggest || [];
  const nMissing = S.images.filter((i) => (i.checks || []).some((x) => x.key === "missing")).length;
  setHTML($("#m-report"), (nMissing ? `<div class="notes"><div class="note warn">元画像が見つからない画像が ${nMissing} 枚あります。移動したときは、今の場所を指定すると戻せます。
      <button class="small" id="m-relink">場所を再指定</button></div></div>` : "") + (sug.length ? `<div class="notes"><div class="note info">素材の多くに共通するタグです。覚えさせたい${esc(conceptUI(S.project).noun)}なら、押して追加してください。
      <div class="tags" style="margin-top:4px">${sug.map((x) => `<span class="chip" data-addc="${esc(x.tag)}" title="素材の ${Math.round(x.share * 100)}% に付いています">＋ ${esc(x.tag)}（${Math.round(x.share * 100)}%）</span>`).join("")}</div></div></div>` : "") +
    mossCatalogNote(rep.moss_catalog) +
    (rep.notes && rep.notes.length ? `<div class="notes">${rep.notes.map((n) =>
    `<div class="note ${n.level}">${esc(n.text)}</div>`).join("")}<div class="muted small">偏りの判定は推定です。</div></div>` : ""));
  if ($("#m-relink")) $("#m-relink").onclick = relinkFolder;
  if ($("#m-mosscat")) $("#m-mosscat").onclick = mossCatalogFetch;
  $$("[data-addc]", $("#m-report")).forEach((b) => b.onclick = async () => {
    const words = (S.project.concept_words || []).concat([b.dataset.addc]);
    try {
      S.project = await api(`/api/projects/${S.project.id}`, { method: "PATCH", body: { concept_words: words } });
      if ($("#ph-concept")) $("#ph-concept").value = S.project.concept_words.join(", ");
      toast(`「${b.dataset.addc}」を覚えさせたい${conceptUI(S.project).noun}に加えました（自動説明文を作り直します）`);
      refreshImages();
    } catch (e) { fail(e); }
  });
}

/* Tag list of the whole material: how many images carry each tag, with delete / replace / select for all
   of them at once.  Counted in the browser from the captions already loaded; edits go through bulk_caption
   and are undoable like any other caption edit. */
const TL = { scope: "used", sort: "count", q: "", picked: new Set(), limit: 150 };
function tagIndex() {
  const imgs = S.images.filter((i) => TL.scope === "all" || i.use);
  const m = new Map();
  for (const im of imgs) for (const t of new Set((im.caption || "").split(",").map((x) => x.trim()).filter(Boolean))) {
    if (!m.has(t)) m.set(t, []);
    m.get(t).push(im.id);
  }
  return { n: imgs.length, tags: [...m].map(([tag, ids]) => ({ tag, ids })) };
}
function drawTagList() {
  const el = $("#m-tags"), btn = $("#m-tags-open");
  if (!el) return;
  if (btn) btn.classList.toggle("on", !!S.tagsOpen);
  if (!S.tagsOpen) { el.innerHTML = ""; return; }
  if (S.project.caption_style === "natural") {
    el.innerHTML = `<div class="panel tl"><b>タグの一覧</b><div class="muted small" style="margin-top:4px">このモデルの説明文は文章なので、タグに分けて数えられません。
      説明文の語句は、上の「説明文・タグで絞り込み」と、選んだ画像の「置換」で直せます。</div></div>`;
    return;
  }
  const { n, tags } = tagIndex();
  const trig = (S.project.trigger || "").trim();
  const q = TL.q.toLowerCase();
  let list = tags.filter((t) => !q || t.tag.toLowerCase().includes(q));
  list.sort(TL.sort === "name" ? (a, b) => a.tag.localeCompare(b.tag) : (a, b) => (b.ids.length - a.ids.length) || a.tag.localeCompare(b.tag));
  for (const t of [...TL.picked]) if (!tags.some((x) => x.tag === t)) TL.picked.delete(t);
  const shown = list.slice(0, TL.limit);
  const focusQ = document.activeElement && document.activeElement.id === "tl-q";
  const scroll = el.querySelector(".tl-list") ? el.querySelector(".tl-list").scrollTop : 0;   // redrawn while analysis runs
  el.innerHTML = `<div class="panel tl">
    <div class="row"><b>タグの一覧</b>
      <span class="chips">${[["used", "採用中の画像"], ["all", "すべての画像"]].map(([k, l]) => `<span class="chip ${TL.scope === k ? "on" : ""}" data-tls="${k}">${l}</span>`).join("")}</span>
      <span class="chips">${[["count", "多い順"], ["name", "名前順"]].map(([k, l]) => `<span class="chip ${TL.sort === k ? "on" : ""}" data-tlo="${k}">${l}</span>`).join("")}</span>
      <input type="text" id="tl-q" placeholder="タグを探す" value="${esc(TL.q)}" style="width:150px">
      <div class="spacer"></div><button class="small" id="tl-close">閉じる</button></div>
    <div class="muted small" style="margin:4px 0">${n} 枚の説明文に、${tags.length} 種類のタグがあります。削除・置き換えは、そのタグが付いた${TL.scope === "all" ? "すべての" : "採用中の"}画像に効きます（元に戻せます）。</div>
    ${TL.picked.size ? `<div class="row small" style="margin-bottom:4px"><span>${TL.picked.size} 個のタグを選択中：</span>
      <button class="small" data-tlb="del">まとめて削除</button><button class="small" data-tlb="sel">付いている画像を選ぶ</button>
      <button class="small" data-tlb="clear">選択を解除</button></div>` : ""}
    <div class="tl-list">${shown.map((t) => {
      const isTrig = !!trig && (t.tag === trig || t.tag.startsWith(trig + " "));
      const pct = n ? Math.round(t.ids.length / n * 100) : 0;
      return `<div class="tl-row" data-tag="${esc(t.tag)}">
        <input type="checkbox" data-tlp ${TL.picked.has(t.tag) ? "checked" : ""} ${isTrig ? "disabled" : ""}>
        <span class="tl-tag" title="${esc(t.tag)}">${esc(t.tag)}${isTrig ? ` <span class="kind">呼び出し語</span>` : ""}</span>
        <span class="tl-bar"><span style="width:${pct}%"></span></span>
        <span class="tl-n">${t.ids.length} 枚</span>
        <button class="small" data-tla="sel" title="このタグが付いた画像を選びます">選ぶ</button>
        ${isTrig ? "" : `<button class="small" data-tla="rep">置き換え</button><button class="small" data-tla="del">削除</button>`}</div>`; }).join("")}</div>
    ${list.length > shown.length ? `<button class="small" id="tl-more" style="margin-top:6px">さらに表示（残り ${list.length - shown.length} 個）</button>` : ""}
    ${!list.length ? `<div class="muted small">${tags.length ? "一致するタグはありません。" : "説明文にタグがありません。"}</div>` : ""}</div>`;
  if (focusQ) { const qi = $("#tl-q"); qi.focus(); qi.setSelectionRange(qi.value.length, qi.value.length); }
  if ($(".tl-list", el)) $(".tl-list", el).scrollTop = scroll;
  const idsOf = (tagList) => [...new Set(tags.filter((t) => tagList.includes(t.tag)).flatMap((t) => t.ids))];
  const select = (ids) => {
    S.sel = new Set(ids); S.filter = "all"; S.search = ""; if ($("#m-search")) $("#m-search").value = "";
    drawReport(); drawGrid(); drawBulk();
    toast(`${ids.length} 枚を選びました。上の一括操作で、採用・除外や説明文の編集ができます。`);
  };
  const run = async (body, msg) => {
    try {
      const r = await api(`/api/projects/${S.project.id}/bulk_caption`, { body });
      toast(msg(r.changed), "", { label: "元に戻す", fn: () => undoRedo("undo") });
      TL.picked.clear();
      refreshImages();
    } catch (e) { fail(e); }
  };
  $$("[data-tls]", el).forEach((c) => c.onclick = () => { TL.scope = c.dataset.tls; drawTagList(); });
  $$("[data-tlo]", el).forEach((c) => c.onclick = () => { TL.sort = c.dataset.tlo; drawTagList(); });
  $("#tl-q").oninput = debounce((e) => { TL.q = e.target.value.trim(); TL.limit = 150; drawTagList(); }, 200);
  $("#tl-close").onclick = () => { S.tagsOpen = false; drawTagList(); };
  if ($("#tl-more")) $("#tl-more").onclick = () => { TL.limit += 300; drawTagList(); };
  $$("[data-tlp]", el).forEach((c) => c.onchange = () => { const t = c.closest("[data-tag]").dataset.tag; c.checked ? TL.picked.add(t) : TL.picked.delete(t); drawTagList(); });
  $$("[data-tlb]", el).forEach((b) => b.onclick = () => {
    const picked = [...TL.picked];
    if (b.dataset.tlb === "clear") { TL.picked.clear(); drawTagList(); }
    else if (b.dataset.tlb === "sel") select(idsOf(picked));
    else run({ ids: idsOf(picked), op: "remove", tags: picked.join(",") }, (n) => `${picked.length} 個のタグを ${n} 枚の説明文から削除しました`);
  });
  $$("[data-tla]", el).forEach((b) => b.onclick = async () => {
    const tag = b.closest("[data-tag]").dataset.tag;
    const ids = idsOf([tag]);
    if (b.dataset.tla === "sel") return select(ids);
    if (b.dataset.tla === "del") return run({ ids, op: "remove", tags: tag }, (n) => `「${tag}」を ${n} 枚の説明文から削除しました`);
    const v = await ask([{ label: `「${tag}」を置き換えるタグ（空欄で削除）`, value: tag, width: 260 }], "置き換え", "#m-tags");
    if (!v || v[0].trim() === tag) return;
    if (v[0].includes(",")) return toast("置き換えるタグは 1 つにしてください（カンマは使えません）", "err");
    run({ ids, op: "replace_tag", find: tag, replace: v[0].trim() }, (n) => `${n} 枚の「${tag}」を${v[0].trim() ? `「${v[0].trim()}」に置き換えました` : "削除しました"}`);
  });
}

function visibleImages() {
  let list = S.images.slice();
  const f = S.filter;
  if (f === "used") list = list.filter((i) => i.use);
  else if (f === "unused") list = list.filter((i) => !i.use);
  else if (!["attention", "all"].includes(f)) list = list.filter((i) => i.verdict === f);
  if (S.search) list = list.filter((i) => (i.caption + " " + i.tags.join(" ") + " " + i.name).toLowerCase().includes(S.search));
  if (f === "attention") list.sort((a, b) => (VORDER[a.verdict] - VORDER[b.verdict]) || (a.id - b.id));
  return list;
}

function cardSig(im) {
  return [im.verdict, im.use, im.decision, im.crop, im.fast_status, im.ai_status, im.caption_status, S.sel.has(im.id), S.focus === im.id,
    im.caption.slice(0, 40), S.refs.includes(im.id), im.checks.filter((c) => c.status === "warn" || c.status === "bad").length].join("|");
}

function cardHtml(im) {
  const thumb = im.fast_status === "pending" ? `<span class="spin"></span>` :
    im.fast_status === "failed" && !im.w ? `<span class="muted small">読めません</span>` :
    `<img loading="lazy" decoding="async" src="/api/projects/${S.project.id}/thumb/${im.id}?v=${im.fast_status}" alt="">`;
  const busy = im.ai_status === "running" || im.caption_status === "running" ? ` <span class="spin"></span>` : "";
  const nIssues = im.checks.filter((c) => c.status === "warn" || c.status === "bad").length;
  return `<div class="imgw">${thumb}</div>
    <span class="badge b-${im.verdict}">${esc(im.verdict_label)}${nIssues ? " " + nIssues : ""}</span>
    ${im.decision ? `<span class="man">${im.decision === "use" ? "手動:採用" : "手動:除外"}</span>` : S.refs.includes(im.id) ? `<span class="man">基準</span>` : ""}
    <div class="meta">${im.crop ? "✂ " : ""}${esc(im.caption || im.name)}${busy}</div>`;
}

function drawGrid() {
  const grid = $("#m-grid");
  if (!grid) return;
  const list = visibleImages();
  const keep = new Set(list.map((i) => i.id));
  for (const el of [...grid.children]) if (!el.classList.contains("card")) el.remove();  // "no images" message
  for (const [id, el] of S.cards) if (!keep.has(id)) { el.remove(); S.cards.delete(id); }
  let prev = null;
  for (const im of list) {
    let el = S.cards.get(im.id);
    const sig = cardSig(im);
    if (!el) {
      el = document.createElement("div");
      el.dataset.id = im.id;
      S.cards.set(im.id, el);
    }
    if (el._sig !== sig) {
      el.className = "card" + (im.use ? "" : " unused") + (S.sel.has(im.id) ? " sel" : "") + (S.focus === im.id ? " focus" : "");
      el.innerHTML = cardHtml(im);
      el._sig = sig;
    }
    const want = prev ? prev.nextSibling : grid.firstChild;
    if (want !== el) grid.insertBefore(el, want);
    prev = el;
  }
  if (!list.length) grid.innerHTML = S.images.length ? `<div class="muted">この条件の画像はありません。</div>` :
    `<div class="muted" style="padding:30px">画像をここにドラッグ＆ドロップするか、Ctrl+V で貼り付けるか、「画像を追加」を押してください。取り込みと同時に確認が始まります。</div>`;
  if (!list.length) S.cards.clear();
}

function onGridClick(e) {
  const card = e.target.closest(".card");
  if (!card) return;
  const id = +card.dataset.id;
  const list = visibleImages().map((i) => i.id);
  if (e.shiftKey && S.lastClick != null) {
    const a = list.indexOf(S.lastClick), b = list.indexOf(id);
    if (a >= 0 && b >= 0) list.slice(Math.min(a, b), Math.max(a, b) + 1).forEach((x) => S.sel.add(x));
  } else if (e.ctrlKey || e.metaKey) {
    S.sel.has(id) ? S.sel.delete(id) : S.sel.add(id);
  } else {
    S.sel.clear(); S.sel.add(id);
  }
  S.lastClick = id;
  S.focus = id;
  drawGrid(); drawBulk();
  loadDetail(id);
}

function drawBulk() {
  const n = S.sel.size;
  const el = $("#m-bulk");
  if (!el) return;
  setHTML(el, `
    <button data-b="selall">全選択</button>${n ? `<button data-b="selnone">選択解除</button>` : ""}
    <span class="muted small">${n ? `${n} 枚を選択中：` : "Ctrl / Shift クリックで複数選択"}</span>
    ${n ? `<button data-b="use">採用</button><button data-b="exclude">除外</button><button data-b="auto">自動判定に従う</button>
      <button data-b="addtag">説明文に追記</button><button data-b="rmtag">語句を削除</button><button data-b="replace">置換</button>
      <button data-b="resetcap">自動説明文に戻す</button><button data-b="autocrop">自動切り抜き</button><button data-b="reanalyze">再解析</button>
      <button data-b="remove" class="danger" title="Delete キー・右クリックでも削除できます">削除</button>` : ""}`);
  $$("[data-b]", el).forEach((b) => b.onclick = () => bulk(b.dataset.b));
}

async function bulk(op) {
  const pid = S.project.id;
  const ids = [...S.sel];
  try {
    if (op === "selall") { visibleImages().forEach((i) => S.sel.add(i.id)); drawGrid(); drawBulk(); return; }
    if (op === "selnone") { S.sel.clear(); drawGrid(); drawBulk(); return; }
    await flushCaption();
    if (op === "remove") return removeImages(ids);
    if (["use", "exclude", "auto"].includes(op)) {
      const value = op === "auto" ? null : op;
      optimistic(ids, (im) => { im.decision = value; im.use = value ? value === "use" : im.verdict !== "exclude" && im.verdict !== "unusable"; });
      const r = await api(`/api/projects/${pid}/edit`, { body: { label: `${{ use: "採用", exclude: "除外", auto: "自動判定に戻す" }[op]}（${ids.length}枚）`,
        changes: ids.map((id) => ({ id, field: "decision", value })) } });
      S.history = r.history; drawUndo();
    } else if (op === "addtag" || op === "rmtag") {
      const v = await ask([{ label: op === "addtag" ? `${ids.length}枚に追記する語句（カンマ区切り）` : `${ids.length}枚から削除する語句（カンマ区切り・完全一致）` }],
        op === "addtag" ? "追記" : "削除");
      const tags = v && v[0].trim();
      if (!tags) return;
      const r = await api(`/api/projects/${pid}/bulk_caption`, { body: { ids, op: op === "addtag" ? "add" : "remove", tags } });
      toast(`${r.changed} 枚の説明文を変更しました`, "", { label: "元に戻す", fn: () => undoRedo("undo") });
    } else if (op === "replace") {
      const v = await ask([{ label: "検索", width: 180 }, { label: "置換後（空欄で削除）", width: 180 }], "置換");
      if (!v || !v[0]) return;
      const [find, rep] = v;
      const r = await api(`/api/projects/${pid}/bulk_caption`, { body: { ids, op: "replace", find, replace: rep } });
      toast(`${r.changed} 枚を置換しました`, "", { label: "元に戻す", fn: () => undoRedo("undo") });
    } else if (op === "resetcap") {
      const r = await api(`/api/projects/${pid}/bulk_caption`, { body: { ids, op: "reset_auto" } });
      toast(`${r.changed} 枚を自動説明文に戻しました`, "", { label: "元に戻す", fn: () => undoRedo("undo") });
    } else if (op === "autocrop") {
      const r = await api(`/api/projects/${pid}/auto_crop`, { body: { ids } });
      toast(r.changed ? `${r.changed} 枚を切り抜きました（再判定します）` : "切り抜き候補のある画像がありません");
    } else if (op === "reanalyze") {
      await api(`/api/projects/${pid}/reanalyze`, { body: { ids, force: true } });
      toast(`${ids.length} 枚を再解析します`);
    }
    refreshImages();
  } catch (e) { fail(e); refreshImages(); }
}

// Removing takes the image out of the project (undoable, no confirmation).  The original file and
// the project's copy stay on disk, so Ctrl+Z / 元に戻す brings it back with its edits.
const DETAIL_EMPTY = `<div class="muted">画像を選ぶと、判定の理由・切り抜き・説明文を確認・修正できます。</div>`;

async function removeImages(ids) {
  if (!ids.length) return;
  const gone = new Set(ids);
  S.images = S.images.filter((i) => !gone.has(i.id));
  ids.forEach((id) => S.sel.delete(id));
  if (gone.has(S.focus)) { S.focus = null; S.detail = null; const box = $("#m-detail"); if (box) box.innerHTML = DETAIL_EMPTY; }
  drawGrid(); drawBulk(); drawReport();
  try {
    const r = await api(`/api/projects/${S.project.id}/edit`, { body: { label: `削除（${ids.length}枚）`,
      changes: ids.map((id) => ({ id, field: "removed", value: 1 })) } });
    S.history = r.history; drawUndo();
    toast(`${ids.length} 枚をプロジェクトから削除しました（元の画像ファイルは残っています）`, "", { label: "元に戻す", fn: () => undoRedo("undo") });
  } catch (e) { fail(e); }
  refreshImages();
}

function closeMenu() { const m = $("#ctxmenu"); if (m) m.remove(); }

function onGridMenu(e) {
  const card = e.target.closest(".card");
  if (!card) return;
  e.preventDefault();
  const id = +card.dataset.id;
  if (!S.sel.has(id)) { S.sel.clear(); S.sel.add(id); S.lastClick = id; S.focus = id; drawGrid(); drawBulk(); loadDetail(id); }
  const ids = [...S.sel];
  const n = ids.length > 1 ? `${ids.length} 枚を` : "";
  closeMenu();
  const m = document.createElement("div");
  m.id = "ctxmenu"; m.className = "ctxmenu";
  m.innerHTML = `<button data-m="use">${n}採用</button><button data-m="exclude">${n}除外</button>
    <button data-m="auto">${n}自動判定に従う</button><hr>
    <button data-m="remove" class="danger">${n}削除<span class="key">Delete</span></button>`;
  document.body.append(m);
  const r = m.getBoundingClientRect();
  m.style.left = Math.min(e.clientX, innerWidth - r.width - 8) + "px";
  m.style.top = Math.min(e.clientY, innerHeight - r.height - 8) + "px";
  m.onclick = (ev) => {
    const b = ev.target.closest("[data-m]");
    if (!b) return;
    closeMenu();
    b.dataset.m === "remove" ? removeImages(ids) : bulk(b.dataset.m);
  };
}
window.addEventListener("mousedown", (e) => { if (!e.target.closest("#ctxmenu, [data-pmore]")) closeMenu(); });
window.addEventListener("blur", closeMenu);
window.addEventListener("scroll", closeMenu, true);

function optimistic(ids, fn) {
  ids.forEach((id) => { const im = S.images.find((x) => x.id === id); if (im) fn(im); });
  drawGrid(); drawReport();
}

async function undoRedo(which) {
  await flushCaption();   // an unsaved caption is the last edit: saved before it is undone or redone over
  try {
    const r = await api(`/api/projects/${S.project.id}/${which}`, { body: {} });
    const what = r.undone || r.redone;
    toast(what ? `${which === "undo" ? "元に戻しました" : "やり直しました"}: ${what.label}` : "対象の操作がありません");
    S.detail = null;
    refreshImages();
  } catch (e) { fail(e); }
}

document.addEventListener("keydown", (e) => {
  if (S.route[0] !== "p" || (S.route[2] || "material") !== "material") return;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undoRedo(e.shiftKey ? "redo" : "undo"); }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") { e.preventDefault(); undoRedo("redo"); }
  if (e.key === "Delete" && S.sel.size) { e.preventDefault(); closeMenu(); removeImages([...S.sel]); }
  if (e.key === "Escape") closeMenu();
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const list = visibleImages().map((i) => i.id);
    const i = list.indexOf(S.focus);
    const nx = list[Math.max(0, Math.min(list.length - 1, i + (e.key === "ArrowRight" ? 1 : -1)))];
    if (nx != null) { S.sel.clear(); S.sel.add(nx); S.focus = nx; drawGrid(); drawBulk(); loadDetail(nx); }
  }
});

// ---- detail panel
const detailSig = (im) => [im.verdict, im.decision, im.crop, im.ai_status, im.caption_status, im.caption_edited ? "" : im.caption, JSON.stringify(im.checks.map((c) => c.key)), JSON.stringify(im.mask || null), im.mask_off].join("|");
let _detReq = 0;
/* The caption typed in the detail panel is saved 600 ms after the last key; anything that redraws the panel
   or makes another edit saves it first, so the redrawn panel never shows (and then re-saves) the old text. */
function flushCaption() { const c = S.capPending; return c && c.flush ? c.flush() : S.capSaving || Promise.resolve(); }
async function loadDetail(id, soft = false) {
  const my = ++_detReq;
  const box = $("#m-detail");
  const same = S.detail && S.detail.id === id;
  if (!(soft && $("#d-cap") && document.activeElement === $("#d-cap"))) {   // still typing: only the checks are redrawn
    const f = flushCaption();
    if (same) await f;
  }
  if (!soft) box.innerHTML = `<div class="muted">読み込み中…</div>`;
  let d;
  try { d = await api(`/api/projects/${S.project.id}/images/${id}`); } catch (e) { return fail(e); }
  if (my !== _detReq || S.focus !== id) return;
  // keep the user's in-progress caption edit
  const ta = $("#d-cap");
  const typing = soft && ta && document.activeElement === ta;
  S.detail = d;
  // compared with list items later: taken from the list item (the detail's "mask" has another shape)
  const li = S.images.find((x) => x.id === id);
  S.detailSig = detailSig(li || d);
  if (typing) { drawDetailChecks(d); return; }
  drawDetail(d);
}

function drawDetailChecks(d) {
  const el = $("#d-checks");
  if (!el) return;
  el.innerHTML = d.checks.length ? d.checks.map((c) => `<div class="chk ${c.status}"><span class="src">${esc(c.source)}${c.confidence && c.confidence !== "—" ? "・確信度" + esc(c.confidence) : ""}</span>
      <b>${esc(c.label)}</b> ${esc(c.message)}${c.fix ? ` <span class="muted">→ ${esc(c.fix)}</span>` : ""}</div>`).join("")
    : d.verdict === "pending" ? `<div class="chk"><span class="spin"></span> AI 解析中です。完了まで「問題なし」とは表示しません。</div>`
    : `<div class="chk">確認した項目に問題は見つかりませんでした。</div>`;
}

function drawDetail(d) {
  const pid = S.project.id;
  const box = $("#m-detail");
  const isRef = S.refs.includes(d.id);
  const dom = d.domain ? `${d.domain.label === "photo" ? "写真" : "イラスト"}（推定・確信度${d.domain.confidence}）` : "";
  box.innerHTML = `
    <div class="row" style="justify-content:space-between"><b>#${d.id}</b><span class="badge b-${d.verdict}" style="position:static">${esc(d.verdict_label)}</span></div>
    <div class="muted small">${esc(d.name)} ・ ${d.w || "?"}×${d.h || "?"}px ${dom ? "・ " + dom : ""}</div>
    <div class="pv" id="d-pv"><img id="d-img" src="/api/projects/${pid}/preview/${d.id}" alt="" draggable="false"></div>
    <div class="row small" style="margin-top:6px">
      <span class="muted">ドラッグで切り抜き範囲を指定</span>
      ${d.suggested_crop ? `<button id="d-sugg">切り抜き候補を適用</button>` : ""}
      ${d.crop ? `<button id="d-nocrop">切り抜きを解除</button>` : ""}
    </div>
    ${maskDetailHtml(d)}
    <h3>採用</h3>
    <div class="seg" id="d-dec">
      <button data-v="" class="${!d.decision ? "on" : ""}">自動（${["exclude", "unusable"].includes(d.verdict) ? "不採用" : "採用"}）</button>
      <button data-v="use" class="${d.decision === "use" ? "on" : ""}">採用</button>
      <button data-v="exclude" class="${d.decision === "exclude" ? "on" : ""}">除外</button>
    </div>
    ${S.project.kind === "character" ? `<button id="d-ref" style="margin-left:6px">${isRef ? "基準画像から外す" : "基準画像にする"}</button>` : ""}
    <h3>判定の理由</h3><div class="checks" id="d-checks"></div>
    <h3>説明文 <span class="savestate" id="d-save">${d.caption_edited ? "手動編集済み（自動解析で上書きされません）" : d.caption_status === "pending" || d.caption_status === "running" ? "自動生成中…"
      : d.caption_status === "failed" ? "自動生成に失敗" : d.caption_status === "skipped" ? "自動生成なし（解析できませんでした）" : "自動生成"}</span></h3>
    ${!d.caption_edited && d.caption_status === "failed" ? `<div class="warntext small" style="margin-bottom:4px">説明文の自動生成に失敗しました。今の説明文は古いままの可能性があります。
      <button class="small" id="d-recap">説明文を作り直す</button></div>` : ""}
    <textarea id="d-cap" rows="5">${esc(d.caption)}</textarea>
    <div class="row small" style="margin-top:4px">
      ${d.caption_edited && d.caption_auto != null ? `<button id="d-resetcap">自動説明文に戻す</button>` : ""}
      <span class="muted">${S.project.kind === "enhance" ? "この種類では、素材と加工画像に同じ説明文を使います。説明文は効果にほとんど影響しないので、そのままで構いません。" : `呼び出し語「${esc(S.project.trigger)}」`}${S.project.kind === "enhance" ? "" : S.project.kind === "character" ? "に結びつけたい特徴（髪・目など）は説明文に書かないのが基本です。" :
        S.project.kind === "concept" ? `に覚えさせたい${esc(conceptUI(S.project).noun)}（` + esc((S.project.concept_words || []).join(", ") || "未設定") + "）は書きません。" + esc(conceptUI(S.project).caption) :
        "が画風を担うよう、画風ではなく内容を書きます。"}</span>
    </div>
    ${d.tags && d.tags.length ? `<h3>検出タグ（推定）</h3><div class="tags">${d.tags.slice(0, 40).map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>` : ""}
    ${d.origin_path ? `<div class="muted small" style="margin-top:8px">元の場所: ${esc(d.origin_path)}</div>` : ""}`;
  drawDetailChecks(d);
  setupCrop(d);
  $$("#d-dec button").forEach((b) => b.onclick = async () => {
    const v = b.dataset.v || null;
    $$("#d-dec button").forEach((x) => x.classList.toggle("on", x === b));
    optimistic([d.id], (im) => { im.decision = v; im.use = v ? v === "use" : im.verdict !== "exclude" && im.verdict !== "unusable"; });
    await flushCaption();
    try { const r = await api(`/api/projects/${pid}/edit`, { body: { label: `#${d.id} を${v === "use" ? "採用" : v === "exclude" ? "除外" : "自動判定に戻す"}`, changes: [{ id: d.id, field: "decision", value: v }] } });
      S.history = r.history; drawUndo(); refreshImages(); } catch (e) { fail(e); refreshImages(); }
  });
  if ($("#d-ref")) $("#d-ref").onclick = async () => {
    const cur = new Set((S.project.reference_ids || []));
    isRef ? cur.delete(d.id) : cur.add(d.id);
    try { S.project = await api(`/api/projects/${pid}`, { method: "PATCH", body: { reference_ids: [...cur] } });
      toast(isRef ? "基準画像から外しました（同一性を再判定します）" : "基準画像にしました（同一性を再判定します）"); refreshImages(); } catch (e) { fail(e); }
  };
  const ta = $("#d-cap"), st = $("#d-save");
  if (S.project.caption_style !== "natural") attachTagComplete(ta);   // tag captions: tags as the tagger writes them
  let capTimer = null;
  const save = () => {
    clearTimeout(capTimer); capTimer = null;
    if (S.capPending && S.capPending.flush === save) S.capPending = null;
    st.className = "savestate"; st.textContent = "保存中…";
    const value = ta.value;
    // kept until it is answered: a redraw asked for meanwhile waits for it too (flushCaption)
    const p = S.capSaving = (async () => {
      try {
        const r = await api(`/api/projects/${pid}/edit`, { body: { label: `#${d.id} の説明文を編集`, changes: [{ id: d.id, field: "caption", value }] } });
        st.textContent = "保存済み（手動編集）"; S.history = r.history; drawUndo();
        const im = S.images.find((x) => x.id === d.id); if (im) { im.caption = value; im.caption_edited = true; }
      } catch (e) { st.className = "savestate err"; st.textContent = "保存できませんでした：" + e.message; }
      if (S.capSaving === p) S.capSaving = null;
    })();
    return p;
  };
  ta.oninput = () => {
    st.className = "savestate"; st.textContent = "未保存の変更…";
    S.capPending = { pid, id: d.id, value: ta.value, flush: save };
    clearTimeout(capTimer); capTimer = setTimeout(save, 600);
  };
  if ($("#d-recap")) $("#d-recap").onclick = async (e) => {
    e.target.disabled = true;
    try { await api(`/api/projects/${pid}/reanalyze`, { body: { ids: [d.id], captions: true } }); toast("説明文を作り直します"); refreshImages(); }
    catch (err) { fail(err); e.target.disabled = false; }
  };
  if ($("#d-resetcap")) $("#d-resetcap").onclick = async () => {
    await flushCaption();
    try { await api(`/api/projects/${pid}/bulk_caption`, { body: { ids: [d.id], op: "reset_auto" } }); loadDetail(d.id); refreshImages(); } catch (e) { fail(e); }
  };
  if ($("#d-mask-off")) $("#d-mask-off").onchange = async (e) => {
    await flushCaption();
    try { const r = await api(`/api/projects/${pid}/edit`, { body: { label: `#${d.id} のマスクを${e.target.checked ? "使わない" : "使う"}`, changes: [{ id: d.id, field: "mask_off", value: e.target.checked ? 1 : 0 }] } });
      S.history = r.history; drawUndo(); loadDetail(d.id); refreshImages(); } catch (x) { fail(x); }
  };
  if ($("#d-mask-redo")) $("#d-mask-redo").onclick = async () => {
    try { await api(`/api/projects/${pid}/images/${d.id}/mask_redo`, { body: {} }); toast("マスクを作り直します"); loadDetail(d.id); } catch (x) { fail(x); }
  };
  if ($("#d-sugg")) $("#d-sugg").onclick = () => saveCrop(d, d.suggested_crop);
  if ($("#d-nocrop")) $("#d-nocrop").onclick = () => saveCrop(d, null);
}

function maskDetailHtml(d) {
  const m = d.mask;
  if (!m) return "";
  const what = m.mode === "person" ? "人物" : "背景";
  const area = m.mode === "person" && d.crop ? "切り抜いた範囲" : "画像";   // a person mask is made for the crop
  const st = m.status === "done" ? `学習で効かせる範囲（明るい部分）は${area}の約 ${Math.round((m.coverage || 0) * 100)}% です。${m.mode === "person" && m.people > 1 ? `人物が ${m.people} 人見つかったため、${area}の中でいちばん大きく写っている人物を対象にしています。` : ""}`
    : m.status === "none" ? `${area === "画像" ? "" : "切り抜いた範囲に"}人物が見つかりませんでした。この画像はマスクなしで学習します。`
    : m.status === "failed" ? `マスクを作れませんでした（${esc(m.error || "")}）。この画像はマスクなしで学習します。`
    : `<span class="spin"></span> マスクを作っています…`;
  return `<h3>マスク（${what}を中心に学習）</h3>
    <div class="small">${m.off ? "この画像はマスクを使いません。" : st}</div>
    ${m.status === "done" && !m.off ? `<img class="maskpv" src="/api/projects/${S.project.id}/images/${d.id}/mask_preview?v=${encodeURIComponent((m.coverage || 0) + "-" + (m.people || 0) + "-" + (d.crop || ""))}" alt="" style="max-width:100%;margin-top:4px;border-radius:6px">` : ""}
    <div class="row small" style="margin-top:4px"><label><input type="checkbox" id="d-mask-off" ${m.off ? "checked" : ""}> この画像はマスクを使わない</label>
      ${["done", "none", "failed"].includes(m.status) ? `<button id="d-mask-redo">作り直す</button>` : ""}</div>`;
}

async function saveCrop(d, crop) {
  await flushCaption();
  try {
    const r = await api(`/api/projects/${S.project.id}/edit`, { body: { label: crop ? `#${d.id} を切り抜き` : `#${d.id} の切り抜きを解除`, changes: [{ id: d.id, field: "crop", value: crop }] } });
    S.history = r.history; drawUndo();
    toast(crop ? "切り抜きました。この画像だけ再判定します。" : "切り抜きを解除しました");
    loadDetail(d.id); refreshImages();
  } catch (e) { fail(e); }
}

function setupCrop(d) {
  const pv = $("#d-pv"), img = $("#d-img");
  if (!pv || !d.w) return;
  const sx = () => img.clientWidth / d.w;
  const rect = (r, cls) => {
    const el = document.createElement("div");
    el.className = cls;
    const s = sx();
    Object.assign(el.style, { left: r[0] * s + "px", top: r[1] * s + "px", width: (r[2] - r[0]) * s + "px", height: (r[3] - r[1]) * s + "px" });
    pv.appendChild(el);
    return el;
  };
  const draw = () => {
    $$(".crop,.box", pv).forEach((x) => x.remove());
    if (d.crop) rect(d.crop, "crop");
    else if (d.suggested_crop) rect(d.suggested_crop, "crop sugg");
  };
  img.onload = draw;
  if (img.complete) draw();
  let start = null, live = null;
  pv.onpointerdown = (e) => {
    const b = img.getBoundingClientRect();
    start = [(e.clientX - b.left) / sx(), (e.clientY - b.top) / sx()];
    pv.setPointerCapture(e.pointerId);
  };
  pv.onpointermove = (e) => {
    if (!start) return;
    const b = img.getBoundingClientRect();
    const cur = [(e.clientX - b.left) / sx(), (e.clientY - b.top) / sx()];
    const r = [Math.min(start[0], cur[0]), Math.min(start[1], cur[1]), Math.max(start[0], cur[0]), Math.max(start[1], cur[1])];
    $$(".crop", pv).forEach((x) => x.remove());
    live = r; rect(r, "crop");
  };
  pv.onpointerup = () => {
    const r = live; start = null; live = null;
    if (!r || r[2] - r[0] < 64 || r[3] - r[1] < 64) { draw(); return; }
    const c = [Math.max(0, r[0]), Math.max(0, r[1]), Math.min(d.w, r[2]), Math.min(d.h, r[3])].map(Math.round);
    saveCrop(d, c);
  };
}

// ---- start bar (one click to train when there are no blocking problems)
const refreshStartBar = debounce(async () => {
  const el = $("#m-start");
  if (!el || !S.project) return;
  let plan;
  try { plan = await api(`/api/projects/${S.project.id}/train/plan`); } catch (e) { el.innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  S.plan = plan;
  const pf = plan.preflight;
  const running = S.status && S.status.job && S.status.job.project === S.project.id;
  setHTML(el, `
    <div><b>採用 ${plan.used} 枚</b>${plan.values ? ` <span class="muted small">・ ${plan.values.max_train_steps} ステップ（途中保存を比較できます）</span>` : ""}
      ${pf.errors.map((x) => `<div class="err small">${esc(x)}</div>`).join("")}
      ${pf.warnings.map((x) => `<div class="warntext small">${esc(x)}</div>`).join("")}
      ${running ? `<div class="muted small">学習中です。ここでの編集は次回の学習に反映されます（実行中の学習は変わりません）。</div>` : ""}</div>
    <div class="spacer"></div>
    <a href="#/p/${S.project.id}/train" class="small">設定を確認</a>
    <button class="primary big" id="m-go" ${pf.errors.length || _starting ? "disabled" : ""}>${_starting ? "開始しています…" : "学習を開始"}</button>`);
  if ($("#m-go")) $("#m-go").onclick = startTraining;
}, 400);

let _starting = false;   // a start request on its way: the button stays disabled through redraws
async function startTraining() {
  if (_starting) return;
  const b = $("#m-go") || $("#t-go");
  if (b) { b.disabled = true; b.textContent = "開始しています…"; }
  _starting = true;
  try {
    const r = await api(`/api/projects/${S.project.id}/train/start`, { body: {} });
    toast(r.ahead > 0 ? `学習 #${r.run_id} を予約しました。前に ${r.ahead} 件あり、順番が来たら始まります（素材と設定は今の内容で固定しました）。`
      : `学習 #${r.run_id} を開始しました`);
    location.hash = `#/p/${S.project.id}/train`;
  } catch (e) { fail(e); if (b) { b.disabled = false; b.textContent = "学習を開始"; } }
  finally { _starting = false; }
}

// ------------------------------------------------------------------ drag & drop (anywhere)
let dragDepth = 0;
// Chrome reports a drag of an image on this page as "Files" too: a click on a result image that moves the mouse
// a little becomes such a drag, so drags that start inside the page are not treated as files dropped from outside
let innerDrag = false;
window.addEventListener("dragstart", () => { innerDrag = true; });
window.addEventListener("dragend", () => { innerDrag = false; });
const isZipItem = (it) => it.kind === "file" && /zip/i.test(it.type || "");
window.addEventListener("dragenter", (e) => {
  if (innerDrag || !e.dataTransfer.types.includes("Files")) return;
  const items = [...(e.dataTransfer.items || [])];
  const zips = items.some(isZipItem), others = items.some((it) => !isZipItem(it));
  if (!S.project && !(zips && S.route[0] !== "i2t")) return;   // the i2t page has its own drop area
  dragDepth++;
  $("#dropveil div").textContent = !zips ? "ここに画像をドロップして追加"
    : !S.project || !others ? "ここに zip をドロップすると、別のプロジェクトとして取り込みます（書き出した zip）"
    : "画像はこのプロジェクトに追加し、zip は別のプロジェクトとして取り込みます";
  $("#dropveil").hidden = false;
});
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("#dropveil").hidden = true; } });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", async (e) => {
  const taken = e.defaultPrevented;   // dropped on a box with its own handling (元画像)
  e.preventDefault(); dragDepth = 0; $("#dropveil").hidden = true;
  if (innerDrag) { innerDrag = false; return; }
  if (taken) return;
  const top = [...(e.dataTransfer.files || [])];
  if (!top.length) return;
  const zips = top.filter((f) => /\.zip$/i.test(f.name));
  if (zips.length) importProjects(zips);   // project zips written by "zip に書き出す"
  if (!S.project) {
    if (!zips.length) toast("先にプロジェクトを作成してください", "err");
    else if (zips.length < top.length) toast("zip 以外のファイルは、プロジェクトを開いてからドロップしてください", "err");
    return;
  }
  if (zips.length === top.length) return;
  const files = [];
  const items = [...(e.dataTransfer.items || [])].map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
  if (items.length) {
    const walk = async (ent) => {
      if (ent.isFile) files.push(await new Promise((res, rej) => ent.file(res, rej)));
      else if (ent.isDirectory) {
        const rd = ent.createReader();
        let batch;
        do { batch = await new Promise((res) => rd.readEntries(res)); for (const x of batch) await walk(x); } while (batch.length);
      }
    };
    for (const it of items) if (!(it.isFile && /\.zip$/i.test(it.name))) await walk(it);
  } else files.push(...top.filter((f) => !zips.includes(f)));
  if ((S.route[2] || "material") !== "material") location.hash = `#/p/${S.project.id}/material`;
  uploadFiles(files);
});

// ------------------------------------------------------------------ paste (Ctrl+V)
// Screenshots / images copied in a browser arrive as one unnamed "image.png"; give each a unique
// name.  Files copied in Explorer keep their names.  Text pastes into inputs are left alone.
let pasteSeq = 0;
window.addEventListener("paste", (e) => {
  if (e.defaultPrevented || !S.project || S.route[0] !== "p") return;
  const files = [...(e.clipboardData?.files || [])].filter((f) => /^image\//.test(f.type));
  if (!files.length) return;
  e.preventDefault();
  const d = new Date(), z = (n) => String(n).padStart(2, "0");
  const stamp = `${d.getFullYear()}${z(d.getMonth() + 1)}${z(d.getDate())}-${z(d.getHours())}${z(d.getMinutes())}${z(d.getSeconds())}`;
  const named = files.map((f) => {
    if (f.name && f.name !== "image.png") return f;
    const ext = (f.type.split("/")[1] || "png").replace("jpeg", "jpg");
    return new File([f], `貼り付け_${stamp}_${++pasteSeq}.${ext}`, { type: f.type });
  });
  if ((S.route[2] || "material") !== "material") location.hash = `#/p/${S.project.id}/material`;
  uploadFiles(named);
});

// ------------------------------------------------------------------ material generation
/* The character is defined by the user's own model + LoRAs + prompt.  Framing / angle / expression /
   background are added by the tool from a fixed set.  Generated images are candidates: the user
   picks which become material. */
const GEN_STATUS = { queued: "順番待ち", running: "生成中", paused: "一時停止中", done: "完了", cancelled: "中止", failed: "失敗" };

async function renderGen() {
  const el = $("#m-gen");
  if (!el) return;
  const btn = $("#m-gen-open");
  if (btn) btn.classList.toggle("primary", !!S.genOpen);
  if (!S.genOpen) { el.innerHTML = ""; return; }
  if (!S.gen) el.innerHTML = `<div class="panel gen muted">読み込み中…</div>`;
  try {
    [S.gen, S.lib] = await Promise.all([api(`/api/projects/${S.project.id}/gen`), api("/api/library")]);
  } catch (e) { el.innerHTML = `<div class="panel gen err">${esc(e.message)}</div>`; return; }
  drawGenForm();
  drawGenBatches();
}

function genModelLabel(m) {
  const v = m.variant_info && m.variant_info.label;
  return `${m.name}（${v || (m.family === "sdxl" ? "SDXL 系・系統未確認" : m.family)}）`;
}

const GEN_FAMILIES = ["sdxl", "anima", "qwen_image", "qwen_image21", "krea2", "zimage", "moss_image"];

/* Generation models grouped by the folder their file is in, labelled by file name, so every checkpoint
   placed in models\checkpoints (or an added folder) can be picked individually. */
function genModelOptions(lib, selected) {
  const groups = {};
  const add = (path, html) => {
    const dir = path.split(/[\\/]/).slice(0, -1).join("\\");
    (groups[dir] = groups[dir] || []).push([path.split(/[\\/]/).pop(), html]);
  };
  for (const m of lib.models.filter((x) => GEN_FAMILIES.includes(x.family))) {
    const f = m.files.gen || m.files.turbo || m.files.ckpt || "";   // Krea 2 registers its generation file as "turbo"
    const v = m.variant_info && m.variant_info.label;
    add(f, `<option value="${m.id}" ${m.id === selected ? "selected" : ""}>${esc(f.split(/[\\/]/).pop())}（${esc(v || "SDXL 系・系統未確認")}）</option>`);
  }
  // every other model file in the folders is listed too: one that can generate is set up when chosen
  // (pickFileModel asks for its text encoder / VAE when they are not found by name); others say why not
  for (const c of lib.checkpoints || []) {
    if (c.model_id) continue;
    const name = c.path.split(/[\\/]/).pop();
    add(c.path, GEN_FAMILIES.includes(c.family) && (!c.gguf || ["qwen_image", "qwen_image21"].includes(c.family))
      ? `<option value="file:${esc(c.path)}">${esc(name)}（${esc(c.family_label)}・未設定）</option>`
      : `<option disabled title="${esc(c.evidence || "")}">${esc(name)}（使えません: ${esc(c.evidence || "種類を判別できません")}）</option>`);
  }
  return Object.entries(groups).sort(([a], [b]) => a.localeCompare(b)).map(([dir, items]) =>
    `<optgroup label="${esc(dir)}">${items.sort((a, b) => a[0].localeCompare(b[0])).map((x) => x[1]).join("")}</optgroup>`).join("");
}

// the model id of a model list (a file not set up yet has none until pickFileModel has registered it)
const selModelId = (sel) => (sel && sel.value && !sel.value.startsWith("file:") ? sel.value : null);

/* A model file chosen in a model list that is not set up yet: registered, with its text encoder / VAE found by
   name or, when they are not, chosen here from the files in the text_encoders / vae folders.  then(id) runs once
   the list shows the registered model. */
async function pickFileModel(sel, then) {
  const path = sel.value.slice(5);
  const box = sel.closest(".row").parentElement;
  box.querySelector(".pickfile")?.remove();
  const done = async (id) => {
    box.querySelector(".pickfile")?.remove();
    S.lib = await api("/api/library");
    sel.innerHTML = `<option value="">選んでください</option>${genModelOptions(S.lib, id)}`;
    await then(id);
  };
  try {
    const r = await api("/api/models/from_file", { body: { path } });
    if (r.id) return done(r.id);
    // fitting files first; one known not to fit this model is shown but cannot be chosen
    const opts = (kind) => (r.choices?.[kind] || []).map((c) => c.fits === false
      ? `<option disabled title="${esc(c.path)}">${esc(c.path.split(/[\\/]/).pop())}（このモデルには合いません）</option>`
      : `<option value="${esc(c.path)}" title="${esc(c.path)}">${esc(c.path.split(/[\\/]/).pop())}</option>`).join("");
    const field = (kind, label, dir) => r.needs.includes(kind) ? `<div class="row small"><span style="min-width:9em">${label}</span>${opts(kind)
      ? `<select data-pick="${kind}" style="max-width:100%">${opts(kind)}</select>` : `<span class="err">models\\${dir} にファイルがありません</span>`}</div>` : "";
    const div = document.createElement("div");
    div.className = "pickfile";
    div.innerHTML = `<div class="small muted" style="margin-top:6px">${esc(path.split(/[\\/]/).pop())} と組み合わせるファイルが、決まった名前では見つかりませんでした。フォルダーにあるものから選んでください。</div>
      ${field("te", "テキストエンコーダー", "text_encoders")}${field("vae", "VAE", "vae")}
      <div class="row"><button class="small primary" data-pick-ok>この組み合わせで使う</button></div>`;
    box.appendChild(div);
    div.querySelector("[data-pick-ok]").onclick = async () => {
      const v = (k) => div.querySelector(`[data-pick=${k}]`)?.value || "";
      try {
        const r2 = await api("/api/models/from_file", { body: { path, te: v("te"), vae: v("vae") } });
        if (r2.id) await done(r2.id); else toast(r2.message, "err");
      } catch (e) { fail(e); }
    };
  } catch (e) { fail(e); }
}

// The fast LoRAs of MOSS-Image SFT (8-step DMD / 4-step): shown only while the chosen model has one
// (default_sampling.fast; fast_choices lists the ones there, the first is the default)
const FAST_LABEL = { on: "8 ステップ", 4: "4 ステップ" };
function fastOptions(v, ds) {
  const ch = ds.fast_choices || ["on"];
  return `<option value="">初期値（${FAST_LABEL[ch[0]]}）</option>${ch.map((c) => `<option value="${c}" ${v === c ? "selected" : ""}>${FAST_LABEL[c]}</option>`).join("")}
    <option value="off" ${v === "off" ? "selected" : ""}>使わない</option>`;
}
function fastField(attr, v, ds) {
  return `<div class="field" data-fastfield ${"fast" in ds ? "" : "hidden"}><div class="lab"><span>高速生成</span><span>初期値 <span data-fastdef>${FAST_LABEL[(ds.fast_choices || ["on"])[0]]}</span></span></div>
    <select data-${attr}="fast">${fastOptions(v, ds)}</select>
    <div class="muted small">MOSS-Image SFT 用の高速化 LoRA を重ね、8 ステップまたは 4 ステップ・CFG なしで生成します（4 ステップは取得したときだけ選べます）。比較画像には使いません。</div></div>`;
}

// after a save: the defaults of the chosen model (and of the 8-step choice) as the fields' placeholders
function showDefaultSampling(root, sAttr, dAttr, ds) {
  root = typeof root === "string" ? $(root) : root;
  if (!root) return;
  for (const [k, v] of Object.entries(ds || {})) {
    if (k === "fast" || k === "fast_choices") continue;
    const d = $(`[data-${dAttr}="${k}"]`, root); if (d) d.textContent = v;
    const i = $(`input[data-${sAttr}="${k}"]`, root); if (i) i.placeholder = v;
  }
  $$("[data-fastfield]", root).forEach((x) => {
    x.hidden = !("fast" in (ds || {}));
    if (x.hidden) return;
    const sel = $("select", x), def = $("[data-fastdef]", x);
    if (sel) sel.innerHTML = fastOptions(sel.value, ds);   // the choices follow the chosen model's files
    if (def) def.textContent = FAST_LABEL[(ds.fast_choices || ["on"])[0]];
  });
}

function drawGenForm() {
  const el = $("#m-gen");
  const r = S.gen.recipe || {};
  const models = S.lib.models;
  const ds = S.gen.default_sampling || {};
  const sm = r.sampling || {};
  const ex = new Set(r.exclude_angles || []), exf = new Set(r.exclude_framings || []), exe = new Set(r.exclude_exprs || []);
  const exp = new Set(r.exclude_poses || []);
  const exg = new Set(r.exclude_groups || []);
  const trainSel = S.project.model_id || "";
  const isStyle = S.project.kind === "style";
  el.innerHTML = `<div class="panel gen">
    <div class="row"><h2 style="margin:0">AI で素材を生成</h2>
      <span class="muted small">${isStyle ? "画風（線・塗り・色使い・質感など）だけを書くか、画風の LoRA を入れてください。題材（人物・動物・風景・物など）と構図はツールが 1 枚ずつ入れ替えます。背景は外しません。" :
        "キャラの見た目だけを書いてください。画角・アングル・表情（喜怒哀楽など）・ポーズ（走る・座る・ジャンプ・手の動きなど）は学習向けの組み合わせをツールが足し、背景はプロンプトに書いても外し、白一色にします。"}</span>
      <div class="spacer"></div><span class="savestate" id="g-save"></span><button id="g-close">閉じる</button></div>
    <div class="form" style="margin-top:10px">
      <label>生成モデル</label>
      <div><div class="row"><select id="g-model" style="max-width:100%"><option value="">選んでください</option>${genModelOptions(S.lib, r.gen_model_id)}</select>
        <button class="small" id="g-rescan" title="フォルダーを読み直します">再読み込み</button></div>
        <span class="muted small">models\\checkpoints（と設定で追加した場所）に置いた checkpoint を、ファイルごとに選べます。
          <button class="small" data-opendir="checkpoints">フォルダーを開く</button></span></div>
      <label>LoRA（任意）</label><div><div id="g-loras"></div><button id="g-addlora" class="small">＋ LoRA を追加</button> <button id="g-urllora" class="small" title="Hugging Face や Civitai の LoRA のページの URL から取得して追加します">URL から追加</button>
        <span class="muted small">LoRA は models\\loras に置くか、<a href="#/settings">設定</a>の「リンク・ファイルから追加」で Civitai などから追加すると選べます。「URL から追加」に Hugging Face や Civitai の URL を入れて、ここで直接追加することもできます。</span></div>
      <label>プロンプト</label><textarea id="g-pos" rows="8" placeholder="${isStyle ? "例: watercolor, soft colors, thick outlines, flat color, masterpiece, best quality" : "例: 1girl, silver hair, long hair, red eyes, white dress, masterpiece, best quality"}">${esc(r.positive || "")}</textarea>
      <label></label><div class="small" id="g-trig"></div>
      <label>ネガティブ</label><textarea id="g-neg" rows="5" placeholder="例: lowres, bad anatomy, bad hands, watermark">${esc(r.negative || "")}</textarea>
      <label>LoRA を作るモデル</label><div class="row"><select id="g-train"><option value="">生成モデルと同じ</option>${models.map((m) => `<option value="${m.id}" ${m.id === trainSel ? "selected" : ""}>${esc(genModelLabel(m))}</option>`).join("")}</select>
        <span class="muted small">学習の土台にするモデルです。生成モデルと別のものも選べます。</span></div>
      <label>枚数</label><div class="row">${(() => {
        const n = r.count || 48, preset = [24, 48, 72].includes(n);
        return `<select id="g-count">${[24, 48, 72].map((x) => `<option value="${x}" ${preset && n === x ? "selected" : ""}>${x} 枚</option>`).join("")}
          <option value="custom" ${preset ? "" : "selected"}>枚数を指定…</option></select>
          <input type="number" id="g-count-n" min="1" max="${GEN_MAX}" step="1" value="${preset ? 100 : n}" style="width:90px" ${preset ? "hidden" : ""}>
          <span class="muted small" id="g-count-note" ${preset ? "hidden" : ""}>1〜${GEN_MAX} 枚。多いほどアングル・ポーズの組み合わせが増えますが、生成と学習に時間がかかります。</span>`; })()}</div>
      <label>元画像（任意）</label><div id="g-init"></div>
    </div>
    <details style="margin-top:8px"><summary class="small">詳細設定（いつもの生成条件に合わせたいとき）</summary>
      <div class="fields" style="margin-top:8px">
        ${[["steps", "ステップ数", "number"], ["cfg", "CFG", "number"], ["sampler", "サンプラー", "text"], ["scheduler", "スケジューラー", "text"], ["clip_skip", "Clip skip", "number"], ["seed", "seed（空欄で毎回ランダム）", "number"]].map(([k, label, t]) => {
          const v = k === "seed" ? (r.seed ?? "") : (sm[k] ?? "");
          return `<div class="field"><div class="lab"><span>${label}</span>${k !== "seed" ? `<span>初期値 <span data-gd="${k}">${esc(ds[k] ?? "")}</span></span>` : ""}</div>
            <input type="${t}" step="any" data-gs="${k}" value="${esc(v)}" placeholder="${esc(k === "seed" ? "" : ds[k] ?? "")}"></div>`; }).join("")}
        ${fastField("gs", sm.fast, ds)}
      </div>
      ${isStyle ? `<div style="margin-top:8px" class="small">使わない題材 <span class="excl">${Object.entries(S.gen.groups || {}).map(([k, l]) => `<label><input type="checkbox" data-xg="${k}" ${exg.has(k) ? "checked" : ""}>${esc(l)}</label>`).join("")}</span>
        <div class="muted">題材を外しすぎると、その題材でしか画風が出ない LoRA になりやすくなります。</div></div>` : `
      <div style="margin-top:8px" class="small">使わないアングル <span class="excl">${Object.entries(S.gen.angles).map(([k, l]) => `<label><input type="checkbox" data-xa="${k}" ${ex.has(k) ? "checked" : ""}>${esc(l)}</label>`).join("")}</span></div>
      <div style="margin-top:4px" class="small">使わない画角 <span class="excl">${Object.entries(S.gen.framings).map(([k, l]) => `<label><input type="checkbox" data-xf="${k}" ${exf.has(k) ? "checked" : ""}>${esc(l)}</label>`).join("")}</span></div>
      <div style="margin-top:4px" class="small">使わない表情 <span class="excl">${Object.entries(S.gen.exprs || {}).map(([k, l]) => `<label><input type="checkbox" data-xe="${k}" ${exe.has(k) ? "checked" : ""}>${esc(l)}</label>`).join("")}</span></div>
      <div style="margin-top:4px" class="small">使わないポーズ <span class="excl">${Object.entries(S.gen.poses || {}).map(([k, l]) => `<label><input type="checkbox" data-xp="${k}" ${exp.has(k) ? "checked" : ""}>${esc(l)}</label>`).join("")}</span>
        <div class="muted">外したポーズの画像は、ポーズの指定なしで作ります（画角・アングル・表情はそのままです）。</div></div>`}
    </details>
    <div id="g-check" style="margin-top:8px"></div>
    <div class="row" style="margin-top:8px"><button class="primary big" id="g-start">生成を開始</button>
      <span class="muted small">生成中も一時停止・中止できます。できた画像から順に下に並びます。</span></div>
    <div class="gauto"><div class="row"><button class="big" id="g-auto">生成から学習まで全自動で進める</button>
      <a class="small" href="#/jobs">ジョブのページ</a></div>
      <div class="muted small">素材の生成 → 生成した画像を素材に追加${isStyle ? "" : "（顔立ちがほかと違う候補は外します）"} → 解析と説明文の作成 → 学習 → 比較画像、までを止まらずに自動で進めます。
        学習の設定は、学習を始める時点のこのプロジェクトの設定を使います。進み具合はジョブのページで見られ、途中で一時停止や中止ができます。</div></div>
    <div id="g-batches"></div></div>`;
  drawLoraRows(r.loras || []);
  drawGenCheck(S.gen.check, S.gen.notes);
  $("#g-close").onclick = () => { saveRecipeSoon.flush(); S.genOpen = false; renderGen(); };
  $("#g-addlora").onclick = () => { const l = readRecipe().loras; l.push({ path: "", strength: 0.8 }); drawLoraRows(l); };
  $("#g-urllora").onclick = (e) => loraUrlBox(e.target, (path) => { const l = readRecipe().loras.filter((x) => x.path); l.push({ path, strength: 0.8 }); drawLoraRows(l); saveRecipeSoon(); });
  $("#g-model").addEventListener("change", () => {
    const sel = $("#g-model");
    if (sel.value.startsWith("file:")) pickFileModel(sel, () => { drawLoraRows(readRecipe().loras); saveRecipeSoon(); });
    else { sel.closest(".row").parentElement.querySelector(".pickfile")?.remove(); drawLoraRows(readRecipe().loras); }
  });
  $$("#m-gen .form select:not([data-lp]), #m-gen .form textarea, #m-gen details input, #m-gen details select").forEach((x) => { x.oninput = saveRecipeSoon; x.onchange = saveRecipeSoon; });
  $("#g-pos").addEventListener("input", debounce(() => drawTrigPreview("#g-loras", "#g-pos", "#g-trig"), 200));
  $$("#g-pos, #g-neg").forEach((x) => attachTagComplete(x, { escape: true, sep: true, moss: () => modelFamily($("#g-model")?.value || S.project.model_id) === "moss_image" }));
  $("#g-train").onchange = async (e) => {
    try { S.project = await api(`/api/projects/${S.project.id}`, { method: "PATCH", body: { model_id: e.target.value || readRecipe().gen_model_id || null } });
      toast("LoRA を作るモデルを変更しました"); refreshStartBar(); } catch (err) { fail(err); }
  };
  $("#g-start").onclick = startGen;
  genDrawInit();
  $("#g-auto").onclick = startAuto;
  $("#g-count").addEventListener("change", () => {
    const custom = $("#g-count").value === "custom";
    $("#g-count-n").hidden = $("#g-count-note").hidden = !custom;
    if (custom) $("#g-count-n").focus();
  });
  $("#g-count-n").oninput = saveRecipeSoon;
  $("#g-rescan").onclick = async () => {
    try { S.lib = await api("/api/library"); drawGenForm(); drawGenBatches(); toast("モデルの一覧を読み直しました"); } catch (e) { fail(e); }
  };
  bindOpenDir(el);
}

function drawLoraRows(loras) {
  drawLoraRowsIn($("#g-loras"), loras, saveRecipeSoon, () => drawTrigPreview("#g-loras", "#g-pos", "#g-trig"), modelFamily($("#g-model").value));
}

/* LoRA rows shared by material generation and the image generation page.  A LoRA's trigger words come
   from LoRA 設定 (#/loras); the row's switch says whether they are put in front of the prompt. */
function loraConf(path) { const x = S.lib.loras.find((l) => l.path === path); return x ? x.conf : null; }

function loraTrigHtml(lo) {
  if (!lo.path) return "";
  const c = loraConf(lo.path);
  if (!c || !c.triggers.length) {
    return `<span class="small muted">呼び出し語は未登録です${c && c.candidates.length ? "（候補があります）" : ""}。<a href="#/loras" data-lconf="${esc(lo.path)}">LoRA 設定</a>で登録すると、プロンプトに自動で入れられます。</span>`;
  }
  const on = lo.trigger ?? c.auto;
  if (c.triggers.length < 2) {
    return `<label class="small"><input type="checkbox" data-lt ${on ? "checked" : ""}> 呼び出し語をプロンプトに入れる：<code>${esc(c.triggers.join(", "))}</code></label>
      <a class="small" href="#/loras" data-lconf="${esc(lo.path)}">編集</a>`;
  }
  // several trigger words (e.g. one per character or outfit): the row chooses which ones go in, from the pull-down
  const chosen = Array.isArray(lo.words) ? lo.words : c.triggers;
  return `<label class="small"><input type="checkbox" data-lt ${on ? "checked" : ""}> 呼び出し語をプロンプトに入れる：</label>
    <span class="lwords" data-lw='${esc(JSON.stringify(chosen))}'>${trigChipsHtml(chosen, c)}</span>
    <a class="small" href="#/loras" data-lconf="${esc(lo.path)}">編集</a>`;
}

function trigChipsHtml(chosen, c) {
  const low = chosen.map((w) => w.toLowerCase());
  const rest = c.triggers.filter((w) => !low.includes(w.toLowerCase()));
  const cands = (c.candidates || []).filter((w) => !low.includes(w.toLowerCase()));
  return chosen.map((w, i) => `<span class="lchip"><code>${esc(w)}</code><button class="small" data-lwx="${i}" title="この語を入れない">×</button></span>`).join(" ")
    + (rest.length || cands.length ? ` <select data-lwadd class="small"><option value="">＋ 呼び出し語を追加</option>
        ${rest.length ? `<optgroup label="登録した呼び出し語">${rest.map((w) => `<option value="${esc(w)}">${esc(w)}</option>`).join("")}</optgroup>` : ""}
        ${cands.length ? `<optgroup label="候補（未登録）">${cands.map((w) => `<option value="${esc(w)}">${esc(w)}</option>`).join("")}</optgroup>` : ""}</select>` : "")
    + (chosen.length ? "" : ` <span class="small muted">（どれも入れません）</span>`);
}

function modelFamily(id) { const m = id && S.lib.models.find((x) => x.id === id); return m ? m.family : null; }

/* LoRAs made with this tool first, then the ones placed or added from elsewhere.  With a model chosen,
   only the LoRAs made for its family are listed (and those whose family cannot be read from the file).
   A LoRA already picked for another family stays in the row, marked, so it is not dropped silently. */
const LORA_GROUPS = [[true, "このツールで作った LoRA"], [false, "別に置いた・追加した LoRA"]];
function loraOptions(selected, family) {
  const fits = (x) => !family || !x.family || x.family === family;
  const opt = (x) => `<option value="${esc(x.path)}" ${x.path === selected ? "selected" : ""}>${esc(x.name)}（${esc(x.family_label)}）</option>`;
  const other = S.lib.loras.find((x) => x.path === selected && !fits(x));
  return LORA_GROUPS.map(([made, label]) => {
    const items = S.lib.loras.filter((x) => !!(x.conf && x.conf.made) === made && fits(x));
    return items.length ? `<optgroup label="${label}">${items.map(opt).join("")}</optgroup>` : "";
  }).join("") + (other ? `<optgroup label="このモデルでは使えない LoRA">${opt(other)}</optgroup>` : "");
}

/* 「URL から追加」: a Hugging Face / Civitai link typed beside the LoRA list.  The files behind it are checked from
   their headers first (as in 設定 → リンク・ファイルから追加); the LoRA is downloaded into models\loras (a file
   already here is used as it is) and added as a LoRA row with addRow(path). */
function loraUrlBox(btn, addRow) {
  const host = btn.parentElement;
  if (host.querySelector(".lurl")) { host.querySelector(".lurl input").focus(); return; }
  const box = document.createElement("div");
  box.className = "lurl";
  box.style.marginTop = "6px";
  box.innerHTML = `<div class="row"><input type="text" placeholder="例: https://huggingface.co/作者/リポジトリ　または　https://civitai.com/models/12345" style="flex:1;min-width:260px">
      <button class="small primary" data-go>調べて追加</button><button class="small" data-close>閉じる</button></div>
    <div class="row" data-tokrow hidden><input type="password" data-tok placeholder="アクセストークン / API キー（保存しません）" style="width:300px"></div>
    <div class="small" data-msg></div><div data-pick></div>`;
  host.appendChild(box);
  const input = $("input", box), msg = $("[data-msg]", box), pick = $("[data-pick]", box);
  input.focus();
  $("[data-close]", box).onclick = () => box.remove();
  // the server's messages are written for the settings page (「下の欄にドロップ」 is its drop box)
  const say = (t, err) => { msg.textContent = String(t).replace("下の欄に", "「設定」の「リンク・ファイルから追加」の欄に"); msg.className = "small" + (err ? " err" : ""); };
  const done = async (path, note) => {
    try { S.lib = await api("/api/library"); } catch { /* the row still names the file */ }
    addRow(path);
    say(note);
    pick.innerHTML = "";
    input.value = "";
  };
  const fetchOne = async (res, f) => {
    if (f.have) return done(f.have, `${f.have.split(/[\\/]/).pop()} はもう入っているので、それを使います`);
    const tok = $("[data-tok]", box).value.trim() || null;
    let st;
    try { st = await api("/api/import/download", { body: { url: f.url, name: f.name, size: f.size, site: res.source, token: tok } }); }
    catch (e) { return say(e.message, true); }
    for (;;) {   // the download runs on the server; its state is read about once a second
      await new Promise((r) => setTimeout(r, 1000));
      if (!box.isConnected) return;
      let cur;
      try { cur = (await api("/api/import/state")).imports.find((x) => x.id === st.id); } catch { continue; }
      if (!cur) return say("取得の状態がわからなくなりました。「設定」の「リンク・ファイルから追加」で確かめてください。", true);
      if (cur.state === "running") {
        const pct = cur.total_bytes ? Math.round((cur.done_bytes / cur.total_bytes) * 100) : null;
        say(`${f.name} を取得中${pct == null ? "" : `（${pct}%）`}…`);
        continue;
      }
      if (cur.state !== "done") return say(cur.error || cur.message || "取得できませんでした", true);
      if (cur.kind !== "lora") return say(`${f.name} は LoRA ではなかったため、${cur.dest_label || "モデル"}として追加しました（LoRA の欄には入れていません）`);
      return done(cur.path, `${cur.path.split(/[\\/]/).pop()} を追加しました`);
    }
  };
  $("[data-go]", box).onclick = async () => {
    const link = input.value.trim();
    if (!link) return say("URL を入れてください", true);
    say("調べています…");
    pick.innerHTML = "";
    const tok = $("[data-tok]", box).value.trim();
    let res;
    try {
      const r = await fetch("/api/import/resolve", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ link, hf_token: tok || undefined, civitai_token: tok || undefined }) });
      res = await r.json();
      if (r.status === 401) { $("[data-tokrow]", box).hidden = false; return say(res.detail, true); }
      if (!r.ok) return say(res.detail || `HTTP ${r.status}`, true);
    } catch (e) { return say(e.message, true); }
    if (res.need_token) $("[data-tokrow]", box).hidden = false;
    const loras = res.files.filter((f) => f.probe && (f.probe.kind === "lora" ? f.probe.ok !== false : f.probe.ok == null));
    if (!loras.length) {
      const why = res.files.map((f) => `${f.name}: ${f.probe && (f.probe.reason || f.probe.evidence) || "LoRA ではありません"}`).join(" / ");
      return say(`このリンクには、使える LoRA が見つかりませんでした${why ? `（${why}）` : ""}`, true);
    }
    if (loras.length === 1) return fetchOne(res, loras[0]);
    say(`${res.title} に LoRA が ${loras.length} 個あります。使うものを選んでください。`);
    pick.innerHTML = `<div class="row"><select data-f style="max-width:100%">${loras.map((f, i) => `<option value="${i}">${esc(f.name)}${f.probe.family_label ? `（${esc(f.probe.family_label)} 用）` : ""}${f.size ? `・${(f.size / 1e6).toFixed(0)}MB` : ""}${f.have ? "・入っています" : ""}</option>`).join("")}</select>
      <button class="small primary" data-get>これを追加</button></div>`;
    $("[data-get]", pick).onclick = () => fetchOne(res, loras[+$("[data-f]", pick).value]);
  };
  input.onkeydown = (e) => { if (e.key === "Enter") $("[data-go]", box).click(); };
}

function drawLoraRowsIn(box, loras, save, onChange, family) {
  box.innerHTML = loras.map((lo) => `<div class="lorarow">
    <select data-lp><option value="">${family && !S.lib.loras.some((x) => !x.family || x.family === family) ? "このモデル用の LoRA はありません" : "選んでください"}</option>${loraOptions(lo.path, family)}</select>
    <span class="small muted">強さ</span><input type="number" step="0.05" data-ls value="${esc(lo.strength ?? 0.8)}">
    <button data-lx title="外す">×</button><div class="loratrig" data-ltw>${loraTrigHtml(lo)}</div></div>`).join("");
  const bindTrig = (row) => {
    const t = $("[data-lt]", row);
    if (t) t.onchange = () => { save(); onChange(); };
    const lw = $("[data-lw]", row);
    if (lw) {   // the chosen words of a LoRA with several: × takes one out, the pull-down adds one
      const set = (words) => { lw.dataset.lw = JSON.stringify(words); lw.innerHTML = trigChipsHtml(words, loraConf($("[data-lp]", row).value)); bindTrig(row); save(); onChange(); };
      const cur = () => JSON.parse(lw.dataset.lw || "[]");
      $$("[data-lwx]", lw).forEach((b) => b.onclick = () => set(cur().filter((_, i) => i !== +b.dataset.lwx)));
      const add = $("[data-lwadd]", lw);
      if (add) add.onchange = () => { if (add.value) set([...cur(), add.value]); };
    }
    $$("[data-lconf]", row).forEach((a) => a.onclick = () => { S.loraFocus = a.dataset.lconf; save(); });
  };
  $$(".lorarow", box).forEach((row) => {
    bindTrig(row);
    $("[data-lp]", row).onchange = () => {
      const c = loraConf($("[data-lp]", row).value);
      if (c && c.strength != null) $("[data-ls]", row).value = c.strength;   // the strength set (or tested) for this LoRA
      $("[data-ltw]", row).innerHTML = loraTrigHtml({ path: $("[data-lp]", row).value });
      bindTrig(row); save(); onChange();
    };
    $("[data-ls]", row).oninput = save; $("[data-ls]", row).onchange = save;
    $("[data-lx]", row).onclick = () => { row.remove(); save(); onChange(); };
  });
  onChange();
}

function readLoraRows(box) {
  const num = (v) => v === "" || v == null ? null : Number(v);
  return $$(".lorarow", box).map((row) => {
    const t = $("[data-lt]", row);
    const lw = $("[data-lw]", row);
    return { path: $("[data-lp]", row).value, strength: num($("[data-ls]", row).value) ?? 1, ...(t ? { trigger: t.checked } : {}),
      ...(lw ? { words: JSON.parse(lw.dataset.lw || "[]") } : {}) };
  });
}

/* Which words generation will put in front of the prompt (the server does the same; words already in
   the prompt are not added twice). */
function drawTrigPreview(boxSel, promptSel, outSel, text) {
  const out = $(outSel), box = $(boxSel), pr = $(promptSel);
  if (!out || !box || !pr) return;
  const prompt = (text ?? pr.value).toLowerCase(), words = [];
  for (const lo of readLoraRows(box)) {
    const c = loraConf(lo.path);
    if (!c || !(lo.trigger ?? c.auto)) continue;
    for (const w of Array.isArray(lo.words) ? lo.words : c.triggers) {
      const re = new RegExp("(?<![\\w])" + w.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "(?![\\w])", "u");
      if (!re.test(prompt) && !words.some((x) => x.toLowerCase() === w.toLowerCase())) words.push(w);
    }
  }
  out.innerHTML = words.length ? `生成するときに、プロンプトの先頭へ呼び出し語 <code>${esc(words.join(", "))}</code> を足します。` : "";
}

function readRecipe() {
  const num = (v) => v === "" || v == null ? null : Number(v);
  const sampling = {};
  $$("[data-gs]").forEach((i) => { if (i.dataset.gs !== "seed" && i.value !== "") sampling[i.dataset.gs] = i.type === "number" ? num(i.value) : i.value; });
  const seed = $("[data-gs=seed]");
  return {
    gen_model_id: selModelId($("#g-model")), positive: $("#g-pos").value, negative: $("#g-neg").value,
    count: genCount(), sampling, seed: seed && seed.value !== "" ? num(seed.value) : null,
    loras: readLoraRows($("#g-loras")),
    exclude_angles: $$("[data-xa]:checked").map((x) => x.dataset.xa), exclude_framings: $$("[data-xf]:checked").map((x) => x.dataset.xf),
    exclude_exprs: $$("[data-xe]:checked").map((x) => x.dataset.xe), exclude_poses: $$("[data-xp]:checked").map((x) => x.dataset.xp),
    exclude_groups: $$("[data-xg]:checked").map((x) => x.dataset.xg),
    init_images: (S.gen.recipe || {}).init_images || [],
    denoise: $("#g-den") && $("#g-den").value !== "" ? num($("#g-den").value) : (S.gen.recipe || {}).denoise ?? null,
  };
}

/* image-to-image material: the shots go through these images in turn (each scaled and centre-cropped to the shot's size) */
const GEN_SRC_MAX = 20;
function genDrawInit() {
  const el = $("#g-init");
  if (!el) return;
  const r = S.gen.recipe || {}, list = r.init_images || [];
  const src = (n) => `/api/projects/${S.project.id}/gen/source/${encodeURIComponent(n)}`;
  el.innerHTML = `${list.length ? `<div class="row" style="flex-wrap:wrap;gap:6px">${list.map((n, i) => `<div style="position:relative">
        <img src="${src(n)}" alt="" style="width:84px;height:84px;object-fit:cover;border-radius:6px;border:1px solid var(--line)">
        <button class="small" data-gsx="${i}" title="この元画像を外す" style="position:absolute;top:2px;right:2px;padding:0 6px">×</button></div>`).join("")}</div>
      <div class="row" style="margin-top:6px"><span class="small">変化の強さ</span>
        <input type="number" id="g-den" min="0.05" max="1" step="0.05" style="width:80px" value="${esc(r.denoise ?? 0.75)}">
        <span class="muted small">小さいほど元画像に近くなります。小さすぎると、どの画像も元画像と同じポーズ・アングルになり、学習の素材として偏ります（0.7〜0.85 を目安に）。</span></div>` : ""}
    <div class="row" style="margin-top:4px">${list.length < GEN_SRC_MAX ? `<button class="small" id="g-init-pick">${list.length ? "元画像を追加" : "画像を選ぶ"}</button>` : ""}
      <span class="muted small">画像をここにドロップするか、貼り付ける（Ctrl+V）と、その画像をもとに素材を生成します（I2I、${GEN_SRC_MAX} 枚まで）。
        複数入れると 1 枚ずつ順に使います。アングル・表情などの指定はそのまま足します。背景のある元画像では、背景が残ることがあります。</span>
      <input type="file" id="g-init-file" accept="image/*" multiple hidden></div>`;
  const pick = $("#g-init-pick");
  if (pick) pick.onclick = () => $("#g-init-file").click();
  $("#g-init-file").onchange = (e) => genAddInit([...e.target.files]);
  if ($("#g-den")) { $("#g-den").oninput = saveRecipeSoon; $("#g-den").onchange = saveRecipeSoon; }
  $$("[data-gsx]", el).forEach((b) => b.onclick = () => {
    S.gen.recipe = { ...readRecipe(), init_images: list.filter((_, i) => i !== +b.dataset.gsx) };
    genDrawInit(); saveRecipe();
  });
  if (!el.dataset.drop) {
    el.dataset.drop = "1";
    el.addEventListener("dragover", (e) => { e.preventDefault(); });
    el.addEventListener("drop", (e) => { e.preventDefault(); genAddInit([...(e.dataTransfer.files || [])]); });
  }
}

async function genAddInit(files) {
  files = files.filter((f) => f && f.type.startsWith("image/"));
  if (!files.length) return;
  const names = [...((S.gen.recipe || {}).init_images || [])], pid = S.project.id;
  try {
    for (const file of files) {   // one after another, then one save
      if (names.length >= GEN_SRC_MAX) { toast(`元画像は ${GEN_SRC_MAX} 枚までです`); break; }
      const r = await fetch(`/api/projects/${pid}/gen/source`, { method: "POST", body: file });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "画像を読み込めませんでした");
      if (!names.includes(j.name)) names.push(j.name);
    }
  } catch (e) { fail(e); }
  if (!$("#g-init") || !S.project || S.project.id !== pid) return;
  S.gen.recipe = { ...readRecipe(), init_images: names };
  genDrawInit(); saveRecipe();
}
document.addEventListener("paste", (e) => {
  if (!$("#g-init") || ["INPUT", "TEXTAREA"].includes((document.activeElement || {}).tagName)) return;
  const files = [...(e.clipboardData ? e.clipboardData.items : [])].filter((x) => x.type.startsWith("image/")).map((x) => x.getAsFile());
  if (files.length) { e.preventDefault(); genAddInit(files); }
});

function drawGenCheck(errs, notes) {
  const el = $("#g-check");
  if (!el) return;
  el.innerHTML = (errs || []).map((x) => `<div class="err small">${esc(x)}</div>`).join("") +
    (notes || []).map((x) => `<div class="muted small">${esc(x)}</div>`).join("");
}

const saveRecipeSoon = saveDebounce(saveRecipe, 500);
/* false when it could not be saved */
async function saveRecipe() {
  const st = $("#g-save");
  if (!st) return true;
  st.className = "savestate"; st.textContent = "保存中…";
  const body = readRecipe(), pid = S.project.id;   // read now: the form may be gone when the answer comes
  try {
    const r = await api(`/api/projects/${pid}/gen/recipe`, { method: "PUT", body });
    if (!S.project || S.project.id !== pid) return true;
    S.gen.recipe = body; S.gen.check = r.check; S.gen.notes = r.notes; S.project = r.project;
    S.gen.default_sampling = r.default_sampling;
    st.textContent = "保存済み";
    if (!$("#g-save")) return true;
    showDefaultSampling("#m-gen", "gs", "gd", r.default_sampling);
    drawGenCheck(r.check, r.notes);
    if ($("#g-train") && !$("#g-train").value && S.project.model_id) refreshStartBar();
    return true;
  } catch (e) { st.className = "savestate err"; st.textContent = "保存できませんでした：" + e.message; return false; }
}

async function startGen() {
  const b = $("#g-start");
  b.disabled = true;
  try {
    if (!(await saveRecipe())) throw new Error("生成の設定を保存できなかったため、開始しませんでした");
    const r = await api(`/api/projects/${S.project.id}/gen/start`, { body: { count: genCount() } });
    toast(`生成 #${r.batch_id} を開始しました`);
    S.genBatch = r.batch_id;
    await reloadGen();
  } catch (e) { fail(e); }
  b.disabled = false;
}

const GEN_MAX = 200;   // server.gen_start and the LoRA creation task accept 1-200 images
function genCount() {
  const v = $("#g-count").value;
  return v === "custom" ? Math.round(Number($("#g-count-n").value) || 0) : +v;
}

/* One button for the whole way: registers a LoRA creation task (generate -> add -> analyse -> train). */
async function startAuto() {
  const b = $("#g-auto"), n = genCount();
  if (!(n >= 1 && n <= GEN_MAX)) return toast(`枚数は 1〜${GEN_MAX} 枚で指定してください`, "err");
  b.disabled = true;
  try {
    if (!(await saveRecipe())) throw new Error("生成の設定を保存できなかったため、登録しませんでした");
    const r = await api(`/api/projects/${S.project.id}/tasks`, { body: { gen_count: n } });
    toast(r.ahead > 0 ? `全自動の作成を登録しました（${n} 枚）。前に ${r.ahead} 件あり、順番が来たら始まります。` : `全自動の作成を始めました（${n} 枚）。`,
      "", { label: "ジョブを開く", fn: () => { location.hash = "#/jobs"; } });
  } catch (e) { fail(e); }
  b.disabled = false;
}

async function reloadGen() {
  try { S.gen = { ...S.gen, ...(await api(`/api/projects/${S.project.id}/gen`)), recipe: S.gen.recipe }; } catch (e) { return fail(e); }
  drawGenBatches();
}
const reloadGenSoon = debounce(reloadGen, 400);

function onGenEvent(m) {
  if (m.status === "consistency") { if (S.project && m.project === S.project.id) toast(m.message); }
  if (!$("#g-batches")) return;
  const b = S.gen && S.gen.batches.find((x) => x.id === m.batch);
  if (b && m.message) {
    b.status = m.status; b.message = m.message;
    const st = $(`#gb-msg`);
    if (st && S.genBatch === m.batch) st.textContent = `${GEN_STATUS[m.status] || m.status}・${m.message}${m.eta_s != null ? `・残り ${fmtDur(m.eta_s)}` : ""}`;
  }
  reloadGenSoon();
}

function drawGenCards(grid, b, active, url) {
  const cards = grid._cards || (grid._cards = new Map());
  const keep = new Set(b.images.map((im) => im.id));
  for (const [id, c] of cards) if (!keep.has(id)) { c.remove(); cards.delete(id); }
  let prev = null;
  for (const im of b.images) {
    let c = cards.get(im.id);
    if (!c) { c = document.createElement("div"); c.dataset.gi = im.id; cards.set(im.id, c); }
    const sig = [im.image_id ? 1 : 0, im.picked, im.differs, im.ccip_diff, im.mismatch, im.file, im.label, im.seed, im.idx].join("|");
    if (c._sig !== sig) {
      c.className = `gcard ${im.image_id ? "done" : im.picked ? "on" : ""}`;
      c.innerHTML = `<img loading="lazy" src="${url(im.file, true)}" alt="">
        ${im.image_id ? `<span class="gdone">素材に追加済み</span>` : `<span class="gchk">${im.picked ? "✓" : ""}</span>`}
        ${im.differs ? `<span class="gdiff" title="キャラ判定 AI による推定。中心の画像との差 ${im.ccip_diff}（0.178 を超えると別のキャラと判定される目安）">顔立ちが違う可能性</span>` : ""}
        ${im.mismatch ? `<span class="gdiff gmis" title="${esc(`タグ判定 AI による推定です。${im.mismatch}`)}">プロンプトと違う可能性</span>` : ""}
        <button class="gzoom" data-gz="${esc(url(im.file))}" data-cap="${esc(im.label)} / seed ${im.seed}" title="拡大">⤢</button>
        <div class="gl"><span>${esc(im.label)}</span><span>#${im.idx + 1}</span></div>`;
      c._sig = sig;
    }
    const want = prev ? prev.nextSibling : grid.firstChild;
    if (want !== c) grid.insertBefore(c, want);
    prev = c;
  }
  let sp = $(".gspin", grid);
  if (active && b.images.length < b.total) {
    if (!sp) {
      sp = document.createElement("div"); sp.className = "gcard gspin";
      sp.style.cssText = "display:flex;align-items:center;justify-content:center;min-height:140px"; sp.innerHTML = `<span class="spin"></span>`;
    }
    grid.appendChild(sp);
  } else if (sp) sp.remove();
}

function drawGenBatches() {
  const el = $("#g-batches");
  if (!el) return;
  const bs = S.gen.batches;
  if (!bs.length) { el.innerHTML = ""; return; }
  const b = bs.find((x) => x.id === S.genBatch) || bs[0];
  S.genBatch = b.id;
  const active = ["queued", "running"].includes(b.status);
  const open = b.images.filter((i) => !i.image_id);
  const picked = open.filter((i) => i.picked);
  const url = (f, t) => `/api/projects/${S.project.id}/gen/${b.id}/img/${encodeURIComponent(f)}${t ? "?thumb=1" : ""}`;
  // the image grid is kept and updated card by card (an event comes for every image, a click for every pick)
  let wrap = $(".gbwrap", el);
  if (!wrap || +wrap.dataset.batch !== b.id) {
    el.innerHTML = `<div class="gbwrap" data-batch="${b.id}" style="margin-top:14px;border-top:1px solid var(--line);padding-top:10px">
      <div class="gbhead"></div><div class="ggrid"></div></div>`;
    wrap = $(".gbwrap", el);
    $(".ggrid", wrap).onclick = (e) => {
      const c = e.target.closest(".gcard[data-gi]");
      if (!c) return;
      const z = e.target.closest("[data-gz]");
      if (z) { openLightbox([{ src: z.dataset.gz, cap: z.dataset.cap }], 0); return; }
      const im = wrap._b.images.find((x) => x.id === +c.dataset.gi);
      if (im && !im.image_id) wrap._pick([im.id], !im.picked);
    };
  }
  $(".gbhead", wrap).innerHTML = `<div class="row">
      ${bs.length > 1 ? `<select id="gb-sel">${bs.map((x) => `<option value="${x.id}" ${x.id === b.id ? "selected" : ""}>生成 #${x.id}（${esc(GEN_STATUS[x.status] || x.status)}・${x.images.length}/${x.total}）</option>`).join("")}</select>` : `<b>生成 #${b.id}</b>`}
      <span class="st ${b.status === "running" ? "st-training" : b.status === "done" ? "st-done" : b.status === "failed" ? "st-failed" : ""}">${esc(GEN_STATUS[b.status] || b.status)}</span>
      <span class="small" id="gb-msg">${esc(b.message || "")}</span>
      <span class="muted small">${b.images.length} / ${b.total} 枚 ・ ${esc(b.gen_model_name || "")} ・ ${b.size_mb} MB</span>
      <div class="spacer"></div>
      ${active ? `<button data-gc="pause">一時停止</button><button class="danger" data-gc="stop">中止</button>` : ""}
      ${b.status === "paused" ? `<button class="primary" data-gc="resume">再開</button><button class="danger" data-gc="stop">中止</button>` : ""}
      ${["failed", "cancelled"].includes(b.status) && b.remaining > 0 ? `<button data-gc="resume">残りを生成</button>` : ""}
      ${!active && open.length > picked.length ? `<button data-gdel="unpicked" data-n="${open.length - picked.length}">選んでいない画像を削除</button>` : ""}
      ${!active ? `<button class="danger" data-gdel="all" data-n="${b.images.length}">この生成を削除</button>` : ""}
    </div>
    ${active ? `<div class="prog" style="margin:8px 0"><div style="width:${Math.round(b.images.length / b.total * 100)}%"></div></div>` : ""}
    ${b.images.length ? `<div class="row" style="margin-top:8px">
      <span class="small">素材にする画像を選んでください（クリックで切り替え）：<b>${picked.length}</b> / ${open.length} 枚を選択中</span>
      <button data-gp="all">全部選ぶ</button><button data-gp="none">全部外す</button>
      ${!active && b.images.length >= 3 && S.project.kind === "character" ? `<button id="g-cons" title="すべての生成候補の顔立ちを、キャラ判定 AI で比べます">顔立ちのそろい具合を確認</button>` : ""}
      <div class="spacer"></div>
      <button class="primary" id="g-import" ${picked.length ? "" : "disabled"}>選んだ ${picked.length} 枚を素材に追加</button></div>
      <div class="muted small">${S.project.kind === "style" ? "画風が出ていないもの・崩れたものを外してから追加してください。題材がかたよらないように残すと、別の被写体にも画風が出る LoRA になりやすくなります。追加後は通常の素材と同じように判定・説明文の自動作成が行われます。" : `同じキャラに見えないもの・崩れたものを外してから追加してください。生成が終わると、キャラ判定 AI がほかと顔立ちが違う可能性のある画像を見つけて選択を外します（自分で選び直したものはそのままにします）。顔立ちがそろった素材ほど、キャラの特徴がはっきりした LoRA になります。追加後は通常の素材と同じように判定・説明文の自動作成が行われます。`}</div>` : ""}`;
  drawGenCards($(".ggrid", wrap), b, active, url);
  if ($("#gb-sel")) $("#gb-sel").onchange = (e) => { S.genBatch = +e.target.value; drawGenBatches(); };
  $$("[data-gc]", el).forEach((x) => x.onclick = async () => {
    x.disabled = true;
    try { const r = await api(`/api/projects/${S.project.id}/gen/${b.id}/${x.dataset.gc}`, { body: {} }); toast(r.message); } catch (e) { fail(e); }
    reloadGen();
  });
  // file deletion cannot be undone: the first click arms the button, the second one deletes
  $$("[data-gdel]", el).forEach((x) => x.onclick = async () => {
    if (!x.dataset.armed) {
      const label = x.textContent;
      x.dataset.armed = "1";
      x.textContent = x.dataset.gdel === "all" ? `もう一度押すと ${x.dataset.n} 枚の生成画像を削除（素材に追加済みの分は残ります）` : `もう一度押すと ${x.dataset.n} 枚を削除`;
      x.classList.add("danger");
      setTimeout(() => { if (x.isConnected) { delete x.dataset.armed; x.textContent = label; } }, 8000);
      return;
    }
    x.disabled = true;
    try {
      const r = await api(`/api/projects/${S.project.id}/gen/${b.id}/files/delete`, { body: { scope: x.dataset.gdel } });
      toast(`${r.deleted} 枚を削除しました（${r.freed_mb} MB）`);
    } catch (e) { fail(e); }
    reloadGen();
  });
  const setPick = async (ids, picked) => {
    b.images.forEach((im) => { if (ids.includes(im.id)) im.picked = picked; });
    drawGenBatches();
    try { await api(`/api/projects/${S.project.id}/gen/pick`, { body: { ids, picked } }); } catch (e) { fail(e); reloadGen(); }
  };
  wrap._b = b; wrap._pick = setPick;
  $$("[data-gp]", el).forEach((x) => x.onclick = () => setPick(open.map((i) => i.id), x.dataset.gp === "all"));
  if ($("#g-cons")) $("#g-cons").onclick = async () => {
    $("#g-cons").disabled = true;
    try { await api(`/api/projects/${S.project.id}/gen/consistency`, { body: {} }); toast("顔立ちのそろい具合を確認しています（GPU の順番待ちに入りました）"); } catch (e) { fail(e); }
  };
  if ($("#g-import")) $("#g-import").onclick = async () => {
    const btn = $("#g-import"); btn.disabled = true; btn.textContent = "追加しています…";
    try {
      const r = await api(`/api/projects/${S.project.id}/gen/import`, { body: { ids: picked.map((i) => i.id) } });
      toast(`${r.added} 枚を素材に追加しました。判定と説明文の作成を始めます。`);
      refreshImages(); reloadGen();
    } catch (e) { fail(e); btn.disabled = false; }
  };
}

// ------------------------------------------------------------------ training tab
/* Settings varied in one go: each value becomes its own reserved run (server: train/batch). */
const BATCH_SUGGEST = {
  learning_rate: (v) => [v / 2, v, v * 2],
  max_train_steps: (v) => [Math.round(v * 0.6 / 100) * 100, v, Math.round(v * 1.5 / 100) * 100],
  network_dim: (v) => [Math.max(4, v / 2), v, Math.min(128, v * 2)],
  seed: (v) => [v, v + 1, v + 2],
  resolution: (v) => [Math.max(512, v - 256), v],
  caption_dropout_rate: () => [0, 0.05, 0.1],
};
const BATCH_HELP = {
  learning_rate: "特徴が弱いときは大きく、崩れるときは小さくすると変わることがあります。",
  max_train_steps: "途中保存も比べられるので、長めの回を 1 つ入れると比べられる範囲が広がります。",
  network_dim: "alpha は、今の rank との比を保ったまま一緒に変えます。",
  seed: "設定は同じまま、学習ごとの偶然の差がどのくらいあるかを確かめられます。",
  resolution: "",
  caption_dropout_rate: "説明文を使わずに学習する割合です。",
};
function drawBatch(plan) {
  const el = $("#t-batch-body");
  if (!el || !plan.values) return;
  const fields = Object.keys(BATCH_SUGGEST).filter((k) => k in plan.values);
  const cur = S.batchField && fields.includes(S.batchField) ? S.batchField : fields[0];
  const fmt = (x) => (typeof x === "number" && !Number.isInteger(x)) ? +x.toPrecision(3) : x;
  const sugg = [...new Set(BATCH_SUGGEST[cur](plan.values[cur]).map(fmt))].join(", ");
  el.innerHTML = `<div class="muted small" style="margin:6px 0">選んだ設定だけを変えた学習を、値の数だけ順番待ちに入れます。ほかの設定と素材は今の内容のまま、予約した時点で固定します。
      終わった回は、結果の画面の「回どうしを比べる」で並べて比べられます。</div>
    <div class="row small">変える設定 <select id="b-field">${fields.map((k) => `<option value="${k}" ${k === cur ? "selected" : ""}>${esc(plan.fields[k][0])}（今 ${esc(plan.values[k])}）</option>`).join("")}</select>
      試す値 <input type="text" id="b-vals" value="${esc(sugg)}" style="width:200px"> <span class="muted">カンマ区切り・最大 8 件</span></div>
    ${BATCH_HELP[cur] ? `<div class="muted small" style="margin-top:4px">${esc(BATCH_HELP[cur])}</div>` : ""}
    <div class="row" style="margin-top:6px"><button id="b-go" ${plan.preflight.errors.length ? "disabled" : ""}>予約する</button><span class="muted small" id="b-count"></span></div>`;
  const vals = () => $("#b-vals").value.split(",").map((x) => x.trim()).filter(Boolean);
  const count = () => { $("#b-count").textContent = `${vals().length} 件を予約します（1 件ずつ順番に学習します）`; };
  $("#b-vals").oninput = count; count();
  $("#b-field").onchange = (e) => { S.batchField = e.target.value; drawBatch(plan); };
  $("#b-go").onclick = async () => {
    const b = $("#b-go"); b.disabled = true;
    try {
      const r = await api(`/api/projects/${S.project.id}/train/batch`, { body: { field: cur, values: vals() } });
      toast(`${r.run_ids.length} 件の学習を予約しました（#${r.run_ids.join(", #")}）`);
      loadRuns(); drawQueue();
    } catch (e) { fail(e); }
    b.disabled = false;
  };
}

/* The job queue (all projects): what runs now and what waits, with reordering and cancelling of reserved runs. */
async function drawQueue() {
  const el = $("#queue");
  if (!el) return;
  let q;
  try { q = await api("/api/queue"); } catch { return; }
  if (!$("#queue")) return;
  if (!q.current && !q.waiting.length) {
    el.innerHTML = S.route[0] === "jobs" ? `<div class="panel" style="margin:10px 0"><h2 style="margin-top:0">実行の順番</h2>
      <div class="muted">今動いている GPU の処理と、待っている処理はありません。</div></div>` : "";
    return;
  }
  const name = (j) => j.kind === "train" ? `${esc(j.project_name || j.project)} の学習 ${esc(j.run_name || "#" + j.run)}${j.resume ? "（続きから）" : ""}`
    : `${j.project_name ? esc(j.project_name) + " の" : ""}${esc(j.label || "GPU の処理")}${j.detail ? ` <span class="muted small">「${esc(j.detail)}」</span>` : ""}`;
  const link = (j) => j.project && j.project !== "__imagegen__" ? `#/p/${j.project}/train` : "#/gen";
  const n = q.waiting.length;
  el.innerHTML = `<div class="panel" style="margin:10px 0"><h2 style="margin-top:0">実行の順番</h2>
    <div class="muted small">GPU を使う処理（学習・素材の生成・比較画像・画像生成）は 1 つずつ順番に動きます。待っている処理は、並べ替えと取り消しができます。アプリを終了しても、まだ始まっていない学習の予約は残ります。${S.route[0] === "jobs" ? "" : ` <a href="#/jobs">ジョブ一覧</a>`}</div>
    ${q.current ? `<div class="queue-row"><span class="qn">▶</span><span class="qt"><a href="${link(q.current)}">${name(q.current)}</a>
      <span class="muted small">${esc((q.current.progress || {}).message || "実行中")}</span></span></div>` : ""}
    ${q.waiting.map((j, i) => `<div class="queue-row" data-uid="${j.uid}"><span class="qn">${i + 1}</span>
      <span class="qt"><a href="${link(j)}">${name(j)}</a></span>
      <button class="small" data-qm="top" ${i === 0 ? "disabled" : ""} title="いちばん先にする">先頭へ</button>
      <button class="small" data-qm="up" ${i === 0 ? "disabled" : ""} title="1 つ前へ">↑</button>
      <button class="small" data-qm="down" ${i === n - 1 ? "disabled" : ""} title="1 つ後へ">↓</button>
      ${j.cancellable ? `<button class="small" data-qc>取り消す</button>` : ""}</div>`).join("")}</div>`;
  $$("[data-qm]", el).forEach((b) => b.onclick = async () => {
    try { await api(`/api/queue/${b.closest("[data-uid]").dataset.uid}/move`, { body: { to: b.dataset.qm } }); drawQueue(); } catch (e) { fail(e); drawQueue(); }
  });
  $$("[data-qc]", el).forEach((b) => b.onclick = async () => {
    b.disabled = true;
    try { const r = await api(`/api/queue/${b.closest("[data-uid]").dataset.uid}/cancel`, { body: {} }); toast(r.message); drawQueue(); if ($("#t-runs")) loadRuns(); }
    catch (e) { fail(e); drawQueue(); }
  });
}
const refreshQueueSoon = debounce(drawQueue, 400);

async function renderTrain() {
  $("#tab").innerHTML = `<div class="col">
    <div class="panel" id="t-plan"><div class="muted">確認中…</div></div>
    <div class="panel" id="t-task"></div>
    <div id="queue"></div>
    <div class="panel"><h2>学習の履歴</h2><div id="t-runs"><div class="muted">読み込み中…</div></div></div></div>`;
  loadPlan();
  loadRuns();
  drawQueue();
  drawTaskForm();
}

/* Register a LoRA creation task for this project: generate the material (optional), wait for the analysis and
   captions, then train.  Tasks run one after another; the list is on the job page. */
async function drawTaskForm() {
  const el = $("#t-task");
  if (!el) return;
  const pid = S.project.id;
  let gen = null;
  try { gen = await api(`/api/projects/${pid}/gen`); } catch { /* no generation for this project */ }
  if (!$("#t-task") || !S.project || S.project.id !== pid) return;
  const hasRecipe = gen && gen.recipe && gen.recipe.gen_model_id;
  const canGen = hasRecipe && !(gen.check || []).length;
  el.innerHTML = `<h2 style="margin-top:0">LoRA 作成タスクに登録</h2>
    <div class="muted small">${canGen ? "素材の生成、解析と説明文の作成、学習を、順番が来たら続けて自動で進めます。" : "解析と説明文の作成が終わるのを待ってから、自動で学習を始めます。"}
      いくつかのプロジェクトを登録しておくと、上から 1 つずつ進みます。学習の設定は、学習を始める時点のこのプロジェクトの設定を使います。</div>
    <div class="row" style="margin-top:8px">
      ${canGen ? `<label class="row small"><input type="checkbox" id="tk-gen" ${S.images && S.images.length ? "" : "checked"}> 素材を生成してから学習する</label>
        <label class="row small">生成する枚数 <input type="number" id="tk-n" min="1" max="200" value="${esc(gen.recipe.count || 48)}" style="width:80px"></label>` : ""}
      <label class="row small">名前 <input type="text" id="tk-name" placeholder="省略可" style="width:160px"></label>
      <button id="tk-go">登録する</button> <a class="small" href="#/jobs">ジョブ一覧</a></div>
    ${hasRecipe && !canGen ? `<div class="muted small">素材の生成の設定に足りないところがあるため、今は「今の素材で学習」だけ登録できます（${esc(gen.check.join(" / "))}）。</div>` : ""}`;
  $("#tk-go").onclick = async () => {
    const b = $("#tk-go");
    b.disabled = true;
    const n = $("#tk-gen") && $("#tk-gen").checked ? parseInt($("#tk-n").value, 10) || 0 : 0;
    try {
      const r = await api(`/api/projects/${pid}/tasks`, { body: { gen_count: n, name: $("#tk-name").value } });
      toast(r.ahead > 0 ? `タスクを登録しました。前に ${r.ahead} 件あり、順番が来たら始まります。` : "タスクを登録しました。すぐに始まります。");
      $("#tk-name").value = "";
    } catch (e) { fail(e); }
    b.disabled = false;
  };
}

const TASK_STATUS = { waiting: "順番待ち", generating: "素材を生成中", analyzing: "解析と説明文の作成", training: "学習中",
  done: "完了", failed: "失敗", cancelled: "取り消し" };
const JOB_KIND = { train: "学習", gen: "素材の生成", eval: "比較画像", imagegen: "画像生成", other: "確認" };

/* The job page: what runs now (with stop / pause), the GPU queue (reorder, cancel, pause) and the LoRA creation
   tasks (reorder, cancel, pause).  Redrawn from the server on every queue / progress event while it is open. */
async function renderJobs() {
  S.project = null;
  $("#crumbs").innerHTML = `<span>›</span><span>ジョブ</span>`;
  main.innerHTML = `<div class="jobs">
    <div class="row" style="justify-content:space-between;align-items:baseline"><h1 style="margin:0">ジョブ</h1>
      <span class="muted small">GPU を使う処理（学習・素材の生成・比較画像・画像生成）は、1 つずつ順番に動きます。画像からプロンプトの処理は、その横で動きます。</span></div>
    <div id="jobs-body"><div class="muted">読み込み中…</div></div></div>`;
  drawJobs();
}

let _jobsBusy = false, _jobsAgain = false;
async function drawJobs() {
  if (S.route[0] !== "jobs" || !$("#jobs-body")) return;
  if (_jobsBusy) { _jobsAgain = true; return; }
  _jobsBusy = true;
  let q, d;
  // an app started before the job page existed has no task list: the page still shows the GPU queue
  try { [q, d] = await Promise.all([api("/api/queue"), api("/api/tasks").catch(() => ({ tasks: [], paused: false, old: true }))]); }
  catch (e) { if ($("#jobs-body")) $("#jobs-body").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
  finally { _jobsBusy = false; }
  if (_jobsAgain) { _jobsAgain = false; return drawJobs(); }
  const el = $("#jobs-body");
  if (!el || !q) return;
  const openDetails = !!$("#jb-ended[open]");
  el.innerHTML = (d.old ? `<div class="jpaused">LoRAtelier を起動し直すと、このページのすべての機能（止める・一時停止・LoRA 作成タスク）が使えるようになります。</div>` : "")
    + jobsCurrentHtml(q) + jobsI2tHtml(q) + jobsQueueHtml(q) + jobsTasksHtml(d, openDetails);
  bindJobs(el, q);
}
/* Progress events can come several times a second: redraw at most about once a second, and not while a
   stop button waits for its second click (a redraw would reset it). */
let _jobsTimer = null;
function refreshJobsSoon() {
  if (_jobsTimer || S.route[0] !== "jobs") return;
  _jobsTimer = setTimeout(() => { _jobsTimer = null; if ($("#jobs-body .armed")) return refreshJobsSoon(); drawJobs(); }, 800);
}

const jobKind = (k) => `<span class="jkind jk-${k}">${JOB_KIND[k] || "処理"}</span>`;
const jobLink = (j) => j.project === "__imagegen__" ? "#/gen" : j.kind === "gen" ? `#/p/${j.project}/material` : `#/p/${j.project}/train`;
function jobTitle(j) {
  if (j.kind === "train") return `${esc(j.project_name || j.project)} <span class="muted">の学習 ${esc(j.run_name || "#" + j.run)}${j.resume ? "（続きから）" : ""}</span>`;
  if (j.kind === "imagegen") return `画像生成${j.detail ? ` <span class="muted">「${esc(j.detail)}」</span>` : ""}`;
  return `${esc(j.project_name || "")} <span class="muted">${esc(j.label || "")}</span>`;
}

function jobsCurrentHtml(q) {
  const c = q.current;
  if (!c) return `<section class="panel jb"><h2>いま動いている処理</h2><div class="muted">動いている処理はありません。${q.paused && q.waiting.length ? "順番待ちを一時停止しているため、次の処理は始まりません。" : ""}</div></section>`;
  const pct = c.fraction == null ? null : Math.round(c.fraction * 100);
  return `<section class="panel jb jb-now"><h2>いま動いている処理</h2>
    <div class="jnow">
      <div class="jnow-head">${jobKind(c.kind)}<a href="${jobLink(c)}" class="jnow-title">${jobTitle(c)}</a></div>
      <div class="jbar ${pct == null ? "indet" : ""}"><span style="${pct == null ? "" : `width:${pct}%`}"></span></div>
      <div class="row small"><span>${esc(c.message || "実行中")}</span>${pct == null ? "" : `<span class="muted">${pct}%</span>`}</div>
      ${(c.controls || []).length ? `<div class="row jctl">${c.controls.map((x) => `<button class="${x.danger ? "danger" : ""}" data-cur="${x.action}" title="${esc(x.title || "")}">${esc(x.label)}</button>`).join("")}
        <span class="muted small" id="jctl-note">${esc(c.controls[0].title || "")}</span></div>`
        : `<div class="muted small">この処理は途中で止められません。すぐに終わります。</div>`}
    </div></section>`;
}

/* 画像からプロンプト: runs in its own worker beside the GPU queue (an analysis that needs the GPU waits for it) */
function jobsI2tHtml(q) {
  if (!("i2t" in q)) return "";   // an app started before this section existed
  const j = q.i2t;
  if (!j) return `<section class="panel jb"><h2>画像からプロンプト</h2><div class="muted">動いている処理はありません。</div></section>`;
  const pct = j.fraction == null || j.fraction < 0 ? null : Math.round(j.fraction * 100);
  return `<section class="panel jb jb-now"><h2>画像からプロンプト</h2>
    <div class="jnow">
      <div class="jnow-head"><span class="jkind jk-i2t">画像からプロンプト</span><a href="#/i2t" class="jnow-title">${esc(j.label)}</a></div>
      <div class="jbar ${pct == null ? "indet" : ""}"><span style="${pct == null ? "" : `width:${pct}%`}"></span></div>
      <div class="row small"><span>${esc(j.paused ? "一時停止中" : j.message || "実行中")}</span>${pct == null ? "" : `<span class="muted">${pct}%</span>`}</div>
      <div class="row jctl">${j.pausable ? `<button data-i2t="${j.paused ? "resume" : "pause"}">${j.paused ? "▶ 再開" : "⏸ 一時停止"}</button>` : ""}
        ${j.cancellable ? `<button class="danger" data-i2t="cancel">中止</button>` : `<span class="muted small">この処理は今は止められません。</span>`}</div>
    </div></section>`;
}

function jobsQueueHtml(q) {
  const n = q.waiting.length;
  return `<section class="panel jb">
    <div class="jb-head"><h2>順番待ち <span class="muted">${n} 件</span></h2>
      <button class="${q.paused ? "primary" : ""}" data-qpause="${q.paused ? 0 : 1}">${q.paused ? "▶ 再開する" : "⏸ 一時停止"}</button></div>
    ${q.paused ? `<div class="jpaused">一時停止中です。いま動いている処理は最後まで進みますが、次の処理は「再開する」を押すまで始まりません。</div>` : ""}
    ${n ? `<div class="jlist">${q.waiting.map((j, i) => `<div class="jrow" data-uid="${j.uid}">
        <span class="jnum">${i + 1}</span>${jobKind(j.kind)}<a class="jtitle" href="${jobLink(j)}">${jobTitle(j)}</a>
        <span class="jbtns"><button class="small" data-qm="top" ${i === 0 ? "disabled" : ""} title="いちばん先にする">先頭へ</button>
          <button class="small" data-qm="up" ${i === 0 ? "disabled" : ""} title="1 つ前へ">▲</button>
          <button class="small" data-qm="down" ${i === n - 1 ? "disabled" : ""} title="1 つ後へ">▼</button>
          ${j.cancellable ? `<button class="small danger" data-qc>取り消す</button>` : ""}</span></div>`).join("")}</div>`
      : `<div class="muted">待っている処理はありません。</div>`}
  </section>`;
}

function taskStages(t) {
  const st = t.gen_count ? [["generating", "素材の生成"], ["analyzing", "解析・説明文"], ["training", "学習"]] : [["analyzing", "解析・説明文"], ["training", "学習"]];
  const cur = st.findIndex(([k]) => k === t.status);
  return `<div class="jsteps">${st.map(([, label], i) => `<span class="jstep ${i < cur ? "done" : i === cur ? "now" : ""}">${i < cur ? "✓ " : ""}${label}</span>`).join(`<span class="jarrow">›</span>`)}</div>`;
}

function jobsTasksHtml(d, openEnded) {
  const active = d.tasks.filter((t) => ["generating", "analyzing", "training"].includes(t.status));
  const waiting = d.tasks.filter((t) => t.status === "waiting");
  const ended = d.tasks.filter((t) => ["done", "failed", "cancelled"].includes(t.status)).reverse();
  const title = (t) => `<a class="jtitle" href="#/p/${esc(t.project)}/train">${esc(t.project_name || t.project)}${t.name ? ` <span class="muted">${esc(t.name)}</span>` : ""}</a>
    <span class="muted small">${t.gen_count ? `${t.gen_count} 枚生成してから学習` : "今の素材で学習"}</span>`;
  const n = waiting.length;
  return `<section class="panel jb">
    <div class="jb-head"><h2>LoRA 作成タスク <span class="muted">${active.length + n} 件</span></h2>
      <button class="${d.paused ? "primary" : ""}" data-tpause="${d.paused ? 0 : 1}">${d.paused ? "▶ 再開する" : "⏸ 一時停止"}</button></div>
    <div class="muted small">プロジェクトの学習画面の「LoRA 作成タスクに登録」で登録したものです。素材の生成、解析と説明文の作成、学習を 1 つずつ順番に進めます。</div>
    ${d.paused ? `<div class="jpaused">一時停止中です。いま進んでいる段階（生成・解析・学習）は最後まで進みますが、次の段階と次のタスクは「再開する」を押すまで始まりません。</div>` : ""}
    ${active.map((t) => `<div class="jtask now" data-tid="${t.id}">
        <div class="jnow-head"><span class="jkind jk-task">${TASK_STATUS[t.status]}</span>${title(t)}
          <span class="jbtns"><button class="small danger" data-tc>取り消す</button></span></div>
        ${taskStages(t)}<div class="small">${esc(t.message || "")}</div></div>`).join("")}
    ${n ? `<div class="jlist">${waiting.map((t, i) => `<div class="jrow" data-tid="${t.id}"><span class="jnum">${i + 1}</span>${title(t)}
        <span class="jbtns"><button class="small" data-tm="top" ${i === 0 ? "disabled" : ""} title="いちばん先にする">先頭へ</button>
          <button class="small" data-tm="up" ${i === 0 ? "disabled" : ""} title="1 つ前へ">▲</button>
          <button class="small" data-tm="down" ${i === n - 1 ? "disabled" : ""} title="1 つ後へ">▼</button>
          <button class="small danger" data-tc>取り消す</button></span></div>`).join("")}</div>` : ""}
    ${!active.length && !n ? `<div class="muted" style="margin-top:8px">進行中や待っているタスクはありません。</div>` : ""}
    ${ended.length ? `<details id="jb-ended" ${openEnded ? "open" : ""} style="margin-top:10px"><summary class="small">終わったタスク（${ended.length} 件）</summary>
      ${ended.map((t) => `<div class="jrow"><span class="jkind jk-${t.status}">${TASK_STATUS[t.status]}</span>${title(t)}<div class="small muted jmsg">${esc(t.message || "")}</div></div>`).join("")}
      <button class="small" id="tk-clear" style="margin-top:6px">終わったタスクを一覧から消す</button></details>` : ""}
  </section>`;
}

/* Buttons that stop something take two clicks (the first one asks), the others act at once. */
function confirmClick(b, ask, fn) {
  if (b.dataset.armed) return fn();
  const label = b.textContent;
  b.dataset.armed = "1"; b.textContent = ask; b.classList.add("armed");
  setTimeout(() => { if (b.isConnected) { delete b.dataset.armed; b.textContent = label; b.classList.remove("armed"); } }, 4000);
}

function bindJobs(el, q) {
  const act = async (b, path, body) => {
    b.disabled = true;
    try { const r = await api(path, { body: body || {} }); if (r && r.message) toast(r.message); } catch (e) { fail(e); }
    drawJobs();
  };
  $$("[data-cur]", el).forEach((b) => {
    b.onmouseenter = () => { if ($("#jctl-note")) $("#jctl-note").textContent = b.title; };
    b.onclick = () => {
      const go = () => act(b, `/api/queue/current/${b.dataset.cur}`, { uid: q.current.uid });
      b.classList.contains("danger") ? confirmClick(b, "もう一度押すと実行", go) : go();
    };
  });
  $$("[data-qpause]", el).forEach((b) => b.onclick = () => act(b, "/api/queue/pause", { on: b.dataset.qpause === "1" }));
  $$("[data-i2t]", el).forEach((b) => b.onclick = () => {
    const go = () => act(b, `/api/i2t/${b.dataset.i2t}`);
    b.dataset.i2t === "cancel" ? confirmClick(b, "もう一度押すと中止", go) : go();
  });
  $$("[data-tpause]", el).forEach((b) => b.onclick = () => act(b, "/api/tasks/pause", { on: b.dataset.tpause === "1" }));
  $$("[data-qm]", el).forEach((b) => b.onclick = () => act(b, `/api/queue/${b.closest("[data-uid]").dataset.uid}/move`, { to: b.dataset.qm }));
  $$("[data-qc]", el).forEach((b) => b.onclick = () => confirmClick(b, "もう一度押すと取り消し",
    () => act(b, `/api/queue/${b.closest("[data-uid]").dataset.uid}/cancel`)));
  $$("[data-tm]", el).forEach((b) => b.onclick = () => act(b, `/api/tasks/${b.closest("[data-tid]").dataset.tid}/move`, { to: b.dataset.tm }));
  $$("[data-tc]", el).forEach((b) => b.onclick = () => confirmClick(b, "もう一度押すと取り消し",
    () => act(b, `/api/tasks/${b.closest("[data-tid]").dataset.tid}/cancel`)));
  if ($("#tk-clear", el)) $("#tk-clear", el).onclick = (e) => act(e.target, "/api/tasks/clear");
}

/* Masked training (masks.py): the loss counts mainly inside an automatic mask of the person (or of the background
   for place LoRAs).  The setting is per project; the masks are made in the background after it is turned on. */
function maskPanelHtml(m) {
  if (!m || !m.mode) return "";
  const what = m.mode_label;
  if (m.unsupported) return `<div class="muted small" style="margin-top:8px">マスク学習: ${esc(m.unsupported)}</div>`;
  const sm = m.summary;
  return `<div class="note" style="margin-top:10px">
    <label><input type="checkbox" id="t-mask-on" ${m.on ? "checked" : ""}> <b>マスク学習: ${esc(what)}を中心に学習する</b></label>
    <div class="small">${m.mode === "person"
      ? "画像ごとに人物の範囲を自動で見つけ、学習をその範囲に集中させます。背景や、ほかに写っている人物はあまり覚えません。人物が複数いるときは、いちばん大きく写っている人物を対象にします。"
      : "画像ごとに人物の範囲を自動で見つけ、そこを除いた背景を中心に学習します。写っている人物はあまり覚えません。"}</div>
    ${m.on ? `<div class="row small" style="margin-top:6px">マスクの外の効き方 <input type="number" id="t-mask-floor" min="0" max="1" step="0.05" value="${m.floor}" style="width:70px">
      <span class="muted">0 で外を無視、1 でマスクなしと同じです（初期値 0.1）。</span></div>
      ${sm ? `<div class="small" style="margin-top:6px">マスク: できた ${sm.done} 枚${sm.waiting ? ` ・ <span class="spin"></span> 作成中 ${sm.waiting} 枚` : ""}${sm.none ? ` ・ 人物が見つからない ${sm.none} 枚` : ""}${sm.failed ? ` ・ 作れなかった ${sm.failed} 枚` : ""}${sm.off ? ` ・ 使わない ${sm.off} 枚` : ""}
        <span class="muted">（素材の画面で、画像ごとにマスクを確かめたり、使わないようにしたりできます）</span></div>` : ""}` : ""}
    <div class="muted small" style="margin-top:4px">最初に使うときは、マスクを作る AI をダウンロードします（イラスト用 約 176MB、写真用 約 177MB）。効果は素材によって変わり、このツールではまだ確かめていません。</div>
  </div>`;
}

function bindMaskPanel(m) {
  const save = async (mask) => {
    try { S.project = await api(`/api/projects/${S.project.id}`, { method: "PATCH", body: { mask } }); loadPlan(); } catch (e) { fail(e); }
  };
  if ($("#t-mask-on")) $("#t-mask-on").onchange = (e) => save({ on: e.target.checked, floor: m.floor });
  if ($("#t-mask-floor")) $("#t-mask-floor").onchange = (e) => {
    const v = parseFloat(e.target.value);
    if (isNaN(v) || v < 0 || v > 1) { toast("0 から 1 の間で入れてください"); e.target.value = m.floor; return; }
    save({ on: true, floor: v });
  };
}

async function loadPlan() {
  const el = $("#t-plan");
  let plan;
  try { plan = S.plan = await api(`/api/projects/${S.project.id}/train/plan`); }
  catch (e) { if ($("#t-plan") === el) el.innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  if (!el || $("#t-plan") !== el) return;   // the tab was left meanwhile
  // the field being typed in (another than the one just saved) and the open sections survive the redraw
  const fs = formState(el), typing = fs.focus && fs.focus[0];
  const keep = { v: typing && typing in fs.v ? { [typing]: fs.v[typing] } : {}, open: fs.open, focus: fs.focus };
  const pf = plan.preflight;
  if (!plan.values) { el.innerHTML = `<h2>学習の準備</h2>${pf.errors.map((x) => `<div class="err">${esc(x)}</div>`).join("")}<div class="small">上の「LoRA を作るモデル」で選んでください。選べるモデルがないときは、<a href="#/settings">設定</a>でモデルを追加します。</div>`; return; }
  const v = plan.values, dflt = plan.defaults, changed = new Set(plan.changed);
  const keys = Object.keys(plan.fields).filter((k) => k in v);
  const rel = plan.relative || {}, imp = plan.imported;
  const skipped = new Set(imp ? imp.skipped : []);
  const relNote = (k) => k === "max_train_steps" && rel.steps_scale && changed.has(k) ? `自動の ${+rel.steps_scale.toFixed(2)} 倍です。素材の枚数が変わると、この値も変わります。`
    : k === "save_every_n_steps" && rel.save_count && changed.has(k) ? `途中保存が約 ${Math.round(rel.save_count)} 回になる間隔です。学習ステップ数が変わると、この値も変わります。` : "";
  const RELKEY = { max_train_steps: "steps_scale", save_every_n_steps: "save_count" };
  el.innerHTML = `
    <div class="row" style="justify-content:space-between"><h2>学習の準備</h2>
      <button class="primary big" id="t-go" ${pf.errors.length ? "disabled" : ""}>学習を開始</button></div>
    <div class="row small muted">${S.project.kind === "enhance" && (S.project.enhance || {}).method === "words" ? "言葉から作る（画像は使いません）" : `採用 ${plan.used} 枚`} ・ モデル ${esc(plan.model.name)} ・ ${esc(plan.model.variant_info.label || "")}
      ${pf.info.gpu ? ` ・ GPU ${esc(pf.info.gpu.name)} (${Math.round(pf.info.gpu.vram_total_mb / 1024)}GB)` : ""}
      ${pf.info.disk_need_gb ? ` ・ 必要容量 約${pf.info.disk_need_gb}GB / 空き ${pf.info.disk_free_gb}GB` : ""}</div>
    ${pf.errors.map((x) => `<div class="err">${esc(x)}</div>`).join("")}
    ${pf.warnings.map((x) => `<div class="warntext">${esc(x)}</div>`).join("")}
    ${maskPanelHtml(plan.mask)}
    <details style="margin-top:10px"><summary>学習後の比較で使うほかのモデル <span class="muted small">（マージモデルなどでの出方も自動で確かめるとき）</span></summary>
      <div id="t-xm" style="margin-top:6px"></div></details>
    <details ${changed.size ? "open" : ""} style="margin-top:10px"><summary>詳細設定 ${changed.size ? `<span class="warntext small">（${changed.size} 項目を変更中）</span>` : `<span class="muted small">（自動設定のまま使えます）</span>`}</summary>
      <div class="fields" style="margin-top:10px">${keys.map((k) => {
        const [label, type, help] = plan.fields[k];
        return `<div class="field ${changed.has(k) ? "changed" : ""}"><div class="lab"><span>${esc(label)}</span>
          ${changed.has(k) ? `<a href="#" data-reset="${k}">初期値 ${esc(dflt[k])} に戻す</a>` : `<span>初期値</span>`}</div>
          <input type="${type === "str" ? "text" : "number"}" step="any" data-k="${k}" data-type="${type}" value="${esc(v[k])}">
          ${relNote(k) ? `<div class="muted small">${esc(relNote(k))}</div>` : ""}
          ${skipped.has(k) && !changed.has(k) ? `<div class="warntext small">読み込んだ設定は ${esc(imp.from_family)} 用のため、この項目は写さず初期値にしています。</div>` : ""}
          ${help ? `<div class="muted small">${esc(help)}</div>` : ""}</div>`; }).join("")}</div>
      <div class="row" style="margin-top:8px">${changed.size ? `<button id="t-resetall">すべて初期値に戻す</button>` : ""}
        <span class="muted small">プリセット ${esc(plan.preset_version)}。初期値はエンジンの資料と検証記録から決めています。</span></div>
    </details>
    ${imp ? `<div class="small" style="margin-top:8px">${esc(imp.from)}から設定を読み込みました。${skipped.size ? `<span class="warntext">${esc(imp.from_family)} 用の設定のため、${esc([...skipped].map((k) => plan.fields[k] ? plan.fields[k][0] : k).join("・"))} は写さず、このモデルの初期値にしています。</span>` : ""}</div>` : ""}
    <details id="t-batch" style="margin-top:10px" ${S.batchOpen ? "open" : ""}><summary>設定を変えて何通りか予約する</summary><div id="t-batch-body"></div></details>
    <div class="row small" id="t-presetrow" style="margin-top:8px">
      <button id="t-load">別のプロジェクト・プリセットから設定を読み込む</button>
      ${changed.size ? `<button id="t-savepreset">今の設定をプリセットとして保存</button>` : ""}</div>
    <div id="t-sources"></div>`;
  restoreForm(el, keep);
  if (S.project.kind === "enhance" && (S.project.enhance || {}).method !== "words") drawPairPreview(el);
  $("#t-go").onclick = startTraining;
  if (plan.model && $("#t-xm")) drawExtraModels($("#t-xm"), plan.model.family, plan.model.id);
  bindMaskPanel(plan.mask);
  $("#t-batch").ontoggle = (e) => { S.batchOpen = e.target.open; };
  drawBatch(plan);
  const save = async (ov) => { try { S.project = await api(`/api/projects/${S.project.id}`, { method: "PATCH", body: { train_overrides: ov } }); loadPlan(); } catch (e) { fail(e); } };
  $$("[data-k]", el).forEach((inp) => inp.onchange = () => {
    const ov = { ...(S.project.train_overrides || {}) };
    const t = inp.dataset.type;
    ov[inp.dataset.k] = t === "int" ? parseInt(inp.value, 10) : t === "float" ? parseFloat(inp.value) : inp.value;
    delete ov[RELKEY[inp.dataset.k]];   // a typed value is a fixed value
    save(ov);
  });
  $$("[data-reset]", el).forEach((a) => a.onclick = (e) => {
    e.preventDefault();
    const ov = { ...(S.project.train_overrides || {}) };
    delete ov[a.dataset.reset]; delete ov[RELKEY[a.dataset.reset]];
    save(ov);
  });
  if ($("#t-resetall")) $("#t-resetall").onclick = () => save({});
  $("#t-load").onclick = () => $("#t-sources").innerHTML ? ($("#t-sources").innerHTML = "") : drawSources();
  if ($("#t-savepreset")) $("#t-savepreset").onclick = async () => {
    const r = await ask([{ label: "プリセットの名前", value: `${S.project.name}の設定` }], "保存", "#t-presetrow");
    if (!r || !r[0].trim()) return;
    try { await api(`/api/projects/${S.project.id}/train/presets`, { body: { name: r[0].trim() } }); toast(`プリセット「${r[0].trim()}」を保存しました。同じ名前で保存すると置き換わります。`); if ($("#t-sources").innerHTML) drawSources(); }
    catch (e) { fail(e); }
  };
}

/* Pair method: one material image next to its processed partner, so it is visible what difference is learned. */
async function drawPairPreview(el) {
  let imgs = S.images;
  if (!imgs || !imgs.length) { try { imgs = await api(`/api/projects/${S.project.id}/images`); } catch { return; } }
  const list = Array.isArray(imgs) ? imgs : (imgs.images || []);
  const im = list.find((x) => x.use) || list[0];
  if (!im || !$("#t-plan")) return;
  let cat;
  try { cat = await enhanceCatalog(); } catch { return; }
  const ef = cat.effects[(S.project.enhance || {}).effect] || {};
  const u = (side) => `/api/projects/${S.project.id}/enhance/preview/${im.id}?side=${side}`;
  const box = document.createElement("div");
  box.className = "panel"; box.style.marginTop = "10px";
  box.innerHTML = `<b>学習する違いの例</b>
    <div class="muted small">左が素材、右がツールの作った加工画像です。LoRA はこの 2 枚の違い（${esc(ef.label || "")}）だけを覚えます。${ef.direction === "remove" ? "右は質感を弱めた画像です。" : "右は質感を足した画像です。"}拡大すると違いが分かりやすくなります。</div>
    <div class="row" style="margin-top:6px;align-items:flex-start">
      <img src="${u("orig")}" data-zoom="${u("orig")}" style="max-width:48%;cursor:zoom-in" alt="素材">
      <img src="${u("partner")}" data-zoom="${u("partner")}" style="max-width:48%;cursor:zoom-in" alt="加工画像"></div>`;
  el.append(box);
  $$("[data-zoom]", box).forEach((x, i) => x.onclick = () => openLightbox([{ src: u("orig"), cap: "素材" }, { src: u("partner"), cap: "加工画像" }], i));
}

/* Picker for "copy settings from a saved preset or another project".  Family-dependent values
   (learning rate, rank, ...) are copied only between the same model family (server side). */
async function drawSources() {
  const el = $("#t-sources");
  el.innerHTML = `<div class="muted small">読み込み中…</div>`;
  let src;
  try { src = await api(`/api/projects/${S.project.id}/train/sources`); } catch (e) { el.innerHTML = ""; return fail(e); }
  const item = (x, type) => `<div class="row" style="justify-content:space-between;gap:8px;padding:4px 0;border-top:1px solid var(--line)">
    <div style="flex:1;min-width:0"><b>${esc(x.name)}</b> <span class="muted small">${esc(x.family_label)} 用${type === "project" ? ` ・ ${esc(kindShort(x))}` : ""}</span>
      <div class="muted small">${esc(x.summary)}</div>
      ${x.same_family ? "" : `<div class="warntext small">モデルの系列が違うため、学習率・rank などは写らず、このモデルの初期値のままになります。</div>`}</div>
    <span class="row"><button class="small" data-apply="${type}:${esc(x.id)}">読み込む</button>
      ${type === "preset" ? `<button class="small" data-delpreset="${esc(x.id)}">削除</button>` : ""}</span></div>`;
  el.innerHTML = `<div class="panel" style="margin-top:8px">
    <div class="small muted">読み込むと、このプロジェクトで変えていた設定は読み込んだ設定に置き換わります（直後なら元に戻せます）。</div>
    <label class="row small" style="margin:6px 0"><input type="checkbox" id="t-fixed"> 学習ステップ数と保存間隔を、素材の枚数に合わせず元の値のまま写す</label>
    <h3 style="margin:8px 0 2px">保存したプリセット</h3>
    ${src.presets.length ? src.presets.map((x) => item(x, "preset")).join("") : `<div class="muted small">まだありません。設定を変えたあとで「今の設定をプリセットとして保存」を押すと、ここに並びます。</div>`}
    <h3 style="margin:10px 0 2px">ほかのプロジェクト</h3>
    ${src.projects.length ? src.projects.map((x) => item(x, "project")).join("") : `<div class="muted small">設定を初期値から変えたプロジェクトはありません。</div>`}</div>`;
  $$("[data-apply]", el).forEach((b) => b.onclick = async () => {
    const [type, id] = b.dataset.apply.split(":");
    b.disabled = true;
    try {
      const r = await api(`/api/projects/${S.project.id}/train/apply`, { body: { [type === "preset" ? "preset_id" : "project_id"]: id, fixed_steps: $("#t-fixed").checked } });
      S.project = r.project;
      el.innerHTML = "";
      loadPlan();
      toast("設定を読み込みました。詳細設定で内容を確かめてください。", "", { label: "元に戻す", fn: async () => {
        try { S.project = await api(`/api/projects/${S.project.id}/train/restore`, { body: r.previous }); loadPlan(); } catch (e) { fail(e); }
      } });
    } catch (e) { fail(e); b.disabled = false; }
  });
  $$("[data-delpreset]", el).forEach((b) => b.onclick = async () => {
    if (!b.dataset.armed) { b.dataset.armed = "1"; b.textContent = "もう一度押すと削除"; return; }
    try { await api(`/api/train/presets/${b.dataset.delpreset}`, { method: "DELETE" }); drawSources(); } catch (e) { fail(e); }
  });
}

const refreshRunsSoon = debounce(() => { if ($("#t-runs")) loadRuns(); if ($("#r-body")) renderResults(true); }, 500);

async function loadRuns() {
  const el = $("#t-runs");
  if (!el) return;
  try { S.runs = await api(`/api/projects/${S.project.id}/runs`); } catch (e) { return fail(e); }
  if (!S.runs.length) { el.innerHTML = `<div class="muted">まだ学習していません。</div>`; return; }
  el.innerHTML = S.runs.map(runHtml).join("");
  bindRunButtons(el);
}

function spark(hist) {
  if (!hist || hist.length < 3) return "";
  const xs = hist.map((h) => h[0]), ys = hist.map((h) => h[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pts = hist.map(([x, y]) => `${((x - x0) / Math.max(1, x1 - x0) * 218 + 1).toFixed(1)},${(39 - (y - y0) / Math.max(1e-9, y1 - y0) * 37).toFixed(1)}`).join(" ");
  return `<svg class="spark" viewBox="0 0 220 40"><polyline points="${pts}"/></svg><span class="muted small">平均 loss（参考。低さだけで成否は判断できません）</span>`;
}

function runHtml(r) {
  const p = r.progress || {};
  const pct = p.total ? Math.min(100, (p.step || 0) / p.total * 100) : 0;
  const active = ["queued", "preparing", "training", "evaluating"].includes(r.status);
  const ck = r.checkpoints.filter((c) => c.kind === "lora").length;
  const states = r.checkpoints.filter((c) => c.kind === "state" && c.complete !== false);
  return `<div class="run panel" data-run="${r.id}">
    <div class="row"><b>${esc(r.name || "#" + r.id)}</b><span class="st st-${r.status}">${esc(r.status_label)}</span>
      <span class="muted small">${new Date(r.created * 1000).toLocaleString()} ・ ${r.n_images} 枚 ・ ${esc(r.model || "")} ・ ${esc(r.engine)} ${esc(r.engine_commit || "")}</span>
      <div class="spacer"></div>
      ${active ? `<button data-stop="${r.id}">停止</button>${r.status !== "evaluating" ? `<button class="danger" data-kill="${r.id}">強制停止</button>` : ""}` : ""}
      ${r.can_resume ? `<button data-resume="${r.id}">続きから再開</button>` : r.resume_note ? `<span class="muted small">${esc(r.resume_note)}</span>` : ""}
      ${r.status === "done" || ck ? `<a href="#/p/${S.project.id}/results" data-res="${r.id}"><button class="primary">結果を見る</button></a>` : ""}
      <button data-log="${r.id}">技術ログ</button>
      ${active ? "" : `<button class="danger" data-rdel="${r.id}" title="保存版・比較画像・ログを削除します。書き出し済みの LoRA は残ります。">削除</button>`}
    </div>
    ${active ? `<div class="prog" style="margin:8px 0"><div style="width:${pct}%"></div></div>
      <div class="row small"><span>${esc(p.message || "")}</span>${p.total ? `<span class="muted">${p.step || 0} / ${p.total}</span>` : ""}
      ${p.phase === "training" || p.phase === "evaluating" ? `<span class="muted">残り ${fmtDur(p.eta_s)}${p.phase === "training" ? "（＋評価の時間）" : ""}</span>` : ""}
      ${p.loss != null ? `<span class="muted">loss ${p.loss.toFixed(4)}</span>` : ""}</div>` : ""}
    ${p.warning ? `<div class="err" style="margin-top:6px">${esc(p.warning)}</div>` : ""}
    ${spark(p.loss_hist)}
    ${r.error ? `<div class="err" style="margin-top:6px">${esc(r.error.message)}</div>${(r.error.fixes || []).map((f) => `<div class="small">・${esc(f)}</div>`).join("")}` : ""}
    ${!active && r.status !== "done" && p.message ? `<div class="small muted">${esc(p.message)}</div>` : ""}
    <div class="small muted" style="margin-top:4px">途中保存 ${ck} 個${states.length ? ` ・ 再開用データ（オプティマイザー状態込み）: ステップ ${states.map((s) => s.step ?? "最終").join(", ")}` : " ・ 再開用データなし（LoRA のみ）"}
      ${r.changed && r.changed.length ? ` ・ 変更した設定: ${r.changed.map(esc).join(", ")}` : ""}</div>
    ${r.changes && r.changes.length ? `<div class="changes"><span class="muted">前の回から：</span>${r.changes.map(esc).join("。")}。</div>` : ""}
    <pre class="log" id="log-${r.id}" hidden></pre>
  </div>`;
}

function bindRunButtons(el) {
  const pid = S.project.id;
  $$("[data-stop]", el).forEach((b) => b.onclick = async () => {
    b.disabled = true; b.textContent = "停止要求を送信…";
    try { const r = await api(`/api/projects/${pid}/runs/${b.dataset.stop}/stop`, { body: {} }); toast(r.message); loadRuns(); } catch (e) { fail(e); }
  });
  $$("[data-kill]", el).forEach((b) => b.onclick = async () => {
    b.disabled = true;
    try { const r = await api(`/api/projects/${pid}/runs/${b.dataset.kill}/stop`, { body: { force: true } }); toast(r.message); loadRuns(); } catch (e) { fail(e); }
  });
  $$("[data-resume]", el).forEach((b) => b.onclick = async () => {
    try { const r = await api(`/api/projects/${pid}/runs/${b.dataset.resume}/resume`, { body: {} }); toast(`ステップ ${r.resumed_from_step ?? "最終"} の保存地点から再開します`); loadRuns(); } catch (e) { fail(e); }
  });
  $$("[data-rdel]", el).forEach((b) => b.onclick = async () => {
    if (!b.dataset.armed) { b.dataset.armed = "1"; b.textContent = "もう一度押すと削除"; return; }
    b.disabled = true;
    try { await api(`/api/projects/${pid}/runs/${b.dataset.rdel}`, { method: "DELETE" }); toast("学習の履歴を削除しました"); loadRuns(); }
    catch (e) { fail(e); b.disabled = false; }
  });
  $$("[data-res]", el).forEach((a) => a.onclick = () => { S.resultsRun = +a.dataset.res; });
  $$("[data-log]", el).forEach((b) => b.onclick = async () => {
    const pre = $(`#log-${b.dataset.log}`);
    if (!pre.hidden) { pre.hidden = true; return; }
    try { const r = await api(`/api/projects/${pid}/runs/${b.dataset.log}/log?tail=300`); pre.textContent = `${r.file || ""}\n\n${r.text}`; pre.hidden = false; pre.scrollTop = 1e9; } catch (e) { fail(e); }
  });
}

function onRunEvent(m) {
  refreshStatusSoon();
  const el = document.querySelector(`[data-run="${m.run}"]`);
  // the results tab shows no progress: it is redrawn when a run ends (new saved versions come as "checkpoints")
  if (!el) { if ($("#t-runs") || ["done", "failed", "cancelled"].includes(m.status)) refreshRunsSoon(); return; }
  const r = S.runs.find((x) => x.id === m.run);
  if (!r) return;
  if (m.status) { r.status = m.status; r.status_label = { queued: "待機", preparing: "準備", training: "学習", evaluating: "評価", done: "完了", cancelled: "中止", failed: "失敗" }[m.status]; }
  if (m.progress) r.progress = { ...(r.progress || {}), ...m.progress };
  const ended = m.status && ["done", "failed", "cancelled"].includes(m.status);
  if (ended) refreshRunsSoon();
  const logOpen = !$(`#log-${m.run}`).hidden;
  if (logOpen) return; // don't disturb an open log
  if (_pressed && el.contains(_pressed)) return;   // a click under way (停止): the next event redraws
  const tmp = document.createElement("div");
  tmp.innerHTML = runHtml(r);
  el.replaceWith(tmp.firstElementChild);
  bindRunButtons($("#t-runs"));
}
let _pressed = null;   // the element under a mouse button held down
window.addEventListener("mousedown", (e) => { _pressed = e.target; }, true);
window.addEventListener("mouseup", () => { setTimeout(() => { _pressed = null; }); }, true);

// ------------------------------------------------------------------ results tab
/* Each comparison image made for the comparison shown redraws only the comparison table; a new comparison,
   or one that ends, redraws the whole tab. */
function onEvalEvent(m) {
  const R = S.resShown;
  if (m.status === "running" && R && $("#r-cmp") && R.el === $("#r-cmp") && m.run === R.run.id && R.evIds.includes(m.eval)) {
    if (m.eval === R.evId) refreshCmpSoon();
    return;
  }
  refreshRunsSoon();
}
const refreshCmpSoon = debounce(async () => {
  const R = S.resShown;
  if (!R || !S.project || S.project.id !== R.pid || !R.el.isConnected) return;
  let evals;
  try { evals = await api(`/api/projects/${R.pid}/runs/${R.run.id}/evals`); } catch { return; }
  const ev = evals.find((e) => e.id === R.evId);
  if (ev && R.el.isConnected && S.resShown === R) drawCompare(R.run, ev, R.ckpts);
}, 500);

/* A background redraw of the results tab (new saved versions, comparison images) keeps what the user has typed,
   ticked or opened there, and the focused field. */
const formKey = (x) => x.dataset.pick !== undefined || x.id === "r-run" || x.id === "r-ev" ? null
  : x.id ? "#" + x.id : x.type === "checkbox" || x.type === "radio" ? `${x.className}|${x.name}|${x.value}` : null;
function formState(root) {
  const st = { v: {}, open: {}, focus: null };
  if (!root) return st;
  $$("input, textarea, select", root).forEach((x) => { const k = formKey(x); if (k) st.v[k] = x.type === "checkbox" || x.type === "radio" ? x.checked : x.value; });
  $$("details", root).forEach((x) => { const s = $("summary", x); if (s) st.open[s.textContent] = x.open; });
  const a = document.activeElement;
  if (a && root.contains(a) && formKey(a)) st.focus = [formKey(a), a.selectionStart, a.selectionEnd];
  return st;
}
function restoreForm(root, st) {
  if (!root || !st) return;
  $$("input, textarea, select", root).forEach((x) => {
    const k = formKey(x);
    if (!k || !(k in st.v)) return;
    if (x.type === "checkbox" || x.type === "radio") x.checked = st.v[k]; else x.value = st.v[k];
    if (st.focus && st.focus[0] === k) { x.focus(); try { x.setSelectionRange(st.focus[1], st.focus[2]); } catch { /* number inputs */ } }
  });
  $$("details", root).forEach((x) => { const s = $("summary", x); if (s && s.textContent in st.open) x.open = st.open[s.textContent]; });
}

/* 「別のモデル向けに作る」: the same material, captions and crops, trained on another base model (the project's
   model is switched; the results so far stay here). */
async function drawRemodel(pid) {
  const el = $("#r-remodel");
  let d;
  try { d = await api("/api/models"); } catch (e) { return; }
  if (!el || !el.isConnected || S.project.id !== pid) return;
  const cur = S.project.model_id;
  const others = d.models.filter((m) => m.id !== cur && !m.missing.length && m.variant_info.train !== false && m.variant_info.available !== false);
  const curName = (d.models.find((m) => m.id === cur) || {}).name;
  el.innerHTML = `<h2>別のモデル向けに作る</h2>
    <div class="muted small">LoRA は、モデルの系統（SDXL 系、Anima、MOSS-Image など）が違うと使えません。この素材のまま、別のモデルで使う LoRA を作れます。</div>
    ${others.length ? `<div class="row" style="margin-top:8px"><span class="small">今のモデル: ${esc(curName || "未設定")}</span>
        <select id="rm-model">${others.map((m) => `<option value="${m.id}">${esc(m.name)}（${esc(m.variant_info.label || m.family)}）</option>`).join("")}</select>
        <button class="primary" id="rm-go">このモデルで作る</button></div>
      <ul class="small muted" style="margin:6px 0 0;padding-left:1.2em">
        <li>素材、切り抜き、採用はそのまま使います。</li>
        <li>自動で作った説明文は、新しいモデルに合う形に作り直します。自分で直した説明文はそのままなので、必要なら「素材」で確かめてください。</li>
        <li>学習率など、モデルの系統ごとに値が違う詳細設定は、系統が変わるときに初期値に戻します。</li>
        <li>今までの学習結果は、この「結果」にそのまま残ります。上の一覧で回を切り替えると見られます。</li>
      </ul>`
    : `<div class="muted small" style="margin-top:6px">学習に使えるほかのモデルが登録されていません。「設定」画面でモデルを追加すると、ここで選べます。</div>`}`;
  const go = $("#rm-go");
  if (!go) return;
  go.onclick = async () => {
    go.disabled = true;
    try {
      S.project = await api(`/api/projects/${pid}`, { method: "PATCH", body: { model_id: $("#rm-model").value } });
      const reset = S.project.reset_settings || [];
      toast(`LoRA を作るモデルを ${S.project.model ? S.project.model.name : "選んだモデル"} に変えました。「学習」で学習を始めてください` +
        (reset.length ? `（前のモデル向けの設定 ${reset.join("、")} は初期値に戻しました）` : ""));
      location.hash = `#/p/${pid}/train`;
    } catch (e) { go.disabled = false; fail(e); }
  };
}

async function renderResults(soft = false) {
  const pid = S.project.id;
  if (!soft) $("#tab").innerHTML = `<div id="r-body"><div class="muted">読み込み中…</div></div>`;
  let runs;
  try { runs = S.runs = await api(`/api/projects/${pid}/runs`); } catch (e) { return fail(e); }
  const usable = runs.filter((r) => r.checkpoints.some((c) => c.kind === "lora"));
  const body = $("#r-body");
  if (!body) return;
  if (!usable.length) { body.innerHTML = `<div class="panel muted">学習が終わると、ここで LoRA の有無・保存時点・強さを並べて比較できます。</div>`; return; }
  const run = usable.find((r) => r.id === S.resultsRun) || usable[0];
  S.resultsRun = run.id;
  let evals = [];
  try { evals = await api(`/api/projects/${pid}/runs/${run.id}/evals`); } catch (e) { fail(e); }
  if (!body.isConnected || !S.project || S.project.id !== pid) return;
  const ev = evals.find((e) => e.id === S.resultsEv) || evals.find((e) => e.status === "done" || e.status === "running") || evals[0];
  const ckpts = run.checkpoints.filter((c) => c.kind === "lora");
  const kept = soft ? formState(body) : null;
  S.rcForm = kept;   // "回どうしを比べる" is drawn after a request: it restores its part itself
  body.innerHTML = `
    <div class="row" style="margin-bottom:10px">
      <select id="r-run">${usable.map((r) => `<option value="${r.id}" ${r.id === run.id ? "selected" : ""}>${esc(r.name)}（${r.model ? `${esc(r.model)}・` : ""}${esc(r.status_label)}）</option>`).join("")}</select>
      ${evals.length > 1 ? `<select id="r-ev">${evals.map((e) => `<option value="${e.id}" ${ev && e.id === ev.id ? "selected" : ""}>比較 #${e.id}（${esc(e.status)}）</option>`).join("")}</select>` : ""}
      <span class="muted small">ベースモデル・プロンプト・seed・解像度・サンプラー・ステップは固定し、LoRA の有無・保存時点・強さだけを変えています。</span>
    </div>
    <div class="panel cmp" id="r-cmp"></div>
    <div class="col" style="margin-top:12px">
      <div class="panel" id="r-more"></div>
      <div class="panel" id="r-runs"></div>
      <div class="panel" id="r-export"></div>
      <div class="panel" id="r-fix"></div>
      <div class="panel" id="r-remodel"></div>
    </div>`;
  $("#r-run").onchange = (e) => { S.resultsRun = +e.target.value; S.resultsEv = null; S.picked = null; renderResults(); };
  if ($("#r-ev")) $("#r-ev").onchange = (e) => { S.resultsEv = S.resShown.evId = +e.target.value; drawCompare(run, evals.find((x) => x.id === S.resultsEv), ckpts); };
  drawCompare(run, ev, ckpts);
  drawMore(run, ckpts);
  drawRunPicker(usable);
  drawExport(run, ckpts);
  drawFix(run, ev);
  drawRemodel(pid);
  S.resShown = { pid, run, ckpts, evId: ev ? ev.id : null, evIds: evals.map((e) => e.id), el: $("#r-cmp") };
  if (kept) { restoreForm(body, kept); if ($("#m-str")) $("#m-str").dispatchEvent(new Event("input")); }   // recounts the images
}

/* Kinds of comparison prompts (the same choice in "比較を追加" and in "回どうしを比べる"). */
function promptSetsHtml(prefix) {
  const sets = S.project.prompt_sets || [{ key: "standard", label: "標準" }];
  return `<div class="row small">プロンプトの種類 ${sets.map((x) => `<label title="${x.n ? `${x.n} 種類のプロンプト` : ""}"><input type="checkbox" class="${prefix}-set" value="${x.key}" ${x.key === "standard" ? "checked" : ""}> ${esc(x.label)}</label>`).join(" ")}</div>
    <div class="small" style="margin-top:6px">自分のプロンプト（任意・1 行に 1 つ。{T} と書いた所は${prefix === "rg" ? "、それぞれの回の" : ""}呼び出し語に置き換えます）</div>
    <textarea id="${prefix}-prompts" rows="2" style="width:100%" placeholder="例: {T}, solo, holding umbrella, rain, street"></textarea>`;
}
function readPromptSets(prefix) {
  const sets = $$(`.${prefix}-set:checked`).map((c) => c.value);
  const own = $(`#${prefix}-prompts`).value.split("\n").map((x) => x.trim()).filter(Boolean);
  const n = sets.reduce((a, k) => a + (((S.project.prompt_sets || []).find((x) => x.key === k) || {}).n || 4), 0) + own.length;
  return { sets, own, n };
}
const randomSeeds = (n) => Array.from({ length: n }, () => Math.floor(Math.random() * 1e9)).join(", ");

/* Two or more runs side by side: what was changed and the comparison images made under the same conditions. */
function drawRunPicker(usable) {
  const el = $("#r-runs");
  if (!el) return;
  if (usable.length < 2) {
    el.innerHTML = `<h2>回どうしを比べる</h2><div class="muted small">学習が 2 回以上終わると、設定・素材・説明文の違いと、同じ条件で作った比較画像を並べて比べられます。</div>`;
    return;
  }
  if (!S.cmp || S.cmp.pid !== S.project.id) {
    const i = usable.findIndex((r) => r.id === S.resultsRun);
    const other = usable[i + 1] || usable[i - 1];
    S.cmp = { pid: S.project.id, ids: [other.id, S.resultsRun].filter((x) => x != null).sort((a, b) => a - b), mode: "final", strength: 1.0, all: false, open: false };
  }
  const C = S.cmp;
  C.ids = C.ids.filter((id) => usable.some((r) => r.id === id));   // a run deleted meanwhile
  el.innerHTML = `<h2>回どうしを比べる</h2>
    <div class="muted small">選んだ回の、設定・素材・説明文の違いと、同じ条件（同じプロンプト・seed・生成設定）で作った比較画像を並べます。いちばん古い回を基準にして違いを示します。</div>
    <div class="row small" style="margin-top:6px">${usable.map((r) => `<label><input type="checkbox" class="rc-run" value="${r.id}" ${C.ids.includes(r.id) ? "checked" : ""}> ${esc(r.name || "#" + r.id)}</label>`).join(" ")}</div>
    <div class="row small" style="margin-top:6px">並べる版
      <label><input type="radio" name="rc-mode" value="final" ${C.mode === "final" ? "checked" : ""}> 最終の版・強さ <input type="text" id="rc-str" value="${C.strength}" style="width:50px"></label>
      <label><input type="radio" name="rc-mode" value="recommend" ${C.mode === "recommend" ? "checked" : ""}> それぞれのおすすめ</label>
      <button class="primary" id="rc-go">比べる</button></div>
    <div id="rc-out"></div>`;
  const read = () => {
    C.ids = $$(".rc-run:checked").map((c) => +c.value).sort((a, b) => a - b);
    C.mode = ($("[name=rc-mode]:checked") || {}).value || "final";
    C.strength = parseFloat($("#rc-str").value) || 1.0;
  };
  $("#rc-go").onclick = () => { read(); C.open = true; loadRunCompare(); };
  if (C.open) loadRunCompare();
}

async function loadRunCompare() {
  const C = S.cmp, out = $("#rc-out");
  if (!out) return;
  if (C.ids.length < 2) { out.innerHTML = `<div class="warntext small" style="margin-top:6px">比べる回を 2 つ以上選んでください。</div>`; return; }
  const kept = S.rcForm || formState(out);   // seeds / prompts typed below the comparison
  S.rcForm = null;
  if (!out.innerHTML) out.innerHTML = `<div class="muted small" style="margin-top:6px">読み込み中…</div>`;
  let d;
  try { d = await api(`/api/projects/${S.project.id}/compare?runs=${C.ids.join(",")}&mode=${C.mode}&strength=${C.strength}`); }
  catch (e) { out.innerHTML = `<div class="err small">${esc(e.message)}</div>`; return; }
  if (!out.isConnected) return;
  const names = d.runs.map((r) => esc(r.name));
  const nDiff = d.settings.filter((r) => r.differs).length;
  const rows = d.settings.filter((r) => C.all || r.differs);
  const thumb = (iid) => `<img src="/api/projects/${S.project.id}/thumb/${iid}" alt="#${iid}" title="画像 #${iid}" style="width:48px;height:48px;object-fit:cover;border-radius:3px">`;
  const ids = (list) => list.slice(0, 24).map(thumb).join(" ") + (list.length > 24 ? ` <span class="muted small">ほか ${list.length - 24} 枚</span>` : "");
  const mat = d.material.map((m) => {
    const name = esc((d.runs.find((r) => r.id === m.run) || {}).name);
    const parts = [m.added.length ? `${m.added.length} 枚増やした` : "", m.removed.length ? `${m.removed.length} 枚外した` : "",
      m.crop.length ? `${m.crop.length} 枚の切り抜きを変えた` : "", m.caption.length ? `${m.caption.length} 枚の説明文を変えた` : ""].filter(Boolean);
    return `<div style="margin-top:6px"><b>${name}</b>：${parts.length ? parts.join("、") : "素材と説明文は同じ"}
      ${parts.length ? `<details><summary class="small">内容を見る</summary>
        ${m.added.length ? `<div class="small">増やした画像</div><div>${ids(m.added)}</div>` : ""}
        ${m.removed.length ? `<div class="small">外した画像</div><div>${ids(m.removed)}</div>` : ""}
        ${m.crop.length ? `<div class="small">切り抜きを変えた画像</div><div>${ids(m.crop)}</div>` : ""}
        ${m.caption.slice(0, 60).map((c) => `<div class="capdiff row" style="align-items:flex-start">${thumb(c.id)}<div>
          ${c.added.map((t) => `<span class="add">＋${esc(t)}</span>`).join(" ")} ${c.removed.map((t) => `<span class="rm">${esc(t)}</span>`).join(" ")}
          ${!c.added.length && !c.removed.length ? `<span class="muted">${esc(c.before)}</span> → ${esc(c.after)}` : ""}</div></div>`).join("")}
        ${m.caption.length > 60 ? `<div class="muted small">ほか ${m.caption.length - 60} 枚</div>` : ""}</details>` : ""}</div>`;
  }).join("");
  const im = d.images;
  const url = (run, c, t) => `/api/projects/${S.project.id}/runs/${run}/evals/${c.eval}/${encodeURIComponent(c.file)}${t ? "?thumb=1" : ""}`;
  const SC = [["ccip_diff", "同一性差", "小さいほど基準画像のキャラに近い"], ["concept_prob", "特徴の確率", "大きいほど覚えさせた特徴が出ている"],
    ["style_sim", "画風類似", "大きいほど素材の画風に近い"], ["copy_sim", "素材類似", "非常に高いと素材の構図をそのまま写している可能性"], ["effect", "効果", ""]];
  const avg = (rid, k) => { const xs = im.rows.map((r) => r.cells[rid] && r.cells[rid].scores[k]).filter((x) => x != null); return xs.length ? (xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(3) : null; };
  const baseAvg = (k) => { const xs = im.rows.map((r) => r.base && r.base.scores[k]).filter((x) => x != null); return xs.length ? (xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(3) : null; };
  const scoreRows = SC.filter(([k]) => im.cols.some((c) => avg(c.run, k) != null));
  const colHead = (c) => `${esc(c.name)}<br><span class="muted">${c.ckpt ? `${ckLabel(c.ckpt)}・強さ ${c.strength}` : C.mode === "recommend" ? "おすすめがまだありません" : `最終・強さ ${C.strength} の画像がありません`}</span>`;
  const lightList = [];
  const cell = (run, c, cap) => { if (!c) return `<td></td>`; lightList.push({ src: url(run, c), cap }); const i = lightList.length - 1;
    return `<td><img loading="lazy" src="${url(run, c, true)}" data-lb="${i}" alt=""></td>`; };
  const grid = im.rows.length ? `<div class="cmp" style="margin-top:6px"><table><thead><tr><th class="rowh">プロンプト / seed</th><th>LoRA なし</th>${im.cols.map((c) => `<th>${colHead(c)}</th>`).join("")}</tr></thead>
    <tbody>${im.rows.map((r) => `<tr><th class="rowh small">${esc(r.prompt.slice(0, 90))}<br>seed ${r.seed}</th>
      ${cell(r.base ? r.base.run : 0, r.base, `LoRA なし / seed ${r.seed}`)}
      ${im.cols.map((c) => cell(c.run, r.cells[c.run], `${c.name} / seed ${r.seed}`)).join("")}</tr>`).join("")}
    ${scoreRows.map(([k, label, help]) => `<tr><th class="rowh small" title="${esc(help)}">${label} の平均</th><td class="small">${baseAvg(k) ?? ""}</td>
      ${im.cols.map((c) => `<td class="small">${avg(c.run, k) ?? ""}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    ${scoreRows.length ? `<div class="muted small">数値は選択の補助です（成功確率ではありません）。それぞれの回の比較画像だけから計算しています。最終判断は画像を見て行ってください。</div>` : ""}` :
    `<div class="muted small" style="margin-top:6px">同じ条件で作った比較画像が、まだ 2 つ以上の回にそろっていません。下で作れます。</div>`;
  const missing = im.cols.filter((c) => !im.rows.some((r) => r.cells[c.run])).map((c) => esc(c.name));
  out.innerHTML = `
    <h3 style="margin:12px 0 4px">設定の違い</h3>
    ${nDiff ? "" : `<div class="muted small">設定は同じです。</div>`}
    <table class="diff"><thead><tr><th></th>${names.map((n) => `<th>${n}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `<tr class="${r.differs ? "differs" : "same"}"><td>${esc(r.label)}</td>${r.values.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table>
    <label class="small"><input type="checkbox" id="rc-all" ${C.all ? "checked" : ""}> 同じ設定も表示する</label>
    <h3 style="margin:12px 0 4px">素材と説明文の違い（${names[0]} と比べて）</h3>${mat}
    <h3 style="margin:12px 0 4px">比較画像</h3>
    ${d.generating.length ? `<div class="muted small"><span class="spin"></span> 比較画像を生成中です。できた分から並びます。</div>` : ""}
    ${im.conds > 1 ? `<div class="muted small">生成条件（モデル・サンプラー・サイズなど）が違う画像は、別の行に分けています。</div>` : ""}
    ${grid}
    ${missing.length && im.rows.length ? `<div class="warntext small">${missing.join("・")} は、この条件の画像がまだありません。</div>` : ""}
    <details style="margin-top:8px" ${im.rows.length ? "" : "open"}><summary>同じ条件の比較画像を作る</summary>
      <div class="muted small" style="margin:4px 0">選んだすべての回で、同じプロンプト・seed・生成設定の画像を作ります。すでにある画像は作り直しません。GPU の順番待ちに入ります。</div>
      ${promptSetsHtml("rg")}
      <div class="row small" style="margin-top:6px">seed <input type="text" id="rg-seeds" value="1234, 5678" style="width:150px"><button class="small" id="rg-rand">ランダムにする</button>
        <button id="rg-go">作る</button><span class="muted small" id="rg-count"></span></div></details>`;
  $$("[data-lb]", out).forEach((x) => x.onclick = () => openLightbox(lightList, +x.dataset.lb));
  $("#rc-all").onchange = (e) => { C.all = e.target.checked; loadRunCompare(); };
  const count = () => {
    const { n } = readPromptSets("rg");
    const seeds = $("#rg-seeds").value.split(",").filter((x) => x.trim()).length;
    $("#rg-count").textContent = `最大 ${(C.ids.length + 1) * n * seeds} 枚（すでにある分は作りません）`;
  };
  $$("input, textarea", $("#rc-out details:last-of-type")).forEach((x) => x.oninput = count);
  restoreForm(out, kept);
  count();
  $("#rg-rand").onclick = () => { $("#rg-seeds").value = randomSeeds(2); count(); };
  $("#rg-go").onclick = async () => {
    const { sets, own } = readPromptSets("rg");
    const b = $("#rg-go"); b.disabled = true;
    try {
      const r = await api(`/api/projects/${S.project.id}/compare/generate`, { body: { runs: C.ids, mode: C.mode, strength: C.strength,
        seeds: $("#rg-seeds").value.split(",").map((x) => parseInt(x, 10)).filter((x) => !isNaN(x)), prompt_sets: sets, extra_prompts: own } });
      toast(r.queued.length ? `${r.queued.length} 回分（${r.n_images} 枚）の比較画像を順番待ちに追加しました` : "この条件の画像はすべてそろっています");
      loadRunCompare();
    } catch (e) { fail(e); }
    b.disabled = false;
  };
}

function ckLabel(k, ckpts) {
  if (k === "base") return "LoRA なし";
  if (k === "final") return "最終";
  return `ステップ ${k}`;
}

function drawCompare(run, ev, ckpts) {
  const el = $("#r-cmp");
  if (!ev) { el.innerHTML = `<div class="muted">比較画像はまだありません。下の「比較を追加」から生成できます。</div>`; return; }
  if (ev.status === "failed") { el.innerHTML = `<div class="err">比較画像の生成に失敗しました: ${esc(ev.error || "")}</div>`; return; }
  const res = ev.results || {};
  const imgs = res.images || [];
  if (!imgs.length) { el.innerHTML = `<div class="muted"><span class="spin"></span> 比較画像を生成中です…</div>`; return; }
  const sc = res.scores || {};
  const url = (f, t) => `/api/projects/${S.project.id}/runs/${run.id}/evals/${ev.id}/${encodeURIComponent(f)}${t ? "?thumb=1" : ""}`;
  const rec = res.recommend;
  if (!S.picked && rec && ev.status === "done") S.picked = { ckpt: rec.ckpt, strength: rec.strength, auto: true };
  const pickKey = S.picked ? `${S.picked.ckpt}|${S.picked.strength}` : "";
  // one table per model: the training model first, then the other models of the comparison (images with "model")
  const table = (mid, recKey, mname) => {
    const own = imgs.map((r, i) => [r, i]).filter(([r]) => (r.model || null) === mid);
    if (!own.length) return "";
    const cols = [];
    const seen = new Set();
    for (const [r] of own) { const k = `${r.ckpt}|${r.strength}`; if (!seen.has(k)) { seen.add(k); cols.push([r.ckpt, r.strength]); } }
    cols.sort((a, b) => (a[0] === "base" ? -1 : b[0] === "base" ? 1 : 0));
    const rows = [];
    const rs = new Set();
    for (const [r] of own) { const k = `${r.prompt}|${r.seed}`; if (!rs.has(k)) { rs.add(k); rows.push([r.prompt, r.seed]); } }
    const idx = new Map(own.map(([r, i]) => [`${r.prompt}|${r.seed}|${r.ckpt}|${r.strength}`, i]));
    return `<table><thead><tr><th class="rowh">プロンプト / seed</th>${cols.map(([k, s]) => `<th>${ckLabel(k)}${`${k}|${s}` === recKey ? ` <span class="kind">おすすめ</span>` : ""}${k !== "base" ? `<br><span class="muted">強さ ${s}</span><br>
      <label class="small"><input type="radio" name="pick${mid ? "-" + mid : ""}" data-pick="${k}|${s}" ${pickKey === `${k}|${s}` ? "checked" : ""}> これを選ぶ</label>` : ""}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(([pi, seed]) => `<tr><th class="rowh small">${esc((ev.spec.prompts[pi] || "").slice(0, 90))}<br>seed ${seed + pi}</th>
      ${cols.map(([k, s]) => { const i = idx.get(`${pi}|${seed}|${k}|${s}`); if (i == null) return `<td></td>`;
        const r = imgs[i];
        const extra = [sc.effect ? `${{ detail: "細かさ", contrast: "明暗の差", saturation: "彩度" }[sc.effect_metric] || "効果"} ${sc.effect[i]}` : "", sc.ccip_diff ? `同一性差 ${sc.ccip_diff[i]}` : "", sc.style_sim ? `画風類似 ${sc.style_sim[i]}` : "", sc.copy_sim ? `素材類似 ${sc.copy_sim[i]}` : ""].filter(Boolean).join(" ・ ");
        return `<td class="${pickKey === `${k}|${s}` ? "pick" : ""}"><img loading="lazy" src="${url(r.file, true)}" data-zoom="${url(r.file)}" data-cap="${mname ? esc(mname) + " / " : ""}${esc(ckLabel(k))} 強さ${s} / seed ${seed + pi}" alt="">
          ${extra ? `<div class="score">${extra}</div>` : ""}</td>`; }).join("")}</tr>`).join("")}</tbody></table>`;
  };
  const others = (res.models || []).map((m) => {
    const o = (res.other_models || []).find((x) => x.id === m.id) || {};
    const orec = o.recommend;
    const t = table(m.id, orec ? `${orec.ckpt}|${orec.strength}` : "", m.name);
    if (!t) return "";
    return `<h2 style="margin-top:16px">ほかのモデルでの出方: ${esc(m.name)}</h2>
      ${o.reasons && o.reasons.length || orec ? `<div class="note ${o.ratio != null && o.ratio < 0.5 || (orec && orec.weak) ? "warnbox" : ""}" style="margin-bottom:8px">
        ${orec ? `<b>このモデルでのおすすめ: ${esc(ckLabel(orec.ckpt))}・強さ ${orec.strength}</b><br>` : ""}
        <div class="small">${(o.reasons || []).map((x) => `・${esc(x)}`).join("<br>")}</div></div>` : ""}
      <div class="muted small">生成条件: ${esc(JSON.stringify(m.sampling))}。プロンプトは学習したモデル用のものに、このモデルの品質タグを付け替えて使っています。</div>
      ${t}`;
  }).join("");
  el.innerHTML = `
    ${rec ? `<div class="note ${rec.weak ? "warnbox" : ""}" style="margin-bottom:8px"><b>おすすめ: ${esc(ckLabel(rec.ckpt))}・強さ ${rec.strength}</b>
      ${rec.weak ? `<div class="warntext">${{ style: "画風", concept: conceptUI(S.project).noun, enhance: "質感の効果" }[S.project.kind] || "キャラの特徴"}があまり出ていない可能性があります。結果の画面下の「思った結果にならないとき」の「特徴が弱い・出ない」も見てください。</div>` : ""}
      <div class="small">${rec.reasons.map((x) => `・${esc(x)}`).join("<br>")}</div>
      <div class="muted small">自動スコアからの推定です。最初に試す候補として選んでいます。画像を見て、ほかの列を選び直せます。</div></div>` : ""}
    ${ev.status === "running" ? `<div class="muted small"><span class="spin"></span> 生成中… ${imgs.length} 枚完了</div>` : ""}
    ${res.n_lora_key_warnings ? `<div class="err">ComfyUI が LoRA の一部を読み込めませんでした（${res.n_lora_key_warnings} 件）。形式の互換性に問題があります。</div>` : ""}
    ${res.conditions ? `<div class="muted small">生成条件: ${esc(JSON.stringify(res.conditions.sampling))} ${res.conditions.width}×${res.conditions.height} ${res.conditions.evaluated_on === "turbo" ? "・ Turbo モデルで生成" : ""}</div>` : ""}
    ${(res.notes || []).map((x) => `<div class="warntext small">${esc(x)}</div>`).join("")}
    ${res.models && res.models.length ? `<h2>学習したモデル</h2>` : ""}
    ${table(null, rec ? `${rec.ckpt}|${rec.strength}` : "", "")}
    ${others}
    ${sc.effect ? `<div class="muted small" style="margin-top:6px">数値は画像そのものから計算した参考値です（細かさ: 細かい凹凸の量、明暗の差: 明るさのばらつき、彩度: 色の鮮やかさ）。値が大きくても、ノイズや崩れで増えている場合があります。最終判断は画像を見て行ってください。</div>` : ""}
    ${sc.copy_sim ? `<div class="muted small" style="margin-top:6px">数値は選択の補助です（成功確率ではありません）。同一性差: 小さいほど基準画像のキャラに近い。画風類似: 素材の画風に近いほど大きい。素材類似が非常に高い場合は素材の構図をそのまま写している可能性があります。最終判断は画像を見て行ってください。</div>` : ""}
    ${sc.error ? `<div class="muted small">${esc(sc.error)}</div>` : ""}`;
  $$("[data-zoom]", el).forEach((im) => im.onclick = () => openLightbox([{ src: im.dataset.zoom, cap: im.dataset.cap }], 0));
  $$("[data-pick]", el).forEach((r) => r.onchange = () => {
    const [ckpt, strength] = r.dataset.pick.split("|");
    S.picked = { ckpt, strength: parseFloat(strength) };
    drawCompare(run, ev, ckpts); drawExport(run, ckpts);
  });
}
/* Image viewer.  With several images (one generation request), ← → or the side buttons move between them. */
const LB = { list: [], i: 0 };
function openLightbox(list, i) { LB.list = list; LB.i = i; lbShow(); $("#lightbox").hidden = false; }
function lbShow() {
  const lb = $("#lightbox"), it = LB.list[LB.i], many = LB.list.length > 1;
  // 画像をぼかして表示: an image shown with the eye is shown enlarged too (the viewer loads the full-size file)
  if ($$("img.unblur").some((x) => x.dataset.zoom === it.src || x.getAttribute("src") === it.src)) UNBLUR.add(it.src);
  $("img", lb).src = it.src;
  $(".lb-cap", lb).textContent = it.cap + (many ? `　${LB.i + 1} / ${LB.list.length}（← → で移動）` : "");
  $(".lb-prev", lb).hidden = $(".lb-next", lb).hidden = !many;
  $(".lb-prev", lb).disabled = LB.i === 0;
  $(".lb-next", lb).disabled = LB.i === LB.list.length - 1;
}
function lbMove(d) { const n = LB.i + d; if (n >= 0 && n < LB.list.length) { LB.i = n; lbShow(); } }
$("#lightbox").onclick = (e) => {
  if (e.target.closest(".lb-prev")) return lbMove(-1);
  if (e.target.closest(".lb-next")) return lbMove(1);
  $("#lightbox").hidden = true;
};
// capture phase: while the viewer is open the arrow keys belong to it, not to the page underneath
document.addEventListener("keydown", (e) => {
  if ($("#lightbox").hidden) return;
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") lbMove(e.key === "ArrowLeft" ? -1 : 1);
  else if (e.key === "Escape") $("#lightbox").hidden = true;
  else return;
  e.preventDefault(); e.stopPropagation();
}, true);

function drawMore(run, ckpts) {
  const el = $("#r-more");
  const steps = ckpts.map((c) => c.step ?? "final");
  el.innerHTML = `<h2>比較を追加</h2>
    <div class="row small">保存版: ${steps.map((s) => `<label><input type="checkbox" class="m-ck" value="${s}" ${s === "final" ? "checked" : ""}> ${s === "final" ? "最終" : s}</label>`).join(" ")}</div>
    <div class="row small" style="margin-top:6px">強さ <input type="text" id="m-str" value="0.6, 0.8, 1.0" style="width:130px">
      seed <input type="text" id="m-seeds" value="1234, 5678, 9012" style="width:150px"><button class="small" id="m-rand">ランダムにする</button>
      <label><input type="checkbox" id="m-base" checked> LoRA なしも並べる</label></div>
    <div style="margin-top:6px">${promptSetsHtml("m")}</div>
    <div class="muted small">標準のほかに、ポーズ・場所・衣装など条件を 1 つずつ変えたプロンプトを選べます。どの条件でも特徴が保たれるかを確かめられます。</div>
    <div id="m-models" style="margin-top:8px"></div>
    <div class="row" style="margin-top:8px"><button id="m-run">生成する</button><span class="muted small" id="m-count"></span></div>`;
  const count = () => {
    const per = $$(".m-ck:checked").length * $("#m-str").value.split(",").filter((x) => x.trim()).length + ($("#m-base").checked ? 1 : 0);
    const n = per * (1 + $$(".m-xm:checked").length);
    const total = n * readPromptSets("m").n * $("#m-seeds").value.split(",").filter((x) => x.trim()).length;
    $("#m-count").textContent = `約 ${total} 枚を生成します${total > 160 ? "（上限 160 枚を超えています）" : ""}`;
  };
  $$("input, textarea", el).forEach((i) => i.oninput = count);
  $("#m-rand").onclick = () => { $("#m-seeds").value = randomSeeds(3); count(); };
  count();
  drawExtraModels($("#m-models"), run.model_family, run.model_id, count);
  $("#m-run").onclick = async () => {
    try {
      const body = { checkpoints: $$(".m-ck:checked").map((c) => c.value === "final" ? "final" : +c.value),
        extra_models: $$(".m-xm:checked").map((c) => c.value),
        strengths: $("#m-str").value.split(",").map((x) => parseFloat(x)).filter((x) => !isNaN(x)),
        seeds: $("#m-seeds").value.split(",").map((x) => parseInt(x, 10)).filter((x) => !isNaN(x)),
        include_base: $("#m-base").checked, prompt_sets: readPromptSets("m").sets,
        extra_prompts: readPromptSets("m").own.map((x) => x.replaceAll("{T}", S.project.trigger || "")) };
      const r = await api(`/api/projects/${S.project.id}/runs/${run.id}/eval`, { body });
      toast(`比較 #${r.eval_id}（${r.n_images} 枚）を順番待ちに追加しました`);
      setTimeout(() => { if ($("#r-body") && S.project) renderResults(); }, 800);
    } catch (e) { fail(e); }
  };
}

/* Other models (merged / fine-tuned ones of the same family) to compare on as well.  The choice is saved at once
   and used for the comparison made after every training, in every project. */
async function drawExtraModels(el, family, ownId, count = () => {}) {
  let d, saved;
  try { [d, saved] = await Promise.all([api("/api/models"), api("/api/eval_models")]); } catch (e) { return; }
  if (!el.isConnected) return;
  const same = d.models.filter((m) => m.family === family && m.id !== ownId && !m.missing.length && m.variant);
  if (!same.length) {
    el.innerHTML = `<div class="muted small">ほかのモデルでの出方も確かめるときは、同じ系列（${esc(family || "")}）のモデルを設定画面で登録してください。マージモデルなども選べるようになります。</div>`;
    return;
  }
  const ids = new Set(saved.ids);
  el.innerHTML = `<div class="small"><b>ほかのモデルでも試す</b>（${saved.max} つまで）</div>
    <div class="row small">${same.map((m) => `<label><input type="checkbox" class="m-xm" value="${m.id}" ${ids.has(m.id) ? "checked" : ""}> ${esc(m.name)}<span class="muted">（${esc(m.variant_info.label || m.family)}）</span></label>`).join(" ")}</div>
    <div class="muted small">選んだモデルでも、LoRA なしと LoRA ありの画像を同じ条件で作り、LoRA の効き方を学習したモデルと比べます。選んだモデルは保存され、どのプロジェクトでも学習が終わったあとの比較で毎回使います（そのときは最終の保存版だけを試します）。</div>`;
  $$(".m-xm", el).forEach((c) => c.onchange = async () => {
    const mine = $$(".m-xm:checked", el).map((x) => x.value);
    if (mine.length > saved.max) { c.checked = false; toast(`${saved.max} つまで選べます`); return; }
    const keep = saved.ids.filter((i) => !same.some((m) => m.id === i));   // other families stay as they were
    try { saved.ids = (await api("/api/eval_models", { method: "PUT", body: { ids: [...keep, ...mine] } })).ids; } catch (e) { c.checked = !c.checked; fail(e); }
    count();
  });
  count();
}

function drawExport(run, ckpts) {
  const el = $("#r-export");
  let pick = S.picked;
  let ck = pick ? ckpts.find((c) => (c.step ?? "final") + "" === pick.ckpt + "") : null;
  // no comparison to pick from (it failed or was never made): the saved LoRA can still be chosen here
  const direct = !ck && ckpts.length && !$("#r-cmp [data-pick]");
  if (direct) {
    const last = ckpts.find((c) => c.step == null) || ckpts[ckpts.length - 1];
    pick = { ckpt: last.step ?? "final", strength: 1 };
    ck = last;
  }
  el.innerHTML = `<h2>書き出し（生成で使う LoRA）</h2>
    ${direct ? `<div class="row small">比較画像がないため、ここで選んで書き出せます。保存した時点
        <select id="x-ck">${ckpts.map((c) => `<option value="${c.step ?? "final"}" ${c === ck ? "selected" : ""}>${ckLabel(c.step ?? "final")}</option>`).join("")}</select>
        記録する強さ <input type="number" id="x-st" value="1" step="0.05" min="0" max="2" style="width:70px"></div>`
      : ck ? `<div>選択中: <b>${ckLabel(pick.ckpt)}</b>・強さ ${pick.strength}</div>` : `<div class="muted">上の比較表で「これを選ぶ」を選んでください。</div>`}
    <div class="row small" style="margin-top:6px">保存先 <input type="text" id="x-dest" style="flex:1;min-width:240px" placeholder="空欄なら models\\loras の中に、LoRA の名前のフォルダーを作って保存します"></div>
    <div class="row" style="margin-top:8px"><button class="primary" id="x-go" ${ck ? "" : "disabled"}>書き出す</button>
      <span class="muted small">LoRA 本体・使い方・学習条件を保存します。学習画像は含めません。既存ファイルは上書きせず版を分けます。</span></div>
    <div id="x-res"></div>`;
  if (ck) $("#x-go").onclick = async () => {
    if (direct) {
      const v = $("#x-ck").value;
      ck = ckpts.find((c) => (c.step ?? "final") + "" === v) || ck;
      pick = { ckpt: v, strength: Math.max(0, +$("#x-st").value || 1) };
    }
    const b = $("#x-go"); b.disabled = true; b.textContent = "書き出し中…";
    try {
      const r = await api(`/api/projects/${S.project.id}/runs/${run.id}/export`, { body: { ckpt_path: ck.path, strength: pick.strength, dest: $("#x-dest").value } });
      const cl = r.comfy_link;
      $("#x-res").innerHTML = `<div class="note" style="margin-top:8px">書き出しました: <b>${esc(r.file)}</b>
        ${$("#x-dest").value.trim() ? "" : ` <button class="small" data-opendir="loras">loras フォルダーを開く</button>`}
        <br>${r.info.trigger ? `呼び出し語: ${esc(r.info.trigger)} ・ ` : "呼び出し語は要りません ・ "}強さ ${pick.strength} から試してください。
        ${cl ? (cl.path ? `<br>ComfyUI からは次の名前で使えます（リンクなので容量は増えません）: ${esc(cl.path)}` : `<br><span class="muted">${esc(cl.message || "")}</span>`) : ""}</div>
        ${r.leftover_bytes > 0 ? `<div class="note" id="x-discard" style="margin-top:8px">
          <div class="row"><b>学習データ（${fmtSize(r.leftover_bytes)}）を破棄しますか？</b>
            <button id="xd-yes">はい</button><button id="xd-no">いいえ</button></div>
          <div class="muted small">このプロジェクトのすべての回の途中保存の LoRA、再開用の保存データ、学習用キャッシュを消します。書き出した LoRA の元の保存版、素材、説明文、比較画像は残すので、あとで学習し直すこと（別のモデル向けも）はできます。消した保存版は比較や書き出しに使えなくなり、途中からの再開もできなくなります。</div></div>` : ""}`;
      bindOpenDir($("#x-res"));
      if ($("#xd-no")) $("#xd-no").onclick = () => $("#x-discard").remove();
      if ($("#xd-yes")) $("#xd-yes").onclick = async (e) => {
        e.target.disabled = true; e.target.textContent = "破棄しています…";
        try {
          const d = await api(`/api/projects/${S.project.id}/training_leftovers/discard`, { body: {} });
          toast(`学習データを破棄しました${d.freed ? `（${fmtSize(d.freed)} 空きました）` : ""}` +
            (d.skipped_runs.length ? `。学習中・比較画像の作成中の回は残しました` : ""));
          renderResults(true);
        } catch (err) { e.target.disabled = false; e.target.textContent = "はい"; fail(err); }
      };
    } catch (e) { $("#x-res").innerHTML = `<div class="err">${esc(e.message)}（完了していません）</div>`; }
    b.disabled = false; b.textContent = "書き出す";
  };
}

function drawFix(run, ev) {
  const el = $("#r-fix");
  const sc = (ev && ev.results && ev.results.scores) || {};
  const words = (S.project.enhance || {}).method === "words";
  const hints = S.project.kind === "enhance" ? {
    weak: ["効果が弱い・出ない", words ?
      ["強さ 2 以上で比較する", "詳細設定の「効果の強さ（学習時）」を上げる（例: 1.5 → 2.5）", "学習ステップ数を増やす",
       "言葉を、モデルが知っていそうな一般的なタグに言い換える（モデルが知らない質感は強められません）"] :
      ["強さ 1.0 以上や遅い保存版で比較する", "学習ステップ数を増やす（詳細設定）", "強めたい質感がはっきり写った、解像度の高い画像を選ぶ",
       "「学習する違いの例」で、加工画像との違いが見て分かるか確認する"]],
    other: ["質感以外（色・顔・構図）まで変わる", words ?
      ["「対象」を空欄にして、何にでも効くように学習し直す", "強めたいもの・弱めたいものの言葉から、質感以外の言葉を外す", "強さを下げて比較する"] :
      ["詳細設定の「下準備の学習ステップ数」を増やす（下準備が足りないと、加工以外の違いまで覚えます）", "早い保存版や強さ 0.6 前後で比較する"]],
    broken: ["画像が崩れる・ノイズが出る", ["強さを下げる・早い保存版を選ぶ", "学習率を下げる（例: 半分）", words ? "「効果の強さ（学習時）」を下げる" : "学習ステップ数を減らす"]],
  } : {
    weak: ["特徴が弱い・出ない", ["保存時点の遅い版や強さ 1.0 以上で出るか比較する", "学習ステップ数を増やす（詳細設定）", S.project.kind === "concept" ? `説明文に、覚えさせたい${conceptUI(S.project).noun}のタグが残っていないか確認する（語を見直す）` : "説明文に、呼び出し語に覚えさせたい特徴（髪色など）が書かれていないか確認する", "対象が小さく写った素材を切り抜く"]],
    stuck: ["素材の構図・背景に固執する", ["早い保存版や強さ 0.6〜0.8 で比較する", "よく似た画像（要確認の「類似」）を減らす", "構図・背景の偏りの通知を確認し、別の構図の素材を加える", "学習ステップ数を減らす"]],
    broken: ["画像が崩れる・ノイズが出る", ["早い保存版を選ぶ", "学習率を下げる（例: 半分）", "低解像度・強い圧縮の素材を除外する"]],
    other: S.project.kind === "concept" ? ["素材の人物の顔や画風まで出る",
      ["いろいろな人物・画風の素材を混ぜる（同じ人物や同じ絵柄に偏らないようにする）", `説明文に人物の髪${(S.project.concept_target || "body") === "outfit" ? "や顔" : "や服"}が書かれているか確認する（書かれていないと、それも呼び出し語に結びつきます）`, "早い保存版や強さ 0.6〜0.8 で比較する"]] :
      [S.project.kind === "style" ? "画風ではなくキャラが出る" : "別のキャラが混ざる", S.project.kind === "style" ?
      ["特定キャラに偏っている素材を減らす（素材画面の通知）", "説明文に被写体（キャラ名・人物）を書いて内容と画風を分ける"] :
      ["人物混在・同一性の要確認画像を見直す", "基準画像を指定して同一性を再判定する"]],
  };
  const avgCopy = sc.copy_sim ? (sc.copy_sim.reduce((a, b) => a + b, 0) / sc.copy_sim.length).toFixed(3) : null;
  el.innerHTML = `<h2>思った結果にならないとき</h2>
    <div class="muted small">原因は断定できません。根拠と、試せる修正の候補を示します。${avgCopy ? `（参考: 生成画像と素材の平均類似 ${avgCopy}）` : ""}</div>
    <div class="row" style="margin-top:8px">${Object.entries(hints).map(([k, [label]]) => `<button data-h="${k}">${esc(label)}</button>`).join("")}</div>
    <div id="fix-out"></div>`;
  $$("[data-h]", el).forEach((b) => b.onclick = () => {
    const [label, items] = hints[b.dataset.h];
    $("#fix-out").innerHTML = `<h3>${esc(label)}</h3>${items.map((x) => `<div>・${esc(x)}</div>`).join("")}
      <div class="row" style="margin-top:8px"><a href="#/p/${S.project.id}/material"><button>素材を見直す</button></a>
      <a href="#/p/${S.project.id}/train"><button>設定を変えて再学習</button></a></div>`;
  });
}

const workerLabel = (w) => !w ? "—" : w === "stopped" ? "待機中（必要になったら起動します）" : w === "ready" ? "準備完了" :
  w === "ready_cpu" ? "GPU が使えないため CPU で解析中（遅くなります。説明文の生成は止めています。数分ごとに GPU を試し直します）" :
  w === "starting" ? "起動中" : w === "failed" ? "停止（自動再起動を中止）" : w.startsWith("loading") ? `モデル読み込み中: ${w.split(":")[1]}` : w;

// ------------------------------------------------------------------ settings
async function drawAppLogs() {
  const el = $("#s-logs");
  let logs;
  try { logs = await api("/api/app_logs"); } catch (e) { el.innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  const total = logs.reduce((a, x) => a + x.size, 0);
  el.innerHTML = `<div>解析・画像生成（ComfyUI）・エラーの記録です。合計 ${total ? fmtSize(total) : "0MB"}（${logs.map((x) => esc(x.name)).join("、") || "なし"}）。
      学習ごとのログは、学習画面の履歴を削除すると一緒に消えます。</div>
    ${logs.map((x) => `<div class="row" style="margin-top:6px"><span>${esc(x.name)}（${x.size ? fmtSize(x.size) : "空"}）</span>
      <button data-applog="${esc(x.name)}" ${x.size ? "" : "disabled"}>表示</button></div>
      <pre class="log" data-applog-pre="${esc(x.name)}" hidden></pre>`).join("")}
    <div class="row" style="margin-top:6px"><a href="/api/app_logs/zip" download><button ${total ? "" : "disabled"}>まとめて保存（zip）</button></a>
      <button id="s-logclear" ${total ? "" : "disabled"}>ログを空にする</button></div>
    <div class="muted small">不具合を知らせるときは「まとめて保存」の zip を渡してください。ログにはフォルダーの場所などこの PC の情報が含まれます。画像は含まれません。</div>`;
  $$("[data-applog]", el).forEach((b) => b.onclick = async () => {
    const pre = el.querySelector(`[data-applog-pre="${CSS.escape(b.dataset.applog)}"]`);
    if (!pre.hidden) { pre.hidden = true; b.textContent = "表示"; return; }
    try {
      const r = await api(`/api/app_logs/${encodeURIComponent(b.dataset.applog)}?tail=300`);
      pre.textContent = `${r.file}\n（最後の 300 行）\n\n${r.text || "（空です）"}`; pre.hidden = false; pre.scrollTop = 1e9; b.textContent = "閉じる";
    } catch (e) { fail(e); }
  });
  $("#s-logclear").onclick = async () => {
    try {
      const r = await api("/api/app_logs/clear", { body: {} });
      r.failed.length ? toast(`一部を空にできませんでした: ${r.failed.join(" / ")}`, "err") : toast("ログを空にしました");
      drawAppLogs();
    } catch (e) { fail(e); }
  };
}

const STORAGE_LABELS = {
  projects: ["プロジェクト", "素材の画像、説明文、学習した LoRA、比較画像など。いちばん容量が大きくなります。"],
  imagegen: ["画像生成ページで作った画像", "生成の履歴やプリセットは今までどおり data フォルダーに残ります。"],
  i2t: ["i2t に追加した画像", "i2t の履歴（最大 50 件）と設定は今までどおり data フォルダーに残ります。"],
};

function storageHtml(sto) {
  const gb = (b) => b == null ? "（ドライブが見つかりません）" : `空き ${(b / 1e9).toFixed(0)}GB`;
  return `<div class="panel" style="margin-bottom:14px"><h2>データの置き場所</h2>
    <div class="muted small">容量の大きいデータを、SSD 以外のドライブ（HDD など）に置けます。元フォルダーを指定すると、その中に projects・imagegen・i2t のフォルダーを作って保存します。
      変更すると、そのあとに作ったものから新しい場所に保存されます。今までの場所にあるものは移動せず、そのまま一覧に出て使えます。
      HDD に置くと、素材の読み込みや学習の準備に SSD より時間がかかることがあります。</div>
    ${Object.entries(sto).map(([k, s]) => `<div style="padding:8px 0;border-bottom:1px solid var(--line)">
      <div><b>${esc(STORAGE_LABELS[k][0])}</b> <span class="muted small">${esc(STORAGE_LABELS[k][1])}</span></div>
      <div class="small">保存先: ${esc(s.dir)} <span class="muted">${s.base ? "" : "（既定）"}${gb(s.free)}</span></div>
      ${s.others.map((o) => `<div class="small muted">・前の置き場所（中身はそのまま使えます）: ${esc(o.dir)}${o.exists ? "" : ` <span class="warntext">見つかりません（ドライブが外れているかもしれません）</span>`}</div>`).join("")}
      <div class="row" style="margin-top:6px"><input type="text" data-stobase="${k}" style="flex:1;min-width:260px" value="${esc(s.base || "")}" placeholder="元フォルダー（例: D:\LoRAtelier）">
        <button data-stoset="${k}">変更</button>${s.base ? `<button data-storeset="${k}">既定に戻す</button>` : ""}</div>
      ${s.others.length ? `<details data-stomove="${k}" style="margin-top:6px"><summary class="small">置き場所のあいだで移す</summary><div class="small muted">読み込み中…</div></details>` : ""}</div>`).join("")}
    <div id="s-movestate"></div></div>`;
}

const gbText = (b) => `${(b / 1e9).toFixed(b < 1e10 ? 1 : 0)}GB`;

async function drawStorageMove(det) {
  const kind = det.dataset.stomove, box = det.querySelector("div");
  let r;
  try { r = await api(`/api/storage/items?kind=${kind}`); } catch (e) { box.textContent = String(e.message || e); return; }
  const dest = `<label class="small">移し先 <select data-stodest>${r.places.map((p, i) =>
    `<option value="${esc(p.dir)}">${esc(p.dir)}${i === 0 ? "（今の保存先）" : ""}</option>`).join("")}</select></label>`;
  const rows = kind === "projects"
    ? r.projects.map((p) => `<label class="row small" style="padding:2px 0"><input type="checkbox" value="${esc(p.id)}" data-place="${esc(p.place)}"${p.moving ? " disabled" : ""}>
        <b>${esc(p.name)}</b>${p.moving ? `<span class="warntext">移動中</span>` : ""}<span class="muted">更新 ${p.updated ? new Date(p.updated * 1000).toLocaleString("ja-JP", { dateStyle: "short", timeStyle: "short" }) : "不明"} ・ ${esc(p.place)}</span></label>`).join("")
    : r.places.filter((p) => p.exists && p.files).map((p) => `<label class="row small" style="padding:2px 0"><input type="checkbox" value="${esc(p.dir)}" data-place="${esc(p.dir)}">
        ${esc(p.dir)} <span class="muted">${p.files} 個・${gbText(p.bytes)}</span></label>`).join("") ;
  box.className = "";
  box.innerHTML = `<div class="muted small">移すものにチェックを付けて、移し先を選んでください。1 件ずつコピーし、コピーが完了して中身を確かめてから元を消します。同じドライブの中なら、コピーせずにすぐ移ります。
      移動中のプロジェクトは、終わるまで開けません。学習や解析が残っているプロジェクトは移せません。</div>
    ${kind === "projects" && r.places.length > 1 ? `<div class="row small" style="margin:4px 0">置き場所ごとにまとめて選ぶ: ${r.places.map((p) => `<button data-stopick="${esc(p.dir)}">${esc(p.dir)}</button>`).join("")}<button data-stopick="">選択を外す</button></div>` : ""}
    <div style="max-height:320px;overflow:auto">${rows || `<div class="muted small">移せるものはありません。</div>`}</div>
    <div class="row" style="margin-top:6px">${dest}<button class="primary" data-stogo>選んだものを移す</button></div>`;
  box.querySelectorAll("[data-stopick]").forEach((b) => b.onclick = () => box.querySelectorAll("input[type=checkbox]").forEach((c) => { c.checked = !c.disabled && !!b.dataset.stopick && c.dataset.place === b.dataset.stopick; }));
  const go = box.querySelector("[data-stogo]");
  go.onclick = async () => {
    const to = box.querySelector("[data-stodest]").value;
    const items = [...box.querySelectorAll("input[type=checkbox]:checked")].filter((c) => c.dataset.place !== to).map((c) => c.value);
    if (!items.length) return toast("移し先と違う場所にあるものを選んでください", "err");
    go.disabled = true;
    try { await api("/api/storage/move", { body: { kind, items, dest: to } }); toast(`${items.length} 件の移動を始めました`); $("#s-movestate").dataset.wasRunning = "1"; watchStorageMove(); }
    catch (e) { fail(e); } finally { go.disabled = false; }
  };
}

let storageMoveTimer = null;
async function watchStorageMove() {
  clearTimeout(storageMoveTimer);
  const el = $("#s-movestate");
  if (!el) return;
  let s;
  try { s = await api("/api/storage/move"); } catch { return; }
  if (!s.kind) { el.innerHTML = ""; return; }
  const pct = s.bytes_total ? Math.min(100, Math.round(s.bytes_done / s.bytes_total * 100)) : 0;
  el.innerHTML = `<div style="padding:8px 0">
    ${s.running ? `<div class="row small"><b>${s.cancelling ? "中止しています（今のファイルで止めて、途中までのコピーを消します）" : "移動中"}</b>
        ${s.done}/${s.total} 件 ・ ${gbText(s.bytes_done)} / ${s.bytes_total ? gbText(s.bytes_total) : "計算中"}（${pct}%）
        ${s.cancelling ? "" : `<button data-stocancel>中止</button>`}</div>
      ${s.current ? `<div class="small muted">今移しているもの: ${esc(s.current)}</div>` : ""}`
    : `<div class="small"><b>${s.cancelled ? "移動を中止しました" : "移動が終わりました"}</b>（${s.done}/${s.total} 件）。${s.cancelled ? "途中だったものは元の場所にそのまま残っています。" : ""}</div>`}
    ${s.errors.map((e) => `<div class="small err">移せませんでした: ${esc(e.item)} — ${esc(e.message)}</div>`).join("")}
    ${s.notes.map((n) => `<div class="small warntext">${esc(n)}</div>`).join("")}</div>`;
  const c = el.querySelector("[data-stocancel]");
  if (c) c.onclick = async () => { try { await api("/api/storage/move/cancel", { body: {} }); watchStorageMove(); } catch (e) { fail(e); } };
  if (s.running) storageMoveTimer = setTimeout(watchStorageMove, 1000);
  else if (el.dataset.wasRunning) { delete el.dataset.wasRunning; main.querySelectorAll("details[data-stomove][open]").forEach(drawStorageMove); }
  if (s.running) el.dataset.wasRunning = "1";
}

function bindStorage(root) {
  const save = async (kind, base, btn) => {
    btn.disabled = true;
    try {
      const r = await api("/api/storage", { method: "PUT", body: { kind, base } });
      toast(`これからは ${r[kind].dir} に保存します`);
      renderSettings();
    } catch (e) { btn.disabled = false; fail(e); }
  };
  root.querySelectorAll("[data-stoset]").forEach((b) => b.onclick = () =>
    save(b.dataset.stoset, root.querySelector(`[data-stobase="${b.dataset.stoset}"]`).value, b));
  root.querySelectorAll("[data-storeset]").forEach((b) => b.onclick = () => save(b.dataset.storeset, "", b));
  root.querySelectorAll("details[data-stomove]").forEach((d) => d.ontoggle = () => { if (d.open) drawStorageMove(d); });
  watchStorageMove();
}

async function renderSettings() {
  S.project = null;
  $("#crumbs").innerHTML = `<span>›</span><span>設定</span>`;
  let d, st, lib, sto;
  try { lib = await api("/api/library"); [d, st, sto] = await Promise.all([api("/api/models"), api("/api/status"), api("/api/storage")]); } catch (e) { return fail(e); }
  // families whose variant cannot always be read from the file (SDXL: Illustrious / Pony ..., Z-Image: Turbo / Base)
  const famVariants = (fam) => Object.entries(d.variants).filter(([, v]) => v.family === fam && v.available !== false);
  const vopts = Object.entries(d.variants).filter(([, v]) => v.available !== false).map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`).join("");
  main.innerHTML = `
    <h1>モデルと環境</h1>
    <div class="panel" style="margin-bottom:14px"><h2>環境</h2>
      <div>GPU: ${st.gpus.length ? st.gpus.map((g) => `${esc(g.name)}（VRAM ${Math.round(g.vram_total_mb / 1024)}GB）`).join(", ") : `<span class="err">NVIDIA GPU が見つかりません（素材の編集は可能、学習は不可）</span>`}</div>
      <div>学習エンジン環境: ${st.train_env ? "導入済み" : `<span class="err">未導入（setup.bat を実行してください）</span>`} ・ 評価用 ComfyUI: ${st.comfy_available ? "導入済み" : `<span class="err">未導入</span>`}</div>
      <div class="muted small">AI 解析: ${esc(workerLabel(st.worker))}。画像は外部に送信しません（すべてこの PC で処理します）。初回の解析時に解析用 AI モデル（合計 約 15GB）を取得します。</div></div>
    <div class="panel" style="margin-bottom:14px"><h2>表示</h2>
      <label class="row"><input type="checkbox" id="s-blur" ${st.blur_images === true ? "checked" : ""}> 画像をぼかして表示する</label>
      <div class="muted small">画面に出る画像をすべてぼかします。画像にマウスを乗せると右上に目のボタンが出て、押すとその画像だけ表示され、もう一度押すとまたぼけます。画像そのもののクリックは、ぼかしていないときと同じ動き（拡大や選択）になります。目で表示した画像は、拡大したときも表示されます。見た目だけの設定で、学習・解析・生成や画像ファイルには影響しません。</div></div>
    <div class="panel" style="margin-bottom:14px"><h2>学習用キャッシュ</h2>
      <label class="row"><input type="checkbox" id="s-tcache" ${st.keep_train_cache !== false ? "checked" : ""}> 学習用キャッシュを残す</label>
      <div class="muted small">学習の前に、画像と説明文を学習用の形に変換します（キャッシュ）。残しておくと、同じ画像と説明文で次に学習するときに変換を省けるので、学習が早く始まります。そのかわり、プロジェクトのフォルダーの容量が増えます。チェックを外すと、今あるキャッシュをすべて消し、これからは学習が終わるたびに消します。学習中の回のキャッシュは、その回が終わってから消します。画像、説明文、学習した LoRA は消えません。</div>
      <div class="row" style="margin-top:10px" id="s-lo"><button id="s-lo-count">すべてのプロジェクトの学習データを破棄する</button></div>
      <div class="muted small">すべてのプロジェクトの途中保存の LoRA、再開用の保存データ、学習用キャッシュを消します。書き出した LoRA の元の保存版、素材、説明文、比較画像は残すので、学習し直すことはできます。消した保存版は比較や書き出しに使えなくなり、途中からの再開もできなくなります。学習中・予約中の回と、比較画像を作っている回は消しません。</div></div>
    ${storageHtml(sto)}
    ${misplacedHtml(lib.misplaced)}
    <div class="panel" style="margin-bottom:14px"><h2>モデルの置き場所</h2>
      <div class="muted small">ここに置いたモデルは自動で一覧に出て、画像の生成にも学習にも使えます（コピーはしません）。checkpoint は <b>checkpoints</b>、LoRA は <b>loras</b> に置いてください。</div>
      ${lib.default_dirs.map((x) => `<div class="small">・${esc(x)}</div>`).join("")}
      ${lib.roots.map((x) => `<div class="row small">・${esc(x)} <span class="muted">（追加した場所）</span><button data-rmroot="${esc(x)}">外す</button></div>`).join("")}
      <div class="row" style="margin-top:8px"><input type="text" id="s-root" style="flex:1;min-width:260px" placeholder="ComfyUI のフォルダーや、ほかのソフトのモデルフォルダー（例: D:\ComfyUI、D:\forge\models\Lora）">
        <button id="s-addroot">場所を追加</button></div>
      <div class="muted small">ComfyUI のフォルダーを指定すると、その models フォルダーと、extra_model_paths.yaml に書かれた別の置き場所もまとめて読みます。ファイルはコピーせず、ComfyUI と同じファイルを生成にも学習にも使います。指定したフォルダー以外は読みません。</div>
      <details style="margin-top:8px"><summary class="small">見つかったファイル（checkpoint ${lib.checkpoints.length}・LoRA ${lib.loras.length}）</summary>
        ${lib.checkpoints.map((c) => `<div class="small">checkpoint: ${esc(c.name)} <span class="muted">${esc(c.family_label)}</span>${c.family && c.family !== "sdxl" ? ` <span class="muted">（${esc(c.evidence || "")}）</span>` : ""}</div>`).join("")}
        ${lib.loras.map((c) => `<div class="small">LoRA: ${esc(c.name)} <span class="muted">${esc(c.family_label)}</span></div>`).join("")}</details></div>
    <div class="panel" style="margin-bottom:14px" id="s-share"></div>
    <div class="panel" style="margin-bottom:14px"><h2>登録済みのベースモデル</h2>
      ${d.models.length ? d.models.map((m) => `<div class="row" style="padding:6px 0;border-bottom:1px solid var(--line)">
        <b>${esc(m.name)}</b>${m.variant ? `<span class="kind">${esc(m.variant_info.label || m.variant)}</span>` : ""}
        ${famVariants(m.family).length > 1 && (!m.variant || m.needs_variant || (m.auto && !m.variant_confirmed)) ? `<select data-variant="${m.id}">
          <option value="">${m.variant ? "系統（推定）を確認してください" : "系統を選んでください（学習に必要）"}</option>${famVariants(m.family).map(([k, v]) => `<option value="${k}" ${k === m.variant ? "selected" : ""}>${esc(v.label)}</option>`).join("")}</select>` : ""}
        ${m.variant_info.verified ? `<span class="small" style="color:var(--ok)">検証済み</span>` : m.variant_info.experimental ?
          `<span class="small warntext" title="${esc(m.variant_info.note || "")}">試験導入・動作未確認（動かない可能性があります）</span>` : `<span class="small warntext">未検証</span>`}
        ${m.missing.length ? `<span class="err small">ファイルが見つかりません: ${m.missing.join(", ")}</span>` : ""}
        <span class="muted small">${Object.entries(m.files).map(([k, p]) => `${k}: ${esc(p)}${m.hash && m.hash[k] ? "" : "（ハッシュ計算中）"}`).join("<br>")}</span>
        <div class="spacer"></div><button data-del="${m.id}">登録解除</button></div>`).join("") : `<div class="muted">まだありません。</div>`}</div>
    <div class="panel" style="margin-bottom:14px"><h2>手元のモデルファイルを登録</h2>
      <div class="form">
        <label>メインのファイル</label><div class="row"><input type="text" id="s-ckpt" style="flex:1" placeholder="例: D:\\models\\xxx.safetensors"><button id="s-detect">種類を判定</button></div>
        <label>判定結果</label><div id="s-det" class="muted small">—</div>
        <label>種類</label><select id="s-variant">${vopts}</select>
        <label>表示名</label><input type="text" id="s-name">
        <div id="s-extra" style="display:contents"></div>
      </div>
      <div class="row" style="margin-top:10px"><button class="primary" id="s-add">登録</button>
        <span class="muted small">指定したファイルだけを読み取ります。PC 内を自動で探すことはしません。</span></div></div>
    <div class="panel" style="margin-bottom:14px"><h2>リンク・ファイルから追加（モデル・LoRA）</h2>
      <div class="muted small">Hugging Face か Civitai のページのリンクを貼ると、ダウンロードする前にファイルの先頭だけを読んで、
        何のファイルか（SDXL 系の checkpoint、LoRA、VAE など）と、使えるかどうかを判定します。追加すると、種類に合ったフォルダーに置かれます。</div>
      <div class="row" style="margin-top:8px"><input type="text" id="im-link" style="flex:1;min-width:280px" placeholder="例: https://civitai.com/models/12345 / https://huggingface.co/作者/リポジトリ">
        <button class="primary" id="im-go">調べる</button></div>
      <div id="im-token" class="row small" style="margin-top:6px" hidden></div>
      <div id="im-res"></div>
      <div id="im-drop" class="dropzone small">PixAI などでダウンロードしたファイル（.safetensors / .gguf）は、ここにドロップするか
        <label class="linklike">ファイルを選んで<input type="file" id="im-file" accept=".safetensors,.gguf" multiple hidden></label>追加できます。中身を確認して、種類に合ったフォルダーに置きます。</div>
      <div id="im-state"></div></div>
    <div class="panel"><h2>おすすめのモデル</h2>
      <div class="muted small">動作確認に使ったモデルと、本体・テキストエンコーダー・VAE をそろえる必要があるモデル（Anima・Qwen-Image）を、必要なファイル一式で取得します。
        取得先は models フォルダーの中（checkpoints・diffusion_models・text_encoders・vae）です。各モデルの利用条件を確認してから使ってください。</div>
      <div id="s-catalog" class="muted small" style="margin-top:6px">読み込み中…</div></div>
    <div class="panel" style="margin-top:14px"><h2>タグの辞書</h2>
      <div class="muted small">タグの日本語訳と読みを見たり直したりします。ここで直した訳は、タグの予測変換、画像生成の日本語表示、画像からプロンプト（i2t）で使われます。</div>
      <div style="margin-top:6px"><a href="#/dict"><button class="small">辞書を開く</button></a></div></div>
    <div id="s-i2t"></div>
    <div class="panel" style="margin-top:14px"><h2>ログ</h2><div id="s-logs" class="small muted">読み込み中…</div></div>`;
  drawAppLogs();
  drawCatalog();
  drawShare();
  drawI2tSettings();
  bindImport();
  const extra = () => {
    const fam = d.variants[$("#s-variant").value].family;
    $("#s-extra").innerHTML = d.family_files[fam].filter(([k]) => k !== "ckpt").map(([k, label, req]) =>
      `<label>${esc(label)}${req ? "" : "（任意）"}</label><input type="text" data-file="${k}" placeholder="ファイルのパス">`).join("");
  };
  $("#s-variant").onchange = extra; extra();
  $("#s-detect").onclick = async () => {
    try {
      const r = await api("/api/models/detect", { body: { path: $("#s-ckpt").value } });
      $("#s-det").innerHTML = r.family ? `${esc(r.family)}（確信度${esc(r.confidence)}）: ${esc(r.evidence)}${r.note ? `<br><span class="warntext">${esc(r.note)}</span>` : ""}` : `<span class="err">${esc(r.evidence || "判定できません")}</span>`;
      if (r.variant && d.variants[r.variant] && d.variants[r.variant].available !== false) { $("#s-variant").value = r.variant; extra(); }
      else if (r.family === "sdxl") { $("#s-det").innerHTML += "<br>SDXL 系のどれかを選んでください（Illustrious / Pony などはファイルから判別できないことがあります）。"; }
      if (!$("#s-name").value) $("#s-name").value = $("#s-ckpt").value.split(/[\\/]/).pop().replace(/\.safetensors$/i, "");
    } catch (e) { $("#s-det").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  };
  $("#s-add").onclick = async () => {
    const files = { ckpt: $("#s-ckpt").value };
    $$("[data-file]").forEach((i) => { files[i.dataset.file] = i.value; });
    try { await api("/api/models", { body: { name: $("#s-name").value || "model", variant: $("#s-variant").value, files } }); toast("登録しました"); renderSettings(); }
    catch (e) { fail(e); }
  };
  $$("[data-variant]").forEach((sel) => sel.onchange = async () => {
    if (!sel.value) return;
    try { await api(`/api/models/${sel.dataset.variant}`, { method: "PATCH", body: { variant: sel.value } }); toast("系統を保存しました"); renderSettings(); } catch (e) { fail(e); }
  });
  $("#s-addroot").onclick = async () => {
    try {
      const r = await api("/api/library/roots", { body: { path: $("#s-root").value } });
      toast(r.added && r.added.length ? `次の場所を追加しました: ${r.added.join("、")}` : "その場所はすでに追加されています");
      renderSettings();
    } catch (e) { fail(e); }
  };
  $$("[data-rmroot]").forEach((b) => b.onclick = async () => {
    try { await api("/api/library/roots/remove", { body: { path: b.dataset.rmroot } }); toast("場所を外しました（ファイルはそのままです）"); renderSettings(); } catch (e) { fail(e); }
  });
  $$("[data-del]").forEach((b) => b.onclick = async () => {
    try { await api(`/api/models/${b.dataset.del}`, { method: "DELETE" }); toast("登録を解除しました（ファイルは削除していません）"); renderSettings(); }
    catch (e) { fail(e); }
  });
  bindMisplaced(main, renderSettings);
  bindStorage(main);
  $("#s-tcache").onchange = async (e) => {
    try {
      const r = await api("/api/ui", { method: "PUT", body: { keep_train_cache: e.target.checked } });
      toast(r.keep_train_cache ? "学習用キャッシュを残します" : `学習用キャッシュを消しました（${(r.freed / 1e9).toFixed(1)}GB）`);
    } catch (err) { e.target.checked = !e.target.checked; fail(err); }
  };
  $("#s-lo-count").onclick = async (e) => {
    e.target.disabled = true; e.target.textContent = "容量を数えています…";
    let d;
    try { d = await api("/api/training_leftovers"); } catch (err) { e.target.disabled = false; e.target.textContent = "すべてのプロジェクトの学習データを破棄する"; return fail(err); }
    const row = $("#s-lo");
    if (!row) return;
    if (!d.bytes) { row.innerHTML = `<span class="muted">破棄できる学習データはありません。</span>`; return; }
    row.innerHTML = `<b>学習データ（${fmtSize(d.bytes)}、${d.projects} 件のプロジェクト）を破棄しますか？</b><button id="s-lo-yes">はい</button><button id="s-lo-no">いいえ</button>`;
    $("#s-lo-no").onclick = () => renderSettings();
    $("#s-lo-yes").onclick = async (ev) => {
      ev.target.disabled = true; ev.target.textContent = "破棄しています…";
      try {
        const r = await api("/api/training_leftovers/discard", { body: {} });
        toast(`学習データを破棄しました${r.freed ? `（${fmtSize(r.freed)} 空きました）` : ""}` + (r.skipped ? "。使用中の回と、移動中のプロジェクトは残しました" : ""));
        renderSettings();
      } catch (err) { ev.target.disabled = false; ev.target.textContent = "はい"; fail(err); }
    };
  };
  $("#s-blur").onchange = async (e) => {
    document.documentElement.classList.toggle("blurimg", e.target.checked);
    try { await api("/api/ui", { method: "PUT", body: { blur_images: e.target.checked } }); }
    catch (err) { e.target.checked = !e.target.checked; document.documentElement.classList.toggle("blurimg", e.target.checked); fail(err); }
  };
}

/* ------------------------------------------------------------------ add from a link / a dropped file */
let IMPORT = null;   // last resolved link: {link, res}

function bindImport() {
  $("#im-go").onclick = () => resolveLink();
  $("#im-link").onkeydown = (e) => { if (e.key === "Enter") resolveLink(); };
  const dz = $("#im-drop");
  dz.ondragover = (e) => { e.preventDefault(); e.stopPropagation(); dz.classList.add("over"); };
  dz.ondragleave = () => dz.classList.remove("over");
  dz.ondrop = (e) => { e.preventDefault(); e.stopPropagation(); dz.classList.remove("over"); uploadModelFiles([...e.dataTransfer.files]); };
  $("#im-file").onchange = (e) => uploadModelFiles([...e.target.files]);
  drawImportState();
}

async function resolveLink() {
  const link = $("#im-link").value.trim();
  if (!link) return;
  const tok = $("#im-tok");
  const body = { link };
  if (tok && tok.value) body[tok.dataset.site === "hf" ? "hf_token" : "civitai_token"] = tok.value;
  $("#im-res").innerHTML = `<div class="muted small" style="margin-top:8px">調べています…（ファイルごとに先頭だけを読んでいます）</div>`;
  let res;
  try {
    const r = await fetch("/api/import/resolve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    res = await r.json();
    if (r.status === 401) { showTokenField(res.need_token, res.detail); $("#im-res").innerHTML = ""; return; }
    if (!r.ok) throw new Error(res.detail || `エラー ${r.status}`);
  } catch (e) { $("#im-res").innerHTML = `<div class="err small" style="margin-top:8px">${esc(e.message)}</div>`; return; }
  IMPORT = { link, res, token: body.hf_token || body.civitai_token || null };
  if (res.need_token) showTokenField(res.need_token, "このページのファイルの取得にはログインが必要です。" +
    (res.need_token === "hf" ? "Hugging Face のアクセストークンを入れてください。" : "Civitai の API キー（Civitai の設定画面で作れます）を入れてください。"));
  drawResolved();
}

function showTokenField(site, msg) {
  const el = $("#im-token");
  el.hidden = false;
  el.innerHTML = `<span class="warntext">${esc(msg)}</span><input type="password" id="im-tok" data-site="${site}" style="width:240px"
    placeholder="${site === "hf" ? "Hugging Face のアクセストークン" : "Civitai の API キー"}"><button id="im-retry">もう一度調べる</button>
    <span class="muted">キーはこの操作にだけ使い、保存しません。</span>`;
  $("#im-retry").onclick = () => resolveLink();
}

function drawResolved() {
  const { res } = IMPORT;
  const gb = (b) => b ? `${(b / 1e9).toFixed(b < 1e9 ? 2 : 1)}GB` : "容量不明";
  $("#im-res").innerHTML = `<div style="margin-top:10px"><b>${esc(res.title || "")}</b>
      <a href="${esc(res.page)}" target="_blank" rel="noopener" class="small">ページを開く</a>
      ${res.site_info ? `<span class="muted small">${esc(res.site_info)}</span>` : ""}
      ${res.license ? `<span class="muted small">ライセンス: ${esc(res.license)}</span>` : ""}</div>
    ${res.files.length ? res.files.map((f, i) => { const p = f.probe || {};
      const what = p.ok ? `${esc(p.evidence || "")}${p.family_label && !String(p.evidence || "").includes(p.family_label) ? `（${esc(p.family_label)}）` : ""} → <b>${esc(p.dest_label)}</b> に置きます` :
        p.ok === false ? `<span class="err">使えません: ${esc(p.reason || "")}</span>` : `<span class="muted">${esc(p.reason || "")}</span>`;
      return `<div class="row small" style="padding:6px 0;border-bottom:1px solid var(--line)">
        <span style="min-width:200px;word-break:break-all"><b>${esc(f.name)}</b></span><span class="muted">${gb(f.size)}</span>
        <span>${what}</span>${p.note ? `<span class="warntext">${esc(p.note)}</span>` : ""}<div class="spacer"></div>
        ${f.have ? `<span style="color:var(--ok)" title="${esc(f.have)}">手元にあります（${esc(f.have.split(/[\\/]/).pop())}）</span>` : p.ok === false || p.need_token ? "" : `<button data-imdl="${i}">追加する</button>`}</div>`; }).join("")
      : `<div class="muted small">このページに取得できるファイル（.safetensors / .gguf）がありません。</div>`}
    ${res.skipped && res.skipped.length ? `<div class="muted small">安全のため取得しないファイル（読み込むとプログラムが動く形式）: ${res.skipped.map(esc).join(", ")}</div>` : ""}
    ${res.files.length && res.files.some((f) => (f.probe || {}).kind === "model" && (f.probe || {}).family && f.probe.family !== "sdxl") ?
      `<div class="muted small">Anima・Qwen-Image の本体は、テキストエンコーダーと VAE もそろうと選べるようになります（「おすすめのモデル」から一式で取得することもできます）。</div>` : ""}`;
  $$("[data-imdl]").forEach((b) => b.onclick = async () => {
    const f = IMPORT.res.files[+b.dataset.imdl];
    b.disabled = true;
    try {
      await api("/api/import/download", { body: { url: f.url, name: f.name, size: f.size, site: IMPORT.res.source,
        token: IMPORT.token, variant_hint: (f.probe || {}).kind === "model" ? IMPORT.res.variant_hint : null } });
      toast(`${f.name} の取得を始めました`);
      drawImportState();
    } catch (e) { fail(e); b.disabled = false; }
  });
}

async function uploadModelFiles(files) {
  for (const file of files) {
    if (!/\.(safetensors|gguf)$/i.test(file.name)) { toast(`${file.name}: safetensors か GGUF のファイルだけを追加できます`, "err"); continue; }
    const row = document.createElement("div");
    row.className = "small";
    row.textContent = `${file.name}: 受け取り中…`;
    $("#im-state").prepend(row);
    await new Promise((resolve) => {
      const x = new XMLHttpRequest();
      x.open("POST", `/api/import/upload?name=${encodeURIComponent(file.name)}`);
      x.upload.onprogress = (e) => { if (e.lengthComputable) row.textContent = `${file.name}: 受け取り中 ${(e.loaded / 1e9).toFixed(1)} / ${(e.total / 1e9).toFixed(1)} GB`; };
      x.onload = () => {
        let r = {};
        try { r = JSON.parse(x.responseText); } catch { }
        if (x.status === 200) { row.innerHTML = `${esc(file.name)}: <span style="color:var(--ok)">追加しました（${esc(r.dest_label || "")} に置きました）</span>`; }
        else row.innerHTML = `${esc(file.name)}: <span class="err">${esc(r.detail || "追加できませんでした")}</span>`;
        resolve();
      };
      x.onerror = () => { row.innerHTML = `${esc(file.name)}: <span class="err">送れませんでした</span>`; resolve(); };
      x.send(file);
    });
  }
  drawImportState();
}

async function drawImportState() {
  const el = $("#im-state");
  if (!el) return;
  let cat;
  try { cat = await api("/api/catalog"); } catch { return; }
  const items = (cat.imports || []).filter((s) => s.src !== "upload");   // uploads are shown by uploadModelFiles
  el.querySelectorAll(".imst").forEach((x) => x.remove());
  el.insertAdjacentHTML("beforeend", items.map((s) => `<div class="small imst">${esc(s.name)}: ${
    s.state === "running" ? `${esc(s.message)} ${(s.done_bytes / 1e9).toFixed(1)}${s.total_bytes ? ` / ${(s.total_bytes / 1e9).toFixed(1)}` : ""} GB
      ${s.cancel ? "" : `<button class="small" data-imcancel="${esc(s.id)}">キャンセル</button>`}` :
    s.state === "cancelled" ? `<span class="muted">キャンセルしました</span>` :
    s.state === "done" ? `<span style="color:var(--ok)">追加しました（${esc(s.dest_label || "")} に置きました）</span>` :
    `<span class="err">${esc(s.error || "失敗")}</span>`}</div>`).join(""));
  $$("[data-imcancel]", el).forEach((b) => b.onclick = async () => {
    b.disabled = true;
    try { await api(`/api/import/download/${b.dataset.imcancel}/cancel`, { body: {} }); } catch (e) { fail(e); }
    drawImportState();
  });
  if (items.some((s) => s.state === "running")) pollLater("imports", 2000, () => { if (S.route[0] === "settings") drawImportState(); });
}

/* Files shared with ComfyUI: identical model files become one file on disk (hard links), and exported
   LoRAs are also placed in ComfyUI's loras/LoRAStudio. */
async function drawShare() {
  const el = $("#s-share");
  if (!el) return;
  let st;
  try { st = await api("/api/comfy_link"); } catch (e) { el.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
  const sc = st.scan || {}, items = sc.items || [];
  const same = items.filter((x) => x.status === "same"), other = items.filter((x) => x.status === "other_drive"),
    linked = items.filter((x) => x.status === "linked");
  const gb = (b) => (b / 1e9).toFixed(1);
  const typed = $("#s-comfy", el), typing = typed && document.activeElement === typed;   // redrawn every 2 s while scanning
  el.innerHTML = `<h2>ComfyUI とファイルを共用</h2>
    <div class="muted small">ComfyUI のフォルダーを指定すると、作った LoRA を書き出したときに ComfyUI の loras\${"LoRAStudio"} フォルダーにも置きます（手持ちの LoRA とは別のフォルダーです）。また、このツールの models フォルダーと ComfyUI に同じファイルがあるときは、1 つのファイルにまとめられます。どちらもハードリンクなので容量は増えず、ツールも ComfyUI も今までと同じ場所から読めます。</div>
    <div class="row" style="margin-top:8px"><input type="text" id="s-comfy" style="flex:1;min-width:260px" value="${esc(st.comfy_dir || "")}" placeholder="ComfyUI のフォルダー（例: E:\AI\ComfyUI）">
      <button id="s-comfy-save">保存</button></div>
    ${st.export_dir ? `<div class="small" style="margin-top:4px">作った LoRA の置き場所（ComfyUI 側）: ${esc(st.export_dir)}</div>` : ""}
    ${st.comfy_dir ? `<div class="row" style="margin-top:8px"><button id="s-dup" ${sc.state === "running" ? "disabled" : ""}>同じファイルを調べる</button>
      <span class="muted small">${sc.state === "running" ? esc(sc.message) : "大きなファイルは中身を全部読んで比べるので、数分かかることがあります。"}</span></div>` : ""}
    ${sc.state === "failed" ? `<div class="err small">${esc(sc.message)}</div>` : ""}
    ${sc.state === "done" ? `<div style="margin-top:8px">
      ${same.length ? `<div class="small"><b>まとめられるファイル（${same.length} 個・${gb(same.reduce((a, x) => a + x.size, 0))}GB 空きます）</b></div>
        ${same.map((x) => `<div class="small">・${esc(x.tool)}<br><span class="muted">　→ ComfyUI の ${esc(x.comfy)} と中身が同じ</span></div>`).join("")}
        <div class="row" style="margin-top:6px"><button class="primary" id="s-link">1 つのファイルにまとめる</button>
          <span class="muted small">このツール側のファイルを、ComfyUI のファイルへのハードリンクに置き換えます。ComfyUI 側のファイルは変更しません。</span></div>`
        : `<div class="small">まとめられるファイルはありません。</div>`}
      ${linked.length ? `<div class="small muted" style="margin-top:4px">すでに 1 つにまとまっているもの: ${linked.map((x) => esc(x.tool.split(/[\\/]/).pop())).join("、")}</div>` : ""}
      ${other.length ? `<div class="small muted" style="margin-top:4px">別のドライブにあるためまとめられないもの: ${other.map((x) => esc(x.tool.split(/[\\/]/).pop())).join("、")}</div>` : ""}</div>` : ""}`;
  if (typing) { $("#s-comfy").value = typed.value; $("#s-comfy").focus(); }
  $("#s-comfy-save").onclick = async () => {
    try { await api("/api/comfy_link", { method: "PUT", body: { comfy_dir: $("#s-comfy").value } }); toast("ComfyUI のフォルダーを保存しました"); drawShare(); } catch (e) { fail(e); }
  };
  if ($("#s-dup")) $("#s-dup").onclick = async () => { try { await api("/api/comfy_link/scan", { body: {} }); drawShare(); } catch (e) { fail(e); } };
  if ($("#s-link")) $("#s-link").onclick = async () => {
    $("#s-link").disabled = true;
    try {
      const r = await api("/api/comfy_link/apply", { body: {} });
      toast(`${r.linked} 個のファイルを 1 つにまとめました（${gb(r.freed_bytes)}GB 空きました）`);
      (r.errors || []).forEach((x) => toast(x, "err"));
    } catch (e) { fail(e); }
    drawShare();
  };
  if (sc.state === "running") pollLater("share", 2000, () => { if (S.route[0] === "settings") drawShare(); });
}

/* キャンセル on a model download: it stops within a few seconds and the half-fetched file is removed. */
function bindDlCancel(el, redraw) {
  $$("[data-dlcancel]", el).forEach((b) => b.onclick = async (e) => {
    e.preventDefault();   // inside the first-run <label>: do not tick the checkbox
    b.disabled = true;
    try { await api(`/api/downloads/${b.dataset.dlcancel}/cancel`, { body: {} }); } catch (err) { fail(err); }
    redraw();
  });
}

async function drawCatalog() {
  const el = $("#s-catalog");
  if (!el) return;
  let cat;
  try { cat = await api("/api/catalog"); } catch (e) { el.innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  el.className = "";
  // redrawn every 3 s while a download runs: a token being typed and the choices made for other models stay
  const kept = {}, ckey = (x) => x.dataset.tok ? `t:${x.dataset.tok}` : `c:${x.dataset.choice}:${x.dataset.key}`;
  $$("[data-tok], [data-choice]", el).forEach((x) => { kept[ckey(x)] = x.value; });
  const focusKey = document.activeElement && el.contains(document.activeElement) && document.activeElement.dataset.tok ? ckey(document.activeElement) : null;
  el.innerHTML = cat.items.map((x) => {
    const s = x.dl;
    const choices = Object.entries(x.choices || {}).map(([k, opts]) => `<select data-choice="${x.id}" data-key="${k}">${opts.map((o) =>
      `<option value="${o.id}" ${o.id === x.recommended[k] ? "selected" : ""}>${esc(o.label)}${o.id === x.recommended[k] ? "（この PC に推奨）" : ""}</option>`).join("")}</select>`).join("");
    const state = x.deferred ? `<span class="small muted">準備中（積み残し）</span>` :
      x.installed_model ? `<span class="small" style="color:var(--ok)">追加済み</span>` :
      s && s.state === "running" ? `<span class="small">${esc(s.message || "")} ${(s.done_bytes / 1e9).toFixed(1)} / ${(s.total_bytes / 1e9).toFixed(1)} GB</span>
        ${s.cancel ? "" : `<button class="small" data-dlcancel="${x.id}">キャンセル</button>`}` :
      s && s.state === "done" ? `<span class="small" style="color:var(--ok)">完了</span>` :
      `${choices}${x.gated ? `<input type="password" placeholder="HF アクセストークン" data-tok="${x.id}" style="width:170px">` : ""}<button data-dl="${x.id}">追加する</button>`;
    return `<div style="padding:8px 0;border-bottom:1px solid var(--line)"><div class="row"><b>${esc(x.name)}</b>
        ${x.default ? `<span class="kind">標準</span>` : ""}${x.experimental ? `<span class="small warntext">試験導入・動作未確認</span>` : ""}
        <span class="muted small">約 ${x.size_gb}GB ・ ライセンス: ${esc(x.license)}${x.gated ? " ・ ログインと同意が必要" : ""}</span>
        <div class="spacer"></div>${state}</div>
      ${x.note ? `<div class="small ${x.experimental ? "warntext" : "muted"}">${esc(x.note)}</div>` : ""}
      ${x.min_ram && cat.ram_gb && cat.ram_gb < x.min_ram - 2 && !x.installed_model ? `<div class="small warntext">この PC のメインメモリは ${Math.round(cat.ram_gb)}GB です。このモデルの学習には ${x.min_ram}GB を推奨します（足りないと学習が非常に遅くなるか、途中で失敗することがあります）。</div>` : ""}
      ${s && s.error ? `<div class="err small">${esc(s.error)}</div>` : ""}
      ${s && s.state === "cancelled" ? `<div class="muted small">ダウンロードをキャンセルしました。途中までのファイルは消してあります。</div>` : ""}</div>`;
  }).join("") + `<div class="muted small" style="margin-top:8px">ダウンロードせずに使う場合は、ファイルを置くだけで認識されます：
    checkpoint（SDXL 系）は <b>checkpoints</b> <button class="small" data-opendir="checkpoints">開く</button>、
    Anima・Qwen-Image の本体は <b>diffusion_models</b> <button class="small" data-opendir="diffusion_models">開く</button>、
    テキストエンコーダーは <b>text_encoders</b> <button class="small" data-opendir="text_encoders">開く</button>、
    VAE は <b>vae</b> <button class="small" data-opendir="vae">開く</button>。</div>`;
  $$("[data-dl]", el).forEach((b) => b.onclick = async () => {
    const tok = $(`[data-tok="${b.dataset.dl}"]`);
    const choices = {};
    $$(`[data-choice="${b.dataset.dl}"]`, el).forEach((c) => { choices[c.dataset.key] = c.value; });
    try { await api(`/api/downloads/${b.dataset.dl}`, { body: { token: tok ? tok.value : null, choices } }); drawCatalog(); } catch (e) { fail(e); }
  });
  $$("[data-tok], [data-choice]", el).forEach((x) => { const k = ckey(x); if (k in kept) x.value = kept[k]; if (k === focusKey) x.focus(); });
  bindDlCancel(el, drawCatalog);
  bindOpenDir(el);
  cat.items.forEach((x) => { if (x.dl && x.dl.state === "running") _catReloaded.delete(x.id); });
  // a finished download that is not listed as added yet: the page is reloaded once for it (not again and again
  // when the model still does not count as added, e.g. a component it needs is not ready)
  const fresh = cat.items.filter((x) => x.dl && x.dl.state === "done" && !x.installed_model && !_catReloaded.has(x.id));
  if (cat.items.some((x) => x.dl && x.dl.state === "running")) pollLater("catalog", 3000, () => { if (S.route[0] === "settings") drawCatalog(); });
  else if (fresh.length) { fresh.forEach((x) => _catReloaded.add(x.id)); pollLater("catalog", 1000, () => { if (S.route[0] === "settings") renderSettings(); }); }
}
const _catReloaded = new Set();

// ------------------------------------------------------------------ LoRA settings (LoRA 設定)
/* Trigger words, the default of the "put the trigger words in the prompt" switch, and the strength a new
   LoRA row starts with, per LoRA file.  Saved as you type. */
/* タグの辞書: the bundled translations (i2t's tags_ja.json + dict/*.csv) and the words the user registered
   (translations.json), with their readings.  Changing a bundled word registers the user's own translation, which
   wins over the bundled one; removing a registered word brings the bundled one back. */
const DICT_SHOW = 200;
const _dcDirty = new Map();   // タグの辞書: rows typed in but not saved (kept through redraws and searches)
async function renderDict() {
  S.project = null;
  $("#crumbs").innerHTML = `<span>›</span><a href="#/settings">設定</a><span>›</span><span>タグの辞書</span>`;
  main.innerHTML = `<div class="muted">読み込み中…</div>`;
  let v, rd;
  try {
    [v, rd] = await Promise.all([i2tCall("ja_table"), fetch("/static/tag_readings.json").then((r) => r.ok ? r.json() : {}).catch(() => ({}))]);
  } catch (e) {
    main.innerHTML = `<div class="err">辞書を読み込めませんでした（${esc(e.message)}）。画像からプロンプト（i2t）が使える状態か、設定で確認してください。</div>`;
    return;
  }
  const user = v.user || {}, ureads = v.readings || {};
  const rows = Object.entries(v.table).map(([en, ja]) => ({ en, ja, mine: en in user, read: (ureads[en] || (!(en in user) && rd[en]) || "") }))
    .sort((a, b) => a.en.localeCompare(b.en));
  main.innerHTML = `<div class="panel">
    <div class="row"><h2 style="margin:0">タグの辞書</h2><div class="spacer"></div>
      <input type="search" id="dc-q" placeholder="英語・日本語・読みで探す" style="width:240px">
      <label class="small"><input type="checkbox" id="dc-mine"> 登録した語だけ</label></div>
    <div class="muted small" style="margin-top:6px">全 ${rows.length} 語（登録した語 ${Object.keys(user).length} 語）。
      同梱の訳を直すと「登録した語」になり、同梱の訳より優先されます。登録した語を消すと、同梱の訳があればそれに戻ります。
      漢字を含む訳には、読みをひらがなで入れてください（予測変換で、変換前のひらがなから探すのに使います）。</div>
    ${(() => {
      const n = Object.entries(user).filter(([en, ja]) => KANJI_RE.test(ja) && !ureads[en]).length;
      return n ? `<div class="row" style="margin-top:6px"><button class="small" id="dc-rd" title="i2t の設定の Ollama のモデルで読みを調べ、ひらがなで返ってきたものを登録します">読みのない ${n} 語の読みを Ollama で調べる</button>
        <span class="muted small" id="dc-rdmsg"></span></div>` : "";
    })()}
    <div class="row" style="margin-top:10px"><b class="small">追加</b>
      <input type="text" id="dc-en" placeholder="英語（例: cat ears）" style="width:200px">
      <input type="text" id="dc-ja" placeholder="日本語（例: 猫耳）" style="width:200px">
      <input type="text" id="dc-read" placeholder="読み（ひらがな）" style="width:160px">
      <button class="small primary" id="dc-add">登録</button></div>
    <div id="dc-list" style="margin-top:10px"></div></div>`;
  const save = async (en, ja, read) => {
    read = toHira((read || "").trim());
    if (ja && KANJI_RE.test(ja) && !read) { toast(`「${ja}」の読みを、ひらがなで入れてください`); return false; }
    try { await i2tCall("register_ja", en, ja, read); } catch (e) { fail(e); return false; }
    TAGAC.list = null; IGJ.table = null;   // the completion and the 日本語 view read the dictionary again
    return true;
  };
  const draw = () => {
    const q = $("#dc-q").value.trim().toLowerCase(), qk = toHira(q), mine = $("#dc-mine").checked;
    const hit = rows.filter((r) => (!mine || r.mine) && (!q || r.en.toLowerCase().includes(q) || toHira(r.ja).includes(qk) || r.read.includes(qk)));
    $("#dc-list").innerHTML = (hit.length > DICT_SHOW ? `<div class="muted small">${hit.length} 語のうち、先頭の ${DICT_SHOW} 語を表示しています。探す語を入れて絞り込んでください。</div>` : "")
      + (hit.length ? `<table class="dict"><thead><tr><th>英語</th><th>日本語</th><th>読み</th><th></th></tr></thead><tbody>${hit.slice(0, DICT_SHOW).map((r) => `<tr data-de="${esc(r.en)}">
        <td>${esc(r.en)}${r.mine ? ` <span class="badge">登録</span>` : ""}</td>
        <td><input type="text" data-dja value="${esc(r.ja)}"></td><td><input type="text" data-dread value="${esc(r.read)}" placeholder="${KANJI_RE.test(r.ja) ? "読みなし" : ""}"></td>
        <td class="nowrap"><button class="small" data-dsave hidden>保存</button>${r.mine ? `<button class="small" data-ddel title="登録を消します（同梱の訳があればそれに戻ります）">消す</button>` : ""}</td></tr>`).join("")}</tbody></table>`
        : `<div class="muted small">見つかりません。</div>`);
    $$("[data-de]", $("#dc-list")).forEach((tr) => {
      const en = tr.dataset.de, r = rows.find((x) => x.en === en), ja = $("[data-dja]", tr), rdi = $("[data-dread]", tr), sv = $("[data-dsave]", tr);
      const dirty = _dcDirty.get(en);
      if (dirty) { ja.value = dirty.ja; rdi.value = dirty.read; }
      const changed = () => {
        sv.hidden = ja.value.trim() === r.ja && rdi.value.trim() === r.read;
        if (sv.hidden) _dcDirty.delete(en); else _dcDirty.set(en, { ja: ja.value, read: rdi.value });
      };
      changed();
      ja.oninput = rdi.oninput = changed;
      const go = async () => {
        const nja = ja.value.trim();
        if (!nja) { toast("日本語を入れてください（登録を消すときは「消す」を押します）"); return; }
        if (await save(en, nja, rdi.value)) { Object.assign(r, { ja: nja, read: toHira(rdi.value.trim()), mine: true }); _dcDirty.delete(en); toast(`「${en}」を保存しました`); draw(); }
      };
      sv.onclick = go;
      for (const x of [ja, rdi]) x.onkeydown = (e) => { if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); go(); } };
      const del = $("[data-ddel]", tr);
      if (del) del.onclick = async () => {
        if (!(await save(en, "", ""))) return;
        _dcDirty.delete(en);
        toast(`「${en}」の登録を消しました`);
        renderDict();   // the bundled translation (if any) comes back
      };
    });
  };
  const rdb = $("#dc-rd");
  if (rdb) rdb.onclick = async () => {
    rdb.disabled = true;
    let job;
    try { job = await api("/api/i2t/job", { body: { op: "readings" } }); } catch (e) { rdb.disabled = false; $("#dc-rdmsg").textContent = e.message; return; }
    $("#dc-rdmsg").textContent = job.message || "調べています…";
    const done = await i2tJobWait(job.rid, () => !!$("#dc-rd"), (t) => { const m = $("#dc-rdmsg"); if (m) m.textContent = t; });
    if (!done) return;
    if (!done.ok) { $("#dc-rdmsg").textContent = done.error || "調べられませんでした"; rdb.disabled = false; return; }
    const v = done.value || {};
    TAGAC.list = null;
    toast(`読みを登録しました（${v.found} / ${v.asked} 語）` + (v.found < v.asked ? "。ひらがなで返ってこなかった語は登録していません" : ""));
    renderDict();
  };
  $("#dc-q").oninput = debounce(draw, 150);
  $("#dc-mine").onchange = draw;
  $("#dc-add").onclick = async () => {
    const en = $("#dc-en").value.trim(), ja = $("#dc-ja").value.trim();
    if (!en || !ja) { toast("英語と日本語を入れてください"); return; }
    if (await save(en, ja, $("#dc-read").value)) { toast(`「${ja}」＝「${en}」を登録しました`); renderDict(); }
  };
  draw();
}

async function renderLoraSettings() {
  S.project = null;
  $("#crumbs").innerHTML = `<span>›</span><span>LoRA 設定</span>`;
  main.innerHTML = `<div class="muted">読み込み中…</div>`;
  try { S.lib = await api("/api/library"); } catch (e) { main.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
  const focus = S.loraFocus; S.loraFocus = null;
  main.innerHTML = `<div class="panel">
    <div class="row"><h2 style="margin:0">LoRA 設定</h2><div class="spacer"></div>
      <input type="search" id="lc-q" placeholder="名前で絞り込み" style="width:220px">
      <button class="small" id="lc-rescan">再読み込み</button><button class="small" data-opendir="loras">loras フォルダーを開く</button></div>
    <div class="muted small" style="margin-top:6px">LoRA ごとに呼び出し語を登録しておくと、画像生成と素材生成で「呼び出し語をプロンプトに入れる」をオンにした LoRA の呼び出し語が、プロンプトの先頭に自動で入ります（プロンプトにすでに書いてある語は足しません）。
      このツールで作った LoRA と、Civitai のリンクから追加した LoRA は、分かっている呼び出し語が最初から入っています。「候補」は学習時の説明文に多かった語から推測したもので、押すと追加されます。</div>
    <div id="lc-list" style="margin-top:10px"></div></div>`;
  const draw = () => {
    const q = $("#lc-q").value.trim().toLowerCase();
    const all = S.lib.loras.filter((x) => !q || x.name.toLowerCase().includes(q));
    const card = (x) => {
      const c = x.conf;
      return `<div class="note" style="margin-bottom:8px" data-lc="${esc(x.path)}">
        <div class="row"><b>${esc(x.name)}</b><span class="small muted">${esc(x.family_label)}</span><div class="spacer"></div>
          <span class="savestate" data-lcs>${c.saved ? "手入力の設定" : c.source ? esc(c.source) : ""}</span>
          ${c.page ? `<a class="small" href="${esc(c.page)}" target="_blank" rel="noopener">Civitai のページ</a>` : ""}
          ${c.made ? "" : `<button class="small" data-lcv title="ファイルのハッシュ（指紋にあたる値）だけを Civitai に送って、この LoRA のページと呼び出し語を探します">Civitai で調べる</button>`}
          ${c.saved ? `<button class="small" data-lcr title="手入力した設定を消して、ファイルや取得元に書かれた呼び出し語に戻します">初期値に戻す</button>` : ""}</div>
        <div class="small muted">${esc(x.path)}</div>
        <div class="row" style="margin-top:6px"><span class="small">呼び出し語</span>
          <input type="text" data-lcw style="flex:1;min-width:240px" value="${esc(c.triggers.join(", "))}" placeholder="例: abc_chara（複数あるときは , で区切ります）"></div>
        ${c.candidates.length ? `<div class="row small" style="margin-top:4px"><span class="muted">候補</span>${c.candidates.map((w) => `<button class="small" data-lcc="${esc(w)}">＋ ${esc(w)}</button>`).join("")}</div>` : ""}
        <div class="row small" style="margin-top:4px">
          <label><input type="checkbox" data-lca ${c.auto ? "checked" : ""}> この LoRA を選んだとき、「呼び出し語をプロンプトに入れる」を最初からオンにする</label>
          <span class="muted">強さの初期値</span><input type="number" step="0.05" data-lcst style="width:80px" value="${esc(c.strength ?? "")}" placeholder="0.8"></div></div>`; };
    $("#lc-list").innerHTML = all.length ? LORA_GROUPS.map(([made, label]) => {
      const items = all.filter((x) => !!x.conf.made === made);
      return items.length ? `<h3 style="margin:12px 0 6px">${label}（${items.length}）</h3>
        ${made ? "" : `<div class="row small" style="margin-bottom:6px"><span class="muted">「Civitai で調べる」を押すと、ファイルのハッシュ（指紋にあたる値）だけを Civitai に送り、呼び出し語を探します。画像やファイルの中身は送りません。
          Civitai にない LoRA（PixAI など）は、配布ページの呼び出し語をコピーして入れてください。「候補」は説明文に多かった語からの推測です。</span>
          <button class="small" id="lc-vall">呼び出し語が空の LoRA をまとめて調べる</button></div>`}
        ${items.map(card).join("")}` : "";
    }).join("")
      : `<div class="muted small">${S.lib.loras.length ? "一致する LoRA がありません。" : "LoRA が見つかりません。models\loras に置くか、設定画面の「リンク・ファイルから追加」で追加してください。"}</div>`;
    $$("[data-lc]").forEach(bindLoraConfRow);
    const va = $("#lc-vall");
    if (va) va.onclick = async () => {
      const targets = S.lib.loras.filter((x) => !x.conf.made && !x.conf.triggers.length);
      if (!targets.length) return toast("呼び出し語が空の LoRA はありません");
      va.disabled = true;
      let hit = 0, miss = 0;
      for (const [i, x] of targets.entries()) {
        va.textContent = `調べています ${i + 1} / ${targets.length}`;
        try { const r = await civitaiLookup(x.path); if (r.found && r.words.length) hit++; else miss++; }
        catch (e) { toast(e.message, "err"); break; }
      }
      draw();
      toast(`${hit} 件の呼び出し語が見つかりました${miss ? `（見つからない・書かれていないもの ${miss} 件）` : ""}`);
    };
  };
  S.lcDraw = draw;
  $("#lc-q").oninput = debounce(draw, 150);
  $("#lc-rescan").onclick = () => renderLoraSettings();
  bindOpenDir(main);
  draw();
  if (focus) {
    const row = $$("[data-lc]").find((r) => r.dataset.lc === focus);
    if (row) { row.scrollIntoView({ block: "center" }); row.classList.add("flash"); $("[data-lcw]", row).focus(); }
  }
}

async function civitaiLookup(path) {
  const r = await api("/api/loras/civitai", { body: { path } });
  const x = S.lib.loras.find((l) => l.path === path); if (x) x.conf = r.conf;
  _lcDirty.delete(path);   // the words found are shown, not the ones typed before
  return r;
}

/* Rows typed in but not saved yet: the list redraws (filter, Civitai lookup) put these values back in. */
const _lcDirty = new Map();
function bindLoraConfRow(row) {
  const path = row.dataset.lc, st = $("[data-lcs]", row);
  const d = _lcDirty.get(path);
  if (d) { $("[data-lcw]", row).value = d.triggers; $("[data-lca]", row).checked = d.auto; $("[data-lcst]", row).value = d.strength; }
  const read = () => ({ triggers: $("[data-lcw]", row).value, auto: $("[data-lca]", row).checked, strength: $("[data-lcst]", row).value });
  const debounced = saveDebounce(async () => {
    st.className = "savestate"; st.textContent = "保存中…";
    const v = read();
    try {
      const c = await api("/api/loras/settings", { method: "PUT", body: { path, triggers: v.triggers,
        auto: v.auto, strength: v.strength === "" ? null : Number(v.strength) } });
      const x = S.lib.loras.find((l) => l.path === path); if (x) x.conf = c;
      if (JSON.stringify(_lcDirty.get(path)) === JSON.stringify(v)) _lcDirty.delete(path);
      st.textContent = "保存済み";
    } catch (e) { st.className = "savestate err"; st.textContent = "保存できませんでした：" + e.message; }
  }, 400);
  const save = () => { _lcDirty.set(path, read()); debounced(); };
  $("[data-lcw]", row).oninput = save;
  $("[data-lcst]", row).oninput = save;
  $("[data-lca]", row).onchange = save;
  if (d) save();   // saved from this row too, so its status shows when it is done
  $$("[data-lcc]", row).forEach((b) => b.onclick = () => {
    const inp = $("[data-lcw]", row), cur = inp.value.split(",").map((w) => w.trim()).filter(Boolean);
    if (!cur.some((w) => w.toLowerCase() === b.dataset.lcc.toLowerCase())) cur.push(b.dataset.lcc);
    inp.value = cur.join(", "); b.remove(); save();
  });
  const v = $("[data-lcv]", row);
  if (v) v.onclick = async () => {
    v.disabled = true; v.textContent = "調べています…";
    try {
      const r = await civitaiLookup(path);
      if (S.lcDraw) S.lcDraw();
      toast(!r.found ? "Civitai に見つかりませんでした。配布ページの呼び出し語を入れてください。"
        : r.words.length ? `見つかりました：${r.title}` : `${r.title} が見つかりましたが、呼び出し語は書かれていませんでした。`);
    } catch (e) { fail(e); v.disabled = false; v.textContent = "Civitai で調べる"; }
  };
  const r = $("[data-lcr]", row);
  if (r) r.onclick = async () => {
    try {
      const c = await api("/api/loras/settings/reset", { body: { path } });
      const x = S.lib.loras.find((l) => l.path === path); if (x) x.conf = c;
      _lcDirty.delete(path);
      renderLoraSettings(); toast("初期値に戻しました");
    } catch (e) { fail(e); }
  };
}

// ------------------------------------------------------------------ image generation (画像生成)
/* A plain generation page: model, LoRAs, prompt, size.  Used to check settings before material
   generation; a result's settings can be copied into a character project. */
const IG_STATUS = { queued: "順番待ち", running: "生成中", done: "完了", cancelled: "中止", failed: "失敗" };
const IG_SIZES = {
  1024: [["正方形", 1024, 1024], ["縦 2:3", 832, 1216], ["横 3:2", 1216, 832], ["縦 3:4", 896, 1152], ["横 4:3", 1152, 896], ["縦 9:16", 768, 1344], ["横 16:9", 1344, 768]],
  1328: [["正方形", 1328, 1328], ["縦 2:3", 1056, 1584], ["横 3:2", 1584, 1056], ["縦 3:4", 1104, 1472], ["横 4:3", 1472, 1104], ["縦 9:16", 928, 1664], ["横 16:9", 1664, 928]],
};

async function renderImageGen() {
  S.project = null;
  $("#crumbs").innerHTML = `<span>›</span><span>画像生成</span>`;
  if (!$("#ig-history")) main.innerHTML = `<div class="muted">読み込み中…</div>`;
  try {
    [S.ig, S.lib] = await Promise.all([api("/api/imagegen"), api("/api/library")]);
  } catch (e) { main.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
  const f = S.ig.form || {};
  const ds = S.ig.default_sampling || {}, sm = f.sampling || {};
  main.innerHTML = `<div class="panel gen">
    <div class="row"><h2 style="margin:0">画像生成</h2>
      <span class="muted small">モデル・LoRA・プロンプトの組み合わせを試すための、シンプルな生成です。プロンプトは書いたとおりに使います。</span>
      <div class="spacer"></div><span class="savestate" id="ig-save"></span></div>
    <div class="form" style="margin-top:10px">
      <label>保存した設定</label><div id="ig-presets"></div>
      <label>画像から読み込む</label><div id="ig-info"></div>
      <label>モデル</label>
      <div><div class="row"><select id="ig-model" style="max-width:100%"><option value="">選んでください</option>${genModelOptions(S.lib, f.model_id)}</select>
        <button class="small" id="ig-rescan" title="フォルダーを読み直します">再読み込み</button>
        <button class="small" data-opendir="checkpoints">フォルダーを開く</button></div></div>
      <label>LoRA（任意）</label><div><div id="ig-loras"></div><button id="ig-addlora" class="small">＋ LoRA を追加</button> <button id="ig-urllora" class="small" title="Hugging Face や Civitai の LoRA のページの URL から取得して追加します">URL から追加</button></div>
      <label>プロンプト</label><div><div class="row" style="margin-bottom:4px"><span class="small muted" id="ig-jamsg"></span><div class="spacer"></div>
          <button class="small" id="ig-lang" title="プロンプトとネガティブを日本語で表示・入力します。保存と生成には英語を使います">${IGJ.on ? "English" : "日本語"}</button></div>
        <textarea id="ig-pos" rows="8" placeholder="例: 1girl, silver hair, long hair, red eyes, white dress, masterpiece, best quality">${esc(f.positive || "")}</textarea></div>
      <label></label><div class="small" id="ig-trig"></div>
      ${IG_PROMPT_EDIT ? `<label>指示で変える</label><div>
        <div class="row"><input type="text" id="ig-inst" style="flex:1;min-width:240px" placeholder="例: 笑顔にして、背景を夜の街に。眼鏡はなし">
          <button id="ig-rw">プロンプトを変える</button></div>
        <div class="small" id="ig-rwmsg"></div>
        <div class="small" id="ig-chat"></div>
        <div class="row small muted" style="margin-top:2px">パソコンの中の LLM（Ollama）が、指示に合わせてプロンプトとネガティブを書き換えます。文章は外部に送りません。
          <span>使う LLM <select id="ig-llm" class="small"><option value="">読み込み中…</option></select></span></div></div>` : ""}
      <label>ネガティブ</label><div><div class="small warntext" id="ig-negnote" hidden></div><textarea id="ig-neg" rows="5" placeholder="例: lowres, bad anatomy, bad hands, watermark">${esc(f.negative || "")}</textarea></div>
      <label></label><div id="ig-ja" class="small"></div>
      <label>画像サイズ</label><div><div class="row" id="ig-sizes"></div>
        <div class="row" style="margin-top:4px"><span class="small muted">幅</span><input type="number" id="ig-w" step="16" min="512" max="2048" style="width:90px" value="${esc(f.width || S.ig.native)}">
          <span class="small muted">高さ</span><input type="number" id="ig-h" step="16" min="512" max="2048" style="width:90px" value="${esc(f.height || S.ig.native)}">
          <span class="muted small">生成時に 16 の倍数（512〜2048）にそろえます</span></div></div>
      <label>左右で描き分ける（任意）</label><div>
        <div class="row"><textarea id="ig-left" rows="2" style="flex:1;min-width:200px" placeholder="左側だけに描くもの（例: 1girl, red hair, red dress）">${esc((f.regions || {}).left || "")}</textarea>
          <textarea id="ig-right" rows="2" style="flex:1;min-width:200px" placeholder="右側だけに描くもの（例: 1girl, blue hair, black coat）">${esc((f.regions || {}).right || "")}</textarea></div>
        <div class="small muted">2 人を並べるときなどに、左半分と右半分で別の内容を描きます。上のプロンプトは画像全体に効きます（例: 2girls, standing, park）。SDXL 系と Anima で使えます。</div></div>
      <label>元画像（任意）</label><div id="ig-init"></div>
      <label>枚数</label><div class="row"><input type="number" id="ig-count" min="1" max="100" step="1" style="width:90px" value="${esc(f.count || 1)}">
        <span class="small muted">枚（1〜100。seed を 1 ずつ変えて順に生成し、途中で中止できます）</span></div>
    </div>
    <details style="margin-top:8px"><summary class="small">詳細設定</summary>
      <div class="fields" style="margin-top:8px">
        ${[["steps", "ステップ数", "number"], ["cfg", "CFG", "number"], ["sampler", "サンプラー", "text"], ["scheduler", "スケジューラー", "text"], ["clip_skip", "Clip skip", "number"], ["seed", "seed（空欄で毎回ランダム）", "number"]].map(([k, label, t]) => {
          const v = k === "seed" ? (f.seed ?? "") : (sm[k] ?? "");
          return `<div class="field"><div class="lab"><span>${label}</span>${k !== "seed" ? `<span>初期値 <span data-igd="${k}">${esc(ds[k] ?? "")}</span></span>` : ""}</div>
            <input type="${t}" step="any" data-igs="${k}" value="${esc(v)}" placeholder="${esc(k === "seed" ? "" : ds[k] ?? "")}"></div>`; }).join("")}
        ${fastField("igs", sm.fast, ds)}
        <div class="field"><div class="lab"><span>白飛びを抑える</span><span>初期値 使わない</span></div>
          <select data-igs="rescale"><option value="">使わない</option><option value="on" ${sm.rescale === "on" ? "selected" : ""}>使う</option></select>
          <div class="muted small">CFG を高くしたときの白飛びや色の濃くなりすぎを抑えます（NoobAI の v-pred 系などで起きやすい）。</div></div>
        ${igExtraFields(f)}
      </div>
      <datalist id="ig-scheds">${["normal", "karras", "exponential", "sgm_uniform", "simple", "beta", "kl_optimal", "ays", "ays+", "ays_30", "gits"].map((x) => `<option value="${x}">`).join("")}</datalist>
      <div class="muted small" style="margin-top:4px">空欄の項目は、モデルに合わせた初期値で生成します。スケジューラーの ays・gits は、少ないステップでも仕上がりやすい種類です。</div>
    </details>
    <div id="ig-check" style="margin-top:8px"></div>
    <div class="row" style="margin-top:8px"><button class="primary big" id="ig-start">生成</button>
      <span class="muted small">学習や比較画像の生成と同じ順番待ちに入ります。気に入った結果は「この設定で LoRA 素材を作る」を押すと、AI 生成画像からキャラ LoRA を作る設定にコピーできます。</span></div>
  </div>
  <div id="ig-history" style="margin-top:14px"></div>`;
  igDrawLoras(f.loras || []);
  igDrawPresets();
  igDrawSizes();
  igDrawCheck(S.ig.check, S.ig.notes);
  igDrawHistory();
  $$("#ig-pos, #ig-neg, #ig-left, #ig-right, #ig-w, #ig-h, #ig-count, [data-igs], #ig-vae, #ig-te, #ig-attn").forEach((x) => { x.oninput = igSaveSoon; x.onchange = igSaveSoon; });
  $("[data-igs=scheduler]")?.setAttribute("list", "ig-scheds");
  $("#ig-pos").addEventListener("input", debounce(() => igTrig(), 200));
  $("#ig-w").onchange = $("#ig-h").onchange = () => { igDrawSizes(); igSaveSoon(); };
  igDrawInit();
  igDrawInfo();
  igShowExtras();
  $("#ig-model").onchange = async () => {
    const sel = $("#ig-model");
    if (sel.value.startsWith("file:")) return pickFileModel(sel, () => sel.onchange());
    sel.closest(".row").parentElement.querySelector(".pickfile")?.remove();
    igShowExtras();
    const before = S.ig.native;
    igDrawLoras(igRead().loras);
    await igSave();
    if (S.ig.native !== before) igSetSize(S.ig.native, S.ig.native);   // e.g. SDXL 1024 -> Qwen-Image 1328
  };
  $("#ig-addlora").onclick = () => { const l = igRead().loras; l.push({ path: "", strength: 0.8 }); igDrawLoras(l); };
  $("#ig-urllora").onclick = (e) => loraUrlBox(e.target, (path) => { const l = igRead().loras.filter((x) => x.path); l.push({ path, strength: 0.8 }); igDrawLoras(l); igSaveSoon(); });
  igNegNote();
  $$("[data-igs=cfg]").forEach((x) => x.addEventListener("input", igNegNote));
  if (IG_PROMPT_EDIT) {
    $("#ig-rw").onclick = igRewrite;
    S.igChat = S.igChat || [];
    igDrawChat();
    $("#ig-inst").onkeydown = (e) => { if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); igRewrite(); } };
    igLoadLlm();
  }
  $("#ig-start").onclick = igStart;
  $("#ig-lang").onclick = () => igSetJa(!IGJ.on);
  $$("#ig-pos, #ig-neg").forEach((x) => { x.addEventListener("input", igJaNoteSoon); attachTagComplete(x, { escape: true, sep: true, ja: () => IGJ.on, moss: () => modelFamily($("#ig-model").value) === "moss_image" }); });
  if (IGJ.on) igJaShow();
  $("#ig-rescan").onclick = async () => {
    try { await igSave(); await renderImageGen(); toast("モデルの一覧を読み直しました"); } catch (e) { fail(e); }
  };
  bindOpenDir(main);
}

/* 保存した設定: the model, LoRAs, prompt and negative saved under a name.  Choosing one in the list puts it
   in the form; 「保存」 asks for a name in place (no dialog), and the same name replaces that entry. */
function igDrawPresets(msg = "") {
  const el = $("#ig-presets");
  if (!el) return;
  const ps = S.ig.presets || [], cur = ps.find((p) => p.id === S.igPreset);
  el.innerHTML = `<div class="row">
      <select id="ig-psel" style="max-width:100%"><option value="">${ps.length ? "選ぶとフォームに入ります" : "（まだありません）"}</option>
        ${ps.map((p) => `<option value="${esc(p.id)}" ${p.id === S.igPreset ? "selected" : ""}>${esc(p.name)}</option>`).join("")}</select>
      <button class="small" id="ig-psave" title="今のモデル・LoRA・プロンプト・ネガティブを名前を付けて保存します">保存</button>
      ${cur ? `<button class="small" id="ig-pdel" title="一覧からこの設定を消します（生成した画像は消えません）">削除</button>` : ""}
      <a class="btnlink" href="#/presets"><button class="small" title="保存した設定をサムネイル付きで一覧にします">一覧を見る</button></a>
      <span class="small muted" id="ig-pmsg">${esc(msg)}</span></div>
    <div id="ig-pname" hidden><div class="row" style="margin-top:4px">
      <input type="text" id="ig-pn" maxlength="80" style="width:260px" placeholder="設定の名前" value="${esc(cur ? cur.name : "")}">
      <button class="primary small" id="ig-pok">決定</button><button class="small" id="ig-pcancel">やめる</button>
      <span class="small warntext" id="ig-pdup" hidden>同じ名前の設定を上書きします</span></div></div>`;
  const box = $("#ig-pname"), input = $("#ig-pn");
  const dup = () => { $("#ig-pdup").hidden = !ps.some((p) => p.name === input.value.trim()); };
  $("#ig-psel").onchange = (e) => igApplyPreset(e.target.value);
  $("#ig-psave").onclick = () => { box.hidden = false; dup(); input.focus(); input.select(); };
  $("#ig-pcancel").onclick = () => { box.hidden = true; };
  input.oninput = dup;
  input.onkeydown = (e) => {
    if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); igSavePreset(); }
    else if (e.key === "Escape") box.hidden = true;
  };
  $("#ig-pok").onclick = igSavePreset;
  const del = $("#ig-pdel");
  if (del) del.onclick = () => igDeletePreset(cur.id);
}

async function igSavePreset() {
  const name = $("#ig-pn").value.trim();
  if (!name) { $("#ig-pn").focus(); return; }
  if (igJaBlocked()) return;
  const f = igRead(), b = $("#ig-pok");
  b.disabled = true;
  try {
    const r = await api("/api/imagegen/presets", { body: { name, model_id: f.model_id, loras: f.loras, positive: f.positive, negative: f.negative,
      thumb: { width: f.width, height: f.height, sampling: f.sampling, seed: f.seed } } });
    S.ig.presets = r.presets; S.igPreset = r.preset.id;
    const t = r.preset.thumb || {};
    igDrawPresets(t.status === "failed" ? `「${name}」を保存しました（${t.message}）`
      : `「${name}」を保存しました。サムネイル用に 1 枚生成します（下の一覧に順番待ちで入ります）`);
    igReload();
  } catch (e) { b.disabled = false; fail(e); }
}

/* Put a preset into the generation form: only the saved parts change; size, count, seed and the detailed
   settings stay as they are. */
function presetForm(base, p) {
  return { ...base, model_id: p.model_id, loras: p.loras, positive: p.positive, negative: p.negative };
}

async function igApplyPreset(pid) {
  const p = (S.ig.presets || []).find((x) => x.id === pid);
  S.igPreset = p ? p.id : null;
  if (!p) { igDrawPresets(); return; }
  try {
    await api("/api/imagegen/form", { method: "PUT", body: presetForm(igRead(), p) });
  } catch (e) { return fail(e); }
  const before = S.ig.native;
  await renderImageGen();
  if (S.ig.native !== before) igSetSize(S.ig.native, S.ig.native);   // e.g. SDXL 1024 -> Qwen-Image 1328
  toast(`「${p.name}」をフォームに入れました`);
}

async function igDeletePreset(pid) {
  let r;
  try { r = await api(`/api/imagegen/presets/${encodeURIComponent(pid)}`, { method: "DELETE" }); } catch (e) { return fail(e); }
  S.ig.presets = r.presets; S.igPreset = null;
  igDrawPresets();
  const gone = r.deleted;
  toast(`「${gone.name}」を削除しました`, "", { label: "元に戻す", fn: async () => {
    try {
      const back = await api("/api/imagegen/presets", { body: { ...gone, restore: true } });
      S.ig.presets = back.presets; S.igPreset = back.preset.id;
      igDrawPresets(`「${gone.name}」を元に戻しました`);
    } catch (e) { fail(e); }
  } });
}

// ------------------------------------------------------------------ saved generation settings (保存した設定)
/* The presets of the image generation page as a list with their thumbnails, sorted by the chosen order
   (remembered in this browser only).  「使う」 puts one into the generation form and opens that page. */
const PL_SORTS = {
  created_desc: ["作成日（新しい順）", (a, b) => (b.created || b.saved || 0) - (a.created || a.saved || 0)],
  created_asc: ["作成日（古い順）", (a, b) => (a.created || a.saved || 0) - (b.created || b.saved || 0)],
  saved_desc: ["更新日（新しい順）", (a, b) => (b.saved || 0) - (a.saved || 0)],
  name: ["名前順", (a, b) => a.name.localeCompare(b.name, "ja", { numeric: true })],
  model: ["モデル順", (a, b) => (a.model_name || "").localeCompare(b.model_name || "", "ja", { numeric: true }) || a.name.localeCompare(b.name, "ja")],
};
const PL_THUMB_STATE = { queued: "サムネイルの順番待ち", running: "サムネイルを生成中…" };

function plSort() {
  try { const v = localStorage.getItem("presetSort"); if (v && PL_SORTS[v]) return v; } catch (e) { /* storage unavailable */ }
  return "created_desc";
}

async function renderPresetList() {
  S.project = null;
  $("#crumbs").innerHTML = `<span>›</span><a href="#/gen">画像生成</a><span>›</span><span>保存した設定</span>`;
  if (!$("#pl-list")) main.innerHTML = `<div class="muted">読み込み中…</div>`;
  try { S.pl = (await api("/api/imagegen/presets")).presets; } catch (e) { main.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
  if (!$("#pl-list")) {
    const cur = plSort();
    main.innerHTML = `<div class="panel">
      <div class="row"><h2 style="margin:0">保存した設定</h2>
        <span class="muted small">画像生成ページで保存したモデル・LoRA・プロンプトの組み合わせです。サムネイルは保存したときに 1 枚生成したものです。</span>
        <div class="spacer"></div>
        <label class="small">並べ替え <select id="pl-sort">${Object.entries(PL_SORTS).map(([k, [label]]) =>
          `<option value="${k}" ${k === cur ? "selected" : ""}>${label}</option>`).join("")}</select></label></div>
      <div id="pl-list" class="plgrid" style="margin-top:12px"></div></div>`;
    $("#pl-sort").onchange = (e) => {
      try { localStorage.setItem("presetSort", e.target.value); } catch (err) { /* storage unavailable */ }
      plDraw();
    };
  }
  plDraw();
}

async function plReload() {
  if (S.route[0] !== "presets") return;
  try { S.pl = (await api("/api/imagegen/presets")).presets; } catch (e) { return; }
  plDraw();
}
const plReloadSoon = debounce(plReload, 300);

function plDraw() {
  const el = $("#pl-list");
  if (!el) return;
  const items = [...(S.pl || [])].sort(PL_SORTS[$("#pl-sort").value][1]);
  if (!items.length) {
    el.innerHTML = `<div class="muted small">まだありません。<a href="#/gen">画像生成</a>のページで「保存した設定」の「保存」を押すと、ここに並びます。</div>`;
    return;
  }
  const date = (t) => t ? new Date(t * 1000).toLocaleString("ja-JP", { dateStyle: "short", timeStyle: "short" }) : "";
  el.innerHTML = items.map((p) => {
    const t = p.thumb || {};
    const busy = t.status === "queued" || t.status === "running";
    const pic = p.thumb_file ? `<img loading="lazy" src="/api/imagegen/preset_thumb/${encodeURIComponent(p.thumb_file)}" alt="" data-zoom>` : "";
    const lo = p.loras.length ? p.loras.map((l) => `${l.path.split(/[\\/]/).pop()}（${l.strength}）`).join("、") : "なし";
    return `<div class="plcard" data-pl="${esc(p.id)}">
      <div class="plthumb">${pic}${busy ? `<div class="plbadge"><span class="spin"></span> ${PL_THUMB_STATE[t.status]}</div>`
        : !p.thumb_file ? `<div class="muted small" style="padding:8px">${esc(t.message || "サムネイルはありません")}</div>` : ""}</div>
      <div class="plbody">
        <div class="row"><b>${esc(p.name)}</b></div>
        <div class="small muted">作成 ${date(p.created || p.saved)}${p.saved && p.created && p.saved - p.created > 1 ? ` ／ 更新 ${date(p.saved)}` : ""}</div>
        <div class="small">モデル: ${esc(p.model_name || "（不明）")}${p.model_missing ? ` <span class="err">（見つかりません）</span>` : ""}</div>
        <div class="small">LoRA: ${esc(lo)}</div>
        <div class="small plprompt" title="${esc(p.positive || "")}">${esc(p.positive || "")}</div>
        ${t.status === "failed" && p.thumb_file ? `<div class="small warntext">${esc(t.message || "")}</div>` : ""}
        <div class="row" style="margin-top:6px"><button class="small primary" data-pluse>使う</button>
          <button class="small" data-plthumb ${busy ? "disabled" : ""} title="この設定でもう 1 枚生成し、サムネイルを差し替えます（seed は毎回変わります）">サムネイルを作り直す</button>
          <button class="small" data-pldel>削除</button></div></div></div>`;
  }).join("");
  $$("[data-pl]", el).forEach((card) => {
    const p = items.find((x) => x.id === card.dataset.pl);
    const img = $("img[data-zoom]", card);
    if (img) img.onclick = () => openLightbox([{ src: img.src, cap: p.name }], 0);
    $("[data-pluse]", card).onclick = async () => {
      try {
        const st = await api("/api/imagegen");
        await api("/api/imagegen/form", { method: "PUT", body: presetForm(st.form || {}, p) });
      } catch (e) { return fail(e); }
      S.igPreset = p.id;
      location.hash = "#/gen";
      toast(`「${p.name}」をフォームに入れました`);
    };
    $("[data-plthumb]", card).onclick = async (ev) => {
      ev.target.disabled = true;
      try {
        const st = await api("/api/imagegen");
        const f = st.form || {};
        const r = await api(`/api/imagegen/presets/${encodeURIComponent(p.id)}/thumb`, { body: { width: f.width, height: f.height, sampling: f.sampling } });
        S.pl = r.presets; plDraw();
        if (r.thumb.status === "failed") toast(r.thumb.message, "err");
      } catch (e) { ev.target.disabled = false; fail(e); }
    };
    $("[data-pldel]", card).onclick = async () => {
      let r;
      try { r = await api(`/api/imagegen/presets/${encodeURIComponent(p.id)}`, { method: "DELETE" }); } catch (e) { return fail(e); }
      S.pl = r.presets; plDraw();
      if (S.igPreset === p.id) S.igPreset = null;
      const gone = r.deleted;
      toast(`「${gone.name}」を削除しました`, "", { label: "元に戻す", fn: async () => {
        try { S.pl = (await api("/api/imagegen/presets", { body: { ...gone, restore: true } })).presets; plDraw(); } catch (e) { fail(e); }
      } });
    };
  });
}

// ------------------------------------------------------------------ 日本語 view of the prompt (画像生成)
/* The 日本語 button of 画像からプロンプト, for the generation form: each chip of the prompt and negative is
   shown with i2t's dictionary (bundled + dict/*.csv + translations.json) and can be edited in Japanese.
   Only the view is Japanese: whenever the form is read (auto-save, 生成, 保存) the text is turned back into
   English: a chip still showing its label gives back the exact English it came from, a Japanese chip typed
   by the user is looked up in the dictionary the other way, and one that is not in the dictionary has to be
   registered (English for it) before generating.  English words without a Japanese entry can be given one,
   or translated by Ollama as on the i2t page; both go to translations.json, shared with the i2t page. */
const IGJ = { on: false, table: null, rev: null, shown: { pos: new Map(), neg: new Map() }, missing: [], busy: false };
try { IGJ.on = localStorage.getItem("igJa") === "1"; } catch (e) { /* storage unavailable */ }
const JA_RE = /[\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f]/;
const JA_WEIGHT_RE = /^(.*?)\s*×\s*(-?\d+(?:\.\d+)?)$/;   // chip_label_ja writes a weight as 「語 ×1.3」

function igTrig() { drawTrigPreview("#ig-loras", "#ig-pos", "#ig-trig", igToEn("pos").text); }

/* i2t's split_prompt_chips: lines of chips, split at commas outside brackets; empty lines are kept. */
function splitChips(text) {
  const lines = String(text || "").split("\n").map((line) => {
    const chunks = [];
    let depth = 0, start = 0;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i], esc_ = i > 0 && line[i - 1] === "\\";
      if ("([{".includes(ch) && !esc_) depth++;
      else if (")]}".includes(ch) && !esc_) depth = Math.max(0, depth - 1);
      else if (ch === "," && depth === 0) { chunks.push(line.slice(start, i)); start = i + 1; }
    }
    chunks.push(line.slice(start));
    return chunks.map((c) => c.trim()).filter((c) => c);
  });
  while (lines.length && !lines[lines.length - 1].length) lines.pop();
  return lines;
}
const joinChips = (lines) => lines.map((l) => l.join(", ")).join("\n");
const i2tCall = (fn, ...args) => api("/api/i2t/call", { body: { fn, args } }).then((r) => r.value);

/* Tag completion in the prompt and caption boxes, from i2t's chip dictionary (English tag -> Japanese, plus the
   words the user registered).  Typing English or Japanese lists matching tags under the box; ↑↓ choose, Enter / Tab
   put the tag in place of the word being typed, Esc closes.  Without the i2t worker the boxes work as before. */
const TAGAC = { list: null, loading: null, failedAt: 0, box: null, ta: null, opts: null, items: [], sel: 0, start: 0, end: 0 };

// katakana as hiragana: kana typed before the IME conversion (ろんぐへあ) finds ロングヘア
const KANJI_RE = /[\u3400-\u9fff々]/;
const toHira = (t) => t.replace(/[\u30a1-\u30f6]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0x60));
// voiced kana as unvoiced (が -> か, ぱ -> は): はいいろかみ (the reading made) still finds はいいろがみ typed
const seion = (t) => t.normalize("NFD").replace(/[\u3099\u309a]/g, "").normalize("NFC");
const TAG_SEP = /[,\n、，]/;   // a Japanese 「、」 also ends a tag (it becomes ", " when a tag is chosen)

function tagList() {
  if (TAGAC.list || TAGAC.loading || Date.now() - TAGAC.failedAt < 60000) return TAGAC.loading;
  // readings of the bundled translations (made by app/scripts/make_tag_readings.py): えがお finds 笑顔.  Words the
  // user registered use the reading typed with them (v.readings); others they translated have none.
  const readings = fetch("/static/tag_readings.json").then((r) => r.ok ? r.json() : {}).catch(() => ({}));
  TAGAC.loading = Promise.all([i2tCall("ja_table"), readings]).then(([v, rd]) => {
    TAGAC.list = Object.entries({ ...v.table, ...v.user }).filter(([en, ja]) => en && ja).map(([en, ja]) => {
      const kana = toHira(ja), read = (v.readings || {})[en] || (!(en in v.user) && rd[en]) || "";
      return [en, ja, en.toLowerCase(), kana, read, seion(read || kana)];
    });
  }).catch(() => { TAGAC.failedAt = Date.now(); }).finally(() => { TAGAC.loading = null; });
  return TAGAC.loading;
}

/* The 314 styles of hide1's MOSS-Image catalog (name + the prompt the catalog used), offered while a MOSS-Image model
   is chosen; choosing one puts in the catalog's sentence.  Empty until the list was fetched (i2t page / materials). */
const MOSSST = { list: null, at: 0 };
function mossStyles() {
  if ((MOSSST.list && MOSSST.list.length) || Date.now() - MOSSST.at < 60000) return;
  MOSSST.at = Date.now();
  api("/api/moss_catalog/styles").then((r) => { MOSSST.list = (r.styles || []).map((x) => [x.name, x.prompt, x.name.toLowerCase()]); }).catch(() => {});
}

/* opts: escape (prompts: brackets as \( \)), sep (prompts: ", " after the tag), ja (() => true while the box shows Japanese),
   moss (() => true while a MOSS-Image model is chosen: the catalog's styles are offered too) */
function attachTagComplete(ta, opts = {}) {
  if (!ta || ta.dataset.tagac) return;
  ta.dataset.tagac = "1";
  ta.addEventListener("focus", tagList);
  ta.addEventListener("input", () => tagacUpdate(ta, opts));
  ta.addEventListener("keydown", tagacKey);
  ta.addEventListener("blur", () => setTimeout(() => { if (TAGAC.ta === ta) tagacHide(); }, 150));
}

function tagacHide() {
  if (TAGAC.box) TAGAC.box.hidden = true;
  TAGAC.ta = null;
}

function tagacUpdate(ta, opts) {
  const pos = ta.selectionStart;
  if (pos !== ta.selectionEnd) return tagacHide();
  const v = ta.value, before = v.slice(0, pos);
  const after = v.slice(pos), nx = after.search(TAG_SEP);
  if ((nx < 0 ? after : after.slice(0, nx)).trim()) return tagacHide();   // the caret is inside a word
  const s0 = Math.max(...[",", "\n", "、", "，"].map((c) => before.lastIndexOf(c))) + 1;
  const word = before.slice(s0), lead = word.match(/^\s*[(\[{]*/)[0];   // weight brackets stay as typed
  const q = word.slice(lead.length).replace(/\\/g, "").replace(/_/g, " ").toLowerCase();
  if (q.length < (JA_RE.test(q) ? 1 : 2)) return tagacHide();
  const moss = opts.moss && opts.moss() && !(opts.ja && opts.ja());
  if (moss) mossStyles();
  if (!TAGAC.list) {
    const p = tagList();
    if (p) p.then(() => { if (TAGAC.list && document.activeElement === ta) tagacUpdate(ta, opts); });
    if (!moss || !MOSSST.list) return;
  }
  const hits = [], qk = toHira(q), qs = seion(qk);   // Japanese is matched against the translations, the English tag goes in
  if (moss && MOSSST.list && !JA_RE.test(q)) {
    for (const [name, prompt, low] of MOSSST.list) {
      const s = low.startsWith(q) ? -0.5 : low.includes(" " + q) ? 0.5 : low.includes(q) ? 1.5 : null;
      if (s !== null) hits.push([s, prompt, "画風（MOSS-Image の一覧の文）", name]);
    }
  }
  for (const [en, ja, low, kana, read, soft] of TAGAC.list || []) {
    const s = low === q || kana === qk || read === qk ? -1
      : low.startsWith(q) || kana.startsWith(qk) || (read && read.startsWith(qk)) ? 0 : low.includes(" " + q) ? 1
      : low.includes(q) || kana.includes(qk) || (read && read.includes(qk)) ? 2 : soft.startsWith(qs) ? 2.5 : -1.5;
    if (s > -1.5) hits.push([s, en, ja]);
  }
  if (!hits.length) return tagacHide();
  hits.sort((a, b) => a[0] - b[0] || a[1].length - b[1].length || a[1].localeCompare(b[1]));
  Object.assign(TAGAC, { ta, opts, items: hits.slice(0, 8), sel: 0, start: s0 + lead.length, end: pos, bracket: /[(\[{]/.test(lead) });
  tagacDraw();
}

function tagacDraw() {
  if (!TAGAC.box) {
    TAGAC.box = document.createElement("div");
    TAGAC.box.className = "tagac";
    TAGAC.box.onmousedown = (e) => {   // mousedown: before the box loses focus
      const it = e.target.closest("[data-i]");
      e.preventDefault();
      if (it) tagacAccept(Number(it.dataset.i));
    };
    document.body.appendChild(TAGAC.box);
  }
  const b = TAGAC.box, r = TAGAC.ta.getBoundingClientRect();
  b.innerHTML = TAGAC.items.map(([, en, ja, name], i) => `<div class="tagac-i ${i === TAGAC.sel ? "on" : ""}" data-i="${i}" ${name ? `title="${esc(en)}"` : ""}><span>${esc(name || en)}</span><span class="muted">${esc(ja)}</span></div>`).join("");
  b.hidden = false;
  b.style.left = `${Math.max(4, r.left)}px`;
  b.style.width = `${Math.min(Math.max(r.width, 260), 460)}px`;
  const below = window.innerHeight - r.bottom;
  b.style.top = below > b.offsetHeight + 8 ? `${r.bottom + 2}px` : `${Math.max(4, r.top - b.offsetHeight - 2)}px`;
}

function tagacKey(e) {
  if (!TAGAC.ta || TAGAC.ta !== e.target || !TAGAC.box || TAGAC.box.hidden || e.isComposing) return;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    TAGAC.sel = (TAGAC.sel + (e.key === "ArrowDown" ? 1 : -1) + TAGAC.items.length) % TAGAC.items.length;
    tagacDraw();
  } else if (e.key === "Enter" || e.key === "Tab") {
    tagacAccept(TAGAC.sel);
  } else if (e.key === "Escape") {
    tagacHide();
  } else return;
  e.preventDefault();
  e.stopPropagation();
}

function tagacAccept(i) {
  const ta = TAGAC.ta, o = TAGAC.opts || {}, it = TAGAC.items[i];
  if (!ta || !it) return;
  const style = !!it[3];   // a catalog style: its sentence goes in as written, followed by a space
  const ja = !style && o.ja && o.ja();
  let w = ja ? it[2] : it[1];
  if (o.escape && !ja && !style) w = w.replace(/\\?\(/g, "\\(").replace(/\\?\)/g, "\\)");
  const v = ta.value;
  const head = v.slice(0, TAGAC.start).replace(/[、，]\s*$/, ", "), rest = v.slice(TAGAC.end).replace(/^\s*[、，]\s*/, ", ");
  const sep = style ? (/^\s/.test(rest) ? "" : " ")
    : TAGAC.bracket || /^\s*,/.test(rest) || !(o.sep || rest.trim()) ? "" : ", ";   // "(tag" goes on to ":1.2)"
  ta.value = head + w + sep + rest;
  const caret = head.length + w.length + sep.length;
  ta.setSelectionRange(caret, caret);
  tagacHide();
  ta.dispatchEvent(new Event("input", { bubbles: true }));   // autosave and previews, as if typed
}
window.addEventListener("scroll", () => { if (TAGAC.ta) tagacHide(); }, true);

async function igJaLoad(force) {
  if (IGJ.table && !force) return;
  const v = await i2tCall("ja_table");
  const rev = {};
  for (const [en, ja] of Object.entries(v.table)) if (!(ja in rev)) rev[ja] = en;
  for (const [en, ja] of Object.entries(v.user)) rev[ja] = en;   // registered / translated words win
  IGJ.table = v.table; IGJ.rev = rev;
}

/* The English of one box (the box itself when the view is English).  unknown: Japanese chips the
   dictionary does not know (left as typed). */
function igToEn(box) {
  const el = $(`#ig-${box}`);
  if (!el) return { text: "", unknown: [] };
  if (!IGJ.on || !IGJ.rev) return { text: el.value, unknown: [] };
  const unknown = [];
  const lines = splitChips(el.value).map((line) => line.map((c) => {
    const back = IGJ.shown[box].get(c);
    if (back !== undefined && !JA_RE.test(back)) return back;
    if (!JA_RE.test(c)) return c;
    const m = c.match(JA_WEIGHT_RE), core = (m ? m[1] : c).trim(), en = IGJ.rev[core];
    if (!en) { unknown.push(c); return c; }
    return m ? `(${en}:${m[2]})` : en;
  }));
  return { text: joinChips(lines), unknown };
}

/* Show both boxes (now English) in Japanese. */
async function igJaShow() {
  const boxes = ["pos", "neg"].map((b) => $(`#ig-${b}`));
  if (boxes.some((x) => !x)) return;
  IGJ.busy = true;
  boxes.forEach((x) => { x.readOnly = true; });
  $("#ig-jamsg").textContent = "日本語にしています…";
  try {
    await igJaLoad();
    const all = [], done = [];
    for (const box of ["pos", "neg"]) {
      const el = $(`#ig-${box}`), lines = splitChips(el.value), flat = lines.flat();
      const labels = flat.length ? await i2tCall("chips_ja", flat) : [];
      const map = new Map();
      let i = 0;
      const shown = lines.map((line) => line.map((c) => {
        let lab = labels[i++] ?? c;
        if (map.has(lab) && map.get(lab) !== c) lab = c;   // two English chips with one Japanese word: keep this one in English
        map.set(lab, c);
        return lab;
      }));
      done.push([box, el, map, joinChips(shown)]);
      all.push(...flat.filter((c) => !JA_RE.test(c)));
    }
    IGJ.missing = all.length ? await i2tCall("missing_ja", all, []) : [];
    for (const [box, el, map, text] of done) { IGJ.shown[box] = map; el.value = text; }   // both boxes at once, only when all worked
    $("#ig-jamsg").textContent = "日本語で表示中（保存と生成には英語を使います）";
  } catch (e) {
    IGJ.on = false;
    $("#ig-jamsg").textContent = "";
    $("#ig-lang").textContent = "日本語";
    fail(new Error("日本語にできませんでした: " + e.message));
  }
  boxes.forEach((x) => { x.readOnly = false; });
  IGJ.busy = false;
  igJaNote();
}

async function igSetJa(on) {
  if (IGJ.busy || on === IGJ.on) return;
  if (!on) {
    const u = [...igToEn("pos").unknown, ...igToEn("neg").unknown];
    for (const box of ["pos", "neg"]) $(`#ig-${box}`).value = igToEn(box).text;
    IGJ.on = false;
    IGJ.shown = { pos: new Map(), neg: new Map() };
    $("#ig-jamsg").textContent = u.length ? `辞書にない日本語（${u.join("、")}）はそのまま残しました` : "";
  } else {
    IGJ.on = true;
    await igJaShow();
  }
  try { localStorage.setItem("igJa", IGJ.on ? "1" : "0"); } catch (e) { /* storage unavailable */ }
  $("#ig-lang").textContent = IGJ.on ? "English" : "日本語";
  igJaNote();
}

/* Under the negative: Japanese that must be registered before generating, and English words with no
   Japanese yet (optional). */
function igJaNote() {
  const el = $("#ig-ja");
  if (!el) return;
  if (!IGJ.on || IGJ.busy) { el.innerHTML = ""; return; }
  const unknown = [...new Set([...igToEn("pos").unknown, ...igToEn("neg").unknown].map((c) => { const m = c.match(JA_WEIGHT_RE); return (m ? m[1] : c).trim(); }))];
  const en = (igToEn("pos").text + "\n" + igToEn("neg").text).toLowerCase();
  const missing = IGJ.missing.filter((w) => en.includes(w.toLowerCase()));
  el.innerHTML = (unknown.length ? `<div class="note" style="margin-bottom:6px"><div class="warntext">辞書にない日本語があります。英語を登録すると、その英語で生成します（登録するまで生成できません）。</div>
      <div class="row" style="margin:4px 0"><button class="small" id="ig-jaen" title="Ollama で英語のタグに訳し、読みも調べて辞書に登録します">Ollama で英語に訳す</button><span class="muted" id="ig-jaenmsg"></span></div>
      ${unknown.map((w) => `<div class="row" style="margin-top:4px"><span style="min-width:120px">${esc(w)}</span> →
        <input type="text" data-jen="${esc(w)}" placeholder="英語（例: cat ears）" style="width:220px">
        ${KANJI_RE.test(w) ? `<input type="text" data-jread placeholder="読み（ひらがな）" style="width:150px">` : ""}<button class="small primary" data-jreg>登録</button></div>`).join("")}</div>` : "")
    + (missing.length ? `<details><summary class="muted">訳のない英語 ${missing.length} 語（英語のまま表示しています）</summary>
      <div class="row" style="margin:4px 0"><button class="small" id="ig-jatr" title="i2t の設定の「訳のモデル」（Ollama）で訳し、辞書に登録します">Ollama で訳す</button><span class="muted" id="ig-jatrmsg"></span></div>
      ${missing.map((w) => `<div class="row" style="margin-top:4px"><span style="min-width:120px">${esc(w)}</span> →
        <input type="text" data-jja="${esc(w)}" placeholder="日本語" style="width:220px">
        <input type="text" data-jread placeholder="読み（ひらがな、漢字のとき）" style="width:190px"><button class="small" data-jreg>登録</button></div>`).join("")}</details>` : "");
  $$("[data-jreg]", el).forEach((b) => {
    const row = b.parentElement, input = $("[data-jen], [data-jja]", row), read = $("[data-jread]", row);
    const go = () => igJaRegister(input, read);
    b.onclick = go;
    for (const x of [input, read]) if (x) x.onkeydown = (e) => { if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); go(); } };
  });
  const tr = $("#ig-jatr");
  if (tr) tr.onclick = () => igJaTranslate(missing);
  const te = $("#ig-jaen");
  if (te) te.onclick = () => igJaToEnglish(unknown);
}
const igJaNoteSoon = debounce(igJaNote, 400);

async function igJaRegister(input, readInput) {
  const v = input.value.trim();
  if (!v) { input.focus(); return; }
  const [en, ja] = input.dataset.jen ? [v, input.dataset.jen] : [input.dataset.jja, v];
  const reading = toHira(readInput ? readInput.value.trim() : "");
  if (KANJI_RE.test(ja) && !reading) {   // the reading lets the tag completion find it from kana typed before conversion
    toast(`「${ja}」の読みを、ひらがなで入れてください`);
    if (readInput) readInput.focus();
    return;
  }
  try {
    await i2tCall("register_ja", en, ja, reading);
    TAGAC.list = null;   // the completion reads the dictionary again
    await igJaRefresh();
    igSaveSoon(); igTrig();
    toast(`「${ja}」＝「${en}」を登録しました`);
  } catch (e) { fail(e); }
}

/* After the dictionary changed: back to English with the old dictionary, then shown again with the new one. */
async function igJaRefresh() {
  for (const box of ["pos", "neg"]) $(`#ig-${box}`).value = igToEn(box).text;
  await igJaLoad(true);
  await igJaShow();
}

async function igJaTranslate(words) {
  const msg = $("#ig-jatrmsg"), b = $("#ig-jatr");
  b.disabled = true;
  let job;
  try { job = await api("/api/i2t/job", { body: { op: "translate", words } }); } catch (e) { b.disabled = false; msg.textContent = e.message; return; }
  msg.textContent = job.message || "訳しています…";
  for (let errs = 0; ;) {
    await new Promise((r) => setTimeout(r, 1000));
    if (!$("#ig-ja")) return;   // left the page; the words are in the dictionary next time
    let st;
    try { st = await api("/api/i2t/job"); errs = 0; } catch (e) {
      if (++errs < 10) continue;
      const m = $("#ig-jatrmsg"), bb = $("#ig-jatr");
      if (m) m.textContent = "進み具合を確かめられませんでした: " + e.message;
      if (bb) bb.disabled = false;
      return;
    }
    if (st.job && st.job.rid === job.rid) { const m = $("#ig-jatrmsg"); if (m) m.textContent = st.job.message || ""; continue; }
    const done = st.last_done && st.last_done.rid === job.rid ? st.last_done : null;
    if (!$("#ig-ja")) return;   // left the page; the words are in the dictionary next time
    if (done && done.ok) {
      TAGAC.list = null;
      await igJaRefresh();
      const v = done.value || {};
      toast(`日本語に訳しました（${Object.keys(v.translated || {}).length} 語）` + (v.readings_error ? `。読みは調べられませんでした（${v.readings_error}）` : ""));
    }
    else { const m = $("#ig-jatrmsg"); if (m) m.textContent = done ? (done.error || "訳せませんでした") : "訳せませんでした"; const bb = $("#ig-jatr"); if (bb) bb.disabled = false; }
    return;
  }
}

/* Wait for an i2t job started by this page.  onMsg(text) shows its progress; returns the finished job's record,
   or null when the page was left (the job goes on; its results are in the dictionary next time). */
async function i2tJobWait(rid, alive, onMsg) {
  for (let errs = 0; ;) {
    await new Promise((r) => setTimeout(r, 1000));
    if (!alive()) return null;
    let st;
    try { st = await api("/api/i2t/job"); errs = 0; } catch (e) {
      if (++errs < 10) continue;
      return { ok: false, error: "進み具合を確かめられませんでした: " + e.message };
    }
    if (st.job && st.job.rid === rid) { onMsg(st.job.message || ""); continue; }
    if (!alive()) return null;
    return st.last_done && st.last_done.rid === rid ? st.last_done : { ok: false, error: "終わり方を確かめられませんでした" };
  }
}

async function igJaToEnglish(words) {
  const msg = $("#ig-jaenmsg"), b = $("#ig-jaen");
  b.disabled = true;
  let job;
  try { job = await api("/api/i2t/job", { body: { op: "translate_en", words } }); } catch (e) { b.disabled = false; msg.textContent = e.message; return; }
  msg.textContent = job.message || "訳しています…";
  const done = await i2tJobWait(job.rid, () => !!$("#ig-ja"), (t) => { const m = $("#ig-jaenmsg"); if (m) m.textContent = t; });
  if (!done) return;
  if (!done.ok) { const m = $("#ig-jaenmsg"), bb = $("#ig-jaen"); if (m) m.textContent = done.error || "訳せませんでした"; if (bb) bb.disabled = false; return; }
  const v = done.value || {}, known = v.known || {};
  // a word whose English the dictionary already has under another Japanese: write the dictionary's Japanese instead
  if (Object.keys(known).length) {
    for (const box of ["pos", "neg"]) {
      const el = $(`#ig-${box}`);
      el.value = joinChips(splitChips(el.value).map((line) => line.map((c) => {
        const m = c.match(JA_WEIGHT_RE), core = (m ? m[1] : c).trim();
        return known[core] ? (m ? `${known[core][1]} ×${m[2]}` : known[core][1]) : c;
      })));
    }
  }
  TAGAC.list = null;
  await igJaRefresh();
  igSaveSoon(); igTrig();
  const n = Object.keys(v.translated || {}).length, k = Object.keys(known).length;
  toast([n ? `英語に訳して登録しました（${n} 語）` : "", k ? `辞書にある訳に置き換えました（${Object.entries(known).map(([ja, [, j2]]) => `${ja}→${j2}`).join("、")}）` : "",
    (v.failed || []).length ? `訳せなかった語: ${v.failed.join("、")}` : "", v.readings_error ? `読みは調べられませんでした（${v.readings_error}）` : ""].filter(Boolean).join("。"));
}

/* 生成 / 保存 need every chip in English. */
function igJaBlocked() {
  const u = [...igToEn("pos").unknown, ...igToEn("neg").unknown];
  if (!u.length) return false;
  igJaNote();
  toast(`辞書にない日本語（${u.join("、")}）があります。下の欄で英語を登録してください`, "err");
  return true;
}

function igDrawLoras(loras) {
  drawLoraRowsIn($("#ig-loras"), loras, igSaveSoon, () => igTrig(), modelFamily($("#ig-model").value));
}

/* 指示でプロンプトを変える: the local LLM rewrites prompt / negative; the previous text can be restored. */
async function igLoadLlm() {
  const sel = $("#ig-llm");
  let r;
  try { r = await api("/api/llm"); } catch (e) { r = { models: [], error: e.message }; }
  if (!sel || !document.body.contains(sel)) return;
  sel.innerHTML = r.models.length ? r.models.map((m) => `<option value="${esc(m)}" ${m === r.model ? "selected" : ""}>${esc(m)}</option>`).join("")
    : `<option value="">（見つかりません）</option>`;
  if (r.error) $("#ig-rwmsg").innerHTML = `<span class="err">${esc(r.error)}</span>`;
  sel.onchange = async () => { try { await api("/api/llm", { method: "PUT", body: { model: sel.value } }); } catch (e) { fail(e); } };
}

// 指示で変える (LLM prompt editing): hidden until it is good enough to use; the code is kept.
const IG_PROMPT_EDIT = false;

/* CFG 1 models (Krea 2, Z-Image Turbo ...) never use the negative prompt: say so next to the field. */
function igCfg() {
  const i = $("[data-igs=cfg]");
  const v = i && i.value !== "" ? Number(i.value) : Number((S.ig.default_sampling || {}).cfg);
  return isNaN(v) ? null : v;
}
function igNegNote() {
  const n = $("#ig-negnote");
  if (!n) return;
  const cfg = igCfg();
  n.hidden = !(cfg != null && cfg <= 1);
  n.textContent = "このモデルの設定（CFG 1）では、ネガティブプロンプトは生成に使われません。";
}

/* The conversation with the LLM: earlier instructions and what was done are sent with the next one, so
   "さっきのは違う" or "ネガティブは触らないで" refer to them.  Kept until 「やり取りを忘れる」 or reload. */
function igDrawChat() {
  const el = $("#ig-chat");
  if (!el) return;
  const c = S.igChat || [];
  el.innerHTML = c.length ? `<details ${c.length ? "open" : ""}><summary class="muted">これまでの指示（${c.length} 件。次の指示と一緒に LLM に伝えます）</summary>
    ${c.map((h) => `<div style="margin:2px 0 2px 8px">・${esc(h.instruction)}${h.changes.length ? `<span class="muted"> → ${esc(h.changes.join(" ／ "))}</span>` : `<span class="muted"> → 変更なし</span>`}</div>`).join("")}
    <button class="small" id="ig-chatreset">やり取りを忘れる</button></details>` : "";
  const b = $("#ig-chatreset");
  if (b) b.onclick = () => { S.igChat = []; igDrawChat(); };
}

async function igRewrite() {
  const b = $("#ig-rw"), msg = $("#ig-rwmsg"), inst = $("#ig-inst").value.trim();
  if (!inst) { msg.innerHTML = `<span class="err">どう変えたいかを入力してください。</span>`; return; }
  const before = { pos: $("#ig-pos").value, neg: $("#ig-neg").value };
  b.disabled = true;
  msg.innerHTML = `<span class="muted"><span class="spin"></span> 書き換えています…（初回は LLM の読み込みに数十秒かかることがあります）</span>`;
  try {
    const f = igRead();
    const r = await api("/api/imagegen/rewrite", { body: { positive: f.positive, negative: f.negative, instruction: inst,
      model_id: f.model_id, loras: f.loras, sampling: f.sampling, history: S.igChat, llm_model: $("#ig-llm").value || null } });
    S.igChat.push({ instruction: inst, changes: r.changes || [] });
    $("#ig-inst").value = "";
    igDrawChat();
    $("#ig-pos").value = r.positive; $("#ig-neg").value = r.negative;
    igSaveSoon(); igTrig();
    msg.innerHTML = `<span>${esc(r.note || "書き換えました。")}</span> <button class="small" id="ig-undo">元に戻す</button>
      ${r.after ? `<details open class="muted" style="margin-top:2px"><summary>LLM の受け取り方</summary>
        <div style="margin-left:8px">今の絵: ${esc(r.before || "")}</div><div style="margin-left:8px">変更後の絵: ${esc(r.after)}</div></details>` : ""}
      ${r.lost_words.length ? `<div class="warntext">呼び出し語 ${esc(r.lost_words.join(", "))} がプロンプトから消えました。必要なら「元に戻す」を押してください。</div>` : ""}`;
    $("#ig-undo").onclick = () => {
      $("#ig-pos").value = before.pos; $("#ig-neg").value = before.neg;
      igSaveSoon(); igTrig();
      if (S.igChat.length) S.igChat[S.igChat.length - 1].changes = ["（元に戻した）"];
      igDrawChat();
      msg.innerHTML = `<span class="muted">書き換える前に戻しました。</span>`;
    };
  } catch (e) { msg.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  b.disabled = false;
}

function igDrawSizes() {
  const list = IG_SIZES[S.ig.native] || IG_SIZES[1024];
  const w = +$("#ig-w").value, h = +$("#ig-h").value;
  $("#ig-sizes").innerHTML = list.map(([label, pw, ph]) =>
    `<button class="small ${pw === w && ph === h ? "primary" : ""}" data-sz="${pw}x${ph}">${label}<span class="muted"> ${pw}×${ph}</span></button>`).join("");
  $$("[data-sz]").forEach((b) => b.onclick = () => { const [pw, ph] = b.dataset.sz.split("x").map(Number); igSetSize(pw, ph); });
}

function igSetSize(w, h) {
  $("#ig-w").value = w; $("#ig-h").value = h;
  igDrawSizes(); igSaveSoon();
}

function igRead() {
  const num = (v) => v === "" || v == null ? null : Number(v);
  const sampling = {};
  $$("[data-igs]").forEach((i) => { if (i.dataset.igs !== "seed" && i.value !== "") sampling[i.dataset.igs] = i.type === "number" ? num(i.value) : i.value; });
  const seed = $("[data-igs=seed]");
  return {
    model_id: selModelId($("#ig-model")), positive: igToEn("pos").text, negative: igToEn("neg").text,
    width: num($("#ig-w").value), height: num($("#ig-h").value), count: Math.max(1, Math.min(100, Math.round(+$("#ig-count").value || 1))),
    seed: seed && seed.value !== "" ? num(seed.value) : null, sampling,
    regions: { left: $("#ig-left").value.trim(), right: $("#ig-right").value.trim() },
    loras: readLoraRows($("#ig-loras")),
    init_image: (S.ig.form || {}).init_image || null,
    vae: $("#ig-vae") ? $("#ig-vae").value : (S.ig.form || {}).vae || "",
    te: $("#ig-te") ? $("#ig-te").value : (S.ig.form || {}).te || "",
    attention: $("#ig-attn") ? $("#ig-attn").value : (S.ig.form || {}).attention || "",
    workflow: (S.ig.form || {}).workflow || "",
    denoise: $("#ig-den") && $("#ig-den").value !== "" ? num($("#ig-den").value) : (S.ig.form || {}).denoise ?? null,
  };
}

/* image-to-image: a source image the generation starts from (scaled and centre-cropped to the size above) */
function igDrawInit() {
  const el = $("#ig-init");
  if (!el) return;
  const f = S.ig.form || {};
  if (f.init_image) {
    el.innerHTML = `<div class="row" style="align-items:flex-start">
      <img src="/api/imagegen/init/${encodeURIComponent(f.init_image)}" alt="" style="max-width:160px;max-height:160px;border-radius:6px;border:1px solid var(--line)">
      <div><div class="row"><span class="small" ${S.ig.edit ? "hidden" : ""}>変化の強さ</span>
          <input type="number" id="ig-den" min="0.05" max="1" step="0.05" style="width:80px" value="${esc(f.denoise ?? 0.6)}" ${S.ig.edit ? "hidden" : ""}>
          <button class="small" id="ig-init-x">元画像を外す</button></div>
        ${S.ig.edit ? `<div class="small"><b>指示で編集します。</b>編集用の LoRA（MOSS-Image）を選んでいるので、元画像をプロンプトに書いた指示どおりに直します。
          指示は英語で「Change the color of the car to blue.」「Remove the person on the left.」のように書きます。
          出来上がりの大きさは元画像の縦横比に合わせた約 100 万画素になり、上の画像サイズと変化の強さは使いません。
          ネガティブプロンプトは空にしておくのがおすすめです。</div>`
        : `<div class="muted small">小さいほど元画像に近くなります（0.3〜0.5 で色や細部の手直し、0.6〜0.8 で構図を残して描き直し、1 で元画像をほとんど使いません）。
          元画像は上の画像サイズに合わせて拡大・縮小し、はみ出した部分は中央を残して切ります。</div>`}</div></div>`;
    $("#ig-den").oninput = igSaveSoon;
    $("#ig-init-x").onclick = () => { S.ig.form = { ...igRead(), init_image: null }; igDrawInit(); igSave(); };
  } else {
    el.innerHTML = `<div class="row"><button class="small" id="ig-init-pick">画像を選ぶ</button>
      <span class="muted small">画像をここにドロップするか、貼り付ける（Ctrl+V）と、その画像をもとに生成します（I2I）。</span>
      <input type="file" id="ig-init-file" accept="image/*" hidden></div>`;
    $("#ig-init-pick").onclick = () => $("#ig-init-file").click();
    $("#ig-init-file").onchange = (e) => { if (e.target.files[0]) igSetInit(e.target.files[0]); };
  }
  if (!el.dataset.drop) {
    el.dataset.drop = "1";
    el.addEventListener("dragover", (e) => { e.preventDefault(); });
    el.addEventListener("drop", (e) => {
      e.preventDefault();
      const file = [...(e.dataTransfer.files || [])].find((x) => x.type.startsWith("image/"));
      if (file) igSetInit(file);
    });
  }
}

/* 詳細設定の VAE / CLIP / アテンション: the files in the vae / text_encoders folders instead of the model's own, and
   ComfyUI's attention backend (a change restarts ComfyUI before the generation). */
const IG_ATTN = [["", "標準（ComfyUI が自動で選ぶ）"], ["pytorch", "PyTorch（SDPA）"], ["sage", "SageAttention"],
  ["flash", "FlashAttention"], ["split", "Split（省メモリ・遅い）"], ["quad", "Sub-quadratic（省メモリ・遅い）"]];
function igExtraFields(f) {
  const comp = (S.lib && S.lib.companions) || {}, inst = S.ig.attention || {};
  const opts = (kind, cur) => {
    const files = [...(comp[kind] || [])];
    if (cur && !files.includes(cur)) files.unshift(cur);   // chosen before and moved / removed since: still shown
    return `<option value="">モデルのもの（初期値）</option>` + files.map((p) =>
      `<option value="${esc(p)}" title="${esc(p)}" ${p === cur ? "selected" : ""}>${esc(p.split(/[\\/]/).pop())}</option>`).join("");
  };
  return `<div class="field"><div class="lab"><span>VAE</span><span>初期値 モデルのもの</span></div>
      <select id="ig-vae">${opts("vae", f.vae || "")}</select>
      <div class="muted small">vae フォルダーのファイルから選びます。SDXL 系で別の VAE（sdxl_vae など）を使うときに選びます。</div></div>
    <div class="field" id="ig-te-field"><div class="lab"><span>CLIP（テキストエンコーダー）</span><span>初期値 モデルのもの</span></div>
      <select id="ig-te">${opts("te", f.te || "")}</select>
      <div class="muted small">text_encoders フォルダーのファイルから選びます。SDXL 系はモデルの中の CLIP を使うため選べません。</div></div>
    <div class="field"><div class="lab"><span>アテンション</span><span>初期値 標準</span></div>
      <select id="ig-attn">${IG_ATTN.map(([v, label]) => {
        const off = v in inst && !inst[v];
        return `<option value="${v}" ${v === (f.attention || "") ? "selected" : ""} ${off && v !== (f.attention || "") ? "disabled" : ""}>${label}${off ? "（入っていません）" : ""}</option>`; }).join("")}</select>
      <div class="muted small">生成の計算方法です。変えると、生成の前に ComfyUI を起動し直します（10〜30 秒ほど）。SageAttention・FlashAttention は、ComfyUI の環境に入っているときだけ選べます。</div></div>`;
}
function igShowExtras() {
  const el = $("#ig-te-field");
  if (el) el.hidden = modelFamily($("#ig-model").value) === "sdxl";
}

/* 画像から読み込む: an image made by A1111 / Forge / ComfyUI / NovelAI ... (or by this page) carries its prompt,
   model, LoRAs and parameters; they replace the form's, and 「元に戻す」 puts the form back as it was. */
function igDrawInfo() {
  const el = $("#ig-info");
  if (!el) return;
  const m = S.igInfo;
  el.innerHTML = `<div class="row"><button class="small" id="ig-info-pick">画像を選ぶ</button>
      <span class="muted small">生成した画像（PNG など）をここにドロップすると、画像に書き込まれたプロンプト・モデル・LoRA・設定をフォームに入れます。</span>
      <input type="file" id="ig-info-file" accept="image/*" hidden></div>
    ${m ? `<div class="small" style="margin-top:4px">${esc(m.source)} の設定を読み込みました。
        ${m.undo ? `<button class="small" id="ig-info-undo">元に戻す</button>` : ""}
      ${(m.notes || []).map((x) => `<div class="warntext">${esc(x)}</div>`).join("")}</div>` : ""}
    ${igWorkflowHtml()}`;
  const wfSet = async (id) => {
    S.ig.form = { ...igRead(), workflow: id };
    await igSave();
    await renderImageGen();
  };
  if ($("#ig-wf-off")) $("#ig-wf-off").onclick = () => wfSet("");
  if ($("#ig-wf-on")) $("#ig-wf-on").onclick = () => wfSet($("#ig-wf-on").dataset.id);
  $("#ig-info-pick").onclick = () => $("#ig-info-file").click();
  $("#ig-info-file").onchange = (e) => { if (e.target.files[0]) igLoadInfo(e.target.files[0]); };
  if ($("#ig-info-undo")) $("#ig-info-undo").onclick = async () => {
    try { await api("/api/imagegen/form", { method: "PUT", body: m.undo }); } catch (e) { return fail(e); }
    S.igInfo = null;
    await renderImageGen();
    toast("読み込む前の設定に戻しました");
  };
  if (!el.dataset.drop) {
    el.dataset.drop = "1";
    el.addEventListener("dragover", (e) => { e.preventDefault(); });
    el.addEventListener("drop", (e) => {
      e.preventDefault();
      const file = [...(e.dataTransfer.files || [])].find((x) => x.type.startsWith("image/"));
      if (file) igLoadInfo(file);
    });
  }
}

/* workflow のまま生成: a ComfyUI image's workflow is run as it is (its model, LoRAs, size, samplers, hires ...);
   the page's prompt, negative, seed, count and attention go in.  It can be turned off to use the form instead. */
function igWorkflowHtml() {
  const on = (S.ig.form || {}).workflow && S.ig.workflow, loaded = S.igInfo && S.igInfo.workflow;
  if (on) {
    const w = S.ig.workflow;
    return `<div class="panel" style="margin-top:6px;padding:8px"><div class="row"><b>workflow のまま生成します</b>
        <span class="small muted">${esc(w.source)}・ノード ${w.nodes} 個${w.summary ? `・${esc(w.summary)}` : ""}</span>
        <div class="spacer"></div><button class="small" id="ig-wf-off">workflow を使わない</button></div>
      <div class="small muted">モデル・LoRA・サンプラーなどの設定は workflow のものを使い、下の欄のモデル・LoRA・左右の描き分け・元画像・詳細設定（アテンション以外）は使いません。
        プロンプト・ネガティブ・画像サイズ・seed・枚数・アテンションは、この画面のものを使います。「workflow を使わない」を押すと、下の欄の設定で生成します。</div></div>`;
  }
  if (!loaded) return "";
  if ((loaded.missing_nodes || []).length)
    return `<div class="small warntext" style="margin-top:4px">この画像の workflow には、ここの ComfyUI に入っていないノード（${esc(loaded.missing_nodes.join("、"))}）があるため、
      workflow のままでは生成できません。読み取れた設定を下の欄に入れました。</div>`;
  return `<div class="row" style="margin-top:4px"><button class="small" id="ig-wf-on" data-id="${esc(loaded.id)}">workflow のまま生成する</button>
    <span class="small muted">いまは下の欄の設定で生成します。</span></div>`;
}

async function igLoadInfo(file) {
  const before = igRead();
  let j;
  try {
    const r = await fetch("/api/imagegen/pnginfo", { method: "POST", body: file });
    j = await r.json();
    if (!r.ok) throw new Error(j.detail || "画像を読み込めませんでした");
    await api("/api/imagegen/form", { method: "PUT", body: { ...before, ...j.form } });
  } catch (e) { return fail(e); }
  S.igInfo = { source: j.source, notes: j.notes, undo: before, workflow: j.workflow || null };
  await renderImageGen();
  const sel = $("#ig-model");
  if (j.model_file && sel) {   // a model file not set up yet: chosen as in the list, which sets it up
    sel.value = "file:" + j.model_file;
    if (sel.value) sel.onchange();
  }
  toast(`${j.source} の設定をフォームに入れました`);
}

async function igSetInit(file) {
  try {
    const r = await fetch("/api/imagegen/init", { method: "POST", body: file });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "画像を読み込めませんでした");
    // the size follows the image's shape at about the model's own pixel count (multiples of 16)
    const [w, h] = j.size, n = S.ig.native || 1024, k = Math.sqrt((n * n) / (w * h));
    const fit = (v) => Math.max(512, Math.min(2048, Math.round((v * k) / 16) * 16));
    $("#ig-w").value = fit(w); $("#ig-h").value = fit(h);
    S.ig.form = { ...igRead(), init_image: j.name };
    igDrawInit(); igDrawSizes(); igSave();
    toast(`元画像を設定しました（${w}×${h}）。画像サイズを元画像の縦横比に合わせました`);
  } catch (e) { fail(e); }
}
document.addEventListener("paste", (e) => {
  if (!$("#ig-init") || ["INPUT", "TEXTAREA"].includes((document.activeElement || {}).tagName)) return;
  const item = [...(e.clipboardData ? e.clipboardData.items : [])].find((x) => x.type.startsWith("image/"));
  if (item) { e.preventDefault(); igSetInit(item.getAsFile()); }
});

function igDrawCheck(errs, notes) {
  const el = $("#ig-check");
  if (!el) return;
  el.innerHTML = (errs || []).map((x) => `<div class="err small">${esc(x)}</div>`).join("") +
    (notes || []).map((x) => `<div class="muted small">${esc(x)}</div>`).join("");
}

const igSaveSoon = saveDebounce(() => igSave(), 500);
async function igSave() {
  const st = $("#ig-save");
  if (!st) return;
  st.className = "savestate"; st.textContent = "保存中…";
  const body = igRead();   // read now: the form may be gone when the answer comes
  try {
    const r = await api("/api/imagegen/form", { method: "PUT", body });
    if (!$("#ig-save")) { S.ig.form = body; return; }
    S.ig.form = body; S.ig.native = r.native; S.ig.default_sampling = r.default_sampling;
    if ((r.edit || null) !== (S.ig.edit || null)) { S.ig.edit = r.edit || null; igDrawInit(); }
    st.textContent = "保存済み";
    igDrawCheck(r.check, r.notes);
    showDefaultSampling(main, "igs", "igd", r.default_sampling);
    igDrawSizes();
  } catch (e) { st.className = "savestate err"; st.textContent = "保存できませんでした：" + e.message; }
}

async function igStart() {
  if (igJaBlocked()) return;
  const b = $("#ig-start");
  b.disabled = true;
  try {
    await api("/api/imagegen/start", { body: igRead() });
    toast("生成の順番待ちに入りました");
    await igReload();
  } catch (e) { fail(e); }
  b.disabled = false;
}

async function igReload() {
  if (S.route[0] !== "gen" || !S.ig) return;
  try { const r = await api("/api/imagegen"); S.ig.history = r.history; } catch (e) { return fail(e); }
  igDrawHistory();
}
const igReloadSoon = debounce(igReload, 300);

function igSettingsText(s) {
  if (s.workflow) return `workflow のまま（${s.workflow_source}${s.workflow_summary ? `・${s.workflow_summary}` : ""}）${s.width ? ` ／ ${s.width}×${s.height}` : ""} ／ seed ${s.seed}`;
  const lo = s.loras.length ? s.loras.map((l) => `${l.path.split(/[\\/]/).pop()}（${l.strength}）`).join("、") : "なし";
  return `${s.model_name} ／ LoRA: ${lo} ／ ${s.width}×${s.height} ／ seed ${s.seed} ／ ${s.sampling.steps} steps・CFG ${s.sampling.cfg}・${s.sampling.sampler}・${s.sampling.scheduler}${s.sampling.rescale === "on" ? "・白飛び抑制" : ""}${s.regions ? ` ／ 左: ${s.regions[0] || "なし"}・右: ${s.regions[1] || "なし"}` : ""}`
    + (s.edit ? " ／ 元画像を指示で編集" : s.init_image ? ` ／ 元画像から（変化の強さ ${s.denoise}）` : "");
}

/* The settings of a result as text to paste for someone else: the prompt as sent (with the LoRAs' trigger words). */
function igShareText(s) {
  return `モデル：\n${igSettingsText(s)}\n\nポジティブ：\n${s.prompt_sent || s.positive}\n\nネガティブ：\n${s.negative || "なし"}`;
}

/* Progress of a running request: images done, the sampler's steps of the current image (from ComfyUI),
   and the remaining time from measured speed.  Updated in place from the events, without reloading. */
function igProgHtml(p) {
  if (!p) return "";
  const busy = p.phase === "loading" || !p.steps;
  const frac = p.count ? ((p.image - 1) + (p.steps ? p.step / p.steps : 0)) / p.count : 0;
  const label = p.phase === "loading" ? (p.image <= 1 ? "モデルを読み込んでいます…" : "準備中…")
    : `${p.image} / ${p.count} 枚目${p.steps ? `　ステップ ${p.step} / ${p.steps}` : ""}`;
  return `<div class="prog ${busy && frac === 0 ? "busy" : ""}" style="margin:6px 0 4px"><div style="width:${busy && frac === 0 ? 30 : Math.max(2, Math.round(frac * 100))}%"></div></div>
    <div class="row small"><span>${esc(label)}</span>${frac > 0 ? `<span class="muted">${Math.round(frac * 100)}%</span>` : ""}
      <span class="muted">残り ${p.eta_s == null ? "計測中" : fmtDur(p.eta_s)}</span></div>`;
}

function igOnProgress(m) {
  const box = $(`[data-ie="${m.id}"] [data-iprog]`);
  if (!box) { igReloadSoon(); return; }
  box.innerHTML = igProgHtml(m.progress);
}

function igDrawHistory() {
  const el = $("#ig-history");
  if (!el) return;
  const hs = S.ig.history || [];
  if (!hs.length) { el.innerHTML = `<div class="muted small">生成した画像は、ここに新しい順で並びます。</div>`; return; }
  const openBoxes = new Map($$("[data-ie]", el).map((b) => [b.dataset.ie, $("[data-icopybox]", b)]).filter(([, h]) => h && h.childElementCount));
  el.innerHTML = hs.map((e) => {
    const s = e.settings, active = e.status === "queued" || e.status === "running";
    return `<div class="panel" style="margin-bottom:10px" data-ie="${e.id}">
      <div class="row"><b>${IG_STATUS[e.status] || e.status}</b>${e.message && e.message !== IG_STATUS[e.status] ? `<span class="small muted">${esc(e.message)}</span>` : ""}
        <span class="small muted">${new Date(e.created * 1000).toLocaleString("ja-JP")}</span>
        ${s.thumb_for ? `<a class="small" href="#/presets">「${esc(s.thumb_name || "")}」のサムネイル</a>` : ""}<div class="spacer"></div>
        ${active ? `<button class="small" data-icancel>中止</button>` : `<button class="small" data-iuse title="このときの設定（seed も含む）をフォームに戻します">この設定に戻す</button>
          ${e.images.length ? `<a class="btnlink" href="${e.images.length === 1 ? `/api/imagegen/img/${encodeURIComponent(e.images[0].file)}?dl=1` : `/api/imagegen/${e.id}/download.zip`}" download>
            <button class="small">${e.images.length === 1 ? "ダウンロード" : `まとめてダウンロード（${e.images.length} 枚・zip）`}</button></a>
            ${s.workflow ? "" : `<button class="small primary" data-icopy>この設定で LoRA 素材を作る</button>`}` : ""}
          <button class="small" data-idel title="この結果の画像を消します">削除</button>`}</div>
      <div class="row" style="align-items:flex-start;margin:4px 0"><div class="small igshare">${esc(igShareText(s))}</div>
        <button class="small" data-ishare title="モデル・設定・プロンプトを、そのまま貼り付けられる文章でコピーします">コピー</button></div>
      ${active ? `<div data-iprog>${e.status === "running" ? igProgHtml(e.progress) : ""}</div>` : ""}
      ${(e.warnings || []).map((w) => `<div class="err small">${esc(w)}</div>`).join("")}
      <div class="igimgs">${e.images.map((im) => `<div class="igimg"><img loading="lazy" src="/api/imagegen/img/${encodeURIComponent(im.file)}?thumb=1" alt=""
        data-zoom="/api/imagegen/img/${encodeURIComponent(im.file)}" data-cap="seed ${im.seed}" title="クリックで拡大（seed ${im.seed}）">
        <a class="igdl" href="/api/imagegen/img/${encodeURIComponent(im.file)}?dl=1" download title="この画像をダウンロード">↓</a></div>`).join("")}</div>
      <div data-icopybox></div></div>`;
  }).join("");
  $$(".igimgs", el).forEach((row) => {
    const ims = $$("[data-zoom]", row), list = ims.map((x) => ({ src: x.dataset.zoom, cap: x.dataset.cap }));
    ims.forEach((im, i) => im.onclick = () => openLightbox(list, i));
  });
  $$("[data-ie]", el).forEach((box) => {
    const id = box.dataset.ie, e = hs.find((x) => x.id === id);
    const on = (sel, fn) => { const b = $(sel, box); if (b) b.onclick = fn; };
    on("[data-icancel]", async () => { try { await api(`/api/imagegen/${id}/cancel`, { body: {} }); toast("中止を受け付けました"); igReload(); } catch (err) { fail(err); } });
    on("[data-ishare]", async () => {
      try { await navigator.clipboard.writeText(igShareText(e.settings)); toast("設定とプロンプトをコピーしました"); }
      catch (err) { toast("コピーできませんでした。文章を選んでコピーしてください"); }
    });
    on("[data-idel]", async () => { try { await api(`/api/imagegen/${id}`, { method: "DELETE" }); igReload(); } catch (err) { fail(err); } });
    on("[data-iuse]", async () => {
      const s = e.settings;
      try {
        await api("/api/imagegen/form", { method: "PUT", body: { model_id: s.model_id, loras: s.loras, positive: s.positive, negative: s.negative,
          width: s.width, height: s.height, count: s.count, seed: s.seed, sampling: s.sampling_overrides,
          init_image: s.init_image || null, denoise: s.denoise ?? null,
          regions: s.regions ? { left: s.regions[0] || "", right: s.regions[1] || "" } : null,
          vae: s.vae || "", te: s.te || "", attention: s.attention || "", workflow: s.workflow || "" } });
      } catch (err) { return fail(err); }
      await renderImageGen(); window.scrollTo(0, 0);
      toast("フォームをこの結果の設定に戻しました（seed も同じです）");
    });
    on("[data-icopy]", () => igCopyBox(box, e));
    const old = openBoxes.get(id), host = $("[data-icopybox]", box);
    if (old && host) host.replaceWith(old);   // keeps the chosen project and the typed name
  });
}

async function igCopyBox(box, e) {
  const host = $("[data-icopybox]", box);
  let list = [];
  try { list = (await api("/api/projects")).filter((p) => p.kind === "character" && !p.error); } catch (err) { return fail(err); }
  host.innerHTML = `<div class="note" style="margin-top:8px">
    <div class="small">モデル・LoRA・プロンプト・ネガティブ・詳細設定を、AI 生成画像からキャラ LoRA を作る設定にコピーします。
      画像サイズと seed はコピーしません。素材の生成では、アングル・画角ごとのサイズと seed をツールが決め、プロンプトにアングル・画角の指定を足し、背景の指定を外します。</div>
    <div class="row" style="margin-top:6px"><select data-cp>
      <option value="">新しいプロジェクトを作る</option>
      ${list.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}${p.source === "generate" ? "（AI 生成）" : ""}</option>`).join("")}</select>
      <input type="text" data-cn placeholder="プロジェクト名（空欄なら自動）" style="width:220px">
      <button class="primary small" data-cgo>コピーして開く</button><button class="small" data-cx>やめる</button></div>
    <div class="muted small" data-cwarn hidden>選んだプロジェクトの生成設定（モデル・LoRA・プロンプトなど）は上書きされます。枚数と使わないアングルの設定はそのまま残ります。</div></div>`;
  host.scrollIntoView({ behavior: "smooth", block: "nearest" });
  const sel = $("[data-cp]", host);
  sel.onchange = () => { $("[data-cn]", host).hidden = !!sel.value; $("[data-cwarn]", host).hidden = !sel.value; };
  $("[data-cx]", host).onclick = () => { host.innerHTML = ""; };
  $("[data-cgo]", host).onclick = async () => {
    try {
      const r = await api(`/api/imagegen/${e.id}/copy`, { body: { project_id: sel.value || null, name: $("[data-cn]", host).value } });
      (r.notes || []).forEach((n) => toast(n));
      S.genFor = null; S.gen = null; S.project = null;
      location.hash = `#/p/${r.project_id}/material`;
      toast("設定をコピーしました。内容を確かめてから「生成を開始」を押してください");
    } catch (err) { fail(err); }
  };
}

// ------------------------------------------------------------------ help
/* The page body is static/help.html.  #/help/<section id> opens the page at that section. */
async function renderHelp(section) {
  S.project = null;
  $("#crumbs").innerHTML = `<span>›</span><span>ヘルプ</span>`;
  try {
    const r = await fetch("/static/help.html", { cache: "no-cache" });
    if (!r.ok) throw new Error(`ヘルプを読み込めませんでした（${r.status}）`);
    main.innerHTML = await r.text();
  } catch (e) { main.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
  $$("[data-goto]", main).forEach((a) => a.onclick = (e) => {
    e.preventDefault();
    history.replaceState(null, "", `#/help/${a.dataset.goto}`);
    const t = document.getElementById(a.dataset.goto);
    if (t) t.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  const t = section && document.getElementById(section);
  if (t) t.scrollIntoView({ block: "start" }); else window.scrollTo(0, 0);
}

/* 画像をぼかして表示: the eye button on the image shows / hides that one image; a click on the image itself
   does what it does without blur (enlarging, selecting).  Shown images are kept by address, so they stay
   shown when the list is redrawn. */
const UNBLUR = new Set();
const unblurKey = (img) => img.getAttribute("src") || "";
const blurOn = () => document.documentElement.classList.contains("blurimg");
function toggleUnblur(img) {
  const k = unblurKey(img), on = !UNBLUR.has(k);
  on ? UNBLUR.add(k) : UNBLUR.delete(k);
  $$("img").forEach((x) => { if (unblurKey(x) === k) x.classList.toggle("unblur", on); });
  eyeDraw();
}

/* The eye at the top-right of the image under the pointer: open eye = the image is shown, crossed eye =
   blurred.  One button follows the pointer, so no image's layout changes. */
const EYE_OPEN = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z"/><circle cx="12" cy="12" r="3"/></svg>`;
const EYE_SHUT = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z"/><circle cx="12" cy="12" r="3"/><path d="M3 3l18 18"/></svg>`;
const eye = Object.assign(document.createElement("button"), { id: "blureye", type: "button", hidden: true });
let eyeImg = null;
function eyeDraw() {
  if (!eyeImg || !blurOn() || !document.body.contains(eyeImg)) { eye.hidden = true; return; }
  const r = eyeImg.getBoundingClientRect();
  if (r.width < 40 || r.height < 40) { eye.hidden = true; return; }
  const shown = eyeImg.classList.contains("unblur");
  eye.innerHTML = shown ? EYE_OPEN : EYE_SHUT;
  eye.title = shown ? "クリックでぼかす" : "クリックで表示";
  eye.style.left = `${r.right - 34}px`; eye.style.top = `${r.top + 6}px`;
  eye.hidden = false;
}
eye.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); if (eyeImg) toggleUnblur(eyeImg); });
document.addEventListener("DOMContentLoaded", () => document.body.appendChild(eye));
document.addEventListener("mouseover", (e) => {
  if (e.target === eye || eye.contains(e.target)) return;
  const img = e.target.closest && e.target.closest("img");
  eyeImg = img && blurOn() ? img : null;
  eyeDraw();
});
window.addEventListener("scroll", () => { eyeImg = null; eye.hidden = true; }, true);
new MutationObserver((muts) => {
  if (!UNBLUR.size) return;
  for (const m of muts) {
    if (m.type === "attributes") m.target.classList.toggle("unblur", UNBLUR.has(unblurKey(m.target)));
    else for (const n of m.addedNodes) {
      if (n.nodeType !== 1) continue;
      (n.tagName === "IMG" ? [n] : n.querySelectorAll("img")).forEach((x) => { if (UNBLUR.has(unblurKey(x))) x.classList.add("unblur"); });
    }
  }
}).observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ["src"] });

// ------------------------------------------------------------------ boot
// a caption typed just before the tab is closed or reloaded is still saved (the autosave waits 600 ms)
window.addEventListener("pagehide", () => {
  S.unloading = true;
  try { flushSaves(); } finally { S.unloading = false; }
  const c = S.capPending;
  if (!c) return;
  S.capPending = null;
  try {
    fetch(`/api/projects/${c.pid}/edit`, { method: "POST", keepalive: true, headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: `#${c.id} の説明文を編集`, changes: [{ id: c.id, field: "caption", value: c.value }] }) });
  } catch { /* the page is going away */ }
});
connectEvents();
refreshStatus();
// after every script (i2t.js is loaded after this file) has defined its page
window.addEventListener("DOMContentLoaded", route);
