const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const NODE_CLASS = "ReferenceGalleryPPM";
const WIDGET_NAME = "ppm_reference_gallery";
const THUMB = 84;
const ROW_H = THUMB + 18;
const BAR_H = 24;
const STATUS_H = 14;
const PREVIEW = 96;
const FILTER_H = 22;
const ROW_GAP = 4;
// The canvas insets the widget element by DEFAULT_MARGIN (10px) on each side, so the reported
// height has to carry that chrome on top of what the layout renders.
const WIDGET_PAD = 24;
const DEFAULT_NODE_W = 420;
// The bar row and the editor's button row cannot shrink below their text, so narrower than this
// they would be clipped by the widget. This is the floor the canvas respects while resizing.
const MIN_NODE_W = 400;
// Both grids, the filter row and the editor share the room the node is given: stretching the node
// is what turns the second row of each grid and the editor height visible.
const BODY_H = Math.max(2 * ROW_H + ROW_GAP + FILTER_H, PREVIEW);
const WIDGET_MIN_H = BODY_H + BAR_H + STATUS_H + 3 * ROW_GAP + WIDGET_PAD;

// ComfyUI writes the active colour palette onto these CSS variables, so the chrome follows the
// palette - and a live palette switch - instead of fixed greys.
const INK = "var(--input-text,#ddd)";
const DIM = "var(--descrip-text,#999)";
const FILL = "var(--comfy-input-bg,#222)";
const EDGE = "var(--border-color,#4e4e4e)";
const HI = "var(--content-bg,#4e4e4e)";
const HI_INK = "var(--content-fg,#fff)";
const BAD = "var(--error-text,#ff4444)";
const OK = "var(--success-text,#4caf50)";

const BTN = "flex:0 0 auto;padding:2px 8px;border:1px solid " + EDGE + ";border-radius:4px;" +
    "background:" + FILL + ";color:" + INK + ";font:11px monospace;cursor:pointer;";
const FIELD = "box-sizing:border-box;min-width:0;padding:3px 4px;border:1px solid " + EDGE + ";" +
    "border-radius:4px;background:" + FILL + ";color:" + INK + ";font:11px monospace;";
const SLIDER = "flex:1 1 auto;min-width:60px;height:14px;accent-color:" + HI_INK + ";cursor:pointer;";
const GRID = `display:grid;grid-template-columns:repeat(auto-fill,${THUMB}px);gap:${ROW_GAP}px;` +
    "align-content:start;justify-content:start;min-height:0;overflow-y:auto;overflow-x:hidden;";

const galleries = new Map();

function el(tag, css, attrs) {
    const node = document.createElement(tag);
    if (css) node.style.cssText = css;
    for (const [k, v] of Object.entries(attrs ?? {})) node.setAttribute(k, v);
    return node;
}

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget || widget.value === value) return;
    widget.value = value;
    widget.callback?.(value);
}

function characterName(node) {
    return String(getWidget(node, "character")?.value ?? "");
}

// The routes say what went wrong, so the node can show it instead of a bare status code.
async function errorMessage(res) {
    if (!res) return null;
    const body = await res.json().catch(() => ({}));
    return body.error ?? null;
}

function isSelected(node) {
    // The node knows best; selectedItems is only there for a node that predates it. It is a map of
    // item to pointer state, so an entry is a [item, state] pair, not the item itself.
    if (node.selected) return true;
    const items = app.graph?.selectedItems;
    if (!items) return false;
    for (const entry of items) {
        const item = Array.isArray(entry) ? entry[0] : entry;
        if (item === node || item?.node === node) return true;
    }
    return false;
}

// Pseudo elements and the tiles need a stylesheet; inline styles cannot reach them.
function ensureStyles() {
    if (document.getElementById("ppm-ref-style")) return;
    const style = el("style", "", { id: "ppm-ref-style" });
    style.textContent =
        `.ppm-ref-scroll{scrollbar-width:thin;scrollbar-color:${EDGE} transparent;}` +
        `.ppm-ref-scroll::-webkit-scrollbar{width:8px;height:8px;}` +
        `.ppm-ref-scroll::-webkit-scrollbar-thumb{background:${EDGE};border-radius:4px;}` +
        `.ppm-ref-scroll::-webkit-scrollbar-track{background:transparent;}` +
        `.ppm-ref-gallery img{-webkit-user-drag:none;user-select:none;}` +
        `.ppm-ref-tile{position:relative;flex:0 0 auto;width:${THUMB}px;padding:2px 2px 3px;border-radius:4px;` +
        `border:1px solid ${EDGE};background:${FILL};color:${INK};font:10px monospace;cursor:pointer;text-align:left;}` +
        `.ppm-ref-tile:hover{border-color:${HI_INK};}` +
        `.ppm-ref-tile[data-active]{border-color:${HI_INK};background:${HI};color:${HI_INK};}` +
        `.ppm-ref-tile[data-off]{opacity:.45;}` +
        `.ppm-ref-label{margin-top:2px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;opacity:.6;}` +
        `.ppm-ref-label[data-prompt]{opacity:1;font-weight:700;}` +
        `.ppm-ref-flag{position:absolute;top:3px;right:3px;padding:0 3px;border-radius:3px;` +
        `background:rgba(0,0,0,.65);color:${INK};font:9px monospace;cursor:pointer;}` +
        `.ppm-ref-flag:hover{background:${HI};color:${HI_INK};}` +
        `.ppm-ref-flag[data-on]{color:${OK};}` +
        `.ppm-ref-flag[data-star]{color:#ffcc00;}` +
        `.ppm-ref-badge{position:absolute;left:3px;bottom:16px;padding:0 3px;border-radius:3px;` +
        `background:rgba(0,0,0,.65);color:${DIM};font:9px monospace;}` +
        `.ppm-ref-btn:not(:disabled):hover{background:${HI};color:${HI_INK};}` +
        `.ppm-ref-btn[data-on="1"]{border-color:${HI_INK};}` +
        `.ppm-ref-gallery input:disabled,.ppm-ref-gallery textarea:disabled{opacity:.55;}` +
        `.ppm-ref-gallery input:focus,.ppm-ref-gallery textarea:focus{outline:none;border-color:${HI_INK};}` +
        `.ppm-ref-gallery ::placeholder{color:${DIM};opacity:1;}`;
    document.head.append(style);
}

function setStatus(st, text, bad) {
    st.status.textContent = text;
    st.status.style.color = bad ? BAD : DIM;
}

function selectedCharacter(st, node) {
    return st.characters.find((character) => character.name === characterName(node)) ?? null;
}

// The filter matches a character name or prompt, or any of its references, and it never moves the
// selection - filtering cannot change what the node outputs.
function visibleCharacters(st) {
    const term = st.filter.trim().toLowerCase();
    if (!term) return st.characters;
    return st.characters.filter((character) => character.name.toLowerCase().includes(term) ||
        (character.prompt ?? "").toLowerCase().includes(term) ||
        (character.refs ?? []).some((ref) => ref.name.toLowerCase().includes(term) ||
            (ref.prompt ?? "").toLowerCase().includes(term)));
}

function visibleRefs(st, character) {
    const refs = character?.refs ?? [];
    const term = st.filter.trim().toLowerCase();
    if (!term) return refs;
    return refs.filter((ref) => ref.name.toLowerCase().includes(term) ||
        (ref.prompt ?? "").toLowerCase().includes(term));
}

// What the node actually emits: switched on and strong enough to reach the sample.
function outputCount(refs) {
    return (refs ?? []).filter((item) => item.enabled && Number(item.strength ?? 1) > 0).length;
}

function refOf(character, name) {
    return (character?.refs ?? []).find((ref) => ref.name === name) ?? null;
}

function viewUrl(ref) {
    const params = new URLSearchParams({ filename: ref.name, subfolder: ref.subfolder, type: "input" });
    if (ref.mtime) params.set("mtime", String(ref.mtime));
    return api.apiURL(`/view?${params}`);
}

function thumb(ref, size) {
    const img = el("img", `display:block;width:${size}px;height:${size}px;object-fit:cover;` +
        `border-radius:3px;background:${FILL};`, { alt: ref.name, draggable: "false" });
    // Only the thumbs in view are worth a full size /view fetch and decode.
    img.loading = "lazy";
    img.decoding = "async";
    img.src = viewUrl(ref);
    return img;
}

function flag(tile, text, title, on, star) {
    const box = el("div", "", { class: "ppm-ref-flag", title });
    box.textContent = text;
    if (on) box.dataset.on = "1";
    if (star) box.dataset.star = "1";
    tile.append(box);
    return box;
}

function characterTile(st, node, character, active) {
    const portrait = refOf(character, character.portrait) ?? character.refs?.[0] ?? null;
    const tile = el("button", "", { type: "button", title: `${character.name} - double click to edit the character` });
    tile.className = "ppm-ref-tile";
    tile.dataset.character = character.name;
    if (active) tile.dataset.active = "1";
    const on = outputCount(character.refs);
    tile.append(portrait ? thumb(portrait, THUMB - 8) : emptyTile("no image"));

    const badge = el("div", "", { class: "ppm-ref-badge", title: "references that are output" });
    badge.textContent = `${on}/${character.refs?.length ?? 0}`;
    tile.append(badge);

    const label = el("div", "", { class: "ppm-ref-label", title: character.prompt || "no character prompt saved yet" });
    label.textContent = character.name;
    if (character.prompt) label.dataset.prompt = "1";
    tile.append(label);

    tile.addEventListener("click", () => selectCharacter(st, node, character.name));
    tile.addEventListener("dblclick", (event) => {
        event.stopPropagation();
        showMenu(st, node, "image");
    });
    return tile;
}

function refTile(st, node, character, ref, active) {
    const title = `${ref.name} - strength ${Number(ref.strength ?? 1).toFixed(2)} - double click to edit`;
    const tile = el("button", "", { type: "button", title });
    tile.className = "ppm-ref-tile";
    tile.dataset.ref = ref.name;
    if (active) tile.dataset.active = "1";
    if (!ref.enabled) tile.dataset.off = "1";
    tile.append(thumb(ref, THUMB - 8));

    const toggle = flag(tile, ref.enabled ? "on" : "off", ref.enabled ? "output this reference - click to switch it off" : "switched off - click to output it", ref.enabled);
    toggle.addEventListener("click", (event) => {
        event.stopPropagation();
        saveState(st, node, character, { [ref.name]: { enabled: !ref.enabled } });
    });
    const star = flag(tile, character.portrait === ref.name ? "\u2605" : "\u2606", "use as the character portrait", true, true);
    star.style.right = "26px";
    star.addEventListener("click", (event) => {
        event.stopPropagation();
        if (character.portrait !== ref.name) saveState(st, node, character, {}, ref.name);
    });

    const label = el("div", "", { class: "ppm-ref-label", title: ref.prompt || character.prompt || "no prompt - the character prompt is used" });
    label.textContent = ref.name.replace(/\.[^.]+$/, "");
    if (ref.prompt) label.dataset.prompt = "1";
    tile.append(label);

    tile.addEventListener("click", () => selectRef(st, node, ref.name));
    tile.addEventListener("dblclick", (event) => {
        event.stopPropagation();
        st.ref = ref.name;
        showMenu(st, node, "image", true);
    });
    return tile;
}

// Scrolling on every draw would fight the wheel, so follow selection changes only.
function reveal(st, grid, key, name) {
    if (!name || st.revealed[key] === name) return;
    st.revealed[key] = name;
    const safe = CSS.escape ? CSS.escape(name) : name;
    grid.querySelector(`[data-${key}="${safe}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function emptyTile(text) {
    const box = el("div", `display:flex;align-items:center;justify-content:center;width:${THUMB - 8}px;` +
        `height:${THUMB - 8}px;color:${DIM};font:9px monospace;`);
    box.textContent = text;
    return box;
}

function paintGrids(st, node) {
    const character = selectedCharacter(st, node);
    const characters = visibleCharacters(st);
    // The prompt of a tile is only its title and how bold the label is, but it is exactly what a
    // save just changed, so its length belongs in the key too.
    const key = [
        character?.name ?? "",
        characters.map((c) => `${c.name} ${c.portrait} ${c.refs?.length ?? 0} ${(c.prompt ?? "").length}`).join("|"),
        (character?.refs ?? []).map((r) => `${r.name} ${r.enabled} ${r.mtime} ${(r.prompt ?? "").length}`).join("|"),
        st.ref,
        st.filter,
    ].join("#");

    if (key !== st.gridKey) {
        st.gridKey = key;
        const tiles = characters.map((item) => characterTile(st, node, item, item.name === character?.name));
        if (!tiles.length) {
            tiles.push(hint(st.characters.length ? `no character matches "${st.filter.trim()}"`
                : "Drop or paste an image, or press New, to make the first character"));
        }
        st.charGrid.replaceChildren(...tiles);

        const refs = visibleRefs(st, character);
        const cells = refs.map((ref) => refTile(st, node, character, ref, ref.name === st.ref));
        if (!cells.length) {
            cells.push(hint(!character ? "pick a character"
                : refs.length ? "no reference matches the filter" : "Drop images here to add references to " + character.name));
        }
        st.refGrid.replaceChildren(...cells);
    }
    if (st.view !== st.menu) {
        st.view = st.menu;
        st.revealed = {};  // the tiles were hidden, so the selection has to be scrolled to again
    }
    const onCharacters = !imageMenu(st.menu);
    st.charGrid.style.display = onCharacters ? "grid" : "none";
    st.refGrid.style.display = onCharacters ? "none" : "grid";
    if (onCharacters) reveal(st, st.charGrid, "character", character?.name);
    else reveal(st, st.refGrid, "ref", st.ref);
    paintCount(st, node);
}

function hint(text) {
    const box = el("div", `display:flex;align-items:center;justify-content:center;min-height:${ROW_H}px;` +
        `padding:6px;color:${DIM};font:10px monospace;text-align:center;`);
    box.textContent = text;
    return box;
}

function paintCount(st, node) {
    const character = selectedCharacter(st, node);
    const total = st.characters.length;
    if (!total) return setStatus(st, "no characters yet - press New or drop an image on the grid");
    const refs = character?.refs ?? [];
    const on = outputCount(refs);
    const count = `${on}/${refs.length} output`;
    // The image menu is about one character, so how many others exist is noise.
    if (imageMenu(st.menu)) return setStatus(st, character ? `${character.name}: ${count}` : "no character selected");
    const shown = visibleCharacters(st).length;
    const head = st.filter.trim() ? `${shown} of ${total} characters` : `${total} character${total === 1 ? "" : "s"}`;
    setStatus(st, character ? `${head} - ${character.name}: ${count}` : head);
}

// What the open grid edits: the character menu owns the character prompt, the image menu owns the
// prompt of one reference. Browse only selects, so it never edits.
const imageMenu = (menu) => menu === "image";

// What the grid that is showing edits: the character menu owns the character prompt, the image
// menu owns the prompt of one reference. Browse only selects, so it never edits.
function menuLevel(st) {
    return st.menu === "character" ? "character" : "reference";
}

function ensureRef(st, node) {
    const character = selectedCharacter(st, node);
    if (!refOf(character, st.ref)) st.ref = character?.portrait ?? character?.refs?.[0]?.name ?? "";
    return refOf(character, st.ref);
}

// Prompt text only ever lives in the file, so an editor that is closed has nothing left to keep.
function discard(st) {
    st.shownFor = null;
    st.promptArea.value = "";
}

function setButton(button, label, off) {
    if (button.textContent !== label) button.textContent = label;
    button.disabled = off;
}

function paintBar(st, node) {
    const character = selectedCharacter(st, node);
    const ref = refOf(character, st.ref);
    const target = editTarget(st, node);
    const onImages = imageMenu(st.menu);
    // Unsaved text is written by Save and by nothing else, so it is the only button that can say so.
    const mine = st.open && !!st.shownFor && st.shownFor === editorKey(st, node)
        && st.promptArea.value !== st.shownText;
    setButton(st.barEdit, "Edit", !onImages || !ref);
    setButton(st.barRename, "Rename", onImages || !character);
    setButton(st.barImages, "Images", onImages || !character);
    setButton(st.barBack, "Back", !onImages);
    setButton(st.deleteButton, "Delete", !target);
    if (st.saveButton) st.saveButton.disabled = !target;
    if (!st.open) return;
    st.renameButton.disabled = !target;
    st.strength.disabled = !target;
    st.nameInput.disabled = !target;
    st.promptArea.disabled = !target;
    st.levelLabel.textContent = menuLevel(st);
    st.saveButton.textContent = mine ? "Save *" : "Save";
    st.saveButton.title = menuLevel(st) === "character"
        ? "Write this prompt into ref_data.json"
        : `Write this prompt into ${String(ref?.name ?? "the reference").replace(/\.[^.]+$/, "")}.txt`;
}

// A menu is one grid plus the actions of that grid. Only its own buttons show, in the order that
// menu reads in.
function showMenu(st, node, menu, editor) {
    setPanel(st, node, false);
    st.menu = menu;
    st.open = false;
    const order = imageMenu(menu)
        ? [st.barBack, st.refresh, st.add, st.barEdit, st.deleteButton]
        : [st.refresh, st.fresh, st.barRename, st.barImages, st.deleteButton];
    for (const button of st.barButtons) button.style.display = order.includes(button) ? "" : "none";
    st.bar.replaceChildren(...order);
    if (imageMenu(menu)) ensureRef(st, node);
    paint(st, node);
    renderPanel(st, node);
    node.setDirtyCanvas?.(true, true);
    if (editor) setPanel(st, node, true);
}


function paint(st, node) {
    paintGrids(st, node);
    paintBar(st, node);
}

function editorKey(st, node) {
    return `${menuLevel(st)}:${menuLevel(st) === "character" ? characterName(node) : st.ref}`;
}

function editTarget(st, node) {
    const character = selectedCharacter(st, node);
    if (!character) return null;
    return menuLevel(st) === "character" ? character : refOf(character, st.ref);
}

function selectCharacter(st, node, name) {
    setWidgetValue(node, "character", name);
    const character = selectedCharacter(st, node);
    st.ref = character?.portrait ?? character?.refs?.[0]?.name ?? "";
    paint(st, node);
    renderPanel(st, node);
    node.setDirtyCanvas?.(true, false);
}

function selectRef(st, node, name) {
    st.ref = name;
    paint(st, node);
    renderPanel(st, node);
}

// A workflow saved before characters were folders carries "card.png" instead of "card", the name
// of the loose card's image - that one is the same character, anything else is just gone.
function resolveName(names, saved) {
    const stem = String(saved).replace(/\.[^.]+$/, "").toLowerCase();
    return names.find((name) => name.toLowerCase() === stem) ?? names[0] ?? "";
}

// The lists drive both grids, the hidden combo and the editor of whatever is selected.
function applyCharacters(st, node, characters) {
    st.characters = characters ?? [];
    const names = st.characters.map((character) => character.name);
    const combo = getWidget(node, "character");
    if (combo?.options) combo.options.values = names;
    if (!names.includes(characterName(node))) {
        selectCharacter(st, node, resolveName(names, characterName(node)));
        return names;
    }
    ensureRef(st, node);
    paint(st, node);
    renderPanel(st, node);
    return names;
}

// One route carries every action, so the pack registers a single path.
async function request(action, fields) {
    return await api.fetchApi("/ppm/refs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, ...fields }),
    }).catch(() => null);
}

async function reload(st, node) {
    const res = await request("list");
    if (!res?.ok) {
        setStatus(st, "character list unavailable", true);
        return [];
    }
    const body = await res.json().catch(() => ({}));
    st.subfolder = body.subfolder || st.subfolder;
    st.loaded = true;
    return applyCharacters(st, node, body.characters);
}

// The whole list is sent back so a toggle cannot lose the order or the strength of the others.
function stateOf(character, changes, portrait) {
    return {
        character: character.name,
        portrait: portrait ?? character.portrait ?? "",
        refs: (character.refs ?? []).map((ref) => {
            const next = changes?.[ref.name] ?? {};
            return {
                name: ref.name,
                enabled: next.enabled ?? ref.enabled,
                strength: next.strength ?? ref.strength ?? 1,
            };
        }),
    };
}

async function saveState(st, node, character, changes, portrait) {
    if (!character) return;
    const res = await request("save_data", stateOf(character, changes, portrait));
    if (!res?.ok) {
        setStatus(st, (await errorMessage(res)) ?? `not saved (${res?.status ?? "network"})`, true);
        return;
    }
    const body = await res.json().catch(() => ({}));
    if (body.characters) applyCharacters(st, node, body.characters);
    const next = body.characters?.find((c) => c.name === character.name) ?? character;
    const changed = Object.keys(changes ?? {})[0];
    const target = changed ? refOf(next, changed) : null;
    const change = changes?.[changed] ?? {};
    setStatus(st, change.enabled !== undefined ? `${changed} is ${target?.enabled ? "on" : "off"}`
        : change.strength !== undefined ? `${changed} strength ${Number(target?.strength ?? 0).toFixed(2)}`
        : `portrait: ${next.portrait}`);
}

async function saveText(st, node, text) {
    const character = selectedCharacter(st, node);
    if (!character) return;
    const res = menuLevel(st) === "character"
        ? await request("save_character", { character: character.name, prompt: text })
        : await request("save_ref", { character: character.name, ref: st.ref, prompt: text });
    if (!res?.ok) {
        setStatus(st, (await errorMessage(res)) ?? `not saved (${res?.status ?? "network"})`, true);
        return;
    }
    const body = await res.json().catch(() => ({}));
    st.shownText = text;  // the editor and the file are in sync again
    if (body.characters) applyCharacters(st, node, body.characters);
    setStatus(st, menuLevel(st) === "character" ? `saved to ${character.name}` : `saved to ${st.ref}`);
}

async function renameTarget(st, node, typed) {
    const character = selectedCharacter(st, node);
    const next = String(typed ?? "").trim();
    if (!character || !next) return;
    // A rename is not a reason to lose the prompt text typed next to it.
    const prompt = st.shownFor === editorKey(st, node) && st.promptArea.value !== st.shownText
        ? st.promptArea.value : null;
    const res = menuLevel(st) === "character"
        ? await request("rename_character", { character: character.name, new_name: next })
        : await request("rename_ref", { character: character.name, ref: st.ref, new_name: next });
    if (!res?.ok) {
        setStatus(st, (await errorMessage(res)) ?? `not renamed (${res?.status ?? "network"})`, true);
        return;
    }
    const body = await res.json().catch(() => ({}));
    if (menuLevel(st) === "character" && body.name) {
        selectCharacter(st, node, body.name);
    } else if (body.ref) {
        st.ref = body.ref;
    }
    if (body.characters) applyCharacters(st, node, body.characters);
    renderPanel(st, node, true);
    if (prompt !== null) {
        st.promptArea.value = prompt;
        paintBar(st, node);  // the restored text is unsaved again, so say so now
    }
    setStatus(st, `renamed to ${menuLevel(st) === "character" ? body.name : body.ref}`);
}

async function removeTarget(st, node) {
    const character = selectedCharacter(st, node);
    const target = menuLevel(st) === "character" ? character?.name : `${character?.name}/${st.ref}`;
    if (!character || !window.confirm(`Delete ${target}? Files in input/${st.subfolder} go with it.`)) return;
    const res = menuLevel(st) === "character"
        ? await request("delete_character", { character: character.name })
        : await request("delete_ref", { character: character.name, ref: st.ref });
    if (!res?.ok) {
        setStatus(st, (await errorMessage(res)) ?? `not deleted (${res?.status ?? "network"})`, true);
        return;
    }
    const body = await res.json().catch(() => ({}));
    st.ref = "";
    if (body.characters) applyCharacters(st, node, body.characters);
    renderPanel(st, node, true);  // whatever was typed about the deleted target is gone with it
    if (!st.characters.length) showMenu(st, node, "character");
    setStatus(st, `deleted ${target}`);
}

async function newCharacter(st, node) {
    const typed = window.prompt("New character name", "");
    const name = String(typed ?? "").trim();
    if (!name) return;
    const res = await request("add_character", { character: name });
    if (!res?.ok) {
        setStatus(st, (await errorMessage(res)) ?? `not added (${res?.status ?? "network"})`, true);
        return;
    }
    const body = await res.json().catch(() => ({}));
    st.filter = "";
    st.filterInput.value = "";
    if (body.characters) applyCharacters(st, node, body.characters);
    if (body.name) selectCharacter(st, node, body.name);
    setStatus(st, `added ${body.name} - ${imageMenu(st.menu) ? "drop images here" : "press Images for its references"}`);
}

// A pasted image and the character made of it are both named after the clock - a clipboard image
// is always called "image", so the name it carries is worth nothing.
function stamp() {
    return new Date().toISOString().replace(/[-:]/g, "").replace("T", "_").slice(0, 15);
}

// Pasted files have no useful name, dropped ones keep theirs so a file can be replaced by name.
// One stamp covers a whole batch, which is uploaded within one second, so the pasted images of a
// single upload are told apart by a tail - the upload overwrites, so equal names would keep the last.
function uploadName(file, source, index, count) {
    if (source !== "drop") return `ref_${stamp()}${count > 1 ? `_${index + 1}` : ""}.png`;
    const raw = String(file.name || "ref");
    const ext = (raw.match(/\.[A-Za-z0-9]+$/) || [".png"])[0].toLowerCase();
    const stem = raw.replace(/\.[A-Za-z0-9]+$/, "").replace(/[\\/:*?"<>|]+/g, " ").replace(/\s+/g, " ").trim();
    return (stem.slice(0, 60) || "ref") + ext;
}

// A new character is named after the first file that is dropped onto the character grid.
function characterFrom(file) {
    const stem = String(file?.name ?? "").replace(/\.[A-Za-z0-9]+$/, "").replace(/[\\/:*?"<>|]+/g, " ").replace(/\s+/g, " ").trim();
    return (stem || "character").slice(0, 40);
}

function imageFiles(files) {
    return [...files].filter((file) => (file.type || "").startsWith("image/"));
}

async function uploadFiles(st, node, files, source, into) {
    const images = imageFiles(files);
    if (!images.length) return;
    const target = into || characterName(node) || characterFrom(images[0]);
    setStatus(st, `uploading ${images.length}...`);
    let last = null;
    for (const [index, file] of images.entries()) {
        const form = new FormData();
        form.append("image", file, uploadName(file, source, index, images.length));
        form.append("type", "input");
        form.append("subfolder", target ? `${st.subfolder}/${target}` : st.subfolder);
        form.append("overwrite", "true");
        const res = await api.fetchApi("/upload/image", { method: "POST", body: form }).catch(() => null);
        if (!res?.ok) {
            setStatus(st, `upload failed (${res?.status ?? "network"})`, true);
            return;
        }
        last = (await res.json().catch(() => ({})))?.name ?? last;
    }
    const names = await reload(st, node);
    // A new character that a leftover filter would hide looks like an upload that did nothing.
    st.filter = "";
    st.filterInput.value = "";
    if (names.includes(target)) {
        selectCharacter(st, node, target);
        if (refOf(selectedCharacter(st, node), last)) selectRef(st, node, last);
    }
    setStatus(st, `${images.length} image${images.length === 1 ? "" : "s"} added to ${characterName(node)}`);
}

function showFull(st, node) {
    const character = selectedCharacter(st, node);
    // Only a reference is a file: the character itself is a folder, so the character level shows
    // and enlarges its portrait.
    const ref = refOf(character, st.previewRef) ?? refOf(character, st.ref) ?? character?.refs?.[0] ?? null;
    if (!ref) return;
    // The viewer stays in the page on the layer the frontend's own image lightbox uses, rather than
    // taking the browser fullscreen.
    const box = el("div", "position:fixed;inset:0;z-index:1700;display:flex;align-items:center;" +
        "justify-content:center;background:rgba(0,0,0,.92);cursor:zoom-out;");
    const img = el("img", "max-width:100%;max-height:100%;object-fit:contain;", { alt: ref.name, draggable: "false" });
    img.decoding = "async";
    img.src = viewUrl(ref);
    box.append(img);

    const close = () => {
        document.removeEventListener("keydown", onKey, true);
        box.remove();
    };
    const onKey = (event) => {
        if (event.key !== "Escape") return;
        event.preventDefault();
        event.stopPropagation();
        close();
    };
    box.addEventListener("click", close);
    document.addEventListener("keydown", onKey, true);
    document.body.append(box);
}

function bindDrop(st, node) {
    // An OS file drag carries only "Files". A drag of an image that the page itself rendered
    // (a thumbnail) also carries the url, and must never be taken for an import.
    const isImport = (event) => {
        const types = event.dataTransfer?.types;
        return !!types?.includes("Files") && !types.includes("text/uri-list");
    };
    const highlight = (on) => {
        st.root.style.outline = on ? `1px solid ${HI_INK}` : "none";
    };
    st.root.addEventListener("dragenter", (event) => {
        if (!isImport(event)) return;
        event.preventDefault();
        event.stopPropagation();
        highlight(true);
    });
    st.root.addEventListener("dragover", (event) => {
        // Cancelled even without files so the browser never opens a dropped url in its place.
        event.preventDefault();
        if (!isImport(event)) return;
        event.stopPropagation();
    });
    st.root.addEventListener("dragleave", (event) => {
        event.preventDefault();
        event.stopPropagation();
        highlight(false);
    });
    st.root.addEventListener("drop", (event) => {
        const field = event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement;
        if (field && !isImport(event)) return; // dragging selected text inside a field stays native
        event.preventDefault();
        event.stopPropagation();
        highlight(false);
        if (!isImport(event) || !event.dataTransfer.files.length) return;
        // Onto the character grid means a new character; onto the reference grid joins that one.
        const ontoCharacters = event.target?.closest?.("[data-drop='characters']") &&
            !event.target?.closest?.("[data-drop='refs']");
        const into = ontoCharacters ? characterFrom(event.dataTransfer.files[0]) : characterName(node) || undefined;
        uploadFiles(st, node, event.dataTransfer.files, "drop", into);
    });
}

// The panel edits the selected character or the selected reference of it.
function renderPanel(st, node, force) {
    const character = selectedCharacter(st, node);
    const target = editTarget(st, node);
    const key = editorKey(st, node);
    const name = menuLevel(st) === "character" ? (character?.name ?? "") : (refOf(character, st.ref)?.name.replace(/\.[^.]+$/, "") ?? "");
    const stored = menuLevel(st) === "character" ? character?.prompt ?? "" : refOf(character, st.ref)?.prompt ?? "";
    // Do not touch the text while it is being typed, unless the file underneath it changed.
    const typing = document.activeElement === st.promptArea && st.shownFor === key;
    if ((force || st.shownFor !== key) && !typing) {
        st.nameInput.value = name;
        st.promptArea.value = stored;
        st.shownText = stored;
        st.shownFor = target || stored ? key : null;
    }
    const ref = menuLevel(st) === "character"
        ? (refOf(character, character?.portrait) ?? character?.refs?.[0] ?? null)
        : refOf(character, st.ref);
    st.previewRef = ref?.name ?? "";
    const src = ref ? viewUrl(ref) : "";
    if (src !== st.previewKey) {
        st.previewKey = src;
        st.preview.src = src;
        st.preview.alt = ref?.name ?? "";
    }
    const onReference = menuLevel(st) === "reference";
    const strength = Number(ref?.strength ?? 1);
    st.strengthRow.style.display = onReference ? "flex" : "none";
    // Dragging the slider must not be fought by a repaint that arrives mid-drag.
    if (onReference && document.activeElement !== st.strength) {
        st.strength.value = String(strength);
        st.strengthValue.textContent = strength.toFixed(2);
    }
    const placeholder = menuLevel(st) === "character"
        ? "global description, used only for images with empty prompts, optional"
        : "image prompt/description, optional";
    if (st.promptArea.placeholder !== placeholder) st.promptArea.placeholder = placeholder;
    const show = st.open && !!target;
    if (st.editing !== show) {
        st.editing = show;
        st.panel.style.display = show ? "flex" : "none";
        st.grids.style.display = show ? "none" : "flex";
        st.filterRow.style.display = show ? "none" : "flex";
        st.gridKey = null;  // the tiles are back, so they have to be drawn again
    }
    paintBar(st, node);
    return show;
}

function setPanel(st, node, open) {
    const was = st.open;
    st.open = open && !!editTarget(st, node);
    if (was === st.open) return;
    if (!st.open) discard(st);
    renderPanel(st, node, st.open);
    paintBar(st, node);
    node.setDirtyCanvas?.(true, true);
    if (!st.open) return;
    if (!st.promptArea.disabled) {
        st.promptArea.focus?.();
        st.promptArea.setSelectionRange?.(st.promptArea.value.length, st.promptArea.value.length);
    }
}

// The grids and the edit panel replace the stock widgets, so keep those off the node.
function hideStockWidgets(node) {
    for (const name of ["character"]) {
        const widget = getWidget(node, name);
        if (!widget) continue;
        widget.hidden = true;
        const surfaces = widget.visibility?.surfaces;
        if (surfaces && surfaces.canvas !== "never") surfaces.canvas = "never";
    }
}

function sizeNode(node) {
    if (!node.computeSize || !node.setSize) return;
    const [w, h] = node.computeSize();
    node.setSize([Math.max(w, node.size[0], DEFAULT_NODE_W), Math.max(h, node.size[1])]);
}

// The canvas does not tell us about links, undo or workflow loads, so check them on every draw.
function maintain(st, node) {
    hideStockWidgets(node);
    if (!st.loaded) reload(st, node);
    paintGrids(st, node);
    renderPanel(st, node);
    paintBar(st, node);
}

// The canvas takes computeSize() as the floor while dragging a corner, so the minimum width lives
// there. Clamping node.size afterwards instead fights the drag and flickers.
function minWidth(node) {
    const computeSize = node.computeSize?.bind(node);
    if (!computeSize || node._ppmMinWidth) return;
    node._ppmMinWidth = true;
    node.computeSize = (out) => {
        const size = computeSize(out);
        size[0] = Math.max(size[0], MIN_NODE_W);
        return size;
    };
}

function buildUi(st, node) {
    st.root = el("div", "box-sizing:border-box;display:flex;flex-direction:column;gap:" + ROW_GAP + "px;" +
        `width:100%;height:100%;overflow:hidden;color:${INK};font:11px monospace;user-select:none;`);
    st.root.className = "ppm-ref-gallery";

    st.grids = el("div", "display:flex;flex-direction:column;gap:" + ROW_GAP + "px;flex:1 1 auto;min-height:0;");
    // Only one grid shows at a time, so each of them gets the whole room for its rows.
    st.charGrid = el("div", GRID + "flex:1 1 auto;", { "data-drop": "characters" });
    st.charGrid.className = "ppm-ref-scroll";
    st.charGrid.title = "Characters - drop images here, or paste while the node is selected, to make a new one";
    st.refGrid = el("div", GRID + "flex:1 1 auto;", { "data-drop": "refs" });
    st.refGrid.className = "ppm-ref-scroll";
    st.refGrid.title = "References of the selected character - drop images here, or paste while the node is selected, to add some";
    st.grids.append(st.charGrid, st.refGrid);

    // Only the editor holds the full size image, so the big view belongs to this preview alone.
    st.preview = el("img", `flex:0 0 auto;width:${PREVIEW}px;height:100%;min-height:${PREVIEW}px;` +
        `object-fit:contain;border:1px solid ${EDGE};border-radius:3px;background:${FILL};cursor:zoom-in;`,
        { alt: "", draggable: "false", title: "Click for full screen" });
    st.preview.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        showFull(st, node);
    });

    st.levelLabel = el("span", `flex:0 0 auto;color:${DIM};font:10px monospace;text-transform:uppercase;`);
    st.nameInput = el("input", FIELD + "flex:1 1 auto;", { type: "text", spellcheck: "false", title: "name" });
    st.nameInput.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        event.stopPropagation();
        renameTarget(st, node, st.nameInput.value);
    });
    st.renameButton = el("button", BTN, { type: "button", title: "Rename the character folder or this reference image" });
    st.renameButton.className = "ppm-ref-btn";
    st.renameButton.textContent = "Rename";
    st.renameButton.addEventListener("click", () => renameTarget(st, node, st.nameInput.value));

    const nameRow = el("div", "display:flex;gap:4px;align-items:center;flex:0 0 auto;");
    nameRow.append(st.levelLabel, st.nameInput, st.renameButton);

    // Strength belongs to a single reference, so this row only shows in the image editor.
    st.strength = el("input", SLIDER, {
        type: "range", min: "0", max: "2", step: "0.05",
        title: "How strongly this reference steers the sample - 0 drops it from the output",
    });
    st.strength.addEventListener("input", () => {
        st.strengthValue.textContent = Number(st.strength.value).toFixed(2);
    });
    st.strength.addEventListener("change", () => {
        saveState(st, node, selectedCharacter(st, node), { [st.ref]: { strength: Number(st.strength.value) } });
    });
    st.strengthValue = el("span", `flex:0 0 auto;width:32px;text-align:right;color:${DIM};font:11px monospace;`);
    const strengthLabel = el("span", `flex:0 0 auto;color:${DIM};font:10px monospace;text-transform:uppercase;`);
    strengthLabel.textContent = "strength";
    st.strengthRow = el("div", "display:flex;gap:4px;align-items:center;flex:0 0 auto;");
    st.strengthRow.append(strengthLabel, st.strength, st.strengthValue);

    // The stock class makes this a ComfyUI multiline prompt: it picks up the prompt font and the
    // spellcheck setting, and it is the hook text-area helpers such as Autocomplete Plus look for.
    st.promptArea = el("textarea", FIELD + "flex:1 1 auto;width:100%;min-height:48px;resize:none;" +
        "line-height:1.35;", {
        class: "comfy-multiline-input",
        spellcheck: String(!!app.ui?.settings?.getSettingValue?.("Comfy.TextareaWidget.Spellcheck", true)),
    });
    st.promptArea.addEventListener("input", () => paintBar(st, node));

    st.saveButton = el("button", BTN, { type: "button", title: "Write this prompt to its file" });
    st.saveButton.className = "ppm-ref-btn";
    st.saveButton.addEventListener("click", () => saveText(st, node, st.promptArea.value));

    const close = el("button", BTN, { type: "button", title: "Close the editor" });
    close.className = "ppm-ref-btn";
    close.textContent = "Close";
    close.addEventListener("click", () => setPanel(st, node, false));

    const actions = el("div", "display:flex;flex-wrap:wrap;gap:4px;align-items:center;flex:0 0 auto;");
    actions.append(st.saveButton, close);

    const column = el("div", "flex:1 1 auto;min-width:0;display:flex;flex-direction:column;gap:4px;");
    column.append(nameRow, st.promptArea, st.strengthRow, actions);

    st.panel = el("div", "display:none;gap:6px;align-items:stretch;flex:1 1 auto;min-height:0;overflow:hidden;");
    st.panel.append(st.preview, column);

    st.filterInput = el("input", FIELD + "flex:1 1 auto;width:100%;", {
        type: "text", spellcheck: "false", placeholder: "filter characters and references",
        title: "Filter by name or prompt - Enter picks the first match, Esc clears",
    });
    st.filterInput.addEventListener("input", () => {
        st.filter = st.filterInput.value;
        st.gridKey = null;
        paintGrids(st, node);
    });
    st.filterInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            event.stopPropagation();
            const first = visibleCharacters(st)[0];
            if (first) selectCharacter(st, node, first.name);
        } else if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();  // clearing the filter must not also leave the menu
            st.filter = st.filterInput.value = "";
            st.gridKey = null;
            paintGrids(st, node);
        }
    });

    st.filterRow = el("div", `display:flex;gap:${ROW_GAP}px;align-items:center;flex:0 0 auto;` +
        `min-height:${FILTER_H}px;`);
    st.filterRow.append(st.filterInput);

    st.fileInput = el("input", "display:none;", { type: "file", accept: "image/*", multiple: "true" });
    st.fileInput.addEventListener("change", () => {
        const files = [...st.fileInput.files];
        st.fileInput.value = "";
        uploadFiles(st, node, files, "drop");
    });

    // Every menu shows its own set, so all of them live here and showMenu picks the row.
    const bar = (label, title, onClick) => {
        const button = el("button", BTN, { type: "button", title });
        button.className = "ppm-ref-btn";
        button.textContent = label;
        button.addEventListener("click", onClick);
        return button;
    };

    st.add = bar("Add", "Add images as references of the selected character", () => st.fileInput.click());
    st.fresh = bar("New", "Add a character folder", () => newCharacter(st, node));
    st.refresh = bar("Reload", "Reload input/" + st.subfolder, () => reload(st, node));
    st.barBack = bar("Back", "Back to the characters (Esc leaves the menu)", () => showMenu(st, node, "character"));
    st.barRename = bar("Rename", "Rename this character and edit its prompt", () => setPanel(st, node, !st.open));
    st.barImages = bar("Images", "Edit the references of the selected character", () => showMenu(st, node, "image"));
    st.barEdit = bar("Edit", "Edit the selected reference", () => setPanel(st, node, !st.open));
    st.deleteButton = bar("Delete", "Delete this character or reference", () => removeTarget(st, node));
    st.barButtons = [st.barBack, st.refresh, st.fresh, st.barRename, st.barImages, st.add, st.barEdit, st.deleteButton];

    st.status = el("span", "flex:1 1 auto;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;" +
        `text-align:right;color:${DIM};`);
    st.statusRow = el("div", `display:flex;gap:4px;align-items:center;flex:0 0 auto;min-height:${STATUS_H}px;`);
    st.statusRow.append(st.status);

    st.bar = el("div", `display:flex;gap:4px;align-items:center;flex:0 0 auto;min-height:${BAR_H}px;`);
    st.bar.append(...st.barButtons);

    st.root.append(st.filterRow, st.grids, st.panel, st.bar, st.statusRow, st.fileInput);
    st.root.addEventListener("keydown", (event) => {
        // With the editor already closed, Esc leaves the image menu the same way Back does.
        if (event.key !== "Escape" || st.open || !imageMenu(st.menu)) return;
        event.preventDefault();
        event.stopPropagation();
        showMenu(st, node, "character");
    });
    bindWheel(st);
    bindDrop(st, node);
}

// The widget element is stacked above the canvas element, so a wheel only reaches the canvas when
// it is forwarded - the same rule the frontend's own DOM widgets follow.
function bindWheel(st) {
    const scroll = (event, grid) => {
        if (st.editing || Math.abs(event.deltaX) > Math.abs(event.deltaY)) return false;
        if (grid.scrollHeight <= grid.clientHeight) return false;
        grid.scrollTop += event.deltaY;
        return true;
    };
    st.root.addEventListener("wheel", (event) => {
        // Pinch or ctrl/meta zoom and a horizontal swipe always belong to the canvas; a text field
        // under the cursor keeps its own vertical scroll.
        const gesture = event.ctrlKey || event.metaKey || Math.abs(event.deltaX) > Math.abs(event.deltaY);
        if (!gesture && event.target?.closest?.("input,textarea")) return;
        event.preventDefault();
        event.stopPropagation();
        if (!gesture) {
            const grid = event.target?.closest?.(".ppm-ref-scroll") ?? st.refGrid;
            if (scroll(event, grid)) return;
        }
        app.canvas?.processMouseWheel(event);
    }, { passive: false });
}

function attach(node) {
    const previous = galleries.get(node);
    if (previous?.root.isConnected) return previous;

    const st = {
        characters: [], subfolder: "refs_ppm", loaded: false, menu: "character", view: "character", open: false,
        editing: false, filter: "", ref: "", shownFor: null, shownText: null,
        gridKey: null, previewKey: null, revealed: {},
    };
    galleries.set(node, st);
    buildUi(st, node);

    const widget = node.addDOMWidget(WIDGET_NAME, "ppm_reference_gallery", st.root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => WIDGET_MIN_H,
        onDraw: () => maintain(st, node),
    });
    widget.serialize = false;
    const onRemove = widget.onRemove?.bind(widget);
    widget.onRemove = () => {
        onRemove?.();
        if (galleries.get(node) === st) galleries.delete(node);
    };

    // The grids are the selectors and the panel is the text editor; the stock widget only carries
    // the selected character.
    hideStockWidgets(node);
    minWidth(node);
    sizeNode(node);
    showMenu(st, node, "character");
    reload(st, node);
    return st;
}

// A paste follows the menu that is showing, the same way a drop onto a grid does: the character
// menu makes a new character out of it - named after the clock, `Rename` gives it a real one -
// and the image menu adds the image to the selected character.
function pasteFiles(st, node, files) {
    if (imageMenu(st.menu)) return uploadFiles(st, node, files, "paste");
    return uploadFiles(st, node, files, "paste", `char_${stamp()}`);
}

// A paste goes to the node that owns the focused element, or to the only selected gallery node.
function pasteTarget() {
    const active = document.activeElement;
    for (const [node, st] of galleries) {
        if (active && st.root.contains(active)) return node;
    }
    const picked = [...galleries.keys()].filter(isSelected);
    return picked.length === 1 ? picked[0] : null;
}

app.registerExtension({
    name: "PPM.ReferenceGallery",
    setup() {
        ensureStyles();
        // Window capture runs ahead of the frontend's own paste handler on document, which stands
        // down as soon as the event is prevented; stopping the propagation keeps it that way.
        window.addEventListener("paste", (event) => {
            const files = event.clipboardData?.files;
            if (!files?.length || !imageFiles(files).length) return;
            const node = pasteTarget();
            if (!node) return;
            event.preventDefault();
            event.stopImmediatePropagation();
            pasteFiles(galleries.get(node), node, files);
        }, true);
    },
    nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS) return;
        if (node.graph) {
            attach(node);
            return;
        }
        // Runs from the node constructor, before the graph has this node.
        const onAdded = node.onAdded;
        node.onAdded = function (...args) {
            const result = onAdded?.apply(this, args);
            attach(this);
            return result;
        };
    },
});