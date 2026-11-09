import json
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web

import folder_paths
import nodes
from comfy_api.latest import io

from .ref_cond import PPMRefCond

REFS_SUBFOLDER = "refs_ppm"
DATA_NAME = "ref_data.json"
PROMPT_KEY = "prompt"
DATA_LIMIT = 1 << 20
NAME_LIMIT = 120
STRENGTH_MIN = 0.0
STRENGTH_MAX = 2.0


def refs_dir() -> Path:
    return Path(folder_paths.get_input_directory()) / REFS_SUBFOLDER


def safe_name(name: object) -> str:
    text = str(name).strip()
    path = Path(text)
    if (
        not text
        or len(text) > NAME_LIMIT
        or text.startswith(".")
        or text == DATA_NAME
        or ".." in text
        or path.is_absolute()
        or path.drive
        or path.parts != (text,)
        or "\\" in text
        or any(ord(char) < 32 for char in text)
    ):
        raise ValueError(f"Invalid reference name: {name!r}")
    return text


def inside(path: Path) -> bool:
    return folder_paths.is_within_directory(str(refs_dir()), str(path))


def character_dir(name: object) -> Path:
    folder = refs_dir() / safe_name(name)
    if not inside(folder):
        raise ValueError(f"Character {name!r} points outside input/{REFS_SUBFOLDER}.")
    return folder


def is_plain(entry: Path, *, dirs: bool = False) -> bool:
    return (entry.is_dir() if dirs else entry.is_file()) and not entry.is_symlink() and inside(entry)


def plain_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [entry for entry in folder.iterdir() if is_plain(entry)]


def image_names(folder: Path) -> list[str]:
    return sorted(
        folder_paths.filter_files_content_types([entry.name for entry in plain_files(folder)], ["image"]), key=str.lower
    )


def read_text(path: Path) -> str:
    if not path.is_file() or not inside(path):
        return ""
    return path.read_text(encoding="utf-8-sig").strip()


def write_text(path: Path, text: object) -> None:
    if not inside(path):
        raise ValueError(f"Character text points outside input/{REFS_SUBFOLDER}.")
    path.write_text(str(text).strip() + "\n", encoding="utf-8")


def prompt_path(folder: Path, image: str) -> Path:
    return folder / Path(image).with_suffix(".txt")


def character_prompt(data: dict) -> str:
    prompt = data.get(PROMPT_KEY)
    return prompt.strip() if isinstance(prompt, str) else ""


@dataclass(frozen=True)
class Reference:
    name: str
    prompt: str
    enabled: bool
    strength: float
    mtime: int
    loose: bool


@dataclass(frozen=True)
class Character:
    name: str
    folder: Path
    prompt: str
    portrait: str
    loose: bool
    refs: tuple[Reference, ...]

    @property
    def subfolder(self) -> str:
        return REFS_SUBFOLDER if self.loose else f"{REFS_SUBFOLDER}/{self.name}"


def read_data(folder: Path) -> dict:
    path = folder / DATA_NAME
    if not path.is_file() or not inside(path) or path.stat().st_size > DATA_LIMIT:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def strength_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 1.0
    return min(max(float(value), STRENGTH_MIN), STRENGTH_MAX)


def ref_state(entry: dict) -> tuple[str, bool, float]:
    return str(entry.get("name", "")), bool(entry.get("enabled", True)), strength_value(entry.get("strength", 1.0))


def ref_entries(data: dict) -> list[dict]:
    entries = data.get("refs")
    return [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []


def write_data(folder: Path, prompt: str, portrait: object, refs: list[dict]) -> None:
    keep = set(image_names(folder))
    entries = [
        {"name": name, "enabled": enabled, "strength": strength}
        for name, enabled, strength in map(ref_state, refs)
        if name in keep
    ]
    portrait_name = str(portrait)
    body = {
        PROMPT_KEY: prompt.strip(),
        "portrait": portrait_name if portrait_name in keep else "",
        "refs": entries,
    }
    (folder / DATA_NAME).write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def ref_plan(data: dict, names: list[str]) -> list[tuple[str, bool, float]]:
    plan: dict[str, tuple[bool, float]] = {}
    for entry in ref_entries(data):
        name, enabled, strength = ref_state(entry)
        if name not in names or name in plan:
            continue
        plan[name] = (enabled, strength)
    for name in names:
        plan.setdefault(name, (True, 1.0))
    return [(name, enabled, strength) for name, (enabled, strength) in plan.items()]


def resave_data(
    folder: Path, prompt: str | None = None, renamed: tuple[str, str] | None = None, dropped: str = ""
) -> None:
    data = read_data(folder)
    if renamed:
        for entry in ref_entries(data):
            if entry.get("name") == renamed[0]:
                entry["name"] = renamed[1]
    portrait = data.get("portrait")
    portrait = portrait if isinstance(portrait, str) else ""
    if portrait == (renamed[0] if renamed else ""):
        portrait = renamed[1] if renamed else ""
    elif portrait == dropped:
        portrait = ""
    write_data(
        folder,
        character_prompt(data) if prompt is None else prompt,
        portrait,
        [
            {"name": name, "enabled": enabled, "strength": strength}
            for name, enabled, strength in ref_plan(data, image_names(folder))
        ],
    )


def loose_image(name: object) -> str:
    wanted = Path(safe_name(name)).stem.lower()
    for image in image_names(refs_dir()):
        if Path(image).stem.lower() == wanted:
            return image
    return ""


def folder_character(name: str, folder: Path) -> Character:
    names = image_names(folder)
    data = read_data(folder)
    stamps = {image: int((folder / image).stat().st_mtime) for image in names}
    refs = tuple(
        Reference(image, read_text(prompt_path(folder, image)), enabled, strength, stamps[image], False)
        for image, enabled, strength in ref_plan(data, names)
    )
    portrait = data.get("portrait")
    if not isinstance(portrait, str) or portrait not in names:
        portrait = next((ref.name for ref in refs if ref.enabled), names[0] if names else "")
    return Character(name, folder, character_prompt(data), portrait, False, refs)


def loose_character(image: str) -> Character:
    folder = refs_dir()
    prompt = read_text(prompt_path(folder, image))
    ref = Reference(image, prompt, True, 1.0, int((folder / image).stat().st_mtime), True)
    return Character(Path(image).stem, folder, prompt, image, True, (ref,))


def load_character(name: object) -> Character | None:
    text = str(name).strip()
    if not text:
        return None
    for candidate in dict.fromkeys((safe_name(text), Path(safe_name(text)).stem)):
        folder = character_dir(candidate)
        if folder.is_dir():
            return folder_character(candidate, folder)
        image = loose_image(candidate)
        if image:
            return loose_character(image)
    return None


def list_characters() -> list[Character]:
    folder = refs_dir()
    if not folder.is_dir():
        return []
    names = [entry.name for entry in folder.iterdir() if is_plain(entry, dirs=True)]
    for image in image_names(folder):
        stem = Path(image).stem
        if stem.lower() not in {name.lower() for name in names}:
            names.append(stem)
    characters = []
    for name in sorted(names, key=str.lower):
        try:
            character = load_character(name)
        except (ValueError, OSError):
            continue
        if character is not None:
            characters.append(character)
    return characters


def character_stamps(character: Character) -> list[int]:
    if character.loose:
        paths = [character.folder / ref.name for ref in character.refs]
        paths += [prompt_path(character.folder, ref.name) for ref in character.refs]
        return [path.stat().st_mtime_ns if path.is_file() else 0 for path in paths]
    if not character.folder.is_dir():
        return [0]
    stamps = [character.folder.stat().st_mtime_ns]
    stamps += [entry.stat().st_mtime_ns for entry in plain_files(character.folder)]
    return stamps


def load_images(character: Character) -> list[dict]:
    refs = []
    for ref in character.refs:
        if not ref.enabled or ref.strength <= 0.0:
            continue
        image, _ = nodes.LoadImage().load_image(f"{character.subfolder}/{ref.name}")
        refs.append({"image": image, "strength": ref.strength, "caption": ref.prompt or character.prompt})
    return refs


def as_dict(character: Character) -> dict:
    return {
        "name": character.name,
        "subfolder": character.subfolder,
        "prompt": character.prompt,
        "portrait": character.portrait,
        "loose": character.loose,
        "refs": [
            {
                "name": ref.name,
                "subfolder": character.subfolder,
                "prompt": ref.prompt,
                "enabled": ref.enabled,
                "strength": ref.strength,
                "mtime": ref.mtime,
            }
            for ref in character.refs
        ],
    }


def payload() -> dict:
    return {"subfolder": REFS_SUBFOLDER, "characters": [as_dict(character) for character in list_characters()]}


def require_character(name: object) -> Character:
    character = load_character(name)
    if character is None:
        raise FileNotFoundError(f"No character '{str(name).strip()}' in input/{REFS_SUBFOLDER}.")
    return character


def require_image(character: Character, image: object) -> str:
    name = safe_name(image)
    if name not in {ref.name for ref in character.refs}:
        raise FileNotFoundError(f"No reference '{name}' in character '{character.name}'.")
    return name


def require_folder(character: Character) -> Path:
    if character.loose:
        raise ValueError(f"'{character.name}' is a flat card - move it into its own folder to do that.")
    return character.folder


def move_files(pairs: list[tuple[Path, Path]]) -> None:
    for source, target in pairs:
        if inside(source) and inside(target) and source.is_file():
            source.rename(target)


def add_character(name: object) -> str:
    folder = character_dir(name)
    if folder.exists() or loose_image(safe_name(name)):
        raise FileExistsError(f"Character '{folder.name}' already exists.")
    folder.mkdir()
    return folder.name


def rename_character(name: object, new_name: object) -> str:
    character = require_character(name)
    target = character_dir(new_name)
    if target == character.folder:
        return character.name
    if target.exists() or loose_image(target.name):
        raise FileExistsError(f"Character '{target.name}' already exists.")
    if not character.loose:
        character.folder.rename(target)
        return target.name
    image = require_image(character, character.portrait)
    moved = character.folder / (target.name + Path(image).suffix)
    move_files([
        (character.folder / image, moved),
        (prompt_path(character.folder, image), prompt_path(character.folder, moved.name)),
    ])
    return target.name


def rename_image(character: Character, image: object, new_name: object) -> str:
    folder = require_folder(character)
    name = require_image(character, image)
    stem = Path(str(new_name).strip()).with_suffix("")
    target = safe_name(f"{stem}{Path(name).suffix}")
    if target == name:
        return name
    path = folder / target
    if any(entry.name.lower() == target.lower() and entry.name != name for entry in folder.iterdir()):
        raise FileExistsError(f"Reference '{target}' already exists.")
    if not inside(path):
        raise ValueError(f"Reference '{target}' points outside input/{REFS_SUBFOLDER}.")
    move_files([
        (folder / name, path),
        (prompt_path(folder, name), prompt_path(folder, target)),
    ])
    resave_data(folder, renamed=(name, target))
    return target


def delete_character(name: object) -> None:
    character = require_character(name)
    if character.loose:
        folder = character.folder
        remove_files(*[path for ref in character.refs for path in (folder / ref.name, prompt_path(folder, ref.name))])
        return
    entries = list(character.folder.iterdir())
    if any(not is_plain(entry) for entry in entries):
        raise ValueError(f"Character '{character.name}' holds something that is not a reference - delete it by hand.")
    for entry in entries:
        entry.unlink()
    character.folder.rmdir()


def delete_image(character: Character, image: object) -> None:
    folder = character.folder
    name = require_image(character, image)
    remove_files(prompt_path(folder, name), folder / name)
    if not character.loose:
        resave_data(folder, dropped=name)


def remove_files(*paths: Path) -> None:
    for path in paths:
        if inside(path) and path.is_file():
            path.unlink()


class ReferenceGalleryPPM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ReferenceGalleryPPM",
            display_name="Reference Gallery (PPM)",
            category="image",
            search_aliases=["character reference", "character library"],
            description="",
            inputs=[
                io.Combo.Input(
                    "character",
                    options=[character["name"] for character in payload()["characters"]] or [""],
                    tooltip=(
                        "Character to output. Pick it in the character grid on the node. Uses folders in input/refs_ppm."
                    ),
                ),
                PPMRefCond.Input(
                    "ppm_ref_cond",
                    optional=True,
                    tooltip="Optional previous Reference Cond for chaining multiple references.",
                ),
            ],
            outputs=[PPMRefCond.Output(display_name="ppm_ref_cond")],
        )

    @classmethod
    def execute(cls, character: str = "", ppm_ref_cond: list | None = None, **kwargs) -> io.NodeOutput:
        refs = list(ppm_ref_cond) if ppm_ref_cond else []
        loaded = load_character(character)
        if loaded is not None:
            refs += load_images(loaded)
        return io.NodeOutput(refs)

    @classmethod
    def fingerprint_inputs(cls, character: str = "", **kwargs):
        loaded = load_character(character)
        return (character, *(character_stamps(loaded) if loaded else []))


def error_response(exc: Exception) -> web.Response:
    for error, status in ((ValueError, 400), (FileNotFoundError, 404), (FileExistsError, 409), (OSError, 500)):
        if isinstance(exc, error):
            message = f"Could not write the reference library: {exc}" if status == 500 else str(exc)
            return web.json_response({"error": message}, status=status)
    raise exc


def ok_response(**fields: object) -> web.Response:
    return web.json_response({**fields, **payload()})


async def handle_refs(request) -> web.Response:
    try:
        body = await request.json()
        action = body.get("action", "list")
        if action == "list":
            return ok_response()
        if action == "add_character":
            return ok_response(name=add_character(body.get("character", "")))
        if action == "save_character":
            character = require_character(body.get("character", ""))
            text = str(body.get("prompt", ""))
            if character.loose:
                write_text(prompt_path(character.folder, character.portrait), text)
            else:
                resave_data(character.folder, prompt=text)
            return ok_response(name=character.name)
        if action == "save_ref":
            character = require_character(body.get("character", ""))
            name = require_image(character, body.get("ref", ""))
            write_text(prompt_path(character.folder, name), body.get("prompt", ""))
            return ok_response(name=character.name, ref=name)
        if action == "rename_character":
            return ok_response(name=rename_character(body.get("character", ""), body.get("new_name", "")))
        if action == "rename_ref":
            character = require_character(body.get("character", ""))
            ref = rename_image(character, body.get("ref", ""), body.get("new_name", ""))
            return ok_response(name=character.name, ref=ref)
        if action == "save_data":
            character = require_character(body.get("character", ""))
            folder = require_folder(character)
            refs = [
                {"name": name, "enabled": enabled, "strength": strength}
                for name, enabled, strength in map(ref_state, ref_entries(body))
            ]
            write_data(folder, character_prompt(read_data(folder)), str(body.get("portrait", "")), refs)
            return ok_response(name=character.name)
        if action == "delete_character":
            delete_character(body.get("character", ""))
            return ok_response()
        if action == "delete_ref":
            delete_image(require_character(body.get("character", "")), body.get("ref", ""))
            return ok_response()
        raise ValueError(f"Unknown reference action: {action!r}")
    except (ValueError, OSError) as exc:
        return error_response(exc)


def init():
    from server import PromptServer

    routes = PromptServer.instance.routes  # pyrefly: ignore[missing-attribute]
    routes.post("/ppm/refs")(handle_refs)


NODES = [ReferenceGalleryPPM]
