"""Runs an sd-scripts training script with train_network.py's resume step counting fixed (launched by accelerate
from the engine checkout as the working directory: sd_resume.py <script> <script args...>).

sd-scripts with --skip_until_initial_step skips whole epochs and then sets global_step to the remainder inside the
epoch, so a state saved after the first epoch restarts the step count, trains more steps than set and writes over
the earlier step files.  The patch below keeps the step from train_state.json and skips only the remainder.

Adapted from MOSS-LoRA-Trainer v0.1.2 moss_trainer/resume_bridge.py (MIT License, Copyright (c) 2026 suigin10,
https://github.com/suigin10/MOSS-LoRA-Trainer); the LR-recalculation and state-retention parts are left out.
Only the source text is changed in memory; the engine checkout is never modified.  If the engine's code no longer
has the expected shape, the run stops before training instead of resuming with a wrong step count."""
from __future__ import annotations

import ast
import importlib.util
import os
import runpy
import sys
from pathlib import Path


def patch_source(source: str) -> str:
    tree = ast.parse(source)
    trainer = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NetworkTrainer"), None)
    train = next((n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "train"), None) if trainer else None
    if train is None:
        raise RuntimeError("再開の補正ができません: 学習エンジンの NetworkTrainer.train が見つかりません")
    lines = source.splitlines(keepends=True)
    edits = []

    def replace(node, text):
        edits.append((node.lineno - 1, node.end_lineno, text))

    def assign(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)

    start = next((i for i, n in enumerate(train.body) if assign(n, "epoch_to_start")), None)
    if start is None or start + 2 >= len(train.body) or not assign(train.body[start + 2], "global_step"):
        raise RuntimeError("再開の補正ができません: 学習エンジンの再開位置の処理が変わっています")
    old = train.body[start + 1]
    if not isinstance(old, ast.If) or "skip_until_initial_step" not in ast.unparse(old):
        raise RuntimeError("再開の補正ができません: 学習エンジンのバッチを飛ばす処理が変わっています")
    edits.append((train.body[start].lineno - 1, train.body[start + 2].end_lineno,
                  "        resume_global_step = initial_step\n"
                  "        epoch_to_start = resume_global_step // num_update_steps_per_epoch\n"
                  "        initial_step = ((resume_global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps\n"
                  "                        if args.skip_until_initial_step else 0)\n"
                  "        global_step = resume_global_step\n"
                  "        current_step.value = global_step\n"))
    skip = [n for n in train.body if isinstance(n, ast.If)
            and any(assign(x, "global_step") for x in n.body) and "skip_epoch" in ast.unparse(n)]
    if len(skip) != 1:
        raise RuntimeError("再開の補正ができません: 学習エンジンのステップ数の処理が変わっています")
    replace(skip[0], "")
    progress = [n for n in train.body if assign(n, "progress_bar")]
    if len(progress) != 1 or "args.max_train_steps - initial_step" not in ast.unparse(progress[0]):
        raise RuntimeError("再開の補正ができません: 学習エンジンの進捗表示の処理が変わっています")
    replace(progress[0], '        progress_bar = tqdm(total=args.max_train_steps, initial=resume_global_step, smoothing=0, '
                         'disable=not accelerator.is_local_main_process, desc="steps")\n')
    hook = next((n for n in train.body if isinstance(n, ast.FunctionDef) and n.name == "save_model_hook"), None)
    dumps = [n for n in ast.walk(hook) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and ast.unparse(n.value.func) == "json.dump"] if hook else []
    if len(dumps) != 1 or "current_step.value + 1" not in ast.unparse(dumps[0]):
        raise RuntimeError("再開の補正ができません: 学習エンジンの再開用データの保存処理が変わっています")
    replace(dumps[0], '                json.dump({"current_epoch": current_epoch.value, "current_step": global_step}, f)\n')
    for log in [n for n in hook.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and ast.unparse(n.value.func) == "logger.info" and "current_step.value + 1" in ast.unparse(n)]:
        replace(log, '            logger.info(f"save train state to {train_state_file} at epoch {current_epoch.value} step {global_step}")\n')
    for begin, end, text in sorted(edits, reverse=True):
        lines[begin:end] = [text]
    result = "".join(lines)
    compile(result, "<sd-resume>", "exec")
    return result


def fix_train_network(engine: Path) -> None:
    """Load the engine's train_network.py with the fix into sys.modules (the engine folder must be on sys.path)."""
    path = engine / "train_network.py"
    source = patch_source(path.read_text(encoding="utf-8-sig"))
    spec = importlib.util.spec_from_file_location("train_network", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_network"] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    print("LoRAtelier: resume step count fixed (sd_resume.py)", flush=True)


def main():
    engine = Path(os.getcwd()).resolve()
    script = sys.argv.pop(1)
    # this file's folder must not shadow the engine's modules
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(engine))
    fix_train_network(engine)
    sys.argv[0] = str(engine / script)
    runpy.run_path(str(engine / script), run_name="__main__")


if __name__ == "__main__":
    main()
