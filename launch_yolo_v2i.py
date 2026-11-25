"""Start the BLOCKADE preview with the Roboflow v2i checkpoint."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


PROJECT = Path(r"C:\Users\Dartsash\Desktop\Blockade AI")
MODEL_OUTPUTS = Path(r"C:\Users\Dartsash\Documents\Codex\2026-11-19\c-users-dartsash-desktop-blockade-ai-9\outputs")
WEIGHTS_NAME = "blockade_player_yolo26n_roboflow_v2i.pt"


def main():
    script = PROJECT / "preview_yolo.py"
    if not script.is_file():
        raise SystemExit(f"Не найден скрипт предпросмотра: {script}")

    spec = spec_from_file_location("blockade_preview", script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Не удалось открыть скрипт предпросмотра: {script}")
    preview = module_from_spec(spec)
    spec.loader.exec_module(preview)

    # A moved launcher still uses the model and runtime from the training workspace.
    preview.HERE = MODEL_OUTPUTS
    local_weights = Path(__file__).resolve().with_name(WEIGHTS_NAME)
    weights = local_weights if local_weights.is_file() else MODEL_OUTPUTS / WEIGHTS_NAME
    sys.argv[1:1] = ["--weights", str(weights), "--imgsz", "512"]
    preview.main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
