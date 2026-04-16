import json
from pathlib import Path
from typing import Any, List


def _path(stage: str, key: str, base_dir: Path) -> Path:
    return base_dir / stage / f"{key}.json"


def save_checkpoint(stage: str, key: str, data: Any, base_dir: Path = Path("checkpoints")) -> None:
    p = _path(stage, key, base_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_checkpoint(stage: str, key: str, base_dir: Path = Path("checkpoints")) -> Any:
    return json.loads(_path(stage, key, base_dir).read_text())


def checkpoint_exists(stage: str, key: str, base_dir: Path = Path("checkpoints")) -> bool:
    return _path(stage, key, base_dir).exists()


def list_checkpoints(stage: str, base_dir: Path = Path("checkpoints")) -> List[str]:
    stage_dir = base_dir / stage
    if not stage_dir.exists():
        return []
    return [p.stem for p in stage_dir.glob("*.json")]
