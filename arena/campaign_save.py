import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .entities import PlayerLoadout


@dataclass
class CampaignSave:
    level: int = 1
    score: int = 0
    loadout: PlayerLoadout = field(default_factory=PlayerLoadout)


def load_campaign(path: Path) -> CampaignSave:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != 1:
            return CampaignSave()
        loadout = data["loadout"]
        return CampaignSave(
            max(1, min(100, int(data["level"]))), max(0, int(data["score"])),
            PlayerLoadout(bool(loadout["bow"]), bool(loadout["pistol"]),
                          max(0, min(12, int(loadout["arrows"]))),
                          max(0, min(24, int(loadout["energy"])))),
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return CampaignSave()


def save_campaign(path: Path, state: CampaignSave) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"version": 1, "level": state.level, "score": state.score,
                                     "loadout": asdict(state.loadout)}, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    temporary.replace(path)
