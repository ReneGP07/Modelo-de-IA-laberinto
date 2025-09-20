
from pathlib import Path
import torch

def save_checkpoint(path: Path, Q, level_info, rewards_curve):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"Q": Q.cpu(), "level_info": level_info, "rewards_curve": rewards_curve}, path)
