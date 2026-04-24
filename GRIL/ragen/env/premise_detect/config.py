from typing import Optional, List, Dict
from dataclasses import dataclass, field

@dataclass
class PremiseDetectEnvConfig:
    """Configuration for FrozenLake environment"""
    # Map config
    ##修改数据部分
    dataset_path: str = "data/ragen/premise_detect/metamath"
    split: str = field(default="train")