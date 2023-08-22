import yaml
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


DATE_FORMAT = "%Y%m%d"


@dataclass
class RunConfiguration:
    targets_to_ingest: List[str]
    data_prep: Dict[str, Any]
    model: Dict[str, int]

    @classmethod
    def from_yaml(cls, yml_path: Path):
        with open(yml_path, "r") as fp:
            result = yaml.safe_load(fp)
        return cls(**result)
