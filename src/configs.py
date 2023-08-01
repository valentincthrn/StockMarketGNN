import yaml
from pathlib import Path
from typing import List
from dataclasses import dataclass


DATE_FORMAT = "%Y%m%d"


@dataclass
class RunConfiguration:
    targets_to_ingest: List[str]
    targets_to_predict: List[str]

    @classmethod
    def from_yaml(cls, yml_path: Path):
        with open(yml_path, "r") as fp:
            result = yaml.safe_load(fp)
        return cls(**result)
