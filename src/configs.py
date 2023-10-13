import yaml
from pathlib import Path
from typing import List, Dict, Any, Union
from dataclasses import dataclass


DATE_FORMAT = "%Y%m%d"


@dataclass
class RunConfiguration:
    ingest: Dict[str, Any]
    data_prep: Dict[str, int]
    hyperparams: Dict[str, Union[int, str]]

    @classmethod
    def from_yaml(cls, yml_path: Path):
        with open(yml_path, "r") as fp:
            result = yaml.safe_load(fp)
        return cls(**result)
