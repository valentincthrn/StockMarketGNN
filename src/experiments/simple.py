import typing

from model.base import _BaseEXP
from src.configs import RunConfiguration
from src.utils.db import DBInterface


class SimpleEXP(_BaseEXP):
    def __init__(
        self,
        config: RunConfiguration,
        db: DBInterface,
    ) -> None:
        super().__init__(config=config, db=db)

    @property
    def _targets(self) -> typing.List[str]:
        return [
            "RRRP3.SA",
            "BPAC11.SA",
            "ENEV3.SA",
            "B3SA3.SA",
            "PETR4.SA",
            "GOLL4.SA",
            "EZTC3.SA",
            "PETZ3.SA",
        ]

    @property
    def _encodings(self) -> typing.Optional[str]:
        return None

    @property
    def _features(self) -> typing.Dict[str, bool]:
        return {
            "close price": True,
            "technicals": False,
            "technicals + fundamentalist": False,
            "technicals + fundamentalist + economics": False,
        }
