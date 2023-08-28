import typing

from src.model.base import _BaseEXP
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
            # "NUBR33.SA", recent
            "ITUB4.SA",
            "ABCB4.SA",
            # "RPAD5.SA",
            # "BRIV4.SA", flat
            # "BAZA3.SA",
            # recent"BMGB4.SA", recent
            # "BPAN4.SA",
            # "BGIP4.SA",
            "B3SA3.SA",
            "BEES3.SA",
            "BRSR6.SA",
            "BBDC4.SA",
            "BBAS3.SA",
            # "BSLI4.SA",
            # "BPAC11.SA",
            "INBR32.SA",
            # "BMEB4.SA",
            # "BMIN4.SA",
            # "BNBR3.SA",
            "SANB11.SA",
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

    @property
    def _exp_model(self) -> typing.Dict[str, bool]:
        return {
            "nbr_gcn_hidden": 0,
            "nbr_mlp_hidden": 0,
            "has_dropout": False,
        }
