import typing
from enum import Enum

from src.model.base import _BaseEXP
from src.experiments.simple import SimpleEXP


class EXP(Enum):
    SIMPLE = "simple"


EXP_OBJECTS: typing.Dict[EXP, typing.Type[_BaseEXP]] = {EXP.SIMPLE: SimpleEXP}
