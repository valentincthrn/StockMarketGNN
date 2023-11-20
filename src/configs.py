import yaml
from pathlib import Path
from typing import List, Dict, Any, Union
from dataclasses import dataclass


DATE_FORMAT: str = "%Y%m%d"

MACRO_MAPPING = {
    "Risco-Brasil": "JPM366_EMBI366",
    "PIB": "BM12_PIB12",
    "Dolar": "BM12_ERC12",
    "Selic Over": "BM12_TJOVER12",
    "IPCA": "PRECOS12_IPCAG12",
}

# # TOP5 BANKS
# -  # It is the largest private bank in Brazil and has a significant presence in various countries.
# -  # Another major private-sector bank with a vast network of branches and services across the country.
# -  # It is the largest public bank in Brazil and has a strong presence both domestically and internationally.
# -  # The Brazilian unit of the global banking group Santander, it has become one of the main banks in Brazil.
# -  # It is a leading investment bank in Brazil and has expanded its retail banking operations in recent years.

# TOP5 RANDOM
# -  # Energy Sector - Petrobras is Brazil's state-controlled oil company and is a major player in the exploration, production, and refining of oil and natural gas.
# -  # Mining Sector - Vale is one of the world's largest producers of iron ore and nickel. They also produce copper, coal, manganese, and other minerals.
# -  # Beverage Sector - AmBev, or Companhia de Bebidas das Américas, is a major beverage company known for its beers, soft drinks, and other beverages. Brands under its umbrella include Skol, Brahma, and Antarctica.
# -  # Financial Services Sector - B3 is the main financial market infrastructure company in Brazil and offers services related to exchange markets, over-the-counter markets, and infrastructure for the issuance of financial assets.
# -  # Food Sector - JBS is one of the world's leading food industry companies with a vast portfolio that includes meat, poultry, lamb, and leather processing.


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
