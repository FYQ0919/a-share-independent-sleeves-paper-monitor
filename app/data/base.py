from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import pandas as pd

from app.models import MarketIndex, Sector


class MarketDataProvider(ABC):
    name = "base"

    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, List[Sector], List[MarketIndex], Dict]:
        """Return stock snapshot, sectors, indices and market breadth."""
        raise NotImplementedError

