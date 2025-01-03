from langchain.utils.math import cosine_similarity
from abc import abstractmethod
from typing import List, Dict, Any
import numpy as np

class SimSearchBase:

    @abstractmethod
    def get_sim(self,
                query: str,
                top_k: int
                ) -> List[str]:
        """Return the list of top k similar items from question"""
        """['I have fever']"""
        pass

    @abstractmethod
    def get_sim_score(self,
                      query: str,
                      top_k: int,
                      threshold: float
                      ) -> List[Dict[str,Any]]:
        """Return the list of top k similar items with score from question"""
        """[{'I have fever': 0.9}]"""
        pass
