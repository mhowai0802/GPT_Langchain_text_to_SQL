from abc import abstractmethod, ABC
from typing import Dict

class MaskQueryBase(ABC):

    @abstractmethod
    def mask_query(self,
                   query:str
                   ) -> Dict[str, any]:
        """
        Args:
            query (str): the question from user
        Returns:
            output_dict (Dict[str, any]):
            e.g.
            {
                "original_query": "What do you want to",
                "masked_query" [Optional]: "What do {person} want to",
                "extracted_keyword": e.g.
                {
                        "column":{"column_01" : "score"},
                        "score" : {"score_01" : 1}
                }
            }

        """

        pass