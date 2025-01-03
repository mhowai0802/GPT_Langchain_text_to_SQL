from abc import abstractmethod
from typing import Any,Dict,List

class MatchCheckerBase:
    @abstractmethod
    def check_match(self,
                  query: str,
                  keyword: str
                  ) -> Dict[str,Any]:
        """"
            check whether the score would be larger than hard / soft threshold,
            and the extracted keywords match the matched question
        """
        pass