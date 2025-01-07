import numpy as np
from typing import List, Dict, Any
from abc import abstractmethod
from langchain.utils.math import cosine_similarity

class SqlScehmaBase:

    @abstractmethod
    def get_sim(self,
                query: str,
                top_k: int
                ) -> List[str]:
        """Return the list of top k similar items from question"""
        pass

    @abstractmethod
    def get_sim_columns(self,
                        matched_keyword: Dict[str, Any],
                        top_k: int = 3
                        ) -> Dict[str, Any]:
        """
        Args:
            matched_keyword (Dict[str]): extracted keywords from the question
            e.g.
            {'country': {},
            'sales': {},
            'company': {'company_01': 'Toyota'},
            'column': {'column_01': 'sales'}
            }
        Returns:
            output_dict (Dict[str, any]):
            e.g.
            {
             'original_query': "How's the total sales of Toyota?",
             'matched_question': 'Average {column_01} for {company_01}?',
             'matched_SQL': 'SELECT AVG({column_01}) FROM Top_2000_Companies WHERE company '
                            '= {company_01}',
             'matched_keyword': {'company_01': 'Toyota', 'column_01': 'sales'},
             'hard_threshold': 0.7,
             'soft_threshold': 0.6,
             'pass_keyword': 1,
             'pass_hard_threshold': 1,
             'pass_soft_threshold': 1
             'sql_schema': 'xxx'
             }
        """
        pass
