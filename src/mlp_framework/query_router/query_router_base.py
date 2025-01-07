from abc import abstractmethod,ABC
from typing import Dict, Any

class queryRouterBase(ABC):
    @abstractmethod
    def llm_or_template(self, dict_with_sim_marks):
        """
        Route the question to the appropriate LLM or template for answering.
        Args:
            dict_with_sim_marks(Dict[str, Any]): the dictionary with hard/soft threshold passing or not and variables matching or not
        e.g.
            {
            'sim_or_variables_check':
            {
                'original_query': "How's the total sales of Toyota?",
                'matched_question': 'Average {column_01} for {company_01}?',
                'matched_SQL': 'SELECT AVG({column_01}) FROM Top_2000_Companies WHERE company = {company_01}',
                'matched_keyword': {'company_01': 'Toyota', 'column_01': 'sales'},
                'hard_threshold': 0.7,
                'soft_threshold': 0.6,
                'pass_keyword': 1,
                'pass_hard_threshold': 1,
                'pass_soft_threshold': 1
            },
            'sql_schema': ['Company', 'Country', 'Sales']
            }

        Returns:
            Dict[str, Any]: A dictionary containing the LLM/template response.
        """
