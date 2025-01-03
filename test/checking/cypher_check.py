from src.mlp_framework.matched_questions_sim_check.matched_questions_sim_check_cypher import MatchCheckerCypher

dict_input = {'matched_question_score': {'Average {column_01} for {company_01}?': [
    'SELECT AVG({column_01}) FROM Top_2000_Companies WHERE company = {company_01}', 0.7492524930056212]},
              'extracted_keyword': {'country': {}, 'sales': {}, 'company': {'company_01': 'Toyota'},
                                    'column': {'column_01': 'sales'}}, 'original_query': 'Total sales for Toyota?',
              'masked_query': 'Total {column_01} for {company_01}?'}

obj_checking = MatchCheckerCypher()
obj_checking.checking(dict_input)
