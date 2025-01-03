import pprint
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from langchain_core.prompts import PromptTemplate
from src.mlp_framework.matched_questions_sim_check.matched_questions_sim_check_base import MatchCheckerBase


class MatchCheckerSQL(MatchCheckerBase):
    """
    Parameters:
        hard_threshold(float): if it passes and match variable length, direct match
        soft_threshold(float): if it passes, put the most similar question and SQL pair in prompt
    """

    def __init__(self,
                 hard_threshold: float = 0.7,
                 soft_threshold: float = 0.6):
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold

    def checking(self, dict_input):
        d = dict_input['extracted_keyword']
        dict_input_flatten = {}
        for key, value in d.items():
            dict_input_flatten.update(value)
        similar_question = list(dict_input['matched_question_score'].keys())[0]
        list_similar_SQL_with_score = list(dict_input['matched_question_score'].values())[0]
        similar_SQL = list_similar_SQL_with_score[0]
        prompt_similar_SQL = PromptTemplate.from_template(similar_SQL)
        dict_output = {
            'original_query': dict_input['original_query'],
            'matched_question': similar_question,
            'matched_SQL': similar_SQL,
            'matched_keyword': dict_input_flatten,
            'hard_threshold': self.hard_threshold,
            'soft_threshold': self.soft_threshold,
            'pass_keyword': 0,
            'pass_hard_threshold': 0,
            'pass_soft_threshold': 0
        }
        if list_similar_SQL_with_score[1] > self.hard_threshold:
            dict_output.update({
                'pass_hard_threshold': 1,
                'pass_soft_threshold': 1
            })
            if len(list(set(prompt_similar_SQL.input_variables) - set(dict_input_flatten.keys()))) == 0:
                dict_output.update({
                    'pass_keyword': 1
                })
        else:
            if list_similar_SQL_with_score[1] < self.soft_threshold:
                similar_SQL = PromptTemplate.from_template("select * from Top_2000_Companies")
                dict_output.update({
                    'matched_SQL': similar_SQL
                })
            else:
                dict_output.update({
                    'pass_soft_threshold': 1
                })
        print("==============Output Dictionary==================")
        pprint.pp(dict_output)
        print("=================================================")
        return dict_output
