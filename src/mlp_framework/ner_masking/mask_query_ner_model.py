import re
import pprint
from flair.data import Sentence
from src.common.sql_table import read_column_from_table
from src.mlp_framework.ner_masking.mask_query_base import MaskQueryBase

class MaskQueryNER(MaskQueryBase):

    def __init__(self, ner_model, mapping_dict):
        """
        Args: keyword_dict (Dict[List[str]])
        """
        self.ner_model = ner_model
        self.mapping_dict = mapping_dict


    def ner_model_transformation(self,query):
        sentence = Sentence(query)
        self.ner_model.predict(sentence)
        mapping_dict = self.mapping_dict
        dict_master = {}
        for key, value in mapping_dict.items():
            counter = 1
            dict_sub = {}
            for entity in sentence.get_spans('ner'):
                if entity.tag == key:
                    dict_sub[f"{mapping_dict[entity.tag]}_{counter:02d}"] = entity.text
            dict_master[value] = dict_sub
        return dict_master

    def mask_query(self,query):
        """
        Args:
            query (str): the question from user

        Returns:
            Dict[str, any]:
            e.g.
            {
                "original_query": "What is the average score from uber users?",
                "masked_query": "What is the average {column_01} from uber users?",
                "extracted_keyword": e.g.
                {
                    "Name": {"name_01": xxx_Bank},
            }
        """
        extracted_keyword = self.ner_model_transformation(query)
        output_dict = {
            "original_query": query,
            "extracted_keyword": extracted_keyword
        }
        print("==============NER model masking==================")
        pprint.pprint(output_dict)
        print("==============================================")
        return output_dict

