import pprint
from typing import Dict
from src.mlp_framework.ner_masking.mask_query_base import MaskQueryBase
from src.mlp_framework.ner_masking.mask_query_static import MaskQueryStatic
from src.mlp_framework.ner_masking.mask_query_ner_model import MaskQueryNER


class MaskQueryCombine():

    def __init__(self, dict_masking,dict_mapping, tagger):
        """
        Args:
            dict_static(Dict[Dict[str]]) : dictionary from static
            dict_ner(Dict[Dict[str]]) : dictionary from ner
        """
        self.obj_static = MaskQueryStatic(dict_masking)
        self.obj_ner = MaskQueryNER(tagger, dict_mapping)

    def mask_query(self,query) -> Dict[str, any]:
        dict_static = self.obj_static.mask_query(query)
        dict_ner = self.obj_ner.mask_query(query)
        self.dict_static = dict_static
        self.dict_ner = dict_ner
        pprint.pprint(self.dict_static)
        pprint.pprint(self.dict_ner)
        pprint.pprint(self.dict_static['masked_query'])
        masked_query = self.dict_static['masked_query']
        dict_keyword_ner = self.dict_ner['extracted_keyword']
        for item in dict_keyword_ner:
            print(item)
            for column_index, column_value in dict_keyword_ner[item].items():
                masked_query = masked_query.replace(column_value, f'{{{column_index}}}')

        dict_keyword_ner.update(self.dict_static['extracted_keyword'])

        output_dict = {
            "original_query": self.dict_static['original_query'],
            "masked_query": masked_query,
            "extracted_keyword": dict_keyword_ner
        }
        print("==============Static masking==================")
        pprint.pprint(output_dict)
        print("==============================================")

        return output_dict