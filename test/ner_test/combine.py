from src.common.sql_table import *
from flair.models import SequenceTagger
from src.mlp_framework.ner_masking.mask_query_static import MaskQueryStatic
from src.mlp_framework.ner_masking.mask_query_ner_model import MaskQueryNER
from src.mlp_framework.ner_masking.mask_query_combine import MaskQueryCombine
################################################################
dict_masking = {
    'column': read_column_from_table(),
}
dict_mapping = {
    "GPE": "country",
    "MONEY": "sales",
    "ORG": "company"
}

tagger = SequenceTagger.load("flair/ner-english-ontonotes")
query = "What is the company got 1 million dollars in Japan"
################################################################
obj = MaskQueryCombine(dict_masking, dict_mapping, tagger)
obj.mask_query(query)
