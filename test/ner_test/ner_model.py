from flair.models import SequenceTagger
from src.mlp_framework.ner_masking.mask_query_ner_model import MaskQueryNER

dict_mapping = {
    "GPE": "country",
    "MONEY": "sales",
    "ORG": "name"
}
tagger = SequenceTagger.load("flair/ner-english-ontonotes")

query = "Toyota got 1 million dollars in Japan"

obj = MaskQueryNER(tagger,dict_mapping)
obj.mask_query(query)
