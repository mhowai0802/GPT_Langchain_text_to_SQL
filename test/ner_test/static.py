from src.common.sql_table import *
from src.mlp_framework.ner_masking.mask_query_static import MaskQueryStatic

query = "What are the Country?"

dict_masking = {
    'column': read_column_from_table(),
}

obj = MaskQueryStatic(dict_masking)

obj.mask_query(query)