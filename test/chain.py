import json
from src.common.sql_table import *
from operator import itemgetter
from flair.models import SequenceTagger
from langchain_huggingface import HuggingFaceEmbeddings
from src.mlp_framework.ner_masking.mask_query_static import MaskQueryStatic
from src.mlp_framework.ner_masking.mask_query_ner_model import MaskQueryNER
from src.mlp_framework.ner_masking.mask_query_combine import MaskQueryCombine
from src.mlp_framework.similarity_search.simsearch_masked_query import SimSearchMaskedQuery
from src.mlp_framework.matched_questions_sim_check.matched_questions_sim_check_cypher import MatchCheckerSQL
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

################################################################
dict_mapping = {
    "GPE": "country",
    "MONEY": "sales",
    "ORG": "company"
}
dict_masking = {
    'column': read_column_from_table(),
}
tagger = SequenceTagger.load("flair/ner-english-ontonotes")
query = "Total sales for Toyota?"

################################################################
obj = MaskQueryCombine(dict_masking, dict_mapping, tagger)
################################################################
with open('src/data_source/query_SQL.json', 'r') as file:
    data = json.load(file)

model_name = "sentence-transformers/all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

obj_sim = SimSearchMaskedQuery(data, embedding)
################################################################
obj_matchcheck_sql = MatchCheckerSQL()
################################################################
chain = (RunnableLambda(obj.mask_query)
         | RunnableParallel(
            {
                'matched_question_score': itemgetter('masked_query') | RunnableLambda(obj_sim.get_sim_score),
                'extracted_keyword': itemgetter('extracted_keyword'),
                'original_query': itemgetter('original_query'),
                'masked_query': itemgetter('masked_query')
            }
        )
         | RunnableLambda(obj_matchcheck_sql.checking)
         )
################################################################
chain.invoke(query)
