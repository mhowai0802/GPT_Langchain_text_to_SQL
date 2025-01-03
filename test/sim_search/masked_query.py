import json
from transformers import AutoModel
from langchain_huggingface import HuggingFaceEmbeddings
from src.mlp_framework.similarity_search.simsearch_masked_query import SimSearchMaskedQuery

with open('src/data_source/query_SQL.json', 'r') as file:
    data = json.load(file)

list_masked_query = list(data.keys())

model_name = "sentence-transformers/all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

obj_sim = SimSearchMaskedQuery(data, embedding)
print(obj_sim.get_sim_score('average sales?'))

