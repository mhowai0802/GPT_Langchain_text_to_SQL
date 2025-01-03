from typing import List, Dict, Any
import numpy as np
from src.mlp_framework.similarity_search.simsearch_base import SimSearchBase
from langchain.utils.math import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings

class SimSearchMaskedQuery(SimSearchBase):
    def __init__(self, query_SQL, embedding) -> None:
        self.embedding = embedding
        self.dict_query_SQL = query_SQL
        self.list_query_SQL = list(self.dict_query_SQL.keys())
        self.vectors_query_SQL = self.embedding.embed_documents(self.list_query_SQL)

    def get_sim(self,
                query: str,
                top_k: int = 3
                ) -> List[str]:
        query_vector = self.embedding.embed_query(query)
        sim = cosine_similarity([query_vector], self.vectors_query_SQL)[0]
        sorted_sim_i = np.argsort(sim)[::-1]
        best_match = {self.list_query_SQL[i]:self.dict_query_SQL[self.list_query_SQL[i]] for i in sorted_sim_i[0:top_k]}
        return best_match

    def get_sim_score(self,
                query: str,
                top_k: int = 1,
                threshold: int = 0.5
                ) -> List[str]:
        query_vector = self.embedding.embed_query(query)
        sim = cosine_similarity([query_vector], self.vectors_query_SQL)[0]
        sorted_sim_i = np.argsort(sim)[::-1]
        best_match = {self.list_query_SQL[i]:[self.dict_query_SQL[self.list_query_SQL[i]],sim[i]] for i in sorted_sim_i[0:top_k]}
        return best_match