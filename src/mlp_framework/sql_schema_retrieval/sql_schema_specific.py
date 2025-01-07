import numpy as np
from typing import List, Dict, Any
from langchain.utils.math import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.mlp_framework.sql_schema_retrieval.sql_schema_base import SqlScehmaBase


class SqlScehmaSpecific:

    def __init__(self, mysqldb, embedding) -> None:
        list_columns = []
        cur = mysqldb.cursor()
        cur.execute("SHOW COLUMNS FROM Top_2000_Companies;")
        results = cur.fetchall()
        for row in results:
            list_columns.append(row[0])
        self.embedding = embedding
        self.list_columns = list_columns
        self.vectors_list_columns = self.embedding.embed_documents(list_columns)

    #
    def get_sim(self,
                query: str,
                top_k: int = 3
                ) -> List[str]:
        query_vector = self.embedding.embed_query(query)
        sim = cosine_similarity([query_vector], self.vectors_list_columns)[0]
        sorted_sim_i = np.argsort(sim)[::-1]
        best_match = [self.list_columns[i] for i in sorted_sim_i[0:top_k]]
        return best_match

    def get_sim_columns(self,
                        matched_keyword: Dict[str, Any],
                        top_k: int = 3
                        ) -> Dict[str, Any]:
        return self.list_columns
