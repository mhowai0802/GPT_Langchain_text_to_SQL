import mysql.connector
from langchain_huggingface import HuggingFaceEmbeddings
from src.mlp_framework.sql_schema_retrieval.sql_schema_specific import SqlScehmaSpecific

mysqldb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="joniwhfe",
  database="Text2SQL"
)

model_name = "sentence-transformers/all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

obj_sqlschema = SqlScehmaSpecific(mysqldb, embedding)

input = {'matched_question_score': {'Average {column_01} for {company_01}?': [
    'SELECT AVG({column_01}) FROM Top_2000_Companies WHERE company = {company_01}', 0.7323916400298288]},
         'extracted_keyword': {'country': {}, 'sales': {}, 'company': {'company_01': 'Toyota'},
                               'column': {'column_01': 'sales'}}, 'original_query': "How's the total sales of Toyota?",
         'masked_query': "How's the total {column_01} of {company_01}?"}


obj_sqlschema.get_sim_columns(input['extracted_keyword'])

