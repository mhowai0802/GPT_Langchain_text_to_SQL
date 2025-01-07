import pprint
from datetime import datetime
from typing import Any, Dict, List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.mlp_framework.query_router.query_router_base import queryRouterBase


class queryRouterSQL(queryRouterBase):

    def __init__(self, llm):
        self.llm = llm

    def llm_or_template(self, dict_with_sim_marks):
        prompt_sql = ''
        if dict_with_sim_marks['sim_or_variables_check']['pass_keyword'] == 1:
            generated_sql = self.gen_sql_by_template(dict_with_sim_marks)
        else:
            input_variable_prompt = {
                'date': datetime.today(),
                'query': dict_with_sim_marks['sim_or_variables_check']['original_query'],
                'schema': dict_with_sim_marks['sql_schema'],
                'sql': dict_with_sim_marks['sim_or_variables_check']['matched_SQL'],
                'custom_instructions': ''
            }
            prompt_sql = self.generate_prompt()
            chain = prompt_sql | self.llm | StrOutputParser()
            generated_sql = chain.invoke(input_variable_prompt)
            prompt_sql = prompt_sql.invoke(input_variable_prompt).text

        dict_with_sim_marks.update({
            "generated_sql": generated_sql,
            "sql_prompt": prompt_sql
        })

        pprint.pp(dict_with_sim_marks)

        return dict_with_sim_marks

    def generate_prompt(self):
        PROMPT_SQL = """You are an expert Cypher query generator.      
        Follow the instructions to generate a complete and valid SQL query to answer the user's question.      
        - You generate complete, broad SQL queries that answer users questions.     
        - You can only use tables in the schema to answer the question. You cannot use any system tables.     
        - Never translate or modify table or database names. Use them literally as you have received them in the schema.     
        {custom_instructions}          
        
        Today's date is: <date>{date}</date>     
        Here is the user's question: <question>{query}</question>.     
        Here is the relevant schema: <schema>{schema}</schema>.      
        Here is the reference sql: <sql>{sql}</sql>
        
        If the schema doesn't have the necessary labels or relations or properties to generate a SQL query to answer the question, simply explain so in between <thoughts></thoughts> tags.                  
        
        Follow the SQL rules and return the valid SQL query to answer the user's question in between <cypher></cypher> tags.     
        Do not use backticks like '```' or markdown in your response.      
        
        Limit your response to:     
        - In 200 characters or less, explain to the user directly, being specific about the tables/columns used (but not redundant), the SQL query you are going to generate, in between <thoughts></thoughts> tags.     
        - The SQL query in between <cypher></cypher> tags.     
        - Is the user asking explicitly for the SQL query? Answer <execute>0</execute> if yes, <execute>1</execute> if not."""

        prompt_question_to_sql = PromptTemplate.from_template(PROMPT_SQL)
        return prompt_question_to_sql

    def gen_sql_by_template(self, dict_with_sim_marks):
        prompt_matched_SQL = PromptTemplate.from_template(dict_with_sim_marks['sim_or_variables_check']['matched_SQL'])
        return prompt_matched_SQL.invoke(dict_with_sim_marks['sim_or_variables_check']['matched_keyword']).text