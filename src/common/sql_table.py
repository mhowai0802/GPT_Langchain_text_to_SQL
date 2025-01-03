import sqlalchemy
import pymysql
from sqlalchemy import text

pymysql.install_as_MySQLdb()


def read_column_from_table():
    engine = sqlalchemy.create_engine('mysql://root:joniwhfe@localhost/Text2SQL')
    # Query to get column names
    query = """
            DESCRIBE Top_2000_Companies;
    """
    query = text(query)
    # Execute the query and fetch column names
    with engine.connect() as connection:
        result = connection.execute(query)
        result = result.fetchall()
        columns = [row[0] for row in result]

    return columns