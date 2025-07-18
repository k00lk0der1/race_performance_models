import os
import sys

def get_sql_query(sql_query_filename):
    sql_queries_folder_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        "sql_queries"
    )

    sql_queries_file_list = os.listdir(sql_queries_folder_path)
    
    if(sql_query_filename not in sql_queries_file_list):
        raise ValueError(f"{sql_query_filename} file not found in {sql_queries_folder_path}")
    
    sql_query = open(
        os.path.join(
            sql_queries_folder_path,
            sql_query_filename
        ),
        "r"
    ).read().strip()

    return sql_query