import os
import sqlite3
import pandas as pd
from portkey_ai import Portkey
from dotenv import load_dotenv
import re
from sql_agent_core import ask_llm_for_sql, run_sql_query

# Load environment variables from .env
load_dotenv()

# Read Portkey API keys from environment
PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY", "")
PORTKEY_VIRTUAL_KEY = os.getenv("PORTKEY_VIRTUAL_KEY", "")
PORTKEY_BASE_URL = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/"

# Initialize Portkey client
client = Portkey(
    api_key=PORTKEY_API_KEY,
    virtual_key=PORTKEY_VIRTUAL_KEY,
    base_url=PORTKEY_BASE_URL
)

db_path = 'publications.db'
table_name = 'publications'

# Table schema for prompt context
schema = '''
Table: publications
Columns:
- authors (TEXT)
- title (TEXT)
- year (INTEGER)
- abstract (TEXT)
- author_keywords (TEXT)
- index_keywords (TEXT)
- cluster_label (TEXT)
'''

def main():
    print("Ask a question about the publications database (type 'exit' to quit):")
    while True:
        question = input("\nYour question: ")
        if question.lower() in ['exit', 'quit']:
            break
        try:
            sql = ask_llm_for_sql(question)
            print(f"\nGenerated SQL:\n{sql}")
            result, error = run_sql_query(sql)
            print("\nResult:")
            if error:
                print(error)
            else:
                print(result.head(10).to_string(index=False))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main()