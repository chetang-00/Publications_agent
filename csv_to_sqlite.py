import pandas as pd
import sqlite3
import os

# File paths
csv_path = os.path.join('data', 'papers_with_cluster_labels.csv')
db_path = './publications.db'
table_name = 'publications'

# Columns to keep 
columns_to_keep = [
    'authors',
    'title',
    'year',
    'abstract',
    'author_keywords',
    'index_keywords',
    'authors_with_affiliations',
    'cluster_label'
]

def main():
    # Read CSV, only keep specified columns
    df = pd.read_csv(csv_path, usecols=lambda c: c in columns_to_keep)

    # Connect to SQLite database (creates if not exists)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Drop table if exists
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")

    # Create table with id as primary key
    cur.execute(f'''
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            authors TEXT,
            title TEXT,
            year INTEGER,
            abstract TEXT,
            author_keywords TEXT,
            index_keywords TEXT,
            authors_with_affiliations TEXT,
            cluster_label TEXT
        )
    ''')

    # Insert data (without id, so SQLite auto-generates it)
    df.to_sql(table_name, conn, if_exists='append', index=False)

    print(f"Data from {csv_path} stored in {db_path} (table: {table_name}) with id as primary key and columns: id, {', '.join(columns_to_keep)}")
    conn.close()

if __name__ == '__main__':
    main() 