import sqlite3
import pandas as pd
from config import DATA_PATH


# Connect to SQLite db
def connect_db(db_path):
    con = sqlite3.connect(db_path)
    return con


# Get the table name
def get_table_name(con):
    curr = con.cursor()
    curr.execute("SELECT name FROM sqlite_master WHERE type = 'table';")
    tables = curr.fetchall()
    for table in tables:
        return table[0]


# Load dataset from the selected table
def load_data(con):
    dataset_raw = pd.read_sql_query(f"SELECT * FROM {get_table_name(con)}", con)
    return dataset_raw


# Connect to the Database
con = connect_db(DATA_PATH)

# Load dataset
dataset_raw = load_data(con)

# Close connection after use
connect_db(DATA_PATH).close()

if __name__ == "__main__":
    print(dataset_raw.head())
