import sqlite3
import pprint

try:
    conn = sqlite3.connect('data/inventory.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:", tables)
    for t in tables:
        name = t[0]
        cursor.execute(f"PRAGMA table_info({name})")
        print(f"Table {name} info:", cursor.fetchall())
except Exception as e:
    print("Error:", e)
