import sqlite3
import json
from contextlib import contextmanager

@contextmanager
def get_db():
    conn = sqlite3.connect('fashion.db')
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_outfits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            top_image TEXT,
            top_description TEXT,
            bottom_image TEXT,
            bottom_description TEXT,
            score REAL,
            prompt TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()

def save_outfit(outfit):
    with get_db() as conn:
        cursor = conn.cursor()
        # Use .get() for safer access, providing None or an empty string as default
        top_item = outfit.get("top", {})
        bottom_item = outfit.get("bottom", {})

        cursor.execute('''
        INSERT INTO saved_outfits 
        (top_image, top_description, bottom_image, bottom_description, score, prompt)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            top_item.get("image_path"),  # Default to None if key missing
            top_item.get("product_name"), # Default to None if key missing
            bottom_item.get("image_path"), # Default to None if key missing
            bottom_item.get("product_name"),# Default to None if key missing
            outfit.get("score"),          # Default to None if key missing
            outfit.get("prompt")          # Default to None if key missing
        ))
        conn.commit()

def get_saved_outfits():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM saved_outfits ORDER BY created_at DESC')
        columns = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        return [dict(zip(columns, row)) for row in results]
    

def clear_saved_outfits():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM saved_outfits')
        conn.commit()