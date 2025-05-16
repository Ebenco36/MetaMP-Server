import psycopg2
from typing import List, Tuple, Any
from database.db import db
from sqlalchemy import text

def execute_sql(sql: str):
    """Run only SELECTs via SQLAlchemy and return a list of dicts."""
    if not sql.lstrip().lower().startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")
    result = db.session.execute(text(sql))
    # SQLAlchemy RowMapping → dict
    return [dict(row) for row in result]