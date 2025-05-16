from src.sql2db.views import SqlGenerator

def text_to_db(api):
    api.add_resource(SqlGenerator, '/generate_sql')