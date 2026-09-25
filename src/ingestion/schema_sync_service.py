import logging

from database.db import db
from sqlalchemy import inspect, text
from src.ingestion.model_registry import get_dataset_model_bindings

logger = logging.getLogger(__name__)


class SchemaSyncService:
    @staticmethod
    def _physical_identifier_name(connection, identifier_name):
        max_length = getattr(connection.dialect, "max_identifier_length", None)
        if max_length and len(identifier_name) > max_length:
            return identifier_name[:max_length]
        return identifier_name

    @staticmethod
    def _add_missing_columns(connection, table):
        inspector = inspect(connection)
        existing_columns = {column["name"] for column in inspector.get_columns(table.name)}
        existing_physical_names = {
            SchemaSyncService._physical_identifier_name(connection, column_name)
            for column_name in existing_columns
        }
        missing_columns = [
            column
            for column in table.columns
            if (
                column.name not in existing_columns
                and SchemaSyncService._physical_identifier_name(connection, column.name)
                not in existing_physical_names
                and not column.primary_key
            )
        ]
        if not missing_columns:
            return []

        preparer = connection.dialect.identifier_preparer
        quoted_table = preparer.quote(table.name)
        added_column_names = []

        for column in missing_columns:
            column_type_sql = column.type.compile(dialect=connection.dialect)
            quoted_column = preparer.quote(column.name)
            statement = text(
                f"ALTER TABLE {quoted_table} ADD COLUMN {quoted_column} {column_type_sql}"
            )
            logger.info(
                "Adding missing column %s.%s (%s)",
                table.name,
                column.name,
                column_type_sql,
            )
            connection.execute(statement)
            added_column_names.append(column.name)

        return added_column_names

    def sync(self):
        bindings = get_dataset_model_bindings()
        table_names = [binding.model_class.__table__.name for binding in bindings]
        sync_report = []

        logger.info(
            "Ensuring protein dataset tables exist from stable SQLAlchemy models: %s",
            ", ".join(table_names),
        )

        with db.engine.begin() as connection:
            for binding in bindings:
                table = binding.model_class.__table__
                logger.info("Ensuring dataset table %s exists", table.name)
                table.create(bind=connection, checkfirst=True)
                added_columns = self._add_missing_columns(connection, table)
                sync_report.append(
                    {
                        "table_name": table.name,
                        "added_columns": added_columns,
                    }
                )

        return {
            "tables": table_names,
            "sync_report": sync_report,
        }
