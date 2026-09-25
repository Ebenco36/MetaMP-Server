from database.db import db
from src.Commands.Migration.classMigrate import Migration
from src.ingestion.database_seed_service import DatabaseSeedService
from src.ingestion.model_registry import get_dataset_model_bindings


class DatabaseLoadService:
    def __init__(self):
        self.seed_service = DatabaseSeedService()

    def load_current_datasets(self, clear_db=False, seed_defaults=True):
        if clear_db:
            db.drop_all()

        db.create_all()

        bindings = get_dataset_model_bindings()
        for binding in bindings:
            retained_keys = Migration.load_csv_data(binding.model_class, binding.csv_path)
            self._prune_missing_records(binding.model_class, retained_keys)

        if seed_defaults:
            self.seed_service.seed_defaults()

        return bindings

    @staticmethod
    def _clear_dataset_tables(bindings):
        with db.engine.begin() as connection:
            for binding in reversed(bindings):
                table = binding.model_class.__table__
                connection.execute(table.delete())

    @staticmethod
    def _prune_missing_records(model_class, retained_keys):
        if not retained_keys:
            return

        retained_lookup = {
            tuple(str(part or "").strip() for part in record_key)
            for record_key in retained_keys
            if record_key
        }
        for record in model_class.query.all():
            record_key = Migration.get_record_key_from_instance(model_class, record)
            if all(not part for part in record_key):
                continue
            if record_key not in retained_lookup:
                db.session.delete(record)

        db.session.commit()
