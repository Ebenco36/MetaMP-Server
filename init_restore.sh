#!/bin/bash
set -e

echo "Restoring PostgreSQL database from dump..."
pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" /docker-entrypoint-initdb.d/all_tables.dump
echo "Restore complete."
