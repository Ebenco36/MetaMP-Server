## Example Optional TM Wrappers

These files are examples only. They are not auto-discovered by MetaMP because they
are not placed in `vendor/optional_tm_tools/bin` or `vendor/optional_tm_tools/wrappers`.

To enable a predictor locally:

1. Copy the relevant example into `vendor/optional_tm_tools/wrappers/`
2. Rename it to one of the expected names, for example:
   - `metamp-run-phobius`
   - `metamp-run-tmhmm`
   - `metamp-run-signalp`
   - `metamp-run-cctop`
3. Implement the actual predictor invocation
4. Rebuild and restart the containers
5. Verify with:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app \
  env FLASK_APP=manage.py flask optional-tm-runtime-status
```

Each wrapper must write a normalized CSV to `--output` with:

```csv
pdb_code,tm_count,tm_regions
```
