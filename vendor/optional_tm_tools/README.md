## Optional TM Tools

This directory is the production integration point for optional topology and signal-peptide
predictors that are not bundled directly in the MetaMP Python runtime.

MetaMP supports three execution patterns:

1. Built-in local predictors
   - `TMbed`
   - `DeepTMHMM`

2. Vendor-provided local commands or wrappers
   - Place executables in `vendor/optional_tm_tools/bin/`
   - Or place wrapper scripts in `vendor/optional_tm_tools/wrappers/`
   - Place manually downloaded package bundles in `vendor/optional_tm_tools/packages/`

3. External import workflow
   - Use `export-optional-tm-prediction-inputs`
   - Populate `results.csv`
   - Import with `import-optional-tm-prediction-results`

### Wrapper contract

MetaMP auto-discovers commands in `bin/` and `wrappers/` using names such as:

- `metamp-run-phobius`
- `metamp-run-tmhmm`
- `metamp-run-signalp`
- `metamp-run-cctop`

Each wrapper is expected to accept:

```text
--input <FASTA>
--output <CSV>
--reference <CSV>
```

The output CSV must use this schema:

```csv
pdb_code,tm_count,tm_regions
1ABC,2,"[{""index"":1,""start"":10,""end"":31,""length"":22,""label"":""Membrane""}]"
```

For signal-peptide methods such as `SignalP`, `tm_count` may be blank and `tm_regions`
should contain the normalized region list.

### Vendored package install hooks

MetaMP currently has a build-time install hook for the official portable `SignalP 6` package.
It also has a public-archive extraction hook for the official `CCTOP` standalone package.

To enable it:

1. Download the official package from the DTU Health Tech SignalP 6 downloads page.
2. Place either:
   - the extracted directory at `vendor/optional_tm_tools/packages/signalp-6-package/`, or
   - the downloaded archive as `vendor/optional_tm_tools/packages/signalp-6-package*.tar.gz`
3. Rebuild `flask-app`.

The Docker image will install SignalP into an isolated virtual environment at:

```text
/opt/metamp-optional-tools/signalp_venv
```

The wrapper `metamp-run-signalp` will then use that installation automatically.

For CCTOP:

1. Download the official standalone archive from the CCTOP site.
2. Place either:
   - the extracted directory at `vendor/optional_tm_tools/packages/cctop/`, or
   - the downloaded archive as `vendor/optional_tm_tools/packages/cctop*.tgz`
3. Rebuild `flask-app`.

The Docker image will unpack the archive to:

```text
/opt/metamp-optional-tools/cctop
```

The wrapper `metamp-run-cctop` will then resolve:

- `METAMP_CCTOP_BIN` or `/opt/metamp-optional-tools/cctop/Standalone/bin/cctop`
- `METAMP_CCTOP_PARAM_XML` or `/opt/metamp-optional-tools/cctop/Standalone/cctop_param.xml`

Important:

- CCTOP still depends on a licensed `HMMTOP` installation.
- MetaMP does not auto-install `HMMTOP`.
- If `HMMTOP` headers and libraries are present at build time, MetaMP will attempt to compile the vendored
  `CCTOP` source automatically during the Docker build.
- If your site-managed CCTOP environment already handles HMMTOP, you can bypass the wrapper's
  shallow HMMTOP presence check with `METAMP_CCTOP_SKIP_HMMTOP_CHECK=1`.

### Environment knobs

The SignalP wrapper supports these optional environment variables:

- `METAMP_SIGNALP_BIN`: explicit path to `signalp6`
- `METAMP_SIGNALP_ORGANISM`: defaults to `other`
- `METAMP_SIGNALP_MODE`: defaults to `fast`
- `METAMP_SIGNALP_MODEL_DIR`: optional override for model weights

The CCTOP wrapper supports these optional environment variables:

- `METAMP_CCTOP_BIN`: explicit path to the `cctop` executable
- `METAMP_CCTOP_PARAM_XML`: explicit path to `param.xml`
- `METAMP_CCTOP_COMMAND_TEMPLATE`: optional site-specific command template
- `METAMP_CCTOP_SKIP_HMMTOP_CHECK`: set to `1` only if a site-managed CCTOP install already resolves HMMTOP
- `METAMP_HMMTOP_BIN`, `METAMP_HMMTOP_HOME`, `METAMP_HMMTOP_LIB`: optional hints for shallow runtime validation

### Runtime inspection

Use this command to inspect what MetaMP can actually run inside the current container:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app \
  env FLASK_APP=manage.py flask optional-tm-runtime-status
```

### Notes

- MetaMP does not pretend unsupported tools are locally runnable.
- If a tool requires manual license acceptance or proprietary installation, place the
  accepted binary or wrapper in this directory and rebuild/restart the containers.
- `CCTOP` now has a real MetaMP wrapper and public-archive extraction hook, but it still requires
  a licensed `HMMTOP` installation in addition to the public CCTOP standalone source.
