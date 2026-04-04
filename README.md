# MetaMP

MetaMP is a membrane-protein reconciliation and benchmarking platform. It brings MPstruc, PDB, OPM, UniProt, expert annotations, and topology-prediction outputs into one reviewable application so users can inspect disagreements, benchmark structures, and record expert decisions.

## Quick Start

Clone and run the full published app locally with one command:

```bash
git clone https://github.com/Ebenco36/MetaMP-Server.git && cd MetaMP-Server && ./scripts/metamp-reviewer-start.sh
```

This one command:
- clones the repository
- pulls the latest published backend, frontend, and snapshot PostgreSQL images
- pulls the published runtime snapshot assets
- restores the published database state
- starts the full app at `http://localhost/`
- keeps background jobs off by default

## Main URLs

- Frontend: [http://localhost/](http://localhost/)
- Backend health: [http://localhost:5400/api/v1/health/ready](http://localhost:5400/api/v1/health/ready)

## Maintainer Commands

Run the reviewer stack after the repository is already cloned:

```bash
./scripts/metamp-reviewer-start.sh
```

Build and run the full stack from source:

```bash
./scripts/metamp-production-bootstrap.sh run --with-frontend
```

Publish the current application images and snapshot distribution:

```bash
./scripts/metamp-publish-snapshot.sh push
```

## What MetaMP Includes

- harmonized membrane-protein data from MPstruc, PDB, OPM, and UniProt
- discrepancy review workflows for group labels, TM counts, and benchmark suitability
- expert notes and formal discrepancy adjudication
- topology comparison across predictors including TMbed, DeepTMHMM, TMHMM, TMDET, and TMAlphaFold-linked methods
- exportable review and benchmarking outputs

## Requirements

- Docker
- Docker Compose

That is enough for the one-command reviewer workflow above.

## More Commands

For operator commands, snapshot workflows, fallback prediction runs, image publishing, and maintenance utilities, see [COMMANDS.md](/Users/awotoroebenezer/Desktop/MetaMP-Server/COMMANDS.md).

## License

Licensed under the GNU General Public License, Version 3.0. See [LICENSE.md](/Users/awotoroebenezer/Desktop/MetaMP-Server/LICENSE.md).
