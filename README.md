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


## Connecting to the Database

The complete MetaMP dataset—including raw source tables from MPstruc, PDB, OPM, and UniProt, the harmonised `membrane_proteins` tables resides in a PostgreSQL database that is initialised automatically when the Docker container starts. Users who wish to query the data directly or perform custom exports can connect to this database using any standard PostgreSQL client.

### Connection Details

When the container is running (after executing `./scripts/metamp-reviewer-start.sh`), the PostgreSQL instance is exposed on the host machine with the following default parameters:

| Parameter   | Value              |
|-------------|--------------------|
| Host        | `localhost`        |
| Port        | `5432`             |
| Database    | `mpvis_db`        |
| Username    | `mpvis_user`      |
| Password    | `mpvis_user`  |

> **Note:** These credentials are intended for local development and review only. For production deployments, change the password using the environment variables documented in `docker-compose.yml`.

### Connecting with `psql` (Command Line)

If you have the PostgreSQL client installed, you can connect directly from your terminal:

```bash
psql -h localhost -p 5432 -U mpvis_user -d mpvis_db
```

### Connecting with `pgAdmin` (GUI)

Download pgAdmin from [https://www.pgadmin.org/download/](https://www.pgadmin.org/download/).

## Repositories

- **Backend (this repo):** [https://github.com/Ebenco36/MetaMP-Server](https://github.com/Ebenco36/MetaMP-Server)
- **Frontend:** [https://github.com/Ebenco36/MPVisualization](https://github.com/Ebenco36/MPVisualization)

## Maintainer Commands

Run the reviewer stack after the repository is already cloned:

```bash
./scripts/metamp-reviewer-start.sh
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

For operator commands, snapshot workflows, fallback prediction runs, image publishing, and maintenance utilities, see [COMMANDS.md](./COMMANDS.md).

## License

Licensed under the GNU General Public License, Version 3.0. See [LICENSE.md](./LICENSE.md).
