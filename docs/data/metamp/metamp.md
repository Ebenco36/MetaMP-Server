# MetaMP Data 2026

MetaMP harmonises membrane-protein annotations from MPstruc, RCSB PDB, OPM, and UniProt into a searchable resource. It preserves source-specific information, makes conflicting annotations visible, and supports expert review, benchmark assembly, and metadata analysis.

[Read the paper](https://doi.org/10.34133/csbj.0165) | [Source code and data access](https://github.com/Ebenco36/MetaMP-Server) | [Platform overview](https://ebenco36.github.io/MetaMP-Server/#overview)

## Dataset Characteristics

The following counts describe the study snapshot of 13 April 2026; later application releases may differ.

| Characteristic | Detail |
| --- | --- |
| Data type | Integrated annotation metadata in PostgreSQL; tabular exports through the application |
| Unit of integration | PDB accession, with linked source annotations and sequence-related metadata |
| Number of records | 4,089 distinct PDB accessions across 4,095 harmonised table rows |
| Annotation attributes | 282 fields: 107 categorical and 175 quantitative |
| Primary sources | MPstruc, RCSB PDB, OPM, and UniProt |
| Reference annotations | 121 entries with expert broad-group labels; 106 with explicit transmembrane-segment counts |
| Study snapshot | 13 April 2026; MPstruc anchor data retrieved on 27 March 2026 |
| Publication date | 20 August 2026 |
| Associated tasks | Annotation reconciliation, curation triage, benchmark qualification, exploratory topology comparison, and metadata analysis |
| Publication DOI | [10.34133/csbj.0165](https://doi.org/10.34133/csbj.0165) |

A PDB entry is a deposited structure record, not necessarily a unique protein. Coverage is anchored on MPstruc-linked records; field availability varies across sources.

## Data Sources

MetaMP retains source-specific annotations alongside harmonised fields, allowing users to trace a label to its origin and inspect conflicting interpretations.

| Source | Main contribution |
| --- | --- |
| [MPstruc](https://blanco.biomol.uci.edu/mpstruc/) | Curated membrane-protein groups, subgroups, and the initial set of PDB-linked entries |
| [RCSB PDB](https://www.rcsb.org/) | Structure, experimental-method, resolution, and bibliographic metadata |
| [OPM](https://opm.phar.umich.edu/) | Membrane orientation, thickness, tilt, transfer energy, and topology annotations |
| [UniProt](https://www.uniprot.org/) | Sequence-linked functional and taxonomic annotations |

## Data Variables

- **Identifiers:** PDB accessions, linked source identifiers, UniProt accessions, and replacement mappings where available.
- **Source annotations:** Broad structural groups, experimental metadata, membrane descriptors, organism, and functional context.
- **Reference labels:** Expert broad-group assignments and transmembrane-segment counts for the discrepancy benchmark.
- **Comparison outputs:** Topology predictions, assistive classifier outputs, and disagreement in groups, segment counts, or boundaries.
- **Review information:** Review state, contextual caution flags, benchmark decisions, and supporting reasons where available.

## Expert Reference Set

Comparison of MPstruc and OPM broad-group labels identified 121 conflicting entries, representing 2.96% of the 4,089 harmonised PDB entries. These records form the discrepancy benchmark.

A single domain expert reviewed the source annotations, structural and topology evidence, and relevant literature. All 121 entries received a broad-group reference label; 106 also received an explicit transmembrane-segment count.

MetaMP supports inspection of these disagreements through source comparisons, review information, and caution flags for cases such as state-dependent membrane association, non-canonical constructs, multichain structures, and replaced accessions.

**Scope of use:** This deliberately selected subset supports evaluation on contested annotations. It is not a representative sample for estimating overall classifier accuracy or annotation quality. Its labels are expert-informed reference assignments from one annotator. Topology comparisons are exploratory because methods differ in sequence, chain, or entry representation and in available overlap. Benchmark-suitability flags are transparent heuristics and require interpretation for the intended task.

## Accessing the Data

The [MetaMP repository](https://github.com/Ebenco36/MetaMP-Server) provides the source code and deployment workflow. Published Docker images supply the application and its populated PostgreSQL database.

### Start a Local Instance

Requires Git, Docker, and Docker Compose, with Docker running.

~~~bash
git clone https://github.com/Ebenco36/MetaMP-Server.git
cd MetaMP-Server
./scripts/metamp-reviewer-start.sh --keep-data
~~~

After startup, open http://localhost/ to browse the application. Search and filter records, inspect discrepancies, and export the relevant tables. Direct database access is described in the [repository instructions](https://github.com/Ebenco36/MetaMP-Server#connecting-to-the-database).

The `--keep-data` option preserves any existing local stack data. On a first run, the workflow initialises the database from the published image. Available images may be updated; record the repository revision and image versions when reproducing an analysis.

### Dataset Organisation

~~~text
PostgreSQL database
  source_mpstruc     Original MPstruc metadata
  source_pdb        Original PDB metadata
  source_opm        Original OPM metadata
  source_uniprot    Original UniProt metadata
  membrane_proteins Harmonised records

Associated annotation layers
  Expert reference assignments and review information
  Discrepancy and benchmark-decision fields
  Topology predictions and model outputs
~~~

The [platform overview](https://ebenco36.github.io/MetaMP-Server/#overview) provides screenshots and an introduction to the interface. The interactive analysis tools and data exports are provided by the locally deployed MetaMP application.

## Publication

Awotoro, E., Anyabolu, C., Schwarz, F., Tauscher, J., Heider, D., Ladewig, K., Le Bon, C., Moncoq, K., Miroux, B., & Hattab, G. (2026). **MetaMP Ecosystem for Unified, Auditable, and Benchmark-Ready Data for Reliable Membrane Protein Annotation.** *Computational and Structural Biotechnology Journal*, 35(1), 0165. [https://doi.org/10.34133/csbj.0165](https://doi.org/10.34133/csbj.0165).

[PubMed record](https://pubmed.ncbi.nlm.nih.gov/42626122/)

## Licensing and Attribution

MetaMP software is released under the [MIT License](https://github.com/Ebenco36/MetaMP-Server/blob/main/LICENSE.md). Source annotations originate from MPstruc, RCSB PDB, OPM, and UniProt; consult their respective terms and attribution requirements when reusing data. The software licence does not replace the terms governing upstream content.

Please cite the MetaMP publication when using this resource and acknowledge the primary databases relevant to the records and annotations used in your analysis.
