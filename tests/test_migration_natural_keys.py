import unittest

import pandas as pd

from src.Commands.Migration.classMigrate import Migration
from src.ingestion.valid_dataset_cleaner import deduplicate_valid_dataset
from src.MP.model import MembraneProteinData
from src.MP.model_mpstruct import MPSTURC
from src.MP.model_uniprot import Uniprot


class MigrationNaturalKeyTests(unittest.TestCase):
    def test_model_key_fields_are_model_specific(self):
        self.assertEqual(
            Migration.get_key_fields(MembraneProteinData),
            ("pdb_code", "group", "subgroup"),
        )
        self.assertEqual(
            Migration.get_key_fields(MPSTURC),
            ("pdb_code", "group", "subgroup"),
        )
        self.assertEqual(
            Migration.get_key_fields(Uniprot),
            ("uniprot_id", "pdb_code"),
        )

    def test_drop_duplicate_rows_keeps_unique_group_and_subgroup_pairs(self):
        dataframe = pd.DataFrame(
            [
                {
                    "pdb_code": "2GMH",
                    "group": "MONOTOPIC MEMBRANE PROTEINS",
                    "subgroup": "Oxidoreductases (Monotopic)",
                    "resolution": 3.0,
                },
                {
                    "pdb_code": "2GMH",
                    "group": "MONOTOPIC MEMBRANE PROTEINS",
                    "subgroup": "Oxidoreductases (Monotopic)",
                    "resolution": 3.0,
                },
                {
                    "pdb_code": "2GMH",
                    "group": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL",
                    "subgroup": "Oxidoreductases",
                    "resolution": 3.0,
                },
                {
                    "pdb_code": "2GMH",
                    "group": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL",
                    "subgroup": "Oxidoreductases",
                    "resolution": 3.0,
                },
            ]
        )

        deduplicated = Migration.drop_duplicate_rows(dataframe, MembraneProteinData)

        self.assertEqual(len(deduplicated), 2)
        kept_pairs = {
            (row["pdb_code"], row["group"], row["subgroup"])
            for row in deduplicated.to_dict(orient="records")
        }
        self.assertEqual(
            kept_pairs,
            {
                ("2GMH", "MONOTOPIC MEMBRANE PROTEINS", "Oxidoreductases (Monotopic)"),
                ("2GMH", "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL", "Oxidoreductases"),
            },
        )

    def test_valid_dataset_cleaner_deduplicates_quantitative_natural_keys(self):
        dataframe = pd.DataFrame(
            [
                {"Pdb Code": "3HYW", "Group": "MONOTOPIC MEMBRANE PROTEINS", "Subgroup": "Oxidoreductases (Monotopic)"},
                {"Pdb Code": "3HYW", "Group": "MONOTOPIC MEMBRANE PROTEINS", "Subgroup": "Oxidoreductases (Monotopic)"},
                {"Pdb Code": "3HYW", "Group": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL", "Subgroup": "Oxidoreductases"},
                {"Pdb Code": "3HYW", "Group": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL", "Subgroup": "Oxidoreductases"},
            ]
        )

        deduplicated = deduplicate_valid_dataset(dataframe, "Quantitative_data.csv")

        self.assertEqual(len(deduplicated), 2)


if __name__ == "__main__":
    unittest.main()
