import re
from typing import Dict, List

def parse_schemas(schema_strs: List[str]) -> Dict[str, List[str]]:
    tbls: Dict[str, List[str]] = {}
    for s in schema_strs:
        m = re.match(r"\s*Table\s+(\w+)\s*\((.*)\)\s*$", s)
        if not m:
            raise ValueError(f"Bad schema: {s}")
        name, cols = m.groups()
        tbls[name] = [c.strip() for c in cols.split(",")]
    return tbls