import re
from typing import List, Tuple, Dict
from .schema_parser import parse_schemas


def find_joins(tables: List[str], schemas: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
    joins: List[Tuple[str, str, str]] = []
    used = {tables[0]}
    remaining = set(tables[1:])
    while remaining:
        progress = False
        for t in list(remaining):
            for u in used:
                common = set(schemas[u]) & set(schemas[t])
                if common:
                    col = common.pop()
                    joins.append((u, t, col))
                    used.add(t)
                    remaining.remove(t)
                    progress = True
                    break
            if progress:
                break
        if not progress:
            break
    return joins


def nl_to_sql_multi(nl: str, schema_strs):
    schemas = parse_schemas(schema_strs)
    tables = list(schemas.keys())
    base = tables[0]

    # 1) split into filter vs projection
    parts      = re.split(r"\binclude\b", nl, flags=re.IGNORECASE, maxsplit=1)
    filter_part = parts[0]
    proj_part   = parts[1] if len(parts) > 1 else ""

    # 2) build phrase→(table,col) map for projections & filters
    field_map = {
        col.lower().replace("_", " "): (tbl, col)
        for tbl, cols in schemas.items()
        for col in cols
    }

    # 3) PROJECTIONS: only look in proj_part
    select_cols = []
    for phrase, (tbl, col) in field_map.items():
        if re.search(rf"\b{re.escape(phrase)}\b", proj_part, re.IGNORECASE):
            select_cols.append(f"{tbl}.{col}")
    if not select_cols:
        select_cols = [f"{t}.*" for t in tables]

    # 4) FROM + JOIN only on pdb_code and uniprot_id
    from_clause = base
    for fk in ("pdb_code", "uniprot_id"):
        for tbl in tables[1:]:
            if fk in schemas[base] and fk in schemas[tbl]:
                from_clause += f"\n  JOIN {tbl} ON {base}.{fk} = {tbl}.{fk}"

    # 5) WHERE: numeric ops on resolution
    where = []
    # find which table has resolution
    res_tbl = next((t for t,c in schemas.items() if "resolution" in c), base)

    # >=
    m = re.search(
        r"resolution\s+(?:greater\s+than\s+or\s+equals|greater\s+than\s+or\s+equal\s+to|>=|at\s+least)\s+(\d+(\.\d+)?)",
        filter_part, re.IGNORECASE
    )
    if m:
        val = m.group(1)
        where.append(
            f"(CASE WHEN {res_tbl}.resolution ~ '^[0-9]+(\\.[0-9]+)?$' "
            f"THEN CAST({res_tbl}.resolution AS DOUBLE PRECISION) END) >= {val}"
        )

    # >  
    m = re.search(
        r"resolution\s+(?:greater\s+than|over|above)\s+(\d+(\.\d+)?)",
        filter_part, re.IGNORECASE
    )
    if m:
        val = m.group(1)
        where.append(
            f"(CASE WHEN {res_tbl}.resolution ~ '^[0-9]+(\\.[0-9]+)?$' "
            f"THEN CAST({res_tbl}.resolution AS DOUBLE PRECISION) END) > {val}"
        )

    # <=
    m = re.search(
        r"resolution\s+(?:less\s+than\s+or\s+equals|less\s+than\s+or\s+equal\s+to|<=|at\s+most)\s+(\d+(\.\d+)?)",
        filter_part, re.IGNORECASE
    )
    if m:
        val = m.group(1)
        where.append(
            f"(CASE WHEN {res_tbl}.resolution ~ '^[0-9]+(\\.[0-9]+)?$' "
            f"THEN CAST({res_tbl}.resolution AS DOUBLE PRECISION) END) <= {val}"
        )

    # <  
    m = re.search(
        r"resolution\s+(?:under|below|less\s+than)\s+(\d+(\.\d+)?)",
        filter_part, re.IGNORECASE
    )
    if m:
        val = m.group(1)
        where.append(
            f"(CASE WHEN {res_tbl}.resolution ~ '^[0-9]+(\\.[0-9]+)?$' "
            f"THEN CAST({res_tbl}.resolution AS DOUBLE PRECISION) END) < {val}"
        )

    # 6) collect ALL pdb_code / uniprot_id mentions into single IN-clause each
    for phrase in ("pdb code", "uniprot id"):
        vals = []
        pattern = rf"{phrase}\s*(?:with|is)\s*([A-Za-z0-9, ]+?)(?=\s+(?:and|include|$))"
        for m in re.finditer(pattern, filter_part, flags=re.IGNORECASE):
            for v in m.group(1).split(","):
                v = v.strip().upper()
                if v:
                    vals.append(v)
        if vals:
            tbl, col = field_map[phrase]
            vals = sorted(set(vals))
            if len(vals) > 1:
                where.append(f"UPPER({tbl}.{col}) IN ({', '.join(repr(v) for v in vals)})")
            else:
                where.append(f"UPPER({tbl}.{col}) = {repr(vals[0])}")

    # 7) assemble
    sql = (
        f"SELECT {', '.join(select_cols)}\n"
        f"  FROM {from_clause}"
        + ("\n WHERE " + " AND ".join(where) if where else "")
        + ";"
    )

    return sql