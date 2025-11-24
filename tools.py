import subprocess
import re
from pathlib import Path

import pandas as pd


def python_exec(path):
    script_path = Path(path)

    if not script_path.exists():
        raise FileNotFoundError("파일을 찾을 수 없습니다: %s" % path)

    result = subprocess.run(
        ["python", str(script_path)],
        capture_output=True,
        text=True
    )

    stdout = result.stdout
    stderr = result.stderr
    returncode = result.returncode

    numbers = re.findall(r"-?\d+", stdout)
    last_number = int(numbers[-1]) if numbers else None

    return {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "last_number": last_number,
    }


def _load_all_sheets(path):
    xls = pd.ExcelFile(path)
    dfs = []
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        df["_sheet_name"] = sheet_name
        dfs.append(df)
    return dfs


def _normalize_colname(col):
    return col.strip().lower().replace(" ", "_").replace("/", "_")


def xlsx_query(path, query):
    excel_path = Path(path)
    if not excel_path.exists():
        raise FileNotFoundError("엑셀 파일을 찾을 수 없습니다: %s" % path)

    dfs = _load_all_sheets(excel_path)
    q = query.lower()

    result = {
        "raw_query": query,
        "sheets": [],
    }

    for df in dfs:
        sheet_name = df["_sheet_name"].iloc[0]
        info = {
            "sheet": sheet_name,
        }

        norm_map = {}
        for c in df.columns:
            if c == "_sheet_name":
                continue
            norm_map[_normalize_colname(c)] = c

        if "total" in q and "sales" in q and ("location" in norm_map or "city" in norm_map):
            loc_col = norm_map.get("location", norm_map.get("city"))

            locations_in_query = []
            for val in df[loc_col].astype(str).unique():
                if val and val.lower() in q:
                    locations_in_query.append(val)

            num_cols = []
            for c in df.columns:
                if c == "_sheet_name":
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    num_cols.append(c)

            totals = {}
            for loc in locations_in_query:
                sub = df[df[loc_col].astype(str).str.lower() == loc.lower()]
                if not sub.empty:
                    row_sums = sub[num_cols].sum(axis=1)
                    totals[loc] = float(row_sums.sum())

            info["type"] = "location_total_sales"
            info["location_totals"] = totals

        elif "operating status" in q or "operational" in q:
            status_col = None
            for k, v in norm_map.items():
                if "operating_status" in k or "operating" in k:
                    status_col = v
                    break

            if status_col is not None:
                vc = df[status_col].astype(str).value_counts(dropna=True)
                counts = vc.to_dict()

                total_rows = int(df[status_col].notna().sum())
                operational_count = 0
                for key, value in counts.items():
                    if "operational" in str(key).lower():
                        operational_count += int(value)

                info["type"] = "operating_status_counts"
                info["counts"] = counts
                info["total"] = total_rows
                info["operational"] = operational_count

        else:
            info["type"] = "preview"
            info["head"] = df.head(5).to_dict(orient="records")

        result["sheets"].append(info)

    return result
