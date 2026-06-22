"""MAF utilities: query, download, convert to gene TSV, recover manifests, and stats."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
from pathlib import Path

import pandas as pd
import requests

GDC_API = "https://api.gdc.cancer.gov"


def generate_case_list_csv(patches_dir: Path, output_csv: Path) -> bool:
    if not patches_dir.exists():
        print(f"Directory not found: {patches_dir}")
        return False

    files_info = []
    for item in patches_dir.iterdir():
        if item.is_dir():
            files_info.append({"name": item.name, "exists": True})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["name", "exists"])
        writer.writeheader()
        writer.writerows(files_info)

    print(f"Saved case list: {output_csv} ({len(files_info)} entries)")
    return True


def search_maf_files(case_id: str) -> list[dict]:
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": [case_id]}},
            {"op": "like", "content": {"field": "file_name", "value": "*.maf.gz"}},
            {"op": "=", "content": {"field": "data_format", "value": "MAF"}},
            {"op": "=", "content": {"field": "access", "value": "open"}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,data_format,access,file_size,md5",
        "format": "JSON",
        "size": "100",
    }
    try:
        resp = requests.get(f"{GDC_API}/files", params=params, timeout=30)
        resp.raise_for_status()
        hits = resp.json().get("data", {}).get("hits", [])
        return [
            {
                "id": hit.get("id"),
                "file_name": hit.get("file_name"),
                "md5": hit.get("md5", "N/A"),
                "size": hit.get("file_size", "N/A"),
            }
            for hit in hits
        ]
    except Exception as exc:
        print(f"Failed to query {case_id}: {exc}")
        return []


def save_manifest(all_results: list[dict], manifest_file: Path) -> None:
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as handle:
        handle.write("id\tfilename\tmd5\tsize\n")
        for item in all_results:
            handle.write(
                f"{item['id']}\t{item['file_name']}\t{item.get('md5','N/A')}\t{item.get('size','N/A')}\n"
            )
    print(f"Saved MAF manifest: {manifest_file}")


def update_case_csv_with_maf(input_csv: Path, output_csv: Path, manifest_file: Path) -> bool:
    frame = pd.read_csv(input_csv)
    all_maf_files = []

    def check_case_exists(row: pd.Series) -> bool:
        case_id = "-".join(str(row["name"]).split("-")[:3])
        print(f"Querying {case_id}...")
        maf_files = search_maf_files(case_id)
        if maf_files:
            all_maf_files.extend(maf_files)
            print(f"{case_id}: found {len(maf_files)} MAF file(s)")
            return True
        print(f"{case_id}: no MAF found")
        return False

    frame["exists"] = frame.apply(check_case_exists, axis=1)
    save_manifest(all_maf_files, manifest_file)
    frame.to_csv(output_csv, index=False)
    print(f"Saved updated case list: {output_csv}")
    return len(all_maf_files) > 0


def manifest_files_present(manifest_file: Path, download_dir: Path) -> bool:
    if not manifest_file.exists():
        return False

    needed = []
    with open(manifest_file, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            needed.append(parts[1])

    if not needed:
        return False

    for name in needed:
        if not any(download_dir.rglob(name)):
            return False
    return True


def download_with_gdc_client(manifest_file: Path, download_dir: Path) -> bool:
    download_dir.mkdir(parents=True, exist_ok=True)
    if manifest_files_present(manifest_file, download_dir):
        print("MAF files from manifest already present; skipping download")
        return True
    try:
        print("Downloading with gdc-client...")
        subprocess.run(
            ["gdc-client", "download", "-m", str(manifest_file), "-d", str(download_dir)],
            check=True,
        )
        print(f"Download complete: {download_dir}")
        return True
    except Exception as exc:
        print(f"Download failed: {exc}")
        return False


def read_maf_to_df(file_path: Path) -> pd.DataFrame | None:
    try:
        if str(file_path).endswith(".gz"):
            return pd.read_csv(
                file_path, sep="\t", comment="#", compression="gzip", dtype=str
            )
        return pd.read_csv(file_path, sep="\t", comment="#", dtype=str)
    except Exception as exc:
        print(f"pandas read failed for {file_path}: {exc}; trying line-by-line parse")
        headers = None
        rows = []
        open_fn = gzip.open if str(file_path).endswith(".gz") else open
        with open_fn(file_path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                if headers is None:
                    headers = line.strip().split("\t")
                    continue
                rows.append(line.strip().split("\t"))
        if headers and rows:
            return pd.DataFrame(rows, columns=headers)
        return None


def extract_case_id_from_maf(df: pd.DataFrame) -> str | None:
    if "Tumor_Sample_Barcode" in df.columns:
        values = df["Tumor_Sample_Barcode"].dropna().astype(str)
        if not values.empty:
            return "-".join(values.iloc[0].split("-")[:3])
    first_col = df.columns[0] if len(df.columns) > 0 else None
    if first_col:
        values = df[first_col].dropna().astype(str)
        if not values.empty:
            return "-".join(values.iloc[0].split("-")[:3])
    return None


def process_mafs_to_tsv(download_dir: Path, gene_dir: Path) -> int:
    gene_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    maf_files = list(download_dir.rglob("*.maf.gz")) + list(download_dir.rglob("*.maf"))

    for maf_path in maf_files:
        print(f"Processing MAF: {maf_path}")
        frame = read_maf_to_df(maf_path)
        if frame is None or frame.empty:
            print(f"Skipped empty or unreadable MAF: {maf_path}")
            continue

        case_id = extract_case_id_from_maf(frame)
        if not case_id:
            print(f"Could not extract case ID from: {maf_path}")
            continue

        case_dir = gene_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        out_path = case_dir / f"{case_id}.tsv"

        essential_cols = [
            "Hugo_Symbol",
            "Chromosome",
            "Start_Position",
            "End_Position",
            "Reference_Allele",
            "Tumor_Seq_Allele2",
            "Variant_Classification",
            "Variant_Type",
            "Tumor_Sample_Barcode",
        ]
        cols_to_keep = [col for col in essential_cols if col in frame.columns]
        output_frame = frame[cols_to_keep] if cols_to_keep else frame
        output_frame.to_csv(out_path, sep="\t", index=False)
        print(f"Saved TSV: {out_path}")
        processed += 1

    print(f"MAF -> TSV complete: {processed} files")
    return processed


def build_maf_gene_tables(dataset_dir: Path) -> None:
    """End-to-end MAF query/download/conversion for one dataset directory."""
    patches_dir = dataset_dir / "patches"
    input_csv = dataset_dir / "files_list.csv"
    output_csv = dataset_dir / "output_updated.csv"
    manifest_file = dataset_dir / "MANIFEST.txt"
    download_dir = dataset_dir / "maf_files"
    gene_dir = dataset_dir / "gene"

    print("[Step 1] Build case list from patch folders")
    if not generate_case_list_csv(patches_dir, input_csv):
        raise SystemExit(1)

    print("[Step 2] Query GDC and build MAF manifest")
    found = update_case_csv_with_maf(input_csv, output_csv, manifest_file)
    if not found:
        print("No downloadable MAF files found; stopping")
        return

    print("[Step 3] Download MAF files")
    if not download_with_gdc_client(manifest_file, download_dir):
        raise SystemExit(1)

    print("[Step 4] Convert MAF to per-case gene TSV")
    process_mafs_to_tsv(download_dir, gene_dir)


def recover_maf_success_manifest(out_dir: Path) -> Path:
    maf_dir = out_dir / "maf_files"
    manifest_path = out_dir / "manifest_maf.txt"
    success_path = out_dir / "manifest_maf_downloaded.txt"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_maf.txt not found: {manifest_path}")

    frame = pd.read_csv(manifest_path, sep="\t")
    maf_files_real = [
        name
        for name in maf_dir.iterdir()
        if name.name.endswith(".maf") or name.name.endswith(".maf.gz")
    ]
    file_names = [item.name for item in maf_files_real]

    frame_success = frame[frame["filename"].isin(file_names)]
    frame_success.to_csv(success_path, sep="\t", index=False)
    print(f"Rebuilt success manifest: {success_path} ({len(frame_success)} entries)")
    return success_path


def maf_stats_per_case(out_dir: Path) -> pd.DataFrame:
    maf_dir = out_dir / "maf_files"
    manifest_path = out_dir / "manifest_maf.txt"
    frame = pd.read_csv(manifest_path, sep="\t")

    maf_files = [
        item.name
        for item in maf_dir.iterdir()
        if item.name.endswith(".maf") or item.name.endswith(".maf.gz")
    ]
    frame_exist = frame[frame["filename"].isin(maf_files)]
    if "case_id" not in frame_exist.columns:
        frame_exist = frame_exist.copy()
        frame_exist["case_id"] = frame_exist["filename"].str.slice(0, 12)

    count_df = frame_exist.groupby("case_id")["filename"].count().reset_index()
    count_df.columns = ["case_id", "downloaded_maf_count"]

    print("MAF counts per case (top 20):")
    print(count_df.sort_values("downloaded_maf_count", ascending=False).head(20))
    print(f"\nTotal cases: {count_df.shape[0]}")
    print(f"Average MAF files per case: {count_df['downloaded_maf_count'].mean():.2f}")
    return count_df
