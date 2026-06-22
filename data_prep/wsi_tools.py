"""WSI utilities: sync, rename, query, and download from GDC."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

GDC_API = "https://api.gdc.cancer.gov"


def normalize_svs_name(filename: str) -> str:
    if not filename.lower().endswith(".svs"):
        return filename
    stem = filename[:-4]
    if "." not in stem:
        return filename
    return f"{stem.split('.', 1)[0]}.svs"


def copy_dataset_svs(dataset_dir: Path, wsi_dir: Path) -> tuple[int, int]:
    copied = 0
    skipped = 0
    for svs_path in dataset_dir.rglob("*.svs"):
        target_path = wsi_dir / svs_path.name
        if target_path.exists():
            skipped += 1
            continue
        try:
            os.link(svs_path, target_path)
        except OSError:
            shutil.copy2(svs_path, target_path)
        copied += 1
    return copied, skipped


def normalize_wsi_files(wsi_dir: Path) -> int:
    renamed = 0
    for svs_path in sorted(wsi_dir.glob("*.svs")):
        target_name = normalize_svs_name(svs_path.name)
        if target_name == svs_path.name:
            continue

        target_path = svs_path.with_name(target_name)
        if target_path.exists():
            if target_path.stat().st_size != svs_path.stat().st_size:
                raise RuntimeError(
                    f"Rename conflict with different file size: {svs_path} -> {target_path}"
                )
            svs_path.unlink()
        else:
            svs_path.rename(target_path)
        renamed += 1
    return renamed


def sync_dataset_to_wsi(root: Path) -> None:
    """Collect slides from Dataset/ into wsi/ and normalize filenames."""
    if not root.is_dir():
        raise NotADirectoryError(root)

    for cancer_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        dataset_dir = cancer_dir / "Dataset"
        wsi_dir = cancer_dir / "wsi"
        copied = 0
        skipped = 0

        if dataset_dir.is_dir():
            wsi_dir.mkdir(exist_ok=True)
            copied, skipped = copy_dataset_svs(dataset_dir, wsi_dir)
        elif not wsi_dir.is_dir():
            continue

        renamed = normalize_wsi_files(wsi_dir)
        print(
            f"{cancer_dir.name}: added={copied}, skipped={skipped}, renamed={renamed}, dir={wsi_dir}"
        )


def query_caselevel_manifest(project_id: str, save_path: Path) -> pd.DataFrame:
    """Query one diagnostic WSI per case (DX1 preferred) and save a manifest."""
    print(f"Querying WSI files for {project_id}...")
    params = {
        "filters": json.dumps(
            {
                "op": "and",
                "content": [
                    {
                        "op": "in",
                        "content": {
                            "field": "cases.project.project_id",
                            "value": [project_id],
                        },
                    },
                    {
                        "op": "in",
                        "content": {"field": "files.data_format", "value": ["SVS"]},
                    },
                    {
                        "op": "in",
                        "content": {"field": "files.data_type", "value": ["Slide Image"]},
                    },
                ],
            }
        ),
        "fields": "file_id,file_name,cases.submitter_id,md5sum,file_size",
        "format": "JSON",
        "size": "20000",
    }

    resp = requests.get(f"{GDC_API}/files", params=params, timeout=60)
    resp.raise_for_status()
    hits = resp.json().get("data", {}).get("hits", [])
    frame = pd.DataFrame(hits)
    if frame.empty:
        raise RuntimeError(f"No WSI files found for project {project_id}")

    frame["case_id"] = frame["cases"].apply(lambda item: item[0]["submitter_id"][:12])
    rows = []
    for _, group in frame.groupby("case_id"):
        dx1 = group[group["file_name"].str.contains("DX1")]
        rows.append(dx1.iloc[0] if len(dx1) else group.iloc[0])

    caselevel = pd.DataFrame(rows)
    manifest = pd.DataFrame(
        {
            "id": caselevel["file_id"],
            "filename": caselevel["file_name"],
            "md5": caselevel["md5sum"],
            "size": caselevel["file_size"],
            "state": "live",
        }
    )
    manifest.to_csv(save_path, sep="\t", index=False)
    print(f"Saved case-level manifest: {save_path}")
    return manifest


def download_from_manifest(manifest_path: Path, out_dir: Path) -> None:
    """Download WSI files listed in a GDC manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest_path, sep="\t")
    print(f"Files to download: {len(frame)}")

    for _, row in frame.iterrows():
        file_name = str(row["filename"])
        file_id = str(row["id"])
        save_path = out_dir / file_name

        if save_path.exists():
            print(f"Skip existing file: {file_name}")
            continue

        print(f"Downloading: {file_name}")
        url = f"{GDC_API}/data/{file_id}"
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            with open(save_path, "wb") as handle, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=file_name,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        bar.update(len(chunk))
