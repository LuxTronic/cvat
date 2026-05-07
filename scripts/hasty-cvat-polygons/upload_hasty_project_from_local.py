from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from env_utils import load_env_from_parents


API_BASE = "https://api.hasty.ai/v1"
SCRIPT_DIR = Path(__file__).resolve().parent
HASTY_ROOT = SCRIPT_DIR / "hasty_exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a new Hasty project and datasets only (no image or label upload)."
    )
    parser.add_argument("--project-name", required=True, help="New Hasty project name.")
    parser.add_argument("--workspace-id", default=None, help="Optional workspace UUID.")
    parser.add_argument("--project-description", default="", help="Optional project description.")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=HASTY_ROOT / "imgs",
        help="Root folder containing dataset subfolders.",
    )
    parser.add_argument(
        "--import-json",
        type=Path,
        default=HASTY_ROOT / "import_dataset" / "import_dataset_hasty_like.json",
        help="Optional JSON used to discover additional dataset names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without mutating Hasty.",
    )
    return parser.parse_args()


def get_api_key() -> str:
    load_env_from_parents(SCRIPT_DIR, filename=".env", override=False)
    key = os.getenv("HASTY_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing HASTY_API_KEY (set env var or put it in .env).")
    return key


def api_json_request(method: str, url: str, api_key: str, payload: dict | list | None = None) -> dict:
    body = None
    headers = {"X-Api-Key": api_key}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url=url, method=method, headers=headers, data=body)
    try:
        with urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {method} {url}\n{detail}") from exc

    if not raw:
        return {}
    return json.loads(raw)


def discover_datasets(images_root: Path, import_json: Path) -> list[str]:
    names: set[str] = set()

    if images_root.exists():
        folders = sorted([p for p in images_root.iterdir() if p.is_dir()], key=lambda x: x.name.lower())
        if folders:
            for folder in folders:
                if folder.name.strip():
                    names.add(folder.name.strip())
        else:
            names.add("default")

    if import_json.exists():
        payload = json.loads(import_json.read_text(encoding="utf-8"))
        for image_entry in payload.get("images", []):
            ds_name = str(image_entry.get("dataset_name", "")).strip()
            if ds_name:
                names.add(ds_name)

    return sorted(names, key=str.lower)


def main() -> None:
    args = parse_args()
    api_key = get_api_key()

    dataset_names = discover_datasets(args.images_root, args.import_json)
    if not dataset_names:
        raise RuntimeError(
            "No dataset names discovered. Provide dataset folders under --images-root "
            "or a JSON file with images[].dataset_name via --import-json."
        )

    print(f"Datasets to create: {len(dataset_names)}")
    for name in dataset_names:
        print(f"  - {name}")

    if args.dry_run:
        print("Dry-run mode enabled: no API mutations will be performed.")
        return

    project_body: dict[str, object] = {
        "name": args.project_name,
        "description": args.project_description or None,
        "content_type": "IMAGES",
    }
    if args.workspace_id:
        project_body["workspace_id"] = args.workspace_id

    project = api_json_request("POST", f"{API_BASE}/projects", api_key, project_body)
    project_id = str(project.get("id", "")).strip()
    if not project_id:
        raise RuntimeError(f"Project creation failed: {project}")
    print(f"Created project: {args.project_name} ({project_id})")

    created_count = 0
    for i, dataset_name in enumerate(dataset_names):
        ds = api_json_request(
            "POST",
            f"{API_BASE}/projects/{project_id}/datasets",
            api_key,
            {"name": dataset_name, "norder": float(i)},
        )
        ds_id = str(ds.get("id", "")).strip()
        if not ds_id:
            raise RuntimeError(f"Dataset creation failed for '{dataset_name}': {ds}")
        created_count += 1
        print(f"Created dataset: {dataset_name} ({ds_id})")

    print("\nDone")
    print(f"Project ID: {project_id}")
    print(f"Datasets created: {created_count}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
