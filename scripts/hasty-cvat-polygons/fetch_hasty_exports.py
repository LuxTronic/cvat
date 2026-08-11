from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from env_utils import load_env_from_parents, require_http_url

API_BASE = "https://api.hasty.ai/v1"
SCRIPT_DIR = Path(__file__).resolve().parent
HASTY_ROOT = SCRIPT_DIR / "hasty_exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Hasty exports (annotations + images) using API key from env."
    )
    parser.add_argument("--project-id", required=True, help="Hasty project UUID.")
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="Dataset UUID filter. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--annotations-name",
        default="annotations.json",
        help="Export name for json_v1.1 export.",
    )
    parser.add_argument(
        "--images-name",
        default="images",
        help="Export name for images export.",
    )
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=HASTY_ROOT / "raw_exports",
        help="Directory to store downloaded export artifacts.",
    )
    parser.add_argument(
        "--annotations-out",
        type=Path,
        default=HASTY_ROOT / "annotations.json",
        help="Final local path for the annotations JSON file.",
    )
    parser.add_argument(
        "--images-out",
        type=Path,
        default=HASTY_ROOT / "imgs",
        help="Directory to place exported images.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Only fetch annotations export.",
    )
    parser.add_argument(
        "--skip-annotations",
        action="store_true",
        help="Only fetch images export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files/folders.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=3.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Max wait time per export job.",
    )
    return parser.parse_args()


def get_api_key() -> str:
    load_env_from_parents(SCRIPT_DIR, filename=".env", override=False)
    key = os.getenv("HASTY_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing HASTY_API_KEY (set env var or put it in .env).")
    return key


def api_json_request(
    method: str,
    url: str,
    api_key: str,
    payload: dict | None = None,
) -> dict:
    body = None
    headers = {"X-Api-Key": api_key}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url=require_http_url(url), method=method, headers=headers, data=body)
    with urlopen(req, timeout=60) as resp:  # nosec B310 - scheme checked above
        data = resp.read().decode("utf-8")
    if not data:
        return {}
    return json.loads(data)


def start_export(
    project_id: str, api_key: str, fmt: str, export_name: str, dataset_ids: list[str]
) -> str:
    payload: dict[str, object] = {
        "format": fmt,
        "export_name": export_name,
    }
    if dataset_ids:
        payload["dataset_id"] = dataset_ids
    url = f"{API_BASE}/projects/{project_id}/exports"
    res = api_json_request("POST", url, api_key, payload)
    export_id = str(res.get("id", "")).strip()
    if not export_id:
        raise RuntimeError(f"Export start failed for format={fmt}. Response: {res}")
    print(f"Started export format={fmt} id={export_id}")
    return export_id


def wait_for_export_done(
    project_id: str, api_key: str, export_id: str, poll_seconds: float, timeout_seconds: int
) -> dict:
    url = f"{API_BASE}/projects/{project_id}/exports/{export_id}"
    deadline = time.time() + timeout_seconds
    last_status = ""
    while time.time() < deadline:
        job = api_json_request("GET", url, api_key)
        status = str(job.get("status", "")).upper()
        if status != last_status:
            print(f"Export {export_id} status={status}")
            last_status = status
        if status == "DONE":
            return job
        if status in {"FAILED", "CANCELED"}:
            raise RuntimeError(f"Export {export_id} ended with status={status}. Response: {job}")
        time.sleep(max(0.5, poll_seconds))
    raise TimeoutError(f"Timed out waiting for export {export_id} to complete.")


def download_url_to_path(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url=require_http_url(url), method="GET")
    with urlopen(req, timeout=300) as resp, out_path.open("wb") as f:  # nosec B310
        shutil.copyfileobj(resp, f)


def file_ext_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name.lower()
    if name.endswith(".zip"):
        return ".zip"
    if name.endswith(".json"):
        return ".json"
    return ".bin"


def find_first_json(root: Path) -> Path | None:
    for p in sorted(root.rglob("*.json")):
        return p
    return None


def unpack_annotations(artifact_path: Path, annotations_out: Path, overwrite: bool) -> None:
    if annotations_out.exists() and not overwrite:
        raise FileExistsError(f"{annotations_out} exists. Use --overwrite to replace it.")
    annotations_out.parent.mkdir(parents=True, exist_ok=True)

    if artifact_path.suffix.lower() == ".json":
        shutil.copy2(artifact_path, annotations_out)
        return

    if artifact_path.suffix.lower() == ".zip":
        unpack_dir = artifact_path.parent / f"{artifact_path.stem}_unzipped"
        if unpack_dir.exists():
            shutil.rmtree(unpack_dir)
        unpack_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(artifact_path, "r") as zf:
            zf.extractall(unpack_dir)
        json_file = find_first_json(unpack_dir)
        if json_file is None:
            raise RuntimeError(f"No JSON found in extracted annotations archive: {artifact_path}")
        shutil.copy2(json_file, annotations_out)
        return

    raise RuntimeError(f"Unknown annotations artifact type: {artifact_path}")


def unpack_images(artifact_path: Path, images_out: Path, overwrite: bool) -> None:
    if artifact_path.suffix.lower() != ".zip":
        raise RuntimeError(f"Expected .zip for images export, got: {artifact_path}")
    if images_out.exists() and overwrite:
        shutil.rmtree(images_out)
    images_out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(artifact_path, "r") as zf:
        zf.extractall(images_out)
    flatten_single_wrapper_dir(images_out)


def flatten_single_wrapper_dir(root: Path) -> None:
    """
    If extraction created one wrapper directory (e.g. imgs/<dataset_name>/...),
    move its contents up to root so we end at imgs/<files>.
    """
    entries = [p for p in root.iterdir()]
    dirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file()]
    if files or len(dirs) != 1:
        return

    wrapper = dirs[0]
    for child in wrapper.iterdir():
        target = root / child.name
        if target.exists():
            raise RuntimeError(f"Cannot flatten image wrapper; target already exists: {target}")
        shutil.move(str(child), str(target))
    wrapper.rmdir()


def run_export_download(
    project_id: str,
    api_key: str,
    dataset_ids: list[str],
    fmt: str,
    export_name: str,
    exports_dir: Path,
    poll_seconds: float,
    timeout_seconds: int,
) -> Path:
    export_id = start_export(project_id, api_key, fmt, export_name, dataset_ids)
    job = wait_for_export_done(project_id, api_key, export_id, poll_seconds, timeout_seconds)
    meta = job.get("meta") or {}
    download_url = str(meta.get("url", "")).strip()
    if not download_url:
        raise RuntimeError(f"Export {export_id} is DONE but meta.url is missing. Job: {job}")
    ext = file_ext_from_url(download_url)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = export_name.replace("/", "_").replace("\\", "_")
    out_path = exports_dir / f"{timestamp}_{fmt}_{safe_name}{ext}"
    print(f"Downloading export {export_id} -> {out_path}")
    download_url_to_path(download_url, out_path)
    return out_path


def main() -> None:
    args = parse_args()
    api_key = get_api_key()
    args.exports_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_images and args.skip_annotations:
        raise RuntimeError("Both --skip-images and --skip-annotations are set. Nothing to do.")

    if not args.skip_annotations:
        ann_artifact = run_export_download(
            project_id=args.project_id,
            api_key=api_key,
            dataset_ids=args.dataset_id,
            fmt="json_v1.1",
            export_name=args.annotations_name,
            exports_dir=args.exports_dir,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        unpack_annotations(ann_artifact, args.annotations_out, args.overwrite)
        print(f"Annotations written: {args.annotations_out}")

    if not args.skip_images:
        img_artifact = run_export_download(
            project_id=args.project_id,
            api_key=api_key,
            dataset_ids=args.dataset_id,
            fmt="images",
            export_name=args.images_name,
            exports_dir=args.exports_dir,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        unpack_images(img_artifact, args.images_out, args.overwrite)
        print(f"Images extracted to: {args.images_out}")

    print("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
