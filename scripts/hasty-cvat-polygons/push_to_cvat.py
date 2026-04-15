from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from env_utils import load_env_from_parents

SCRIPT_DIR = Path(__file__).resolve().parent
HASTY_ROOT = SCRIPT_DIR / "hasty_exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload prepared Ultralytics dataset zip to CVAT project using token auth."
    )
    parser.add_argument(
        "--dataset-zip",
        type=Path,
        default=HASTY_ROOT / "import_dataset.zip",
        help="Path to the prepared dataset zip (from prepare_import_dataset.py).",
    )
    parser.add_argument(
        "--project-name",
        required=True,
        help="CVAT project name to create/use.",
    )
    parser.add_argument(
        "--format-name",
        default="Ultralytics YOLO Segmentation 1.0",
        help="CVAT dataset import format name.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Polling interval for CVAT async request status.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Timeout waiting for import request completion.",
    )
    return parser.parse_args()


def _is_local_http(host: str) -> bool:
    host_l = host.lower()
    return host_l.startswith("http://localhost") or host_l.startswith("http://127.0.0.1")


def load_env() -> tuple[str, str, str, str, str]:
    load_env_from_parents(SCRIPT_DIR, filename=".env", override=False)
    host = os.getenv("CVAT_HOST", "").strip().rstrip("/")
    token = os.getenv("CVAT_TOKEN", "").strip()
    token_type = os.getenv("CVAT_TOKEN_TYPE", "Token").strip() or "Token"
    username = os.getenv("CVAT_USERNAME", "").strip()
    password = os.getenv("CVAT_PASSWORD", "").strip()
    if not host:
        raise RuntimeError("Missing CVAT_HOST in .env")
    if not (host.startswith("https://") or _is_local_http(host)):
        raise RuntimeError("CVAT_HOST must be https:// (or local http://localhost/127.0.0.1).")
    return host, token, token_type, username, password


def build_session(host: str, token: str, token_type: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Accept": "application/vnd.cvat+json, application/json, */*",
            "Authorization": f"{token_type} {token}",
        }
    )
    s.base_url = host  # type: ignore[attr-defined]
    return s


def build_session_noauth(host: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Accept": "application/vnd.cvat+json, application/json, */*"})
    s.base_url = host  # type: ignore[attr-defined]
    return s


def _raise_on_http(res: requests.Response) -> None:
    if res.ok:
        return
    msg = f"HTTP {res.status_code} {res.request.method} {res.url}\n{res.text[:1500]}"
    raise RuntimeError(msg)


def api_get(session: requests.Session, path: str, **kwargs: Any) -> dict[str, Any]:
    res = session.get(f"{session.base_url}{path}", timeout=60, **kwargs)  # type: ignore[attr-defined]
    _raise_on_http(res)
    return res.json() if res.text else {}


def api_post(session: requests.Session, path: str, **kwargs: Any) -> requests.Response:
    res = session.post(f"{session.base_url}{path}", timeout=300, **kwargs)  # type: ignore[attr-defined]
    _raise_on_http(res)
    return res


def login_and_get_token(host: str, username: str, password: str) -> tuple[str, str]:
    if not username or not password:
        raise RuntimeError("Missing CVAT credentials. Set CVAT_TOKEN or CVAT_USERNAME/CVAT_PASSWORD.")

    s = build_session_noauth(host)
    body: dict[str, Any] | None = None
    last_exc: Exception | None = None

    # Try a few login endpoint/header variants across CVAT deployments.
    login_paths = ["/api/auth/login", "/api/auth/login/"]
    accept_variants = [
        "application/vnd.cvat+json, application/json, */*",
        "application/json",
        "*/*",
    ]
    for path in login_paths:
        for accept in accept_variants:
            try:
                s.headers["Accept"] = accept
                res = api_post(s, path, json={"username": username, "password": password})
                body = res.json() if res.text else {}
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                continue
        if body is not None:
            break
    if body is None:
        raise RuntimeError(f"CVAT login failed: {last_exc}") from last_exc

    # CVAT can return either token style.
    token = str(body.get("key") or body.get("token") or "").strip()
    if not token:
        raise RuntimeError(f"Login succeeded but no token in response: {body}")

    # Most CVAT deployments use "Token", some use Bearer.
    return token, "Token"


def find_or_create_project(session: requests.Session, project_name: str) -> int:
    data = api_get(session, "/api/projects", params={"search": project_name, "page_size": 100})
    for item in data.get("results", []):
        if item.get("name") == project_name:
            return int(item["id"])

    payload = {"name": project_name}
    res = api_post(session, "/api/projects", json=payload)
    body = res.json() if res.text else {}
    if "id" not in body:
        raise RuntimeError(f"Project creation response missing id: {body}")
    return int(body["id"])


def start_project_dataset_import(
    session: requests.Session,
    project_id: int,
    dataset_zip: Path,
    format_name: str,
) -> str | None:
    with dataset_zip.open("rb") as f:
        files = {"dataset_file": (dataset_zip.name, f, "application/zip")}
        res = api_post(
            session,
            f"/api/projects/{project_id}/dataset",
            params={"action": "import", "format": format_name},
            files=files,
        )
    if res.status_code == 202:
        body = res.json() if res.text else {}
        rq_id = body.get("rq_id")
        return str(rq_id) if rq_id else None
    return None


def wait_for_request(session: requests.Session, rq_id: str, poll_seconds: float, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_status = ""
    while time.time() < deadline:
        data = api_get(session, f"/api/requests/{rq_id}")
        status = str(data.get("status", "")).lower()
        if status != last_status:
            print(f"Import request {rq_id}: {status}")
            last_status = status
        if status in {"finished", "completed", "done"}:
            return
        if status in {"failed", "error", "canceled"}:
            message = data.get("message") or data.get("error") or data
            raise RuntimeError(f"CVAT import failed: {message}")
        time.sleep(max(0.5, poll_seconds))
    raise TimeoutError(f"Timed out waiting for request {rq_id}")


def main() -> None:
    args = parse_args()
    if not args.dataset_zip.exists():
        raise FileNotFoundError(f"Dataset zip not found: {args.dataset_zip}")

    host, token, token_type, username, password = load_env()
    if token:
        session = build_session(host, token, token_type)
    else:
        token, token_type = login_and_get_token(host, username, password)
        session = build_session(host, token, token_type)
        print("Auth mode: username/password login (token obtained).")

    # Retry with Bearer if Token fails auth quickly.
    try:
        project_id = find_or_create_project(session, args.project_name)
    except Exception as first_exc:
        if token_type.lower() == "token":
            session = build_session(host, token, "Bearer")
            project_id = find_or_create_project(session, args.project_name)
            print("Auth fallback used: Bearer token type.")
        else:
            raise first_exc

    print(f"Using project id={project_id} name={args.project_name}")
    rq_id = start_project_dataset_import(session, project_id, args.dataset_zip, args.format_name)
    if rq_id:
        print(f"Started dataset import request rq_id={rq_id}")
        wait_for_request(session, rq_id, args.poll_seconds, args.timeout_seconds)
    print("Dataset import completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
