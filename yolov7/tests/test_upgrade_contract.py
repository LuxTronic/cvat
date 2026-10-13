"""Source checks for the Ubuntu 24.04 / Python 3.12 sidecar rebuild.

These tests do not load torch. The image smoke test does that.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _python_files():
    for path in ROOT.rglob("*.py"):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        yield path


class UpgradeContractTest(unittest.TestCase):
    def test_base_image_is_ubuntu_2404_cpu(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("FROM ubuntu:24.04", dockerfile)
        self.assertIn(
            "sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55",
            dockerfile,
        )
        self.assertNotIn("nvcr.io", dockerfile)
        self.assertNotIn("nvidia/pytorch", dockerfile)
        requirements = (ROOT / "requirements.txt").read_text()
        self.assertIn("torch==2.14.1+cpu", requirements)
        self.assertIn("torchvision==0.29.1+cpu", requirements)
        self.assertIn("opencv-python-headless==4.14.0.94", requirements)
        self.assertNotIn("numpy>=1.18.5,<1.24.0", requirements)
        self.assertNotIn("opencv-python-headless==4.5.5.64", requirements)
        self.assertIn(
            "0fa1423606133c06362ec716029dade1169d5631e83195b75d8efa66d2ee0b77",
            dockerfile,
        )
        self.assertIn("HOME=/home/ubuntu", dockerfile)
        self.assertIn('ENTRYPOINT ["/yolov7/docker-entrypoint.sh"]', dockerfile)
        self.assertNotIn("USER 1000:1000", dockerfile)
        self.assertNotIn("Root on purpose", dockerfile)
        entrypoint = (ROOT / "docker-entrypoint.sh").read_text()
        self.assertIn("chown -R 1000:1000 /models", entrypoint)
        self.assertIn("setpriv --reuid=1000 --regid=1000 --clear-groups", entrypoint)
        self.assertIn("chown 1000:1000 /models", dockerfile)
        self.assertNotIn("chown -R 1000:1000 /yolov7", dockerfile)

    def test_compose_keeps_host_ipc_without_nvidia_runtime(self):
        compose = (ROOT.parent / "docker-compose.yml").read_text()
        start = compose.index("  yolov7:\n")
        end = compose.index("  yolov8-cls:\n")
        block = compose[start:end]
        self.assertIn("ipc: host", block)
        self.assertNotIn("runtime: nvidia", block)

    def test_meshgrid_uses_ij_indexing(self):
        mesh_calls = 0
        indexed = 0
        for path in _python_files():
            text = path.read_text()
            self.assertNotIn("torch.meshgrid([", text, path)
            mesh_calls += text.count("torch.meshgrid(")
            indexed += text.count('indexing="ij"')
        self.assertEqual(mesh_calls, 8)
        self.assertEqual(indexed, mesh_calls)

    def test_checkpoints_load_with_weights_only_false(self):
        pattern = re.compile(r"torch\.load\(")
        for path in _python_files():
            lines = path.read_text().splitlines()
            index = 0
            while index < len(lines):
                line = lines[index]
                if not pattern.search(line) or "path/to/model.pt" in line:
                    index += 1
                    continue
                call = line
                start = index + 1
                while call.count("(") > call.count(")") and index + 1 < len(lines):
                    index += 1
                    call += "\n" + lines[index]
                self.assertIn(
                    "weights_only=False",
                    call,
                    f"{path.relative_to(ROOT)}:{start}",
                )
                index += 1

    def test_numpy_int_alias_is_gone(self):
        for path in _python_files():
            self.assertIsNone(
                re.search(r"\bnp\.int\b", path.read_text()),
                path.relative_to(ROOT),
            )

    def test_loss_empty_targets_follow_the_batch_device(self):
        text = (ROOT / "utils" / "loss.py").read_text()
        self.assertNotIn("device='cuda:0'", text)
        self.assertNotIn('device="cuda:0"', text)
        self.assertGreaterEqual(text.count("device=targets.device"), 24)


if __name__ == "__main__":
    unittest.main()
