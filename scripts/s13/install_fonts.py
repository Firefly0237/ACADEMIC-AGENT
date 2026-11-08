"""Build-only exact Debian packages, cross-checked with the signed snapshot."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen


def main() -> None:
    packages = json.loads(Path("/tmp/font-packages.lock.json").read_text())["packages"]
    for package in packages:
        if (
            package["name"] not in {"fonts-arphic-ukai", "fonts-arphic-uming"}
            or Path(package["filename"]).name != package["filename"]
        ):
            raise ValueError("only_the_two_locked_open_font_packages_allowed")
        metadata = subprocess.run(
            ["apt-cache", "show", package["name"] + "=" + package["version"]],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout
        if "SHA256: " + package["sha256"] not in metadata:
            raise ValueError("font_checksum_mismatch_with_signed_snapshot_index")
        if not package["url"].startswith("https://deb.debian.org/debian/pool/main/f/"):
            raise ValueError("explicit_official_Debian_pool_only")
        with urlopen(package["url"], timeout=60) as source:
            data = source.read(16 * 1024**2 + 1)
        if len(data) > 16 * 1024**2 or hashlib.sha256(data).hexdigest() != package["sha256"]:
            raise ValueError("downloaded_font_package_checksum_failed")
        path = Path("/tmp") / package["filename"]
        path.write_bytes(data)
        subprocess.run(["dpkg", "--install", str(path)], check=True, timeout=30)
        path.unlink()


if __name__ == "__main__":
    main()
