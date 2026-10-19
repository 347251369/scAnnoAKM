"""Download the verified retina seed42 checkpoint from the GitHub Release."""
import hashlib
from pathlib import Path
import sys
import tempfile
import urllib.request

URL = (
    "https://github.com/347251369/scAnnoAKM/releases/download/"
    "retina-seed42-demo-v1/train_model_seed42.pth"
)
EXPECTED_BYTES = 980379474
EXPECTED_SHA256 = "ee1e8513cfec466dc2c2ec222bee1af7ac5d26fea3a4bb905a1f62fa4c915a55"
TARGET = Path(__file__).resolve().parent / "weights" / "train_model_seed42.pth"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if TARGET.exists():
        if TARGET.stat().st_size != EXPECTED_BYTES or sha256(TARGET) != EXPECTED_SHA256:
            raise RuntimeError(f"Checkpoint verification failed: {TARGET}. Replace it with the Release asset.")
        print("Checkpoint SHA256 verified.", flush=True)
        return

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "scAnnoAKM-demo"})
        print("Downloading retina seed42 checkpoint (980 MB)...", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=TARGET.parent, suffix=".part", delete=False) as handle:
                temporary = Path(handle.name)
                digest = hashlib.sha256()
                received = 0
                last_progress = -1
                while True:
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > EXPECTED_BYTES:
                        raise RuntimeError("Downloaded checkpoint is larger than expected.")
                    handle.write(chunk)
                    digest.update(chunk)
                    progress = received * 100 // EXPECTED_BYTES
                    if progress // 10 != last_progress:
                        print(f"Download: {progress}%", flush=True)
                        last_progress = progress // 10
        if received != EXPECTED_BYTES or digest.hexdigest() != EXPECTED_SHA256:
            raise RuntimeError("Downloaded checkpoint failed size or SHA256 verification.")
        temporary.replace(TARGET)
        temporary = None
        print("Checkpoint downloaded and SHA256 verified.", flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as error:
        print(f"Checkpoint error: {error}", file=sys.stderr)
        sys.exit(1)
