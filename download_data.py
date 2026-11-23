"""Fetch or verify the exact versioned PhysioNet EDF files used in the study."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from http.client import HTTPException
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
SOURCES = (
    "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/",
    "https://physionet.org/files/eegmmidb/1.0.0/",
)
RUNS = (4, 8, 12)


def parse_subjects(value):
    parts = value.split(",")
    numbers = set()
    for part in parts:
        ends = [int(item.removeprefix("S")) for item in part.upper().split("-")]
        if len(ends) > 2 or not (1 <= ends[0] <= ends[-1] <= 109):
            raise ValueError(f"Invalid subject range: {part}")
        numbers.update(range(ends[0], ends[-1] + 1))
    return [f"S{number:03d}" for number in sorted(numbers)]


def source_manifest():
    result = {}
    for line in (ROOT / "data" / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        result[relative.removeprefix("*").removeprefix("./")] = digest
    return result


def expected_files(subjects):
    manifest = source_manifest()
    return {
        f"{subject}/{subject}R{run:02d}.edf": manifest[f"{subject}/{subject}R{run:02d}.edf"]
        for subject in subjects for run in RUNS
    }


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(relative, expected, root, verify_only):
    path = root / relative
    if path.is_file() and sha256(path) == expected:
        return "verified"
    if verify_only:
        raise ValueError("missing file or SHA-256 mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        temporary = None
        try:
            request = Request(SOURCES[attempt % len(SOURCES)] + relative,
                              headers={"User-Agent": "EEG-MI-course-project/1.0"})
            with urlopen(request, timeout=30) as response, NamedTemporaryFile(
                dir=path.parent, suffix=".part", delete=False
            ) as file:
                temporary = Path(file.name)
                digest = hashlib.sha256()
                while chunk := response.read(1024 * 256):
                    file.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError("SHA-256 mismatch")
            temporary.replace(path)
            return "downloaded"
        except (OSError, ValueError, HTTPException) as error:
            if attempt == 3:
                raise OSError(f"all attempts failed: {error}") from error
            time.sleep(attempt + 1)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="1-109", help="Example: 1-12,24,S031")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    try:
        files = expected_files(parse_subjects(args.subjects))
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    failures = []
    counts = {"downloaded": 0, "verified": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(fetch, name, digest, args.raw_dir, args.verify_only): name
                for name, digest in files.items()}
        for index, job in enumerate(as_completed(jobs), 1):
            try:
                counts[job.result()] += 1
            except (OSError, ValueError) as error:
                failures.append(jobs[job])
                print(f"FAILED {jobs[job]}: {error}", file=sys.stderr, flush=True)
            if index % 12 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} done, {len(failures)} failed", flush=True)
    print(f"Downloaded {counts['downloaded']}, verified {counts['verified']}, failed {len(failures)}")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
