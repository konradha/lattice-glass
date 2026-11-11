#!/usr/bin/env python3
"""Recover one verbatim source receipt/snapshot from the benchmark archive."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import tarfile


def extract(manifest_path, root, source_path, destination):
    archive_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    record = None
    for part in manifest["file_manifests"]:
        with gzip.open(archive_dir / part["file"], "rt", encoding="utf-8") as stream:
            for line in stream:
                candidate = json.loads(line)
                if candidate["root"] == root and candidate["path"] == source_path:
                    record = candidate
                    break
        if record is not None:
            break
    if record is None:
        with (archive_dir / manifest["exclusions"]["file"]).open(encoding="utf-8") as stream:
            for line in stream:
                candidate = json.loads(line)
                if candidate["root"] == root and candidate["path"] == source_path:
                    raise ValueError("Source excluded: " + candidate["reason"])
        raise ValueError("Source not inventoried: " + root + ":" + source_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary_path = Path(output.name)
            requests = {}
            offset = 0
            for chunk in record["chunks"]:
                members = requests.setdefault(chunk["part"], {})
                members.setdefault(chunk["member"], []).append((offset, chunk))
                offset += chunk["size"]
            for part, members in requests.items():
                with tarfile.open(archive_dir / part, "r|xz") as archive:
                    for member in archive:
                        targets = members.pop(member.name, None)
                        if targets is None:
                            continue
                        expected = targets[0][1]
                        if not member.isfile() or member.size != expected["size"]:
                            raise ValueError("Archive chunk has incorrect type or size")
                        digest = hashlib.sha256()
                        chunk_size = 0
                        with archive.extractfile(member) as stream:
                            while data := stream.read(1024 * 1024):
                                for start, chunk in targets:
                                    output.seek(start + chunk_size)
                                    output.write(data)
                                digest.update(data)
                                chunk_size += len(data)
                        if digest.hexdigest() != expected["sha256"] or chunk_size != expected["size"]:
                            raise ValueError("Archive chunk failed SHA256/size validation")
                        if not members:
                            break
                if members:
                    raise ValueError("Archive is missing required chunks")
            output.flush()
            output.seek(0)
            whole = hashlib.sha256()
            total = 0
            while data := output.read(1024 * 1024):
                whole.update(data)
                total += len(data)
            if whole.hexdigest() != record["sha256"] or total != record["size"]:
                raise ValueError("Recovered source failed SHA256/size validation")
        temporary_path.replace(destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--root", choices=("main", "opt3"), required=True)
    parser.add_argument("--path", required=True, help="Original relative path; nested ZIP paths use !/member")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = extract(args.manifest, args.root, args.path, args.output)
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(1, str(error) + "\n")
    print(f"{record['sha256']}  {args.output}")


if __name__ == "__main__":
    main()
