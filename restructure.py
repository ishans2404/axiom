#!/usr/bin/env python3
"""
restructure.py

Post-process a built wheel to move the _axiom extension from
<name>.data/data/_axiom*.pyd to the wheel root (renamed to _axiom.pyd),
and update RECORD.

Usage:
    python restructure.py path/to/axiom-*.whl
"""

import sys
import os
import zipfile
import tempfile
import shutil
import hashlib
import base64

def hash_and_size(path):
    """Return (hash_str, size) for RECORD."""
    h = hashlib.sha256()
    size = 0
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    digest = base64.urlsafe_b64encode(h.digest()).rstrip(b'=').decode('ascii')
    return f"sha256={digest}", str(size)

def main():
    if len(sys.argv) != 2:
        print("Usage: python restructure.py path/to/axiom-*.whl")
        sys.exit(1)

    wheel_path = sys.argv[1]
    if not os.path.isfile(wheel_path):
        print(f"Error: wheel not found: {wheel_path}")
        sys.exit(1)

    # Prepare temp dir
    tempdir = tempfile.mkdtemp(prefix="wheelfix-")
    with zipfile.ZipFile(wheel_path, 'r') as zin:
        zin.extractall(tempdir)

    # Find extension and data dirs
    data_root = None
    for name in os.listdir(tempdir):
        if name.endswith(".data"):
            data_root = name
            break
    if not data_root:
        print("No .data directory found; nothing to do.")
        shutil.rmtree(tempdir)
        sys.exit(0)

    data_dir = os.path.join(tempdir, data_root, "data")
    moved_pyd = None
    for fname in os.listdir(data_dir):
        if fname.startswith("_axiom") and fname.endswith((".pyd", ".so")):
            ext = os.path.splitext(fname)[1]
            src = os.path.join(data_dir, fname)
            dst = os.path.join(tempdir, "_axiom" + ext)
            shutil.move(src, dst)
            moved_pyd = "_axiom" + ext
            break

    # Remove the data directory
    shutil.rmtree(os.path.join(tempdir, data_root))

    # Update RECORD
    dist_info = next(d for d in os.listdir(tempdir) if d.endswith(".dist-info"))
    record_path = os.path.join(tempdir, dist_info, "RECORD")
    lines = []
    with open(record_path, newline='', encoding='utf-8') as recf:
        for line in recf:
            parts = line.rstrip("\n").split(",")
            path = parts[0]
            # skip any entries under data/
            if path.startswith(f"{data_root}/"):
                continue
            lines.append(parts)

    # Add moved pyd entry (renamed)
    if moved_pyd:
        hash_, size = hash_and_size(os.path.join(tempdir, moved_pyd))
        lines.append([moved_pyd, hash_, size])

    # The RECORD itself must have an empty hash and size
    lines.append([f"{dist_info}/RECORD", "", ""])

    # Write back
    with open(record_path, "w", newline='', encoding='utf-8') as recf:
        for parts in lines:
            recf.write(",".join(parts) + "\n")

    # Rebuild wheel
    new_wheel = wheel_path + ".fixed"
    with zipfile.ZipFile(new_wheel, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for root, _, files in os.walk(tempdir):
            for fname in files:
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, tempdir)
                arcname = rel_path.replace(os.path.sep, "/")
                zout.write(abs_path, arcname)

    print(f"Fixed wheel written to {new_wheel}")
    shutil.rmtree(tempdir)

if __name__ == "__main__":
    main()
