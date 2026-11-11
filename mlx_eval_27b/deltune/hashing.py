"""sha256 + size + mtime_ns for every file in a directory (read-only)."""
import argparse, hashlib, json, os, sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def hash_dir(root: str) -> dict:
    root = os.path.abspath(root)
    files = {}
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            st = os.stat(p)
            files[os.path.relpath(p, root)] = {
                "sha256": sha256_file(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    assert files, f"no files under {root}"
    return {"root": root, "files": dict(sorted(files.items()))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = hash_dir(a.dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"{len(res['files'])} files -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
