#!/usr/bin/env python3
"""Upload all files under ~/workspace/ashare/ to Google Drive folder ashare_backup.
Resumable via ~/workspace/ashare/gdrive_manifest.json (skip when size+mtime match).
Logs to ~/workspace/ashare/gdrive_upload.log.
"""
import json
import os
import re
import subprocess
import time

SRC = "/home/hatch/workspace/ashare"
ROOT_ID = "1ncpJB7IiKHqFvAM7ZP3I-DZBovw1-VZ1"
MANIFEST = os.path.join(SRC, "gdrive_manifest.json")
LOG = os.path.join(SRC, "gdrive_upload.log")
CLI = "/opt/hatch/bin/hatch_gws_cli"
PRESERVATION_ID = "1ZLkRuBwgjlbDc4dH2MftWoFMKZlOHU9m"
# Working files created by this task itself; excluded to avoid uploading a moving target.
EXCLUDE = {"gdrive_manifest.json", "gdrive_manifest.json.tmp",
           "gdrive_upload.log", "gdrive_upload.py"}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run_cli(args, timeout=1800):
    p = subprocess.run([CLI] + args, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def save_manifest(m):
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f)
    os.replace(tmp, MANIFEST)


def ensure_folder(rel_dir, folders):
    if not rel_dir or rel_dir == ".":
        return ROOT_ID
    if rel_dir in folders:
        return folders[rel_dir]
    parent_rel = os.path.dirname(rel_dir)
    parent_id = ensure_folder(parent_rel, folders)
    name = os.path.basename(rel_dir)
    body = json.dumps({"name": name,
                       "mimeType": "application/vnd.google-apps.folder",
                       "parents": [parent_id]})
    rc, so, se = run_cli(["drive", "files", "create",
                          "--params", '{"ignoreDefaultVisibility":true}',
                          "--json", body], timeout=120)
    mm = re.search(r'"id"\s*:\s*"([^"]+)"', so)
    if rc != 0 or not mm:
        raise RuntimeError(f"mkdir failed {rel_dir}: rc={rc} "
                           f"out={so[:300]} err={se[:300]}")
    fid = mm.group(1)
    folders[rel_dir] = fid
    log(f"mkdir {rel_dir} -> {fid}")
    return fid


def upload_file(abs_path, parent_id):
    last_err = ""
    for attempt in (1, 2):
        try:
            rc, so, se = run_cli(["drive", "+upload", abs_path,
                                  "--parent", parent_id], timeout=3600)
        except subprocess.TimeoutExpired:
            rc, so, se = -1, "", "timeout"
        mm = re.search(r'"id"\s*:\s*"([^"]+)"', so)
        if rc == 0 and mm:
            return mm.group(1), None
        last_err = (so + se)[-600:]
        log(f"  attempt {attempt} failed {os.path.basename(abs_path)}: "
            f"{last_err[:200]}")
        time.sleep(5)
    return None, last_err


def auth_like(err):
    e = err.lower()
    return any(k in e for k in ("auth", "unauthor", "permission denied",
                               "401", "403", "invalid_grant", "revoked"))


def main():
    m = {"root_folder_id": ROOT_ID, "folders": {}, "files": {}, "failed": []}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            m = json.load(f)
    files = m["files"]
    folders = m["folders"]
    failed = set(m.get("failed", []))

    # PRESERVATION.md already uploaded by parent agent; record without re-upload.
    pres = "PRESERVATION.md"
    if pres not in files:
        st = os.stat(os.path.join(SRC, pres))
        files[pres] = {"id": PRESERVATION_ID, "size": st.st_size,
                       "mtime": int(st.st_mtime)}
        log(f"recorded pre-uploaded PRESERVATION.md id={PRESERVATION_ID}")

    out = subprocess.run(["find", SRC, "-type", "f", "-print"],
                         capture_output=True, text=True).stdout.splitlines()
    rel_paths = sorted(os.path.relpath(p, SRC) for p in out)
    rel_paths = [r for r in rel_paths if r not in EXCLUDE]
    n = len(rel_paths)
    log(f"start: {n} files to consider")

    uploaded = 0
    skipped = 0
    total_bytes = 0
    consec_fail = 0
    since_save = 0

    for i, rel in enumerate(rel_paths, 1):
        abs_p = os.path.join(SRC, rel)
        try:
            st = os.stat(abs_p)
        except FileNotFoundError:
            log(f"missing, skip: {rel}")
            continue
        size, mtime = st.st_size, int(st.st_mtime)
        rec = files.get(rel)
        if rec and rec.get("size") == size and rec.get("mtime") == mtime \
                and rec.get("id"):
            skipped += 1
            continue
        try:
            parent_id = ensure_folder(os.path.dirname(rel), folders)
        except RuntimeError as e:
            log(str(e))
            failed.add(rel)
            consec_fail += 1
        else:
            fid, err = upload_file(abs_p, parent_id)
            if fid:
                files[rel] = {"id": fid, "size": size, "mtime": mtime}
                failed.discard(rel)
                uploaded += 1
                total_bytes += size
                consec_fail = 0
            else:
                failed.add(rel)
                consec_fail += 1
                log(f"FAILED {rel}")
                if consec_fail >= 5 and auth_like(err or ""):
                    log("ABORT: 5 consecutive auth-like failures, stopping")
                    break
        since_save += 1
        if since_save >= 20:
            m["failed"] = sorted(failed)
            save_manifest(m)
            since_save = 0
            log(f"progress {i}/{n} uploaded={uploaded} skipped={skipped} "
                f"failed={len(failed)} bytes={total_bytes}")

    m["failed"] = sorted(failed)
    save_manifest(m)
    log(f"DONE total={n} uploaded={uploaded} skipped={skipped} "
        f"failed={len(failed)} bytes={total_bytes}")


if __name__ == "__main__":
    main()
