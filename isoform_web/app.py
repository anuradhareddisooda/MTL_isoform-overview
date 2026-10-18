"""
Web front-end for tab1_overview_alignments_final.py

Users upload (or paste) a FASTA of isoform sequences (protein or cDNA), give the
canonical UniProt accession, and get the Tab 1 summary table on the page plus
CSV downloads. The pipeline script itself is NOT modified: each job runs it as
a subprocess (same CLI you use locally), so results match the command-line
output exactly and a heavy ESM2 run cannot stall or crash the web process.

Run locally:   python app.py
Production:    gunicorn --workers 1 --threads 8 --timeout 120 app:app
               (keep --workers 1: the concurrency limit is held in-process)
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file

BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = Path(os.environ.get("ISOFORM_SCRIPT", BASE_DIR / "tab1_overview_alignments_final.py"))
JOBS_DIR = Path(os.environ.get("JOBS_DIR", Path(tempfile.gettempdir()) / "isoform_jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 5))
MAX_RECORDS = int(os.environ.get("MAX_RECORDS", 50))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 30000))        # aa or nt per record
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", 1800))      # hard kill after 30 min
JOB_TTL_S = int(os.environ.get("JOB_TTL_S", 3600))             # delete job files after 1 h
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", 1))

SMALL_MODELS = ["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D"]
LARGE_MODELS = ["facebook/esm2_t30_150M_UR50D", "facebook/esm2_t33_650M_UR50D"]
ALLOWED_MODELS = SMALL_MODELS + (LARGE_MODELS if os.environ.get("ALLOW_LARGE_MODELS") == "1" else [])
DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"

# Set DMS_ENABLED=0 to hide ESM2 scoring (lets you deploy without torch/transformers on small servers)
DMS_ENABLED = os.environ.get("DMS_ENABLED", "1") == "1"

FEATURE_TYPES = ["auto", "Transmembrane", "Domain", "Repeat", "Zinc finger", "Coiled coil", "Motif"]

# Official UniProt accession pattern
UNIPROT_RE = re.compile(
    r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"
)
JOB_ID_RE = re.compile(r"[0-9a-f]{32}")

RAW_CSV = "tab1_overview.csv"
DISPLAY_CSV = "tab1_overview_display.csv"

RUN_SLOTS = threading.Semaphore(MAX_CONCURRENT_JOBS)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# ----------------------------------------------------------------------
# Job state lives on disk (status.json) so any request can read it.
# ----------------------------------------------------------------------
def read_status(job_dir: Path) -> dict:
    try:
        return json.loads((job_dir / "status.json").read_text())
    except (OSError, ValueError):
        return {}


def write_status(job_dir: Path, **fields) -> None:
    status = read_status(job_dir)
    status.update(fields)
    tmp = job_dir / "status.json.tmp"
    tmp.write_text(json.dumps(status))
    tmp.replace(job_dir / "status.json")


def get_job_dir(job_id: str) -> Path:
    if not JOB_ID_RE.fullmatch(job_id):
        abort(404)
    job_dir = JOBS_DIR / job_id
    if not job_dir.is_dir():
        abort(404)
    return job_dir


def purge_old_jobs() -> None:
    now = time.time()
    for d in JOBS_DIR.iterdir():
        if not d.is_dir() or not JOB_ID_RE.fullmatch(d.name):
            continue
        state = read_status(d).get("state")
        if state in ("queued", "running"):
            continue
        if now - d.stat().st_mtime > JOB_TTL_S:
            shutil.rmtree(d, ignore_errors=True)


def log_tail(job_dir: Path, max_chars: int = 6000) -> str:
    try:
        text = (job_dir / "run.log").read_text(errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------
def validate_fasta(text: str) -> tuple[str, int]:
    """Returns (clean FASTA text, number of records) or raises ValueError."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    records, header, seq = [], None, []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq)))
            header, seq = line[1:].strip(), []
        else:
            if header is None:
                raise ValueError("FASTA must start with a '>' header line.")
            seq.append(line)
    if header is not None:
        records.append((header, "".join(seq)))

    if not records:
        raise ValueError("No FASTA records found.")
    if len(records) > MAX_RECORDS:
        raise ValueError(f"Too many records ({len(records)}); the limit is {MAX_RECORDS}.")

    out = []
    for i, (h, s) in enumerate(records, start=1):
        h = re.sub(r"[^\x20-\x7E]", "", h)[:200].strip() or f"seq{i}"
        if not s:
            raise ValueError(f"Record '{h[:40]}' has no sequence.")
        if not re.fullmatch(r"[A-Za-z*]+", s):
            raise ValueError(f"Record '{h[:40]}' contains characters other than letters (and a trailing '*').")
        if len(s) > MAX_SEQ_LEN:
            raise ValueError(f"Record '{h[:40]}' is longer than {MAX_SEQ_LEN} characters.")
        out.append(f">{h}\n{s}")
    return "\n".join(out) + "\n", len(records)


# ----------------------------------------------------------------------
# Job runner
# ----------------------------------------------------------------------
def run_job(job_dir: Path, cmd: list[str]) -> None:
    with RUN_SLOTS:
        write_status(job_dir, state="running", started=time.time())
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        try:
            with open(job_dir / "run.log", "ab") as lf:
                proc = subprocess.Popen(cmd, cwd=job_dir, stdout=lf, stderr=subprocess.STDOUT, env=env)
                try:
                    rc = proc.wait(timeout=JOB_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    write_status(job_dir, state="error", finished=time.time(),
                                 error=f"Timed out after {JOB_TIMEOUT_S // 60} minutes.")
                    return
        except Exception as exc:  # could not even start the process
            write_status(job_dir, state="error", finished=time.time(), error=f"Could not start pipeline: {exc}")
            return

        if rc == 0 and (job_dir / DISPLAY_CSV).exists():
            write_status(job_dir, state="done", finished=time.time())
        else:
            lines = [l for l in log_tail(job_dir).splitlines() if l.strip()]
            hint = lines[-1][:300] if lines else ""
            write_status(job_dir, state="error", finished=time.time(),
                         error=f"Pipeline failed (exit code {rc}). {hint}".strip())


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.get("/")
def index():
    return render_template(
        "index.html",
        feature_types=FEATURE_TYPES,
        models=ALLOWED_MODELS,
        dms_enabled=DMS_ENABLED,
        default_model=DEFAULT_MODEL,
        max_records=MAX_RECORDS,
        max_upload_mb=MAX_UPLOAD_MB,
        ttl_min=JOB_TTL_S // 60,
    )


@app.get("/healthz")
def healthz():
    return "ok"


@app.post("/api/jobs")
def create_job():
    purge_old_jobs()

    uniprot_id = request.form.get("uniprot_id", "").strip().upper()
    if not UNIPROT_RE.fullmatch(uniprot_id):
        return jsonify(error="Enter a valid UniProt accession (e.g. Q13286)."), 400

    feature_type = request.form.get("feature_type", "auto")
    if feature_type not in FEATURE_TYPES:
        return jsonify(error="Unknown feature type."), 400

    run_dms = DMS_ENABLED and request.form.get("run_dms") == "on"
    model = request.form.get("dms_model", DEFAULT_MODEL)
    if run_dms and model not in ALLOWED_MODELS:
        return jsonify(error="Unknown or disabled ESM2 model."), 400

    # FASTA: uploaded file wins over pasted text; both optional (falls back to UniProt isoforms)
    fasta_text, n_records = None, 0
    upload = request.files.get("fasta")
    raw_text = ""
    if upload and upload.filename:
        raw_text = upload.read().decode("utf-8", errors="replace")
    elif request.form.get("fasta_text", "").strip():
        raw_text = request.form["fasta_text"]
    if raw_text.strip():
        try:
            fasta_text, n_records = validate_fasta(raw_text)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir()

    cmd = [sys.executable, "-u", str(SCRIPT_PATH),
           "--uniprot-id", uniprot_id,
           "--feature-type", feature_type,
           "--out", RAW_CSV,
           "--out-display", DISPLAY_CSV,
           "--device", "cpu"]
    if fasta_text:
        (job_dir / "input.fasta").write_text(fasta_text)
        cmd += ["--fasta", "input.fasta"]
    if run_dms:
        cmd += ["--dms-model", model]
    else:
        cmd += ["--no-dms"]

    write_status(job_dir, state="queued", uniprot_id=uniprot_id, created=time.time(),
                 source="FASTA upload" if fasta_text else "UniProt isoforms", n_records=n_records)
    threading.Thread(target=run_job, args=(job_dir, cmd), daemon=True).start()
    return jsonify(job_id=job_id), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job_dir = get_job_dir(job_id)
    status = read_status(job_dir)
    payload = {
        "state": status.get("state", "unknown"),
        "error": status.get("error"),
        "uniprot_id": status.get("uniprot_id"),
        "source": status.get("source"),
        "log": log_tail(job_dir),
    }
    if status.get("state") == "done":
        df = pd.read_csv(job_dir / DISPLAY_CSV, dtype=str, keep_default_na=False)
        payload["columns"] = list(df.columns)
        payload["rows"] = df.values.tolist()
    return jsonify(payload)


@app.get("/api/jobs/<job_id>/download/<kind>")
def download(job_id, kind):
    job_dir = get_job_dir(job_id)
    if read_status(job_dir).get("state") != "done":
        abort(404)
    files = {"display": DISPLAY_CSV, "raw": RAW_CSV}
    if kind not in files:
        abort(404)
    uniprot_id = read_status(job_dir).get("uniprot_id", "protein")
    suffix = "display" if kind == "display" else "raw"
    return send_file(job_dir / files[kind], mimetype="text/csv", as_attachment=True,
                     download_name=f"{uniprot_id}_isoform_overview_{suffix}.csv")


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error=f"File too large (limit {MAX_UPLOAD_MB} MB)."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
