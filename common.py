"""Shared settings and helpers for the Gradio app."""

import os
import re
import shutil
import tempfile
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = Path(os.environ.get("ISOFORM_SCRIPT", BASE_DIR / "tab1_overview_alignments_final.py"))
EVIDENCE_SCRIPT = Path(os.environ.get("EVIDENCE_SCRIPT", BASE_DIR / "isoform_evidence.py"))
JOBS_DIR = Path(os.environ.get("JOBS_DIR", Path(tempfile.gettempdir()) / "isoform_jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 5))
MAX_RECORDS = int(os.environ.get("MAX_RECORDS", 50))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 30000))       # aa or nt per record
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", 1800))     # kill a step after 30 min
JOB_TTL_S = int(os.environ.get("JOB_TTL_S", 3600))            # delete job files after 1 h
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", 1))

# ESM2 scoring (Mean Predicted Score). Needs torch + transformers; set DMS_ENABLED=0 to hide it.
DMS_ENABLED = os.environ.get("DMS_ENABLED", "1") == "1"
SMALL_MODELS = ["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D"]
LARGE_MODELS = ["facebook/esm2_t30_150M_UR50D", "facebook/esm2_t33_650M_UR50D"]
ALLOWED_MODELS = SMALL_MODELS + (LARGE_MODELS if os.environ.get("ALLOW_LARGE_MODELS") == "1" else [])
DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"

FEATURE_TYPES = ["auto", "Transmembrane", "Domain", "Repeat", "Zinc finger", "Coiled coil", "Motif"]

# Official UniProt accession pattern
UNIPROT_RE = re.compile(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}")

EVIDENCE_CSV = "isoform_evidence.csv"
RAW_CSV = "tab1_overview.csv"
DISPLAY_CSV = "tab1_overview_display.csv"
FIGURE_PREFIX = "isoform_map"
FIGURE_PNG, FIGURE_SVG = FIGURE_PREFIX + ".png", FIGURE_PREFIX + ".svg"


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


def purge_old_jobs() -> None:
    now = time.time()
    for d in JOBS_DIR.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > JOB_TTL_S:
            shutil.rmtree(d, ignore_errors=True)


def log_tail(job_dir: Path, max_chars: int = 6000) -> str:
    try:
        return (job_dir / "run.log").read_text(errors="replace")[-max_chars:]
    except OSError:
        return ""
