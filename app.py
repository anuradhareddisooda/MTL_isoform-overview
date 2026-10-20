"""
Gradio front-end (for a free Hugging Face Gradio Space) for the isoform pipeline.

Runs two steps, each as a subprocess:
  1. tab1_overview_alignments_final.py  -> overview table (incl. Mean Predicted Score when ESM2 is on)
  2. isoform_evidence.py                -> evidence table + domain map
If step 2 fails, the overview table is still shown.
"""

import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import gradio as gr
import pandas as pd

import common as core

# This tool runs on CPU. A ZeroGPU Space still needs one @spaces.GPU function to start; this one is never
# called. On a normal CPU Space the decorator does nothing, and without the `spaces` package it is skipped.
try:
    import spaces

    @spaces.GPU(duration=10)
    def _zerogpu_placeholder():
        return True
except Exception:
    pass

NOTE_OVERVIEW = (
    "Stability Risk and Coverage Heuristic Risk are descriptive flags, not validated pathogenicity calls "
    "(the coverage cutoffs are round-number heuristics). Mean Predicted Score, when present, is an ESM2 "
    "wt-marginal in-silico score, not an experimental DMS measurement. pLDDT is taken from the canonical "
    "AlphaFold model and mapped onto retained residues."
)
NOTE_MAP = (
    "Each isoform is drawn on the canonical protein. Solid colour = retained part of a feature, dashed outline = "
    "lost part, red \u00d7 = lost site, dashed red line = deleted canonical region, magenta box = novel residues "
    "(not to scale). Only features annotated in UniProt (or InterPro domains when UniProt has none) are drawn; "
    "signal peptides are shown only if UniProt annotates one. Matching blocks shorter than 5 residues are ignored."
)
NOTE_EVIDENCE = (
    "**Event(s)/class** come from the sequence alignment. A deletion with the downstream sequence preserved is "
    "in-frame at the protein level; a novel C-terminal tail suggests a frameshift or alternative terminal exon. "
    "**NMD is not assessed** (it needs exon coordinates). "
    "**Signal peptide / TM / domains / sites** use the canonical UniProt feature table; 'none annotated in "
    "UniProt' means UniProt has no such annotation, not that the protein has none (this tool does not predict "
    "signal peptides). **Topology flag** is a parity heuristic (an odd number of TMs lost upstream of a retained "
    "TM may invert its orientation); verify with a topology predictor. **Junction delta logp** (only if ESM2 "
    "scoring is on) is the mean change in masked log-probability of the residues flanking each new junction, "
    "isoform versus canonical; more negative means a less plausible junction. It is not calibrated against "
    "experimental or clinical outcomes."
)


def _table(path: Path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _run_step(job_dir: Path, cmd: list, title: str):
    """Generator: yields once per poll while the step runs; returns the exit code (None = timed out)."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
           "TOKENIZERS_PARALLELISM": "false"}
    with open(job_dir / "run.log", "ab") as lf:
        lf.write(f"\n=== {title} ===\n".encode())
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=job_dir, stdout=lf, stderr=subprocess.STDOUT, env=env)
    start = time.time()
    try:
        while proc.poll() is None:
            if time.time() - start > core.JOB_TIMEOUT_S:
                proc.kill()
                proc.wait()
                return None
            yield
            time.sleep(2)
    finally:  # also runs if the user closes the page and Gradio cancels the job
        if proc.poll() is None:
            proc.kill()
    return proc.returncode


def _last_line(job_dir: Path, startswith: str = "") -> str:
    lines = [l for l in core.log_tail(job_dir).splitlines() if l.strip() and l.startswith(startswith)]
    return lines[-1][:300] if lines else ""


def analyse(uniprot_id, fasta_path, fasta_text, feature_type, run_dms, model):
    # ---- input checks ----
    uid = (uniprot_id or "").strip().upper()
    if not core.UNIPROT_RE.fullmatch(uid):
        raise gr.Error("Enter a valid UniProt accession (e.g. Q13286).")
    if feature_type not in core.FEATURE_TYPES:
        raise gr.Error("Unknown feature type.")
    run_dms = bool(run_dms) and core.DMS_ENABLED
    if run_dms and model not in core.ALLOWED_MODELS:
        raise gr.Error("Unknown or disabled ESM2 model.")

    raw_text = ""
    if fasta_path:
        if Path(fasta_path).stat().st_size > core.MAX_UPLOAD_MB * 1024 * 1024:
            raise gr.Error(f"File too large (limit {core.MAX_UPLOAD_MB} MB).")
        raw_text = Path(fasta_path).read_text(errors="replace")
    elif (fasta_text or "").strip():
        raw_text = fasta_text
    fasta_clean = None
    if raw_text.strip():
        try:
            fasta_clean, _ = core.validate_fasta(raw_text)
        except ValueError as exc:
            raise gr.Error(str(exc))

    core.purge_old_jobs()
    job_dir = core.JOBS_DIR / uuid.uuid4().hex
    job_dir.mkdir(parents=True)

    cmd = [sys.executable, "-u", str(core.SCRIPT_PATH), "--uniprot-id", uid, "--feature-type", feature_type,
           "--out", core.RAW_CSV, "--out-display", core.DISPLAY_CSV, "--device", "cpu"]
    ev_cmd = [sys.executable, "-u", str(core.EVIDENCE_SCRIPT), "--uniprot-id", uid, "--out", core.EVIDENCE_CSV,
              "--device", "cpu", "--figure-prefix", core.FIGURE_PREFIX]
    if fasta_clean:
        (job_dir / "input.fasta").write_text(fasta_clean)
        cmd += ["--fasta", "input.fasta"]
        ev_cmd += ["--fasta", "input.fasta"]
    if run_dms:
        cmd += ["--dms-model", model]
        ev_cmd += ["--junction-model", model]
    else:
        cmd += ["--no-dms"]

    keep = gr.update()

    def ui(status, overview=keep, figure=keep, evidence=keep, files=keep):
        return status, core.log_tail(job_dir), overview, figure, evidence, files

    # ---- step 1: overview table ----
    msg1 = "Running overview table" + (" with ESM2 scoring (slow; the first run downloads the model)" if run_dms else "") + "\u2026"
    yield ui(msg1)
    gen, rc = _run_step(job_dir, cmd, "Overview table"), None
    while True:
        try:
            next(gen)
            yield ui(msg1)
        except StopIteration as stop:
            rc = stop.value
            break
    if rc != 0 or not (job_dir / core.DISPLAY_CSV).exists():
        reason = "timed out" if rc is None else f"exit code {rc}"
        yield ui(f"**Overview step failed ({reason}).** {_last_line(job_dir)}")
        return
    overview = _table(job_dir / core.DISPLAY_CSV)
    msg2 = "Overview table done. Running evidence table and domain map\u2026"
    yield ui(msg2, overview=overview)

    # ---- step 2: evidence table + domain map (failure keeps the overview table) ----
    gen, rc = _run_step(job_dir, ev_cmd, "Evidence table"), None
    while True:
        try:
            next(gen)
            yield ui(msg2, overview=overview)
        except StopIteration as stop:
            rc = stop.value
            break

    out_dir = job_dir / "downloads"
    out_dir.mkdir(exist_ok=True)
    files = []
    for src, name in [(core.DISPLAY_CSV, "overview_display.csv"), (core.RAW_CSV, "overview_raw.csv"),
                      (core.EVIDENCE_CSV, "evidence.csv"), (core.FIGURE_PNG, "domain_map.png"),
                      (core.FIGURE_SVG, "domain_map.svg")]:
        if (job_dir / src).exists():
            dst = out_dir / f"{uid}_{name}"
            shutil.copy(job_dir / src, dst)
            files.append(str(dst))

    has_ev = (job_dir / core.EVIDENCE_CSV).exists()
    has_fig = (job_dir / core.FIGURE_PNG).exists()
    evidence = _table(job_dir / core.EVIDENCE_CSV) if has_ev else None
    figure = str(job_dir / core.FIGURE_PNG) if has_fig else None

    problems = []
    if not has_ev:
        problems.append("the evidence table could not be produced "
                        f"({'timed out' if rc is None else f'exit code {rc}'}): {_last_line(job_dir)}")
    if not has_fig:
        why = _last_line(job_dir, "Figure not created") or "see the log"
        problems.append(f"the domain map could not be drawn ({why})")
    if problems:
        status = f"**Overview table done ({len(overview)} row(s)), but** " + "; ".join(problems) + "."
    else:
        status = f"**Done** \u2014 {len(overview)} row(s) for {uid}. Download the CSVs and figure at the bottom."
    yield ui(status, overview=overview, figure=figure, evidence=evidence, files=files or None)


with gr.Blocks(title="MTL Isoform Overview") as demo:
    gr.Markdown(
        "# Protein isoform overview & alignments\n"
        "Maps each isoform onto the canonical UniProt sequence and reports coverage, feature retention, "
        "AlphaFold pLDDT and ESM2 Mean Predicted Score, plus a domain map and an evidence table. "
        f"Uploaded sequences and results are deleted after about {core.JOB_TTL_S // 60} minutes; "
        "download your files when the run finishes."
    )
    with gr.Row():
        with gr.Column(scale=1):
            uid_in = gr.Textbox(label="Canonical UniProt accession", value="Q13286",
                                info="The reference your isoforms are compared against")
            fasta_in = gr.File(label=f"Isoform FASTA (protein or cDNA; up to {core.MAX_RECORDS} records)",
                               file_types=[".fa", ".fasta", ".faa", ".fna", ".txt"], type="filepath")
            fasta_txt = gr.Textbox(label="...or paste FASTA text", lines=4,
                                   info="Leave both empty to use the isoforms annotated in UniProt")
            feat_in = gr.Dropdown(core.FEATURE_TYPES, value="auto", label="Feature to track")
            dms_in = gr.Checkbox(label="Compute ESM2 scores (Mean Predicted Score and junction scores) - slower",
                                 value=core.DMS_ENABLED, visible=core.DMS_ENABLED)
            model_in = gr.Dropdown(core.ALLOWED_MODELS, value=core.DEFAULT_MODEL, label="ESM2 model",
                                   visible=core.DMS_ENABLED)
            go = gr.Button("Run analysis", variant="primary")
        with gr.Column(scale=2):
            status = gr.Markdown()
            log = gr.Textbox(label="Log", lines=12, max_lines=12, interactive=False)

    gr.Markdown("## Overview table")
    overview_out = gr.Dataframe(interactive=False, wrap=True)
    gr.Markdown(NOTE_OVERVIEW)
    gr.Markdown("## Domain map")
    fig_out = gr.Image(label="Isoforms on the canonical protein", type="filepath", interactive=False)
    gr.Markdown(NOTE_MAP)
    gr.Markdown("## Evidence table")
    evidence_out = gr.Dataframe(interactive=False, wrap=True)
    gr.Markdown(NOTE_EVIDENCE)
    files_out = gr.File(label="Downloads (CSV tables and figure)", file_count="multiple", interactive=False)

    go.click(analyse, inputs=[uid_in, fasta_in, fasta_txt, feat_in, dms_in, model_in],
             outputs=[status, log, overview_out, fig_out, evidence_out, files_out])

demo.queue(default_concurrency_limit=core.MAX_CONCURRENT_JOBS)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
