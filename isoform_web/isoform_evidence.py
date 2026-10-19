"""
Isoform evidence table (companion to tab1_overview_alignments_final.py)
=======================================================================
Instead of collapsing everything into one risk label, report separate,
independent lines of evidence for each isoform relative to the canonical:

  1. Event type, from the sequence alignment (deletion / insertion / replacement /
     truncation / novel C-terminal tail) and what that implies about the reading
     frame. NMD is NOT assessed: that needs exon coordinates, which a protein
     FASTA does not contain.
  2. What the lost residues contained, using the canonical UniProt feature table:
     signal peptide, TM helices, domains/repeats/motifs, and residue-level sites
     (active/binding sites, disulfides, glycosylation, modified residues, ...),
     plus a simple TM-parity flag for possible topology inversion.
  3. (Optional, needs torch + transformers) ESM2 junction score: for residues
     flanking each new junction, the change in masked log-probability of the true
     residue in the isoform versus the same residue in the canonical sequence.

Nothing here is calibrated against experimental or clinical outcomes.

The pipeline script is imported, not modified: its alignment, UniProt and FASTA
functions are reused so results stay consistent with the Tab 1 table.

Run: python isoform_evidence.py --uniprot-id Q13286 --fasta isoforms.fa \
         --out evidence.csv [--junction-model facebook/esm2_t12_35M_UR50D]
"""

import argparse
import importlib.util
import math
import os
from pathlib import Path

import pandas as pd

SIGNAL_TYPES = ("Signal", "Transit peptide")
TM_TYPE = "Transmembrane"
DOMAIN_TYPES = ("Domain", "Repeat", "Zinc finger", "Motif", "Coiled coil")
SITE_TYPES = ("Active site", "Binding site", "Site", "Disulfide bond", "Cross-link",
              "Modified residue", "Glycosylation", "Lipidation")
ENDPOINT_ONLY = ("Disulfide bond", "Cross-link")  # only the two linked residues matter
ALL_TYPES = set(SIGNAL_TYPES) | {TM_TYPE} | set(DOMAIN_TYPES) | set(SITE_TYPES)

DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"


def load_pipeline_module():
    """Imports the Tab 1 script that sits next to this file (override with PIPELINE_SCRIPT)."""
    path = Path(os.environ.get("PIPELINE_SCRIPT",
                               Path(__file__).with_name("tab1_overview_alignments_final.py")))
    spec = importlib.util.spec_from_file_location("tab1_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# 1. Alignment events
# ----------------------------------------------------------------------
def analyse_alignment(canon: str, iso: str, blocks) -> dict:
    """
    From difflib matching blocks, derive which canonical residues are retained,
    a per-isoform-residue map back to canonical, a list of events, a list of
    event classes, and the isoform positions of new junctions.
    Positions in `events` text are 1-based canonical numbering.
    """
    Lc, Li = len(canon), len(iso)
    blocks = [b for b in blocks if b.size > 0]
    retained = [False] * Lc
    iso_to_can = [None] * Li
    for b in blocks:
        for k in range(b.size):
            retained[b.a + k] = True
            iso_to_can[b.b + k] = b.a + k

    events, classes, junctions = [], [], []

    def add_class(c):
        if c not in classes:
            classes.append(c)

    if not blocks:
        return dict(events=["no shared sequence with canonical"], classes=["no match"],
                    retained=retained, iso_to_can=iso_to_can, junctions=[])

    first, last = blocks[0], blocks[-1]

    # N-terminus
    if first.a > 0 or first.b > 0:
        if first.b == 0:
            events.append(f"N-terminal loss: canonical 1-{first.a}")
            add_class("N-terminal truncation")
        elif first.a == 0:
            events.append(f"N-terminal novel sequence: {first.b} aa")
            add_class("alternative N-terminus")
            junctions.append((first.b, "N-term novel"))
        else:
            events.append(f"N-terminal canonical 1-{first.a} replaced by {first.b} novel aa")
            add_class("alternative N-terminus")
            junctions.append((first.b, "N-term replaced"))

    # Internal gaps between consecutive blocks
    for cur, nxt in zip(blocks, blocks[1:]):
        gap_c = nxt.a - (cur.a + cur.size)
        gap_i = nxt.b - (cur.b + cur.size)
        lo, hi = cur.a + cur.size + 1, nxt.a  # 1-based canonical span of the gap
        if gap_c > 0 and gap_i == 0:
            events.append(f"deletion canonical {lo}-{hi} ({gap_c} aa)")
            add_class("internal deletion, downstream sequence preserved (in-frame)")
            junctions.append((nxt.b, f"del {lo}-{hi}"))
        elif gap_c == 0 and gap_i > 0:
            events.append(f"insertion of {gap_i} novel aa after canonical {lo - 1}")
            add_class("internal insertion")
            junctions.append((cur.b + cur.size, f"ins after {lo - 1}"))
            junctions.append((nxt.b, f"ins end ({gap_i} aa)"))
        elif gap_c > 0 and gap_i > 0:
            events.append(f"canonical {lo}-{hi} ({gap_c} aa) replaced by {gap_i} novel aa")
            add_class("internal replacement with novel residues")
            junctions.append((cur.b + cur.size, f"repl {lo}-{hi} start"))
            junctions.append((nxt.b, f"repl {lo}-{hi} end"))

    # C-terminus
    rem_c = Lc - (last.a + last.size)
    rem_i = Li - (last.b + last.size)
    if rem_c > 0 or rem_i > 0:
        lo = last.a + last.size + 1
        if rem_i == 0:
            events.append(f"C-terminal loss: canonical {lo}-{Lc} ({rem_c} aa)")
            add_class("C-terminal truncation")
        elif rem_c == 0:
            events.append(f"C-terminal extension: {rem_i} novel aa")
            add_class("C-terminal extension")
            junctions.append((last.b + last.size, "C-term extension"))
        else:
            events.append(f"C-terminal novel tail: {rem_i} aa replace canonical {lo}-{Lc}")
            add_class("novel C-terminal tail (frameshift or alternative terminal exon; NMD not assessed)")
            junctions.append((last.b + last.size, "C-term novel tail"))

    if not events:
        events, classes = ["none (identical to canonical)"], ["identical"]

    return dict(events=events, classes=classes, retained=retained,
                iso_to_can=iso_to_can, junctions=junctions)


# ----------------------------------------------------------------------
# 2. Features lost
# ----------------------------------------------------------------------
def collect_features(entry: dict) -> list[dict]:
    feats = []
    for f in entry.get("features", []):
        t = f.get("type")
        if t not in ALL_TYPES:
            continue
        loc = f.get("location", {})
        s = loc.get("start", {}).get("value")
        e = loc.get("end", {}).get("value")
        if s is None or e is None:
            continue
        desc = f.get("description") or (f.get("ligand") or {}).get("name") or ""
        feats.append(dict(type=t, start=int(s), end=int(e), description=desc))
    return feats


def _frac_retained(f: dict, retained: list[bool]) -> float:
    if f["type"] in ENDPOINT_ONLY:
        pos = [f["start"], f["end"]]
    else:
        pos = list(range(f["start"], f["end"] + 1))
    ok = sum(1 for p in pos if 1 <= p <= len(retained) and retained[p - 1])
    return ok / len(pos)


def _label(f: dict) -> str:
    if f["start"] == f["end"]:
        base = f"{f['type']} {f['start']}"
        return f"{base} ({f['description']})" if f["description"] else base
    return f"{f['description'] or f['type']} {f['start']}-{f['end']}"


def _limited(items: list[str], n: int = 6) -> str:
    if len(items) <= n:
        return "; ".join(items)
    return "; ".join(items[:n]) + f"; +{len(items) - n} more"


def feature_summary(feats: list[dict], retained: list[bool]) -> dict:
    # Signal peptide / transit peptide
    sig = [f for f in feats if f["type"] in SIGNAL_TYPES]
    if not sig:
        signal = "none annotated"
    else:
        fr = _frac_retained(sig[0], retained)
        signal = "retained" if fr == 1 else "lost" if fr == 0 else f"partial ({fr:.0%} retained)"

    # TM helices + parity heuristic
    tms = sorted((f for f in feats if f["type"] == TM_TYPE), key=lambda f: f["start"])
    if not tms:
        tm_txt, topo = "no TM annotated", "n/a (no TMs annotated)"
    else:
        fracs = [_frac_retained(f, retained) for f in tms]
        lost = [i + 1 for i, fr in enumerate(fracs) if fr < 0.5]
        partial = [i + 1 for i, fr in enumerate(fracs) if 0.5 <= fr < 1.0]
        tm_txt = f"{len(lost)}/{len(tms)} lost"
        if lost:
            tm_txt += " (" + ", ".join(f"TM{i}" for i in lost) + ")"
        if partial:
            tm_txt += "; partly cut: " + ", ".join(f"TM{i}" for i in partial)
        lost_before, flip = 0, False
        for fr in fracs:
            if fr < 0.5:
                lost_before += 1
            elif lost_before % 2 == 1:
                flip = True
        if flip:
            topo = ("possible orientation change downstream (odd number of TMs lost upstream); "
                    "parity heuristic - verify with DeepTMHMM")
        elif lost:
            topo = "TMs lost; no parity-based inversion predicted"
        else:
            topo = "no TM loss"

    # Domains / repeats / motifs affected
    dom = []
    for f in feats:
        if f["type"] in DOMAIN_TYPES:
            fr = _frac_retained(f, retained)
            if fr < 1:
                dom.append(f"{_label(f)} {fr:.0%} retained")
    dom_txt = _limited(dom) if dom else "none affected"

    # Residue-level functional sites
    sites = [f for f in feats if f["type"] in SITE_TYPES]
    lost_sites = [_label(f) for f in sites if _frac_retained(f, retained) < 1]
    sites_txt = (f"{len(lost_sites)}/{len(sites)}: " + _limited(lost_sites)) if lost_sites \
        else (f"0/{len(sites)}" if sites else "none annotated")

    return dict(signal=signal, tm=tm_txt, topology=topo, domains=dom_txt, sites=sites_txt)


# ----------------------------------------------------------------------
# 3. ESM2 junction score (optional)
# ----------------------------------------------------------------------
class EsmMaskedScorer:
    """Masked log-probability of the true residue at chosen positions (one masked copy per position)."""

    def __init__(self, t1, model_name: str, device: str = "cpu", max_len: int = 1022, batch: int = 8):
        self.tok, self.model = t1.load_esm_model(model_name, device=device)
        self.device, self.max_len, self.batch = device, max_len, batch

    def logp(self, seq: str, positions: list[int]) -> list[float]:
        import torch
        seq = seq.upper()[: self.max_len]
        valid = sorted({p for p in positions if 0 <= p < len(seq)})
        enc = self.tok(seq, return_tensors="pt")["input_ids"].to(self.device)  # [1, L+2]
        out = {}
        for i in range(0, len(valid), self.batch):
            chunk = valid[i:i + self.batch]
            ids = enc.repeat(len(chunk), 1)
            for r, p in enumerate(chunk):
                ids[r, p + 1] = self.tok.mask_token_id  # +1 for CLS
            with torch.no_grad():
                lps = torch.log_softmax(self.model(input_ids=ids).logits, dim=-1)
            for r, p in enumerate(chunk):
                out[p] = lps[r, p + 1, enc[0, p + 1]].item()
        return [out.get(p, float("nan")) for p in positions]


def score_junctions(canon: str, iso: str, junctions, iso_to_can, logp_fn, can_cache: dict,
                    window: int = 10) -> dict:
    """
    For each junction, take isoform positions within +/- `window` that map to a canonical
    residue and compute mean[ logp(residue in isoform) - logp(same residue in canonical) ].
    Negative = the residue is less plausible in its new context. logp_fn(seq, positions) -> list.
    `can_cache` memoises canonical per-position values across isoforms.
    """
    wins = []
    for p, label in junctions:
        lo, hi = max(0, p - window), min(len(iso), p + window)
        pairs = [(i, iso_to_can[i]) for i in range(lo, hi) if iso_to_can[i] is not None]
        wins.append((label, pairs))

    need_iso = sorted({i for _, prs in wins for i, _ in prs})
    need_can = sorted({j for _, prs in wins for _, j in prs if j not in can_cache})
    iso_lp = dict(zip(need_iso, logp_fn(iso, need_iso))) if need_iso else {}
    if need_can:
        can_cache.update(zip(need_can, logp_fn(canon, need_can)))

    deltas, detail = [], []
    for label, pairs in wins:
        d = [iso_lp[i] - can_cache[j] for i, j in pairs
             if not (math.isnan(iso_lp[i]) or math.isnan(can_cache[j]))]
        if d:
            m = sum(d) / len(d)
            deltas.append(m)
            detail.append(f"{label}: {m:+.2f}")
    return dict(
        min=round(min(deltas), 3) if deltas else float("nan"),
        mean=round(sum(deltas) / len(deltas), 3) if deltas else float("nan"),
        n=len(deltas),
        detail=_limited(detail, 8),
    )


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def load_isoforms(t1, entry: dict, fasta_path):
    if fasta_path:
        return t1.load_isoforms_from_fasta(fasta_path)
    isoforms = []
    for iso in t1.get_isoform_accessions(entry):
        iso_id = iso["isoform_id"]
        if iso["sequence_status"] != "Described" or iso_id is None:
            continue
        try:
            seq = t1.fetch_uniprot_fasta(iso_id)
        except Exception:
            print(f"  Skipping {iso_id} (fetch failed)")
            continue
        isoforms.append({"isoform_id": iso_id, "name": iso["name"], "seq": seq})
    return isoforms


def build_evidence_table(t1, uniprot_id: str, fasta_path=None, junction_model=None,
                         device: str = "cpu", window: int = 10, scorer=None) -> pd.DataFrame:
    print(f"Fetching canonical UniProt entry {uniprot_id}...")
    entry = t1.fetch_uniprot_entry(uniprot_id)
    canon = entry["sequence"]["value"]
    feats = collect_features(entry)
    print(f"Canonical length: {len(canon)} aa; {len(feats)} annotated features considered")

    isoforms = load_isoforms(t1, entry, fasta_path)
    print(f"Isoforms to analyse: {len(isoforms)}")

    if scorer is None and junction_model:
        print(f"Loading ESM2 model for junction scoring ({junction_model})...")
        scorer = EsmMaskedScorer(t1, junction_model, device=device)
    can_cache: dict = {}

    rows = []
    for iso in isoforms:
        seq = iso["seq"]
        if not seq:
            print(f"  Skipping {iso['isoform_id']} (empty/untranslatable sequence)")
            continue
        aln = analyse_alignment(canon, seq, t1.get_matching_blocks(canon, seq))
        fs = feature_summary(feats, aln["retained"])
        row = {
            "Isoform Name": iso["name"],
            "Transcript ID": iso["isoform_id"],
            "Length (aa)": len(seq),
            "Canonical aa lost": len(canon) - sum(aln["retained"]),
            "Novel aa gained": sum(1 for x in aln["iso_to_can"] if x is None),
            "Event(s)": "; ".join(aln["events"]),
            "Event class": "; ".join(aln["classes"]),
            "Signal peptide": fs["signal"],
            "TM helices lost": fs["tm"],
            "Topology flag": fs["topology"],
            "Domains/repeats/motifs affected": fs["domains"],
            "Functional sites lost": fs["sites"],
        }
        if scorer is not None:
            js = score_junctions(canon, seq, aln["junctions"], aln["iso_to_can"],
                                 scorer.logp, can_cache, window=window)
            row["Junction delta logp (min)"] = js["min"]
            row["Junction delta logp (mean)"] = js["mean"]
            row["Junctions scored"] = js["n"]
            row["Junction detail"] = js["detail"]
        rows.append(row)
        print(f"  {iso['isoform_id']}: {row['Event(s)']}")
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-isoform evidence table (events, lost features, junction score)")
    ap.add_argument("--uniprot-id", required=True)
    ap.add_argument("--fasta", default=None, help="Isoform FASTA (protein or cDNA); omit to use UniProt isoforms")
    ap.add_argument("--out", default="isoform_evidence.csv")
    ap.add_argument("--junction-model", default=None,
                    help=f"ESM2 checkpoint for junction scoring (e.g. {DEFAULT_MODEL}); omit to skip")
    ap.add_argument("--window", type=int, default=10, help="Residues each side of a junction to score")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    t1 = load_pipeline_module()
    df = build_evidence_table(t1, args.uniprot_id, args.fasta, args.junction_model,
                              args.device, args.window)
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
