"""
CLN3 Isoform Analysis — Tab 1: Overview & Alignments
======================================================
Fetches the canonical UniProt entry for CLN3 (Q13286) plus its annotated
isoforms, pulls domain/TM feature annotations, and maps each isoform back
onto canonical using literal sequence matching (difflib.SequenceMatcher,
autojunk=False) rather than substitution-matrix alignment — appropriate
here because alternative-splicing isoforms are built from exact canonical
exon substrings (deletions/insertions of whole exons), not point-mutated
sequence. autojunk=False is required because difflib's default heuristic
can discard short, frequently-repeated substrings as "junk" and miss short
exons. Builds the Tab 1 summary matrix:

    Isoform | Transcript/Isoform ID | Length | % Canonical Retained |
    TMs Retained | Localisation | Stability Risk | Mean pLDDT

Run: python tab1_overview_alignments.py
Output: cln3_tab1_overview.csv, cln3_tab1_overview_display.csv
"""

import requests
import json
import re
import difflib
import logging
from dataclasses import dataclass
import pandas as pd

UNIPROT_ID = "Q13286"  # CLN3 canonical
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
ALPHAFOLD_BASE = "https://alphafold.ebi.ac.uk/files"  # AFDB model files

LOG = logging.getLogger("tab1_overview_alignments")

# ----------------------------------------------------------------------
# Synthetic (in-silico) DMS scoring constants
# ----------------------------------------------------------------------
# "Mean Predicted Score" = mean, across every possible single amino-acid
# substitution at every position of an isoform's OWN sequence, of:
#     log P(mutant_aa | context) - log P(wild_type_aa | context)
# computed via a single forward pass ("wt-marginal" scoring) through an
# ESM2 protein language model. This is the same style of score reported
# by ESM1b/ESM2 variant-effect studies (Brandes et al. 2023, Nat Genet):
# more negative = the model considers that substitution less likely than
# the isoform's own wild-type residue, i.e. more likely destabilizing.
#
# For a protein of length L this produces L x 19 variant scores (19 = the
# 20 standard amino acids minus the wild-type residue). This is a
# structural/evolutionary plausibility score from a language model, NOT
# an experimental DMS measurement, clinical pathogenicity call, or real
# gnomAD population-frequency lookup — report it as "in-silico DMS
# (ESM2 wt-marginal)" rather than "DMS" alone in any methods section.
# There is no real gnomAD weighting applied (this is a full synthetic
# scan of the isoform sequence, not a lookup against observed human
# variants), which is why the summary matrix's methods note says gnomAD
# frequency defaults to 0.0.
#
# Requires: pip install torch transformers
# First run downloads the ESM2 checkpoint from Hugging Face (cached
# afterwards under ~/.cache/huggingface).
STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_DMS_MODEL = "facebook/esm2_t12_35M_UR50D"
_ESM_MODEL_CACHE = {}


@dataclass
class DMSResult:
    isoform_id: str
    n_variants: int
    mean_score: float
    variant_df: "pd.DataFrame"  # columns: position, wt_aa, mut_aa, score


def load_esm_model(model_name: str = DEFAULT_DMS_MODEL, device: str = "cpu"):
    """Loads (and process-caches) an ESM2 tokenizer + masked-LM model."""
    if model_name in _ESM_MODEL_CACHE:
        return _ESM_MODEL_CACHE[model_name]
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    LOG.info(f"Loading ESM2 model '{model_name}' (first run downloads it)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    _ESM_MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def score_isoform_dms(isoform_id: str, seq: str, tokenizer, model,
                       device: str = "cpu", max_len: int = 1022) -> "DMSResult":
    """
    Computes wt-marginal in-silico DMS scores for every possible amino-acid
    substitution at every position of `seq` (the isoform's OWN sequence).
    """
    import torch

    seq = seq.upper()
    if len(seq) > max_len:
        LOG.warning(f"{isoform_id}: length {len(seq)} exceeds model max ({max_len}); "
                     f"truncating to first {max_len} residues for DMS scoring.")
        seq = seq[:max_len]

    enc = tokenizer(seq, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0]  # [seq_len + 2, vocab] (CLS ... EOS)
    log_probs = torch.log_softmax(logits, dim=-1)

    aa_token_ids = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in STANDARD_AA}
    unk_id = tokenizer.unk_token_id
    for aa, tid in aa_token_ids.items():
        if tid is None or tid == unk_id:
            raise RuntimeError(f"Tokenizer does not recognize amino acid token '{aa}'")

    rows = []
    for i, wt_aa in enumerate(seq):
        if wt_aa not in STANDARD_AA:
            continue  # skip non-standard residues (X, etc.)
        pos_idx = i + 1  # +1 for leading CLS token
        wt_logprob = log_probs[pos_idx, aa_token_ids[wt_aa]].item()
        for mut_aa in STANDARD_AA:
            if mut_aa == wt_aa:
                continue
            mut_logprob = log_probs[pos_idx, aa_token_ids[mut_aa]].item()
            rows.append(dict(position=i + 1, wt_aa=wt_aa, mut_aa=mut_aa,
                              score=round(mut_logprob - wt_logprob, 4)))

    variant_df = pd.DataFrame(rows)
    mean_score = float(variant_df["score"].mean()) if not variant_df.empty else float("nan")
    return DMSResult(isoform_id=isoform_id, n_variants=len(variant_df),
                      mean_score=round(mean_score, 4), variant_df=variant_df)


# ----------------------------------------------------------------------
# 0. AlphaFold pLDDT (per-residue confidence, canonical structure only —
#    AFDB does not host isoform-specific structures, so isoform pLDDT is
#    approximated by mapping retained canonical residues onto the
#    canonical model's per-residue pLDDT track)
# ----------------------------------------------------------------------
def fetch_alphafold_plddt(uniprot_id: str) -> dict[int, float]:
    """
    Queries the AlphaFold DB prediction API to get the current model file
    URL (avoids hardcoding a model version that may go stale), downloads
    the PDB, and extracts per-residue pLDDT (stored in the B-factor
    column). Returns {residue_number (1-based): plddt}.
    """
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    r = requests.get(api_url, timeout=30)
    r.raise_for_status()
    predictions = r.json()
    if not predictions:
        raise requests.HTTPError(f"No AlphaFold prediction found for {uniprot_id}")
    pdb_url = predictions[0]["pdbUrl"]

    r2 = requests.get(pdb_url, timeout=30)
    r2.raise_for_status()

    plddt_by_residue = {}
    for line in r2.text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":  # one entry per residue (alpha carbon)
            resnum = int(line[22:26])
            bfactor = float(line[60:66])  # AFDB stores pLDDT here
            plddt_by_residue[resnum] = bfactor
    return plddt_by_residue


def coverage_heuristic_risk_call(pct_coverage: float, features_retained: float, features_total: int) -> str:
    """
    ORIGINAL heuristic risk classification (kept for comparison/backwards
    compatibility): low overall coverage or loss of a large fraction of
    the tracked structural features flags higher risk.
    IMPORTANT: these coverage/feature-fraction cutoffs (90%/80%, 60%/50%)
    are round-number heuristics, not thresholds derived from literature or
    calibrated against experimental/functional outcomes. There is no
    established percentage-of-protein-lost rule for functional impact in
    the variant-interpretation literature (ACMG/ClinGen PVS1 guidance
    deliberately avoids a fixed numeric cutoff, instead requiring
    case-by-case evidence that the specific lost region is functionally
    critical). Report this column as an unvalidated heuristic, not a
    calibrated risk score.
    """
    feature_frac = features_retained / features_total if features_total else 1.0
    if pct_coverage >= 90 and feature_frac >= 0.8:
        return "LOW RISK"
    if pct_coverage >= 60 and feature_frac >= 0.5:
        return "MODERATE RISK"
    return "HIGH RISK"


def plddt_confidence_tier(mean_plddt: float) -> str:
    """
    AlphaFold's own per-residue pLDDT confidence categories (as used in
    the AlphaFold Protein Structure Database and widely cited in the
    literature): >=90 very high, 70-90 confident, 50-70 low, <50 very low.
    Unlike the coverage/feature-fraction cutoffs above, these ARE an
    established standard, not an invented threshold.
    """
    if mean_plddt is None or (isinstance(mean_plddt, float) and pd.isna(mean_plddt)):
        return "Unknown"
    if mean_plddt >= 90:
        return "Very High"
    if mean_plddt >= 70:
        return "Confident"
    if mean_plddt >= 50:
        return "Low"
    return "Very Low"


def stability_risk_call(pct_coverage: float, features_retained: float, features_total: int,
                         mean_plddt: float) -> str:
    """
    Combined risk classification that separates two distinct claims rather
    than collapsing them into one unvalidated score:
      1. Structural confidence at retained positions, via AlphaFold's own
         pLDDT tiers (plddt_confidence_tier) -- an established standard.
      2. Whether tracked structural features / overall sequence are
         substantially intact, via the same coverage/feature-fraction
         cutoffs as coverage_heuristic_risk_call -- still an unvalidated
         heuristic, but now reported as one input among several rather
         than the sole basis for the call.
    This distinguishes, e.g., "the retained region is confidently folded
    but is missing a critical feature" from "we're simply not confident
    this isoform folds at all" -- two different biological claims that a
    single Low/Moderate/High score would conflate.
    """
    feature_frac = features_retained / features_total if features_total else 1.0
    feature_intact = pct_coverage >= 90 and feature_frac >= 0.8
    tier = plddt_confidence_tier(mean_plddt)

    if tier == "Unknown":
        # No AlphaFold data available -- fall back to the coverage-only heuristic.
        return coverage_heuristic_risk_call(pct_coverage, features_retained, features_total)
    if tier in ("Very High", "Confident"):
        return "High confidence, feature-intact" if feature_intact else "High confidence, feature-disrupted"
    if tier == "Low":
        return "Low structural confidence"
    return "Very low structural confidence"


# ----------------------------------------------------------------------
# 0b. Local FASTA input (e.g. Ensembl isoform cDNA or protein FASTA)
#     Auto-detects nucleotide vs protein sequence and translates if needed.
# ----------------------------------------------------------------------
def parse_fasta(path: str) -> list[dict]:
    """
    Parses a multi-FASTA file into records: {header, seq}.
    Handles the Ensembl-style header format:
        >ENST00000636147.2 gene=CLN3 transcript=CLN3-243 length=438aa
    Extracts transcript_id, gene, name, declared_length when present.
    """
    records = []
    header, seq_lines = None, []

    def flush():
        if header is None:
            return
        seq = "".join(seq_lines).upper().replace("*", "")  # drop trailing stop-codon '*' if translated already
        fields = {"header": header, "seq": seq}
        transcript_id = header.split()[0]
        fields["transcript_id"] = transcript_id
        for tag in ("gene", "transcript", "length"):
            m = re.search(rf"{tag}=(\S+)", header)
            if m:
                fields[tag] = m.group(1)
        records.append(fields)

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip())
    flush()
    return records


NUCLEOTIDE_CHARS = set("ACGTUN")


def is_nucleotide_seq(seq: str) -> bool:
    """Heuristic: if the vast majority of characters are ACGTUN, treat as nucleotide."""
    if not seq:
        return False
    nt_frac = sum(c in NUCLEOTIDE_CHARS for c in seq) / len(seq)
    return nt_frac > 0.95


def translate_longest_orf(nt_seq: str) -> str:
    """
    Finds the longest ATG-to-stop ORF across the 3 forward reading frames
    of a cDNA/mRNA sequence and returns the translated protein (no stop codon).
    Assumes coding (sense) strand input, consistent with Ensembl cDNA FASTA.
    """
    from Bio.Seq import Seq

    best_protein = ""
    dna = Seq(nt_seq.replace("U", "T"))
    for frame in range(3):
        trimmed_len = (len(dna) - frame) - (len(dna) - frame) % 3
        frame_protein = str(dna[frame:frame + trimmed_len].translate())
        for orf in frame_protein.split("*"):
            start = orf.find("M")
            if start == -1:
                continue
            candidate = orf[start:]
            if len(candidate) > len(best_protein):
                best_protein = candidate
    return best_protein


def load_isoforms_from_fasta(path: str) -> list[dict]:
    """
    Loads isoform sequences from a local FASTA file (protein or cDNA/mRNA —
    auto-detected per record). Returns records shaped like the UniProt-API
    isoform records used elsewhere in this script:
        {isoform_id, name, seq}
    """
    raw_records = parse_fasta(path)
    isoforms = []
    for rec in raw_records:
        seq = rec["seq"]
        if is_nucleotide_seq(seq):
            protein_seq = translate_longest_orf(seq)
            print(f"  {rec['transcript_id']}: translated cDNA ({len(seq)} nt) -> "
                  f"protein ({len(protein_seq)} aa)")
        else:
            protein_seq = seq  # already protein

        name = rec.get("transcript", rec["transcript_id"])
        isoforms.append({
            "isoform_id": rec["transcript_id"],
            "name": name,
            "seq": protein_seq,
        })
    return isoforms

# ----------------------------------------------------------------------
# 1. Fetch canonical entry (sequence + features: TM, topology, etc.)
# ----------------------------------------------------------------------
def fetch_uniprot_entry(uniprot_id: str) -> dict:
    url = f"{UNIPROT_BASE}/{uniprot_id}.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_uniprot_fasta(accession: str) -> str:
    """Fetch raw sequence for a given accession (canonical or isoform, e.g. Q13286-2)."""
    url = f"{UNIPROT_BASE}/{accession}.fasta"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = r.text.strip().split("\n")
    return "".join(lines[1:])  # drop header


def get_isoform_accessions(entry: dict) -> list[dict]:
    """Extract isoform IDs + names from the 'comments' -> ALTERNATIVE PRODUCTS block."""
    isoforms = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "ALTERNATIVE PRODUCTS":  # UniProt uses a space, not underscore
            for iso in comment.get("isoforms", []):
                iso_id = iso.get("isoformIds", [None])[0]
                name = iso.get("name", {}).get("value", iso_id)
                seq_status = iso.get("isoformSequenceStatus", "")
                isoforms.append({
                    "isoform_id": iso_id,
                    "name": name,
                    "sequence_status": seq_status,
                })
    return isoforms


def get_structural_features(entry: dict, feature_type: str = "Transmembrane") -> list[dict]:
    """
    Extract structural region features of a given UniProt feature `type`
    from the canonical entry. Defaults to 'Transmembrane' (CLN3-style
    membrane proteins), but works for any type, e.g. 'Domain' (CTC1-style
    OB-fold domains), 'Repeat', 'Zinc finger', etc. — check the protein's
    UniProt feature table to pick the right one.
    """
    features = []
    for feat in entry.get("features", []):
        if feat.get("type") == feature_type:
            loc = feat["location"]
            features.append({
                "start": loc["start"]["value"],
                "end": loc["end"]["value"],
                "description": feat.get("description", ""),
            })
    return features


# Feature types worth tracking retention of, in priority order used only to
# break ties when two types are annotated the same number of times.
AUTO_FEATURE_CANDIDATES = [
    "Transmembrane", "Domain", "Repeat", "Zinc finger", "Coiled coil", "Motif",
]


def auto_detect_feature_type(entry: dict, candidates: list[str] = None) -> tuple[str, int]:
    """
    Scans the canonical entry's UniProt features and picks whichever
    candidate feature `type` has the most annotated instances on this
    protein — e.g. 'Transmembrane' wins for CLN3 (6 TM helices), 'Domain'
    wins for CTC1 (OB-fold domains), with no protein-specific hardcoding.
    Ties are broken by AUTO_FEATURE_CANDIDATES priority order.
    Returns (chosen_type, count). If nothing on the candidate list is
    annotated at all, falls back to ("Domain", 0) and the caller will
    print the existing "no features found" warning.
    """
    candidates = candidates or AUTO_FEATURE_CANDIDATES
    counts = {t: 0 for t in candidates}
    for feat in entry.get("features", []):
        t = feat.get("type")
        if t in counts:
            counts[t] += 1
    best_type = max(candidates, key=lambda t: (counts[t], -candidates.index(t)))
    return best_type, counts[best_type]


# Backwards-compatible alias
def get_tm_features(entry: dict) -> list[dict]:
    return get_structural_features(entry, feature_type="Transmembrane")


def fetch_interpro_domains(accession: str, entry_type: str = "domain", timeout: int = 30) -> list[dict]:
    """
    Fallback data source for proteins with no relevant features curated in
    UniProt's OWN feature table (get_structural_features draws only from
    UniProt; many proteins, e.g. CTC1/Q2NKJ3, have literature/PDB-resolved
    domains -- OB-folds in CTC1's case -- that are represented in InterPro
    (via Pfam/PANTHER/etc. member databases) but not as UniProt sequence
    features). Queries the InterPro7 REST API for entries of the given
    `entry_type` ('domain', 'repeat', 'conserved_site', ...) matching this
    accession, and returns them in the same {start, end, description}
    shape as get_structural_features() so they can be used interchangeably
    downstream. Note: InterPro entries can come from multiple member
    databases and may overlap; overlapping/duplicate (start, end) pairs are
    collapsed, but no attempt is made to pick a single "best" domain
    definition when member databases disagree on exact boundaries.
    """
    features = []
    seen_spans = set()
    url = f"https://www.ebi.ac.uk/interpro/api/entry/InterPro/protein/uniprot/{accession}/?page_size=100"
    while url:
        r = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
        r.raise_for_status()
        payload = r.json()
        for item in payload.get("results", []):
            metadata = item.get("metadata", {})
            if metadata.get("type") != entry_type:
                continue
            name = metadata.get("name", metadata.get("accession", ""))
            for protein in item.get("proteins", []):
                for loc in protein.get("entry_protein_locations", []):
                    for frag in loc.get("fragments", []):
                        start, end = frag.get("start"), frag.get("end")
                        if start is None or end is None:
                            continue
                        span = (start, end)
                        if span in seen_spans:
                            continue
                        seen_spans.add(span)
                        features.append({"start": start, "end": end, "description": name})
        url = payload.get("next")
    return features


def get_topology_localisation(entry: dict) -> str:
    """Pull subcellular location comment as a rough localisation call."""
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "SUBCELLULAR LOCATION":  # UniProt uses a space, not underscore
            locs = comment.get("subcellularLocations", [])
            names = [l["location"]["value"] for l in locs if "location" in l]
            if names:
                return " / ".join(names).upper()
    return "UNKNOWN"


# Priority order for picking the 1-2 most relevant compartments out of a full
# UniProt subcellular-location list (which for CLN3 has ~13 entries). Order
# matters: earlier = higher priority when multiple matches are present.
LOCALISATION_PRIORITY = [
    "Lysosome", "Endosome", "Golgi", "Cell membrane", "Plasma membrane",
    "Endoplasmic reticulum", "Synapse", "Autophagosome",
    "Telomere", "Nucleus", "Nucleoplasm", "Cytoplasm", "Cytosol",
]


def simplify_localisation(raw_localisation: str, max_terms: int = 2) -> str:
    """
    Collapses a full UniProt subcellular-location string (many entries,
    joined by ' / ', e.g. 'LYSOSOME MEMBRANE / LATE ENDOSOME / LYSOSOME / ...')
    down to the top `max_terms` most relevant compartments, matched against
    LOCALISATION_PRIORITY. Falls back to the first raw term if nothing matches.
    """
    if raw_localisation in ("UNKNOWN", ""):
        return "Unknown"

    raw_terms = [t.strip() for t in raw_localisation.split("/")]
    picked = []
    for keyword in LOCALISATION_PRIORITY:
        if any(keyword.upper() in term for term in raw_terms):
            if keyword not in picked:
                picked.append(keyword)
        if len(picked) >= max_terms:
            break

    if not picked:
        picked = [raw_terms[0].title()]

    return "/".join(picked)


# ----------------------------------------------------------------------
# 2. Sequence matching: map canonical domain positions onto isoform via
#    literal matching blocks (exact-substring exon matching), not a
#    substitution-matrix alignment.
# ----------------------------------------------------------------------
def get_matching_blocks(canonical_seq: str, isoform_seq: str):
    """
    Returns difflib matching blocks between canonical and isoform sequence.
    autojunk=False is required: difflib's default autojunk heuristic treats
    substrings that recur very frequently as "popular" and excludes them
    from matching, which can silently drop short exons that happen to
    share short repeated motifs with the rest of the sequence.
    """
    matcher = difflib.SequenceMatcher(None, canonical_seq, isoform_seq, autojunk=False)
    return matcher.get_matching_blocks()  # list of Match(a, b, size), size==0 sentinel at the end


def compute_coverage_and_feature_retention(canonical_seq: str, isoform_seq: str, structural_features: list[dict]):
    """
    Returns:
        pct_coverage: % of canonical residues present in isoform (via matching blocks)
        features_retained: SUM of fractional retention across all tracked structural
                            features (TM helices, domains, etc. — whatever feature_type
                            was passed to get_structural_features) — not a binary count
        n_features_total: total canonical feature count of the tracked type
        retained_map: {canonical_residue_1based: bool}
    """
    blocks = get_matching_blocks(canonical_seq, isoform_seq)

    # Map: canonical residue index (1-based) -> retained in isoform (part of a matching block)?
    retained_map = {i: False for i in range(1, len(canonical_seq) + 1)}
    for block in blocks:
        if block.size == 0:
            continue
        for offset in range(block.size):
            canonical_pos = block.a + offset + 1  # 1-based
            retained_map[canonical_pos] = True

    total_canonical_len = len(canonical_seq)
    retained_count = sum(retained_map.values())
    pct_coverage = 100.0 * retained_count / total_canonical_len

    features_retained_fractional = 0.0
    for feat in structural_features:
        positions = range(feat["start"], feat["end"] + 1)
        retained_residues = sum(retained_map.get(p, False) for p in positions)
        frac = retained_residues / len(positions)
        features_retained_fractional += frac  # sum of fractional retention, not a binary threshold count

    return pct_coverage, features_retained_fractional, len(structural_features), retained_map


# Backwards-compatible alias
def compute_coverage_and_tm_retention(canonical_seq: str, isoform_seq: str, tm_features: list[dict]):
    return compute_coverage_and_feature_retention(canonical_seq, isoform_seq, tm_features)


def compute_mean_plddt(retained_map: dict[int, bool], plddt_by_residue: dict[int, float]) -> float:
    """Mean AlphaFold pLDDT over canonical residues actually retained in this isoform."""
    scores = [plddt_by_residue[pos] for pos, retained in retained_map.items()
              if retained and pos in plddt_by_residue]
    return round(sum(scores) / len(scores), 1) if scores else float("nan")


# ----------------------------------------------------------------------
# 3. Build Tab 1 summary matrix
# ----------------------------------------------------------------------
def build_tab1_summary(uniprot_id: str = UNIPROT_ID, fasta_path: str | None = None,
                        feature_type: str = "Transmembrane",
                        compute_dms: bool = True,
                        dms_model: str = DEFAULT_DMS_MODEL,
                        dms_device: str = "cpu") -> pd.DataFrame:
    print(f"Fetching canonical UniProt entry {uniprot_id}...")
    entry = fetch_uniprot_entry(uniprot_id)
    canonical_seq = entry["sequence"]["value"]

    if feature_type == "auto":
        detected_type, detected_count = auto_detect_feature_type(entry)
        if detected_count > 0:
            print(f"Auto-detected feature type for this protein: '{detected_type}' "
                  f"({detected_count} instances annotated in UniProt)")
            feature_type = detected_type
            structural_features = get_structural_features(entry, feature_type=feature_type)
        else:
            print("No Transmembrane/Domain/Repeat/Zinc finger/Coiled coil/Motif features "
                  "found in UniProt's own feature table for this protein. Falling back to "
                  "InterPro domain annotations (covers cases like CTC1, where domains are "
                  "literature/PDB-characterized but not curated as UniProt sequence features)...")
            try:
                structural_features = fetch_interpro_domains(uniprot_id, entry_type="domain")
            except requests.RequestException as exc:
                print(f"  InterPro lookup failed ({exc}); continuing with 0 features.")
                structural_features = []
            if structural_features:
                feature_type = "InterPro Domain"
                print(f"  Found {len(structural_features)} InterPro domain region(s).")
            else:
                feature_type = "Domain"
                print("  No InterPro domains found either; feature-retention columns will be 0/0.")
    else:
        structural_features = get_structural_features(entry, feature_type=feature_type)
    localisation = simplify_localisation(get_topology_localisation(entry))

    print(f"Canonical length: {len(canonical_seq)} aa")
    print(f"'{feature_type}' features annotated: {len(structural_features)}")
    if not structural_features:
        print(f"  WARNING: no '{feature_type}' features found on this entry — check UniProt's "
              f"feature table for this protein and pick a type that's actually annotated (e.g. "
              f"'Domain', 'Repeat', 'Zinc finger'). Coverage/pLDDT/risk will still compute, but "
              f"the feature-retention columns will be 0/0 and stability risk falls back to coverage-only.")

    if fasta_path:
        print(f"Loading isoforms from local FASTA: {fasta_path}")
        isoforms = load_isoforms_from_fasta(fasta_path)
        print(f"Isoforms loaded: {len(isoforms)}")
    else:
        print("Fetching isoforms from UniProt...")
        iso_meta = get_isoform_accessions(entry)
        print(f"Isoforms found: {len(iso_meta)}")
        isoforms = []
        for iso in iso_meta:
            iso_id = iso["isoform_id"]
            if iso["sequence_status"] != "Described" or iso_id is None:
                continue  # skip isoforms without an explicit sequence (incl. canonical itself)
            try:
                iso_seq = fetch_uniprot_fasta(iso_id)
            except requests.HTTPError:
                print(f"  Skipping {iso_id} (fetch failed)")
                continue
            isoforms.append({"isoform_id": iso_id, "name": iso["name"], "seq": iso_seq})

    print("Fetching AlphaFold canonical model pLDDT...")
    try:
        plddt_by_residue = fetch_alphafold_plddt(uniprot_id)
    except requests.HTTPError:
        print("  AlphaFold model not found for this accession; pLDDT will be NaN.")
        plddt_by_residue = {}

    canonical_mean_plddt = (
        round(sum(plddt_by_residue.values()) / len(plddt_by_residue), 1)
        if plddt_by_residue else float("nan")
    )

    dms_tokenizer, dms_model_obj = (None, None)
    if compute_dms:
        print(f"Loading ESM2 model for in-silico DMS scoring ({dms_model})...")
        dms_tokenizer, dms_model_obj = load_esm_model(dms_model, device=dms_device)

    rows = []
    canonical_dms_mean, canonical_dms_n = float("nan"), 0
    if compute_dms:
        print(f"Scoring in-silico DMS for canonical ({len(canonical_seq)} aa, "
              f"~{len(canonical_seq) * 19} variants)...")
        result = score_isoform_dms(f"{uniprot_id}-1 (canonical)", canonical_seq,
                                    dms_tokenizer, dms_model_obj, device=dms_device)
        canonical_dms_mean, canonical_dms_n = result.mean_score, result.n_variants
        print(f"  canonical: {canonical_dms_n} variants scored, mean score = {canonical_dms_mean}")

    # Include canonical itself as isoform 1 (100% retention baseline)
    canonical_risk = stability_risk_call(100.0, len(structural_features), len(structural_features), canonical_mean_plddt)
    canonical_coverage_risk = coverage_heuristic_risk_call(100.0, len(structural_features), len(structural_features))
    rows.append({
        "isoform_id": f"{uniprot_id}-1 (canonical)",
        "name": "Canonical",
        "length_aa": len(canonical_seq),
        "pct_canonical_retained": 100.0,
        "features_retained": len(structural_features),
        "features_total": len(structural_features),
        "localisation": localisation,
        "mean_plddt": canonical_mean_plddt,
        "stability_risk": canonical_risk,
        "coverage_heuristic_risk": canonical_coverage_risk,
        "mean_predicted_score": canonical_dms_mean,
        "n_dms_variants": canonical_dms_n,
    })

    for iso in isoforms:
        iso_id = iso["isoform_id"]
        iso_seq = iso["seq"]
        if not iso_seq:
            print(f"  Skipping {iso_id} (empty/untranslatable sequence)")
            continue

        pct_cov, feat_ret, feat_tot, retained_map = compute_coverage_and_feature_retention(
            canonical_seq, iso_seq, structural_features
        )
        mean_plddt = compute_mean_plddt(retained_map, plddt_by_residue)
        risk = stability_risk_call(pct_cov, feat_ret, feat_tot, mean_plddt)
        coverage_risk = coverage_heuristic_risk_call(pct_cov, feat_ret, feat_tot)

        iso_dms_mean, iso_dms_n = float("nan"), 0
        if compute_dms:
            result = score_isoform_dms(iso_id, iso_seq, dms_tokenizer, dms_model_obj, device=dms_device)
            iso_dms_mean, iso_dms_n = result.mean_score, result.n_variants

        rows.append({
            "isoform_id": iso_id,
            "name": iso["name"],
            "length_aa": len(iso_seq),
            "pct_canonical_retained": round(pct_cov, 1),
            "features_retained": feat_ret,
            "features_total": feat_tot,
            "localisation": localisation,  # placeholder — refined per-isoform in Tab 2
            "mean_plddt": mean_plddt,
            "stability_risk": risk,
            "coverage_heuristic_risk": coverage_risk,
            "mean_predicted_score": iso_dms_mean,
            "n_dms_variants": iso_dms_n,
        })
        print(f"  {iso_id}: {len(iso_seq)} aa, {pct_cov:.1f}% retained, {feat_ret:.1f}/{feat_tot} "
              f"{feature_type.lower()} features, pLDDT={mean_plddt}, risk={risk} "
              f"(coverage-heuristic: {coverage_risk}), "
              f"mean_predicted_score={iso_dms_mean} ({iso_dms_n} variants)")

    df = pd.DataFrame(rows)
    if compute_dms:
        total_variants = df["n_dms_variants"].sum()
        print(f"\nTotal DMS Variants: {total_variants:,} Scored")
    return df, feature_type


def format_tab1_display(df: pd.DataFrame, feature_label: str = "TMs Retained") -> pd.DataFrame:
    """
    Formats the raw Tab 1 dataframe into the target display layout:
        Isoform Name | Transcript ID | Length (aa) | Sequence Coverage |
        <feature_label> | Localisation | Stability Risk | Mean pLDDT |
        Mean Predicted Score
    `feature_label` should describe whatever feature_type was tracked, e.g.
    'TMs Retained' for Transmembrane, 'Domains Retained' for Domain, etc.
    Mean Predicted Score is an in-silico DMS score (ESM2 wt-marginal) —
    see score_isoform_dms() docstring for exactly what it does and does
    not represent.
    """
    out = pd.DataFrame({
        "Isoform Name": df["name"],
        "Transcript ID": df["isoform_id"],
        "Length (aa)": df["length_aa"].map(lambda x: f"{x} aa"),
        "Sequence Coverage": df["pct_canonical_retained"].map(lambda x: f"{x:.1f}%"),
        feature_label: [
            f"{ret:.1f}/{tot}" for ret, tot in zip(df["features_retained"], df["features_total"])
        ],
        "Localisation": df["localisation"],
        "Stability Risk": df["stability_risk"],
        "Mean pLDDT": df["mean_plddt"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"),
    })
    if "coverage_heuristic_risk" in df.columns:
        out["Coverage Heuristic Risk (unvalidated)"] = df["coverage_heuristic_risk"].str.title()
    if "mean_predicted_score" in df.columns:
        out["Mean Predicted Score"] = df["mean_predicted_score"].map(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Protein isoform overview & alignment stats (any UniProt protein)")
    parser.add_argument("--uniprot-id", default=UNIPROT_ID, help="Canonical UniProt accession (e.g. Q13286)")
    parser.add_argument("--fasta", default=None,
                         help="Optional local FASTA of isoform sequences (protein or cDNA/mRNA, "
                              "auto-detected). If omitted, isoforms are fetched live from UniProt.")
    parser.add_argument("--feature-type", default="auto",
                         help="UniProt feature 'type' to track retention of, e.g. 'Transmembrane' "
                              "(membrane proteins like CLN3) or 'Domain' (e.g. CTC1's OB-fold domains). "
                              "Default 'auto' picks whichever feature type has the most annotated "
                              "instances on this specific protein -- no manual lookup needed.")
    parser.add_argument("--feature-label", default=None,
                         help="Display column name for the feature-retention column, e.g. "
                              "'TMs Retained' or 'Domains Retained'. Defaults to '<feature-type>s Retained'.")
    parser.add_argument("--out", default="tab1_overview.csv", help="Output CSV path (raw values)")
    parser.add_argument("--out-display", default="tab1_overview_display.csv",
                         help="Output CSV path (formatted for the dashboard-style layout)")
    parser.add_argument("--no-dms", action="store_true",
                         help="Skip in-silico DMS scoring (Mean Predicted Score column). "
                              "Useful for a quick run without downloading/running ESM2.")
    parser.add_argument("--dms-model", default=DEFAULT_DMS_MODEL,
                         help="Hugging Face ESM2 checkpoint to use for DMS scoring. Larger = slower "
                              "but more accurate. Options include facebook/esm2_t6_8M_UR50D (fastest), "
                              "facebook/esm2_t12_35M_UR50D (default), facebook/esm2_t30_150M_UR50D, "
                              "facebook/esm2_t33_650M_UR50D (GPU recommended).")
    parser.add_argument("--device", default="cpu", help="Device for DMS scoring: 'cpu' or 'cuda' (default: cpu)")
    args = parser.parse_args()

    feature_label_override = args.feature_label

    df, used_feature_type = build_tab1_summary(
        uniprot_id=args.uniprot_id, fasta_path=args.fasta, feature_type=args.feature_type,
        compute_dms=not args.no_dms, dms_model=args.dms_model, dms_device=args.device
    )
    feature_label = feature_label_override or f"{used_feature_type}s Retained"
    df.to_csv(args.out, index=False)
    print(f"\nSaved (raw): {args.out}")

    display_df = format_tab1_display(df, feature_label=feature_label)
    display_df.to_csv(args.out_display, index=False)
    print(f"Saved (formatted): {args.out_display}")
    print(display_df.to_string(index=False))
