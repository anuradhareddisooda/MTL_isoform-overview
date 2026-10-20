"""
Isoform domain map
==================
One figure: the canonical protein on top, then one row per isoform, all on the
canonical residue axis. For each isoform:
  - grey backbone  = canonical residues retained (from the sequence alignment)
  - dashed red line = canonical residues deleted inside the isoform
  - signal peptide / TM helices (top half) and domains/repeats/motifs (bottom half):
      solid fill = retained part, dashed outline = part that is lost
  - triangles above the bar = functional sites (retained = filled, lost = red x)
  - magenta box "+N aa" = novel residues that have no canonical counterpart
    (drawn at the point where they attach; its width is NOT to scale)

Called from isoform_evidence.py; needs matplotlib only.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

TOP_TYPES = ("Signal", "Transit peptide", "Transmembrane")
DOMAIN_TYPES = ("Domain", "Repeat", "Zinc finger", "Motif", "Coiled coil")
SITE_TYPES = ("Active site", "Binding site", "Site", "Disulfide bond", "Cross-link")
ENDPOINT_ONLY = ("Disulfide bond", "Cross-link")
BAR_TYPES = TOP_TYPES + DOMAIN_TYPES
MARKER_COLORS = {"Glycosylation": "#e03131", "Modified residue": "#2f9e44", "Lipidation": "#d6336c"}
MAX_MARKERS_PER_TYPE = 40  # skip a marker type entirely if it would clutter the figure

COLORS = {
    "Signal": "#e8912d", "Transit peptide": "#e8912d", "Transmembrane": "#2f6fbf",
    "Domain": "#2e9e6b", "Repeat": "#1f9aa8", "Zinc finger": "#8e5bbf",
    "Motif": "#a89a1f", "Coiled coil": "#9a6b3f",
}
NOVEL = "#d6336c"
LOST = "#c0392b"
BACKBONE = "#aab2bd"


def _lanes(feats):
    """Greedy interval packing so overlapping domain-type features get separate lanes."""
    ends, out = [], []
    for f in sorted(feats, key=lambda f: (f["start"], -f["end"])):
        for i, e in enumerate(ends):
            if f["start"] > e:
                ends[i] = f["end"]
                out.append((f, i))
                break
        else:
            ends.append(f["end"])
            out.append((f, len(ends) - 1))
    return out, max(1, len(ends))


def novel_segments(blocks, canon_len, iso_len):
    """[(canonical_x, n_novel_aa)] for novel residues, and [(x0, x1)] for internal deletions."""
    blocks = [b for b in blocks if b[2] > 0]
    novel, deleted = [], []
    if not blocks:
        return novel, deleted
    (a0, b0, _), (al, bl, sl) = blocks[0], blocks[-1]
    if b0 > 0:
        novel.append((a0, b0))
    for (a, b, s), (a2, b2, _) in zip(blocks, blocks[1:]):
        gap_c, gap_i = a2 - (a + s), b2 - (b + s)
        if gap_c > 0:
            deleted.append((a + s, a2))
        if gap_i > 0:
            novel.append(((a + s + a2) / 2 if gap_c > 0 else a + s, gap_i))
    rem_i = iso_len - (bl + sl)
    if rem_i > 0:
        novel.append((al + sl, rem_i))
    return novel, deleted


def _retained_set(blocks):
    return [(a, a + s) for a, _, s in blocks if s > 0]


def _overlap(lo, hi, segs):
    return [(max(lo, s0), min(hi, s1)) for s0, s1 in segs if min(hi, s1) > max(lo, s0)]


def _draw_features(ax, y, h, feats, segs, outline_lost=True, domain_lanes=None):
    """Top half: signal/TM. Bottom half: domain-like features (single lane unless domain_lanes given)."""
    for f in feats:
        if f["type"] not in BAR_TYPES:
            continue
        top = f["type"] in TOP_TYPES
        col = COLORS.get(f["type"], "#777777")
        if top:
            y0, hh = y, h * 0.5
        elif domain_lanes is not None:
            lane = domain_lanes[id(f)]
            y0, hh = y + h * 0.5 + lane * h * 0.5, h * 0.5
        else:
            y0, hh = y + h * 0.5, h * 0.5
        x0, x1 = f["start"] - 1, f["end"]
        kept_parts = _overlap(x0, x1, segs)
        if outline_lost and sum(b - a for a, b in kept_parts) < (x1 - x0):  # outline only if partly/fully lost
            ax.add_patch(Rectangle((x0, y0), x1 - x0, hh, fill=False, ec=col, lw=0.7, ls=(0, (3, 2)), alpha=0.8))
        for s0, s1 in kept_parts:
            ax.add_patch(Rectangle((s0, y0), s1 - s0, hh, fc=col, ec="none", alpha=0.9 if top else 0.6))


def _draw_sites(ax, y, feats, segs):
    for f in feats:
        if f["type"] in MARKER_COLORS:  # glycosylation / modified residue / lipidation: small round markers
            col = MARKER_COLORS[f["type"]]
            kept = bool(_overlap(f["start"] - 1, f["start"], segs))
            ax.plot(f["start"] - 0.5, y - 0.30, marker="o", ms=3.2, mfc=col if kept else "none", mec=col,
                    mew=0.8, lw=0, alpha=0.95 if kept else 0.6, clip_on=False)
            continue
        if f["type"] not in SITE_TYPES:
            continue
        positions = [f["start"], f["end"]] if f["type"] in ENDPOINT_ONLY else [f["start"]]
        for p in positions:
            kept = bool(_overlap(p - 1, p, segs))
            if kept:
                ax.plot(p - 0.5, y - 0.12, marker="v", ms=3.5, color="#222222", lw=0, clip_on=False)
            else:
                ax.plot(p - 0.5, y - 0.12, marker="x", ms=4, color=LOST, mew=1.1, lw=0, clip_on=False)


def draw_isoform_map(canon_len, feats, isoforms, title="", path_png=None, path_svg=None, uniprot_id=""):
    """
    isoforms: list of dict(name, id, length, blocks=[(a, b, size), ...]) using 0-based difflib blocks.
    feats: list of dict(type, start, end, description) in 1-based canonical numbering.
    """
    counts = {}
    for ft in feats:
        counts[ft["type"]] = counts.get(ft["type"], 0) + 1
    feats = [ft for ft in feats
             if ft["type"] in BAR_TYPES or ft["type"] in SITE_TYPES
             or (ft["type"] in MARKER_COLORS and counts[ft["type"]] <= MAX_MARKERS_PER_TYPE)]
    marker_types_drawn = [t for t in MARKER_COLORS if any(ft["type"] == t for ft in feats)]
    dom = [f for f in feats if f["type"] in DOMAIN_TYPES]
    dom_lane_list, n_lanes = _lanes(dom)
    dom_lanes = {id(f): lane for f, lane in dom_lane_list}

    h = 0.9                    # row height
    canon_h = h * (0.5 + 0.5 * n_lanes)
    row_gap = 0.55
    n = len(isoforms)
    total_h = canon_h + row_gap + n * (h + row_gap) + 0.8
    fig_h = max(3.2, 0.55 * total_h + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    # ---- canonical ----
    y = 0.0
    ax.add_patch(Rectangle((0, y + h * 0.5 - 0.04), canon_len, 0.08, fc=BACKBONE, ec="none", zorder=0))
    _draw_features(ax, y, h, feats, [(0, canon_len)], outline_lost=False, domain_lanes=dom_lanes)
    _draw_sites(ax, y, feats, [(0, canon_len)])
    # labels inside wide features
    for f in feats:
        if f["type"] not in BAR_TYPES:
            continue
        width = f["end"] - f["start"] + 1
        if width / canon_len < 0.045:
            continue
        label = f["description"] or f["type"]
        if f["type"] == "Transmembrane":
            label = "TM"
        if f["type"] in TOP_TYPES:
            yy = y + h * 0.25
        else:
            yy = y + h * 0.5 + dom_lanes.get(id(f), 0) * h * 0.5 + h * 0.25
        ax.text((f["start"] + f["end"]) / 2 - 0.5, yy, label[:22].replace("$", r"\$"), ha="center", va="center", fontsize=6.5, color="white",
                fontweight="bold")
    tm_sorted = sorted((f for f in feats if f["type"] == "Transmembrane"), key=lambda f: f["start"])
    for i, f in enumerate(tm_sorted, 1):
        if (f["end"] - f["start"] + 1) / canon_len < 0.045:
            ax.text((f["start"] + f["end"]) / 2 - 0.5, y - 0.28, f"TM{i}", ha="center", va="bottom", fontsize=6, color=COLORS["Transmembrane"])
    ylabels, yticks = [f"Canonical {uniprot_id}".strip() + f" ({canon_len} aa)"], [y + canon_h / 2]

    # ---- isoforms ----
    y = canon_h + row_gap
    for iso in isoforms:
        segs = _retained_set(iso["blocks"])
        kept = sum(s1 - s0 for s0, s1 in segs)
        novel, deleted = novel_segments(iso["blocks"], canon_len, iso["length"])

        for s0, s1 in segs:
            ax.add_patch(Rectangle((s0, y + h * 0.5 - 0.04), s1 - s0, 0.08, fc=BACKBONE, ec="none", zorder=0))
        for d0, d1 in deleted:
            ax.plot([d0, d1], [y + h * 0.5] * 2, color=LOST, lw=1.1, ls=(0, (2, 2)), zorder=1)
            if (d1 - d0) / canon_len > 0.04:
                ax.text((d0 + d1) / 2, y + h * 0.5 - 0.12, f"\u0394{d1 - d0}", ha="center", va="bottom", fontsize=6.5, color=LOST)
        _draw_features(ax, y, h, feats, segs, outline_lost=True, domain_lanes=None)
        _draw_sites(ax, y, feats, segs)
        for x, cnt in novel:
            w = max(canon_len * 0.012, 3)
            ax.add_patch(Rectangle((x - w / 2, y + h * 0.18), w, h * 0.64, fc=NOVEL, ec="none"))
            ax.text(x, y + h + 0.05, f"+{cnt} aa", ha="center", va="top", fontsize=6.5, color=NOVEL)

        name = iso["name"] if iso.get("name") else iso["id"]
        ylabels.append(f"{str(name)[:30].replace('$', chr(92) + '$')}  [{100.0 * kept / canon_len:.0f}% canonical; {iso['length']} aa]")
        yticks.append(y + h / 2)
        y += h + row_gap

    ax.set_xlim(0, canon_len)
    ax.set_ylim(y, -0.55)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.set_xlabel("Canonical residue number", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, loc="left")

    handles = [
        Patch(fc=COLORS["Signal"], label="Signal peptide"),
        Patch(fc=COLORS["Transmembrane"], label="TM helix"),
        Patch(fc=COLORS["Domain"], alpha=0.6, label="Domain / repeat / motif"),
        Patch(fill=False, ec="#555555", ls="--", label="Lost part of a feature"),
        Line2D([0], [0], color=LOST, ls=(0, (2, 2)), lw=1.2, label="Deleted canonical region (\u0394 aa)"),
        Patch(fc=NOVEL, label="Novel residues (+aa; not to scale)"),
        Line2D([0], [0], marker="v", color="#222222", lw=0, ms=4, label="Site retained"),
        Line2D([0], [0], marker="x", color=LOST, lw=0, ms=5, mew=1.1, label="Site lost"),
    ]
    names = {"Glycosylation": "Glycosylation site", "Modified residue": "Modified residue (e.g. phospho)",
             "Lipidation": "Lipidation site"}
    for t in marker_types_drawn:
        handles.append(Line2D([0], [0], marker="o", color=MARKER_COLORS[t], lw=0, ms=4, label=names[t]))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=7, frameon=False)
    fig.tight_layout()
    if path_png:
        fig.savefig(path_png, dpi=150, bbox_inches="tight", facecolor="white")
    if path_svg:
        fig.savefig(path_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
