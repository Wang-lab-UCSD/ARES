"""Helpers for working with MEME-format motif files.

These utilities are designed for use by generated verification code so that
motif IDs do not need to be hard-coded in the data manifest. Instead, the
code can infer motif IDs for a given transcription factor name directly
from a combined MEME file (e.g. JASPAR + HOCOMOCO + CIS-BP).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class MotifHeader:
    """Parsed representation of a MEME 'MOTIF ...' header line."""

    motif_token: str  # what FIMO expects for --motif
    label: str  # segment before first '|', e.g. 'NFYA' or 'AHR::ARNT'
    source: str  # middle field when present, e.g. 'jaspar'
    last_id: str  # last field when present, e.g. 'MA0060.3'


def _parse_motif_header_line(line: str) -> Optional[MotifHeader]:
    """Parse a MEME motif header line.

    Supports both:
      - MOTIF <token>
      - MOTIF <id> <altname>
    We treat the first token after 'MOTIF' as the *motif_token* that FIMO uses.
    """
    if not line.startswith("MOTIF "):
        return None
    tokens = line.strip().split()
    if len(tokens) < 2:
        return None

    motif_token = tokens[1].strip()
    parts = motif_token.split("|")
    label = parts[0].strip() if parts else motif_token
    source = parts[1].strip().lower() if len(parts) >= 2 else ""
    last_id = parts[-1].strip() if parts else motif_token
    return MotifHeader(
        motif_token=motif_token,
        label=label,
        source=source,
        last_id=last_id,
    )


def _tf_variants(tf_name: str) -> list[str]:
    """Generate reasonable TF-name variants for matching (case-insensitive)."""
    t = tf_name.strip()
    if not t:
        return []
    variants = {t.upper()}
    # Common punctuation/spacing variants (e.g. NF-YA vs NFYA)
    variants.add(t.upper().replace("-", ""))
    variants.add(t.upper().replace("_", ""))
    variants.add(t.upper().replace("-", "").replace("_", ""))
    variants.add(t.upper().replace("-", "_"))
    variants.add(t.upper().replace("_", "-"))
    return sorted(v for v in variants if v)


def find_motif_ids_for_tf(
    tf_name: str,
    meme_path: str,
    allowed_sources: Iterable[str] | None = None,
) -> List[str]:
    """Return MEME motif tokens for a TF from a combined MEME file.

    IMPORTANT: despite the legacy name, this function returns motif tokens that are
    appropriate for FIMO's `--motif` flag (i.e., the first token after `MOTIF`),
    not just the trailing database ID segment (like MA0060.3).

    The combined MEME used in this project has MOTIF lines of the form:

        MOTIF AHR::ARNT|jaspar|MA0006.1
        MOTIF AHR|cisbp|M08716_2.00
        MOTIF AHRR|hocomoco12|AHRR.H12CORE.0.P.C

    We treat:
        - The first token after 'MOTIF' as the motif token used by FIMO.
        - Within that token, the segment before the first '|' is the TF label.

    Args:
        tf_name: Transcription factor name (e.g. 'ATF3', 'USF1').
        meme_path: Path to MEME-format motif file.
        allowed_sources: Optional iterable of source tags to keep
            (e.g. ['jaspar', 'hocomoco12', 'cisbp']). When provided,
            motifs whose middle field (between '|' segments) is not in
            this set are skipped.

    Returns:
        List of motif IDs (strings). May be empty if no matching motifs
        are found.
    """
    variants = _tf_variants(tf_name)
    allowed = {s.lower() for s in allowed_sources} if allowed_sources else None
    exact: list[str] = []
    partial: list[str] = []

    with open(meme_path) as f:
        for line in f:
            mh = _parse_motif_header_line(line)
            if mh is None:
                continue
            if allowed is not None and mh.source not in allowed:
                continue

            label_u = mh.label.upper()
            if label_u in variants:
                exact.append(mh.motif_token)
            else:
                # Allow substring match for complexes like AHR::ARNT or NFYA/NFY-A variants
                if any(v in label_u for v in variants):
                    partial.append(mh.motif_token)

    # Prefer exact label matches; fall back to partial matches. De-duplicate preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for tok in exact + partial:
        if tok not in seen:
            out.append(tok)
            seen.add(tok)
    return out


def find_motif_tokens_for_tf(
    tf_name: str,
    meme_path: str,
    allowed_sources: Iterable[str] | None = None,
) -> List[str]:
    """Alias for clarity; returns tokens suitable for FIMO --motif."""
    return find_motif_ids_for_tf(tf_name, meme_path, allowed_sources=allowed_sources)


def find_motif_ids_for_tfs(
    tf_names: Iterable[str],
    meme_path: str,
    allowed_sources: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """Convenience wrapper to get motif IDs for multiple TFs.

    Args:
        tf_names: Iterable of TF names (e.g. ['ATF3', 'USF1']).
        meme_path: Path to MEME-format motif file.
        allowed_sources: Optional iterable of allowed source tags.

    Returns:
        Mapping from TF name to list of motif IDs.
    """
    return {
        tf_name: find_motif_ids_for_tf(tf_name, meme_path, allowed_sources=allowed_sources)
        for tf_name in tf_names
    }

