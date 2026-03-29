"""Helpers for accessing the STRING protein–protein interaction database.

Designed for this pipeline's generated code. Prefer using partner discovery
(`interaction_partners`) when you want to test shared-cofactor hypotheses:

    from src.utils.string_client import fetch_interaction_partners, shared_partners
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests


def _ensure_cache_dir(cache_dir: str | Path | None) -> Path:
    """Return a writable cache directory path, creating it if needed."""
    path = Path(cache_dir or ".string_cache")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path_for_request(
    *,
    cache_dir: str | Path | None,
    endpoint: str,
    params: dict,
) -> Path:
    """Build a stable cache path based on endpoint + request params."""
    cache_root = _ensure_cache_dir(cache_dir)
    payload = {
        "endpoint": endpoint,
        "params": params,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    safe_endpoint = endpoint.strip("/").replace("/", "_")
    return cache_root / f"{safe_endpoint}_{digest}.tsv"


def fetch_interaction_partners(
    proteins: Sequence[str] | Iterable[str],
    *,
    species: int = 9606,
    min_score: int = 700,
    limit: int = 200,
    api_base: str = "https://string-db.org/api",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch interaction partners for each input protein.

    This uses STRING's `interaction_partners` endpoint, which returns partners
    of each query protein (not just edges among your input list). This is the
    right primitive for "shared cofactors" questions.

    Args:
        proteins: Iterable of protein/gene identifiers recognizable by STRING
            (e.g. Ensembl IDs or gene symbols; see STRING documentation).
        species: NCBI taxon ID (9606 = human).
        min_score: Minimum combined evidence score threshold **parameter** (0–1000).
            Note: STRING's returned `score` values in JSON/TSV are typically in 0–1.
            The API parameter `required_score=min_score` uses 0–1000.
        limit: Optional maximum number of partners per protein (STRING parameter: `limit`).
        api_base: STRING API base URL.
        cache_dir: Optional directory for caching TSV responses on disk.

    Returns:
        DataFrame with partner rows. Columns follow STRING TSV schema and
        typically include `queryItem`, `stringId_A`, `stringId_B`,
        `preferredName_A`, `preferredName_B`, `score` (usually 0–1), and evidence subscores
        (e.g. `escore`, `dscore`) when available.
    """
    proteins = sorted({p.strip() for p in proteins if str(p).strip()})
    if not proteins:
        return pd.DataFrame()

    endpoint = "/tsv/interaction_partners"
    identifiers = "%0d".join(proteins)
    data = {
        "identifiers": identifiers,
        "species": species,
        "required_score": min_score,
    }
    if limit is not None:
        data["limit"] = int(limit)

    cache_path = _cache_path_for_request(cache_dir=cache_dir, endpoint=endpoint, params=data)
    if cache_path.exists():
        return pd.read_csv(cache_path, sep="\t")

    url = f"{api_base.rstrip('/')}{endpoint}"

    resp = requests.post(url, data=data, timeout=60)
    resp.raise_for_status()

    cache_path.write_text(resp.text)
    return pd.read_csv(cache_path, sep="\t")


def filter_by_evidence(
    df: pd.DataFrame,
    *,
    min_escore: float | None = None,
    min_dscore: float | None = None,
) -> pd.DataFrame:
    """Filter STRING rows by experimental (`escore`) and curated DB (`dscore`) evidence.

    If a column is missing, the filter is skipped for that column.
    """
    out = df
    if min_escore is not None and "escore" in out.columns:
        out = out[out["escore"].fillna(0) >= float(min_escore)]
    if min_dscore is not None and "dscore" in out.columns:
        out = out[out["dscore"].fillna(0) >= float(min_dscore)]
    return out.reset_index(drop=True)


def clean_partner_rows(
    df: pd.DataFrame,
    *,
    name_a_col: str = "preferredName_A",
    name_b_col: str = "preferredName_B",
) -> pd.DataFrame:
    """Remove self-loops and exact duplicate rows when possible."""
    out = df
    if name_a_col in out.columns and name_b_col in out.columns:
        out = out[out[name_a_col].fillna("") != out[name_b_col].fillna("")]
    out = out.drop_duplicates()
    return out.reset_index(drop=True)


def _choose_partner_col(df: pd.DataFrame) -> str | None:
    """Pick a reasonable partner column name across possible STRING schemas."""
    for c in ("preferredName_B", "stringId_B", "protein2", "partner", "partnerName"):
        if c in df.columns:
            return c
    return None

def shared_partners(
    partners_df: pd.DataFrame,
    *,
    query_col: str = "queryItem",
    partner_col: str | None = "preferredName_B",
) -> set[str]:
    """Return the intersection of partner sets across all query proteins in the table."""
    if partners_df is None or partners_df.empty:
        return set()
    if query_col not in partners_df.columns:
        return set()
    if partner_col is None or partner_col not in partners_df.columns:
        partner_col = _choose_partner_col(partners_df)
    if partner_col is None:
        return set()
    by_query = partners_df.groupby(query_col)[partner_col].apply(
        lambda s: set(x for x in s if pd.notna(x))
    )
    sets = list(by_query.values)
    if not sets:
        return set()
    shared = sets[0].copy()
    for s in sets[1:]:
        shared &= s
    return shared


def fetch_shared_partners(
    proteins: Sequence[str] | Iterable[str],
    *,
    species: int = 9606,
    min_score: int = 700,
    limit: int | None = None,
    min_escore: float | None = 0.2,
    min_dscore: float | None = 0.3,
    clean: bool = True,
    api_base: str = "https://string-db.org/api",
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, set[str]]:
    """Convenience wrapper: fetch partners, apply evidence filters, compute intersection.

    Defaults are intentionally conservative for mechanism claims:
    min_escore=0.2 and min_dscore=0.3 help drop text-mining-only / weak edges.
    """
    df = fetch_interaction_partners(
        proteins,
        species=species,
        min_score=min_score,
        limit=limit,
        api_base=api_base,
        cache_dir=cache_dir,
    )
    df_f = filter_by_evidence(df, min_escore=min_escore, min_dscore=min_dscore)
    if clean:
        df_f = clean_partner_rows(df_f)
    return df_f, shared_partners(df_f)
