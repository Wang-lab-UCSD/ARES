"""Tests for src.utils.string_client helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.utils.string_client import (
    clean_partner_rows,
    fetch_interaction_partners,
    fetch_shared_partners,
    filter_by_evidence,
    shared_partners,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _partner_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


STRING_TSV = (
    "queryItem\tstringId_A\tstringId_B\tpreferredName_A\tpreferredName_B\t"
    "ncbiTaxonId\tscore\tescore\tdscore\n"
    # SP1 partners: EP300 (high evidence), CTCF (low evidence)
    "SP1\t9606.A\t9606.B\tSP1\tEP300\t9606\t0.95\t0.6\t0.5\n"
    "SP1\t9606.A\t9606.C\tSP1\tCTCF\t9606\t0.80\t0.1\t0.1\n"
    # NFYA partners: EP300 (high evidence), HDAC1 (high evidence)
    "NFYA\t9606.D\t9606.B\tNFYA\tEP300\t9606\t0.90\t0.5\t0.4\n"
    "NFYA\t9606.D\t9606.E\tNFYA\tHDAC1\t9606\t0.85\t0.4\t0.3\n"
    # EP300 is shared by both SP1 and NFYA with min_escore=0.3, min_dscore=0.3
    # CTCF fails evidence filter (escore=0.1 < 0.3)
)


# ---------------------------------------------------------------------------
# filter_by_evidence
# ---------------------------------------------------------------------------

class TestFilterByEvidence:
    def test_filters_low_escore(self):
        df = _partner_df([
            {"preferredName_B": "A", "escore": 0.5, "dscore": 0.5},
            {"preferredName_B": "B", "escore": 0.1, "dscore": 0.5},
        ])
        out = filter_by_evidence(df, min_escore=0.3)
        assert list(out["preferredName_B"]) == ["A"]

    def test_filters_low_dscore(self):
        df = _partner_df([
            {"preferredName_B": "A", "escore": 0.5, "dscore": 0.5},
            {"preferredName_B": "B", "escore": 0.5, "dscore": 0.1},
        ])
        out = filter_by_evidence(df, min_dscore=0.3)
        assert list(out["preferredName_B"]) == ["A"]

    def test_missing_column_skips_filter(self):
        df = _partner_df([{"preferredName_B": "A", "escore": 0.1}])
        # dscore column absent — filter should be skipped
        out = filter_by_evidence(df, min_dscore=0.9)
        assert len(out) == 1

    def test_no_filters_returns_all(self):
        df = _partner_df([{"preferredName_B": x, "escore": 0.1} for x in "ABC"])
        out = filter_by_evidence(df)
        assert len(out) == 3


# ---------------------------------------------------------------------------
# clean_partner_rows
# ---------------------------------------------------------------------------

class TestCleanPartnerRows:
    def test_removes_self_loops(self):
        df = _partner_df([
            {"preferredName_A": "SP1", "preferredName_B": "SP1"},
            {"preferredName_A": "SP1", "preferredName_B": "NFYA"},
        ])
        out = clean_partner_rows(df)
        assert len(out) == 1
        assert out.iloc[0]["preferredName_B"] == "NFYA"

    def test_removes_exact_duplicates(self):
        df = _partner_df([
            {"preferredName_A": "SP1", "preferredName_B": "NFYA"},
            {"preferredName_A": "SP1", "preferredName_B": "NFYA"},
        ])
        out = clean_partner_rows(df)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# shared_partners
# ---------------------------------------------------------------------------

class TestSharedPartners:
    def test_intersection_of_two_proteins(self):
        df = _partner_df([
            {"queryItem": "SP1", "preferredName_B": "EP300"},
            {"queryItem": "SP1", "preferredName_B": "CTCF"},
            {"queryItem": "NFYA", "preferredName_B": "EP300"},
            {"queryItem": "NFYA", "preferredName_B": "HDAC1"},
        ])
        result = shared_partners(df)
        # EP300 appears in both SP1's and NFYA's partner lists
        assert result == {"EP300"}

    def test_no_shared_partners_returns_empty(self):
        df = _partner_df([
            {"queryItem": "SP1", "preferredName_B": "CTCF"},
            {"queryItem": "NFYA", "preferredName_B": "EP300"},
        ])
        assert shared_partners(df) == set()

    def test_empty_dataframe_returns_empty(self):
        assert shared_partners(pd.DataFrame()) == set()

    def test_missing_query_col_returns_empty(self):
        df = _partner_df([{"preferredName_B": "CTCF"}])
        assert shared_partners(df) == set()


# ---------------------------------------------------------------------------
# fetch_interaction_partners (mocked HTTP)
# ---------------------------------------------------------------------------

class TestFetchInteractionPartners:
    def test_returns_dataframe_from_api(self, tmp_path):
        with patch("src.utils.string_client.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.text = STRING_TSV
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            df = fetch_interaction_partners(
                ["SP1", "NFYA"],
                species=9606,
                min_score=700,
                cache_dir=tmp_path / "cache",
            )

        assert isinstance(df, pd.DataFrame)
        assert "preferredName_B" in df.columns
        assert len(df) > 0

    def test_uses_cache_on_second_call(self, tmp_path):
        cache_dir = tmp_path / "cache"

        with patch("src.utils.string_client.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.text = STRING_TSV
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            fetch_interaction_partners(["SP1"], cache_dir=cache_dir)
            fetch_interaction_partners(["SP1"], cache_dir=cache_dir)

        # API should only be called once; second call hits disk cache
        assert mock_post.call_count == 1

    def test_empty_input_returns_empty_df(self, tmp_path):
        df = fetch_interaction_partners([], cache_dir=tmp_path / "cache")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# fetch_shared_partners (end-to-end with mocked HTTP)
# ---------------------------------------------------------------------------

class TestFetchSharedPartners:
    def test_returns_shared_set_and_df(self, tmp_path):
        with patch("src.utils.string_client.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.text = STRING_TSV
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            df, shared = fetch_shared_partners(
                ["SP1", "NFYA"],
                species=9606,
                min_score=700,
                min_escore=0.3,
                min_dscore=0.3,
                cache_dir=tmp_path / "cache",
            )

        assert isinstance(df, pd.DataFrame)
        assert isinstance(shared, set)

    def test_shared_partners_is_intersection(self, tmp_path):
        # SP1 partners (after evidence filter): EP300 (escore=0.6, dscore=0.5)
        # NFYA partners (after evidence filter): EP300 (escore=0.5, dscore=0.4), HDAC1 (escore=0.4, dscore=0.3)
        # Intersection: EP300
        with patch("src.utils.string_client.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.text = STRING_TSV
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            _, shared = fetch_shared_partners(
                ["SP1", "NFYA"],
                min_score=700,
                min_escore=0.3,
                min_dscore=0.3,
                cache_dir=tmp_path / "cache",
            )

        assert "EP300" in shared
        assert "CTCF" not in shared   # CTCF fails evidence filter (escore=0.1)

    def test_no_evidence_filter_returns_broader_set(self, tmp_path):
        with patch("src.utils.string_client.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.text = STRING_TSV
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            _, shared_strict = fetch_shared_partners(
                ["SP1", "NFYA"],
                min_score=700,
                min_escore=0.5,
                min_dscore=0.5,
                cache_dir=tmp_path / "cache_strict",
            )

        with patch("src.utils.string_client.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.text = STRING_TSV
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            _, shared_loose = fetch_shared_partners(
                ["SP1", "NFYA"],
                min_score=700,
                min_escore=None,
                min_dscore=None,
                cache_dir=tmp_path / "cache_loose",
            )

        # Loose filter should produce a superset
        assert shared_strict <= shared_loose
