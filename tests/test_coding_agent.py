"""Tests for the coding agent."""

import pytest

from src.agents.coding_agent import (
    CodingAgent,
    _strip_end_turn_tokens,
    _strip_invoke_tags,
    _strip_orphan_execute_closes,
    _strip_runaway_tokens,
)


class TestCodeExtraction:
    """Tests for code extraction from LLM responses."""

    @pytest.fixture
    def agent(self):
        """Create a coding agent with mock LLM."""
        # We'll test the extraction methods without actually calling LLM
        class MockLLM:
            pass

        return CodingAgent(MockLLM())

    def test_extract_plain_code(self, agent):
        """Test extracting plain code without markdown."""
        response = "print('hello')\nx = 1 + 2"
        code = agent._extract_code(response)

        assert code == "print('hello')\nx = 1 + 2"

    def test_extract_from_markdown_block(self, agent):
        """Test extracting code from markdown code block."""
        response = """Here's the code:

```python
print('hello')
x = 1 + 2
```

This should work."""

        code = agent._extract_code(response)

        assert "print('hello')" in code
        assert "x = 1 + 2" in code

    def test_extract_from_unmarked_block(self, agent):
        """Test extracting code from unmarked code block."""
        response = """```
print('hello')
```"""

        code = agent._extract_code(response)
        assert "print('hello')" in code

    def test_extract_multiple_blocks(self, agent):
        """_extract_code returns the first code block when multiple are present."""
        response = """First:
```python
import pandas as pd
```

Second:
```python
df = pd.read_csv('data.csv')
```"""

        code = agent._extract_code(response)

        assert "import pandas as pd" in code


# -----------------------------------------------------------------------------
# _strip_invoke_tags: defense against MiniMax <invoke>-tag hallucination loop
# -----------------------------------------------------------------------------
#
# Regression source: the Apr 14 K562_ELK3_YY1 run crashed after MiniMax M2.7
# emitted 3 consecutive ~64K-token responses, each containing ~150 repetitions
# of <invoke name="lookup_tpm_by_symbol">...</invoke> blocks following a
# legitimate <execute>...</execute> block. The runaway tail blew the
# conversation context past the 204,800 MiniMax limit within 3 REPL turns.
#
# The fix strips <invoke>...</invoke> blocks from every LLM response BEFORE
# downstream processing: the leading legitimate content (<think>, <execute>,
# <solution>) is preserved, only the hallucinated tail is dropped. The
# cleaned content is what reaches conversation history; the raw response
# is still saved to debug_code/ for post-mortem inspection.


class TestStripInvokeTags:
    def test_no_invoke_tags_returns_unchanged(self):
        """Normal responses with no <invoke> tags must pass through untouched."""
        content = (
            "<think>Plan: load peaks, scan motifs, compare.</think>\n"
            "<execute>\n"
            "from src.utils.bioio import read_narrowpeak\n"
            "df = read_narrowpeak(data_files['ELK3_peaks'])\n"
            "print(df.shape)\n"
            "</execute>"
        )
        cleaned, n = _strip_invoke_tags(content)
        assert cleaned == content
        assert n == 0

    def test_strips_trailing_invoke_junk_keeps_execute_block(self):
        """Legitimate <execute> followed by <invoke> junk: keep execute, drop junk.

        Models the exact ELK3 failure shape — one legitimate code block
        followed by ~150 hallucinated tool-call blocks.
        """
        invoke_block = (
            '<invoke name="lookup_tpm_by_symbol">\n'
            '<parameter name="symbol">GAPDH</parameter>\n'
            '<parameter name="rnaseq_path">data_files[\'gene_quantification\']</parameter>\n'
            '<parameter name="gencode_path">data_files[\'gencode\']</parameter>\n'
            '</invoke>\n'
        )
        content = (
            "<think>Let me proceed step by step.</think>\n"
            "<execute>\n"
            "from src.utils.bioio import read_narrowpeak\n"
            "df = read_narrowpeak(data_files['ELK3_peaks'])\n"
            "print(df.shape)\n"
            "</execute>\n"
            + invoke_block * 150  # simulate the ~150 repetitions
        )
        cleaned, n = _strip_invoke_tags(content)

        assert n == 150
        # Legitimate content must survive
        assert "<think>" in cleaned
        assert "read_narrowpeak" in cleaned
        assert "<execute>" in cleaned
        assert "</execute>" in cleaned
        # All invoke junk must be gone
        assert "<invoke" not in cleaned
        assert "lookup_tpm_by_symbol" not in cleaned
        assert "GAPDH" not in cleaned
        # Size should collapse dramatically — the 150 invoke blocks were the bulk
        assert len(cleaned) < len(content) // 10

    def test_strips_invoke_with_single_quoted_attributes(self):
        """Single-quoted tag attributes are valid XML and must also strip."""
        content = (
            "<execute>x = 1</execute>\n"
            "<invoke name='tool_a'>"
            "<parameter name='k'>v</parameter>"
            "</invoke>"
        )
        cleaned, n = _strip_invoke_tags(content)
        assert n == 1
        assert "<invoke" not in cleaned
        assert "<execute>x = 1</execute>" in cleaned

    def test_strips_invoke_with_no_attributes(self):
        """Bare <invoke>...</invoke> with no attributes still strips."""
        content = "before\n<invoke>payload</invoke>\nafter"
        cleaned, n = _strip_invoke_tags(content)
        assert n == 1
        assert cleaned == "before\n\nafter"

    def test_strips_invoke_case_insensitive(self):
        """Uppercase / mixed-case tag variants also strip."""
        content = "a\n<INVOKE name='x'>stuff</INVOKE>\nb\n<Invoke>y</Invoke>\nc"
        cleaned, n = _strip_invoke_tags(content)
        assert n == 2
        assert "invoke" not in cleaned.lower() or "<invoke" not in cleaned.lower()

    def test_strips_multiline_invoke_spanning_body(self):
        """<invoke> blocks can span many lines of parameters."""
        content = (
            "leading\n"
            '<invoke name="foo">\n'
            '  <parameter name="a">\n'
            '    multi\n'
            '    line\n'
            '    value\n'
            '  </parameter>\n'
            '</invoke>\n'
            "trailing"
        )
        cleaned, n = _strip_invoke_tags(content)
        assert n == 1
        assert "multi" not in cleaned
        assert "leading" in cleaned
        assert "trailing" in cleaned

    def test_pure_invoke_junk_response_leaves_no_executable_content(self):
        """If the model ONLY emits <invoke> tags (no <execute>, no <solution>),
        stripping produces an effectively empty response. Downstream run_repl
        will correctly fall into the 'no tags in response' no_tag path and
        eventually short-circuit via the existing no_tag_count >= 3 logic.
        """
        content = (
            '<invoke name="x"><parameter name="a">1</parameter></invoke>\n'
            '<invoke name="y"><parameter name="b">2</parameter></invoke>\n'
        )
        cleaned, n = _strip_invoke_tags(content)
        assert n == 2
        assert "<invoke" not in cleaned
        assert "<execute>" not in cleaned
        assert "<solution>" not in cleaned
        assert cleaned.strip() == ""

    def test_python_code_mentioning_invoke_string_is_untouched(self):
        """Code inside <execute> that happens to contain the literal string
        'invoke' should NOT trigger stripping (we match only XML-tag form)."""
        content = (
            "<execute>\n"
            "# We invoke the helper below\n"
            'result = df.some_method()  # like langchain .invoke()\n'
            'msg = "<invoke>"  # a string containing the word, not a tag body\n'
            "</execute>"
        )
        cleaned, n = _strip_invoke_tags(content)
        # The literal `<invoke>` appears inside a Python string, with no
        # `</invoke>` closing tag — the regex requires a matching close tag,
        # so it doesn't greedily eat the rest of the response.
        assert n == 0
        assert "invoke the helper" in cleaned
        assert ".invoke()" in cleaned
        assert cleaned == content

    def test_empty_and_tagless_inputs(self):
        """Defensive: empty string, None-like, and content with no angle brackets."""
        cleaned, n = _strip_invoke_tags("")
        assert cleaned == "" and n == 0

        cleaned, n = _strip_invoke_tags("just plain text no tags")
        assert cleaned == "just plain text no tags" and n == 0


# -----------------------------------------------------------------------------
# _strip_end_turn_tokens: ZNF445 failure (Apr 15)
# -----------------------------------------------------------------------------
#
# MiniMax's `<end_turn>` marker is never valid in the ARES REPL protocol
# (we use <solution> to terminate). When the model enters a protocol-
# confusion state it spams <end_turn> thousands of times per response,
# ballooning the conversation history. ZNF445 accumulated 46,346 such
# tokens across 10 responses before its job terminated.


class TestStripEndTurnTokens:
    def test_strips_simple_end_turn(self):
        content = "before\n<end_turn>\nafter"
        cleaned, n = _strip_end_turn_tokens(content)
        assert n == 1
        assert "<end_turn" not in cleaned
        assert "before" in cleaned and "after" in cleaned

    def test_strips_repeated_end_turn_znf445_shape(self):
        """ZNF445 pattern: legitimate <think> + <execute>, then end_turn spam."""
        content = (
            "<think>Let me run the final verification.</think>\n"
            "<execute>\nprint('done')\n</execute>\n"
            + "<end_turn>\n" * 4700
        )
        cleaned, n = _strip_end_turn_tokens(content)
        assert n == 4700
        assert "<end_turn" not in cleaned
        # Legitimate content survives
        assert "<think>" in cleaned
        assert "<execute>" in cleaned
        assert "print('done')" in cleaned
        # Response collapses dramatically
        assert len(cleaned) < len(content) // 100

    def test_strips_case_insensitively(self):
        content = "a\n<END_TURN>\nb\n<End_Turn>\nc\n<end_turn>\nd"
        cleaned, n = _strip_end_turn_tokens(content)
        assert n == 3

    def test_strips_self_closing_form(self):
        """XML may write <end_turn /> as self-closing."""
        content = "a\n<end_turn />\nb\n<end_turn/>\nc"
        cleaned, n = _strip_end_turn_tokens(content)
        assert n == 2

    def test_no_end_turn_returns_unchanged(self):
        content = (
            "<think>plan</think>\n"
            "<execute>x = 1; print(x)</execute>"
        )
        cleaned, n = _strip_end_turn_tokens(content)
        assert cleaned == content
        assert n == 0

    def test_python_string_containing_end_turn_literal_is_stripped(self):
        """Tradeoff: we match `<end_turn>` as a literal tag wherever it
        appears, including inside a Python string. This is acceptable
        because legitimate code never embeds this string; valid protocol
        uses <solution> not <end_turn>."""
        # Document current behavior — stripping is aggressive on this
        # specific token since there's no legitimate use case.
        content = '<execute>msg = "<end_turn>"</execute>'
        cleaned, n = _strip_end_turn_tokens(content)
        assert n == 1  # The literal inside the string IS matched


# -----------------------------------------------------------------------------
# _strip_orphan_execute_closes: PTTG1 failure (Apr 15)
# -----------------------------------------------------------------------------
#
# PTTG1 emitted ~15,000 dangling </execute> close tags per response, after
# one legitimate execute block. A well-formed response has balanced
# <execute>...</execute> pairs; a run of 3+ close tags in a row with
# only whitespace between is always the runaway pattern. The 3+ threshold
# leaves single stray closes (rare but occasionally benign) intact.


class TestStripOrphanExecuteCloses:
    def test_strips_long_run_of_orphan_closes(self):
        """PTTG1 pattern: one legit execute + thousands of dangling closes."""
        content = (
            "<execute>print('hello')</execute>\n"
            + "</execute>\n" * 15000
        )
        cleaned, n = _strip_orphan_execute_closes(content)
        assert n == 15000
        # The legitimate balanced pair survives
        assert cleaned.count("<execute>") == 1
        assert cleaned.count("</execute>") == 1
        assert "print('hello')" in cleaned

    def test_strips_short_run_of_three(self):
        """Threshold is exactly 3 consecutive closes."""
        content = "valid\n</execute>\n</execute>\n</execute>\nend"
        cleaned, n = _strip_orphan_execute_closes(content)
        assert n == 3
        assert "</execute>" not in cleaned
        assert "valid" in cleaned and "end" in cleaned

    def test_single_stray_close_is_preserved(self):
        """A single </execute> (e.g., quirky response) is left in place —
        threshold is 3+ consecutive, so one orphan doesn't trigger."""
        content = "some text\n</execute>\nmore text"
        cleaned, n = _strip_orphan_execute_closes(content)
        assert n == 0
        assert cleaned == content

    def test_two_consecutive_closes_also_preserved(self):
        """Two consecutive closes still under threshold, kept intact."""
        content = "text\n</execute>\n</execute>\nmore"
        cleaned, n = _strip_orphan_execute_closes(content)
        assert n == 0
        assert cleaned == content

    def test_balanced_execute_pair_untouched(self):
        """Valid <execute>...</execute> block unchanged — only 1 close."""
        content = "<execute>\nimport pandas as pd\ndf = pd.read_csv(f)\n</execute>"
        cleaned, n = _strip_orphan_execute_closes(content)
        assert n == 0
        assert cleaned == content

    def test_legit_close_plus_run_of_orphans_keeps_first(self):
        """Realistic PTTG1 shape: one legitimate balanced pair, then spam.
        The balanced close is paired with its open and sits alone; the 3+
        run of orphan closes further down is stripped. The regex is
        greedy on consecutive closes, so the single legitimate close
        (separated from orphans by actual content) is left alone.
        """
        content = (
            "<execute>print(42)</execute>\n"
            "Some thinking about results.\n"
            + "</execute>\n" * 50
        )
        cleaned, n = _strip_orphan_execute_closes(content)
        assert n == 50
        # Legitimate balanced pair survives
        assert "<execute>print(42)</execute>" in cleaned
        # Spam block gone
        assert cleaned.count("</execute>") == 1

    def test_case_insensitive(self):
        content = "a\n</EXECUTE>\n</Execute>\n</execute>\nend"
        cleaned, n = _strip_orphan_execute_closes(content)
        assert n == 3


# -----------------------------------------------------------------------------
# _strip_runaway_tokens: integration — all three patterns combined
# -----------------------------------------------------------------------------


class TestStripRunawayTokens:
    def test_clean_response_unchanged(self):
        """A well-formed legitimate response passes through byte-for-byte."""
        content = (
            "<think>Plan: load and check.</think>\n"
            "<execute>\n"
            "df = pd.read_csv('x.csv')\n"
            "print(df.shape)\n"
            "</execute>"
        )
        cleaned, counts = _strip_runaway_tokens(content)
        assert cleaned == content
        assert counts == {"invoke": 0, "end_turn": 0, "orphan_close": 0}

    def test_reports_per_pattern_counts(self):
        """Counts are reported separately per pattern."""
        content = (
            "<think>analysis</think>\n"
            "<execute>x = 1</execute>\n"
            '<invoke name="x">a</invoke>\n'
            '<invoke name="y">b</invoke>\n'
            "<end_turn>\n<end_turn>\n<end_turn>\n"
            + "</execute>\n" * 10
        )
        cleaned, counts = _strip_runaway_tokens(content)
        assert counts["invoke"] == 2
        assert counts["end_turn"] == 3
        assert counts["orphan_close"] == 10
        # Legitimate content survives
        assert "<think>analysis</think>" in cleaned
        assert "<execute>x = 1</execute>" in cleaned

    def test_legitimate_complex_response_unchanged(self):
        """A long, complex but legitimate response — with balanced execute,
        thinking, and solution blocks — is NOT clipped by any stripper."""
        content = (
            "<think>\n"
            "Multi-paragraph analysis of the data.\n"
            "Testing hypothesis H1 with 3 sub-analyses.\n"
            "</think>\n"
            "<execute>\n"
            "# Load data\nimport pandas as pd\ndf = pd.read_csv('peaks.bed', sep='\\t')\n"
            "# Run matched control\nfg_m, bg_m = matched_pair(fg, bg, 'signal')\n"
            "print(fg_m.shape, bg_m.shape)\n"
            "</execute>\n"
            "<solution>\n"
            "support_level: SUPPORTS\nconfidence: 0.85\n"
            "finding: The analysis confirms the predicted direction.\n"
            "reasoning: 1.77x signal increase at motif+ peaks (p=4e-20), "
            "controlling for DNase + GC.\n"
            "</solution>"
        )
        cleaned, counts = _strip_runaway_tokens(content)
        assert cleaned == content
        assert sum(counts.values()) == 0

    def test_exact_elk3_invoke_shape(self):
        """Regression: the exact Apr 14 ELK3 failure shape is cleaned."""
        junk = (
            '<invoke name="lookup_tpm_by_symbol">\n'
            '<parameter name="symbol">GAPDH</parameter>\n'
            '</invoke>\n'
        )
        content = (
            "<think>proceeding</think>\n"
            "<execute>df = read_narrowpeak(f)\nprint(df.shape)</execute>\n"
            + junk * 150
        )
        cleaned, counts = _strip_runaway_tokens(content)
        assert counts["invoke"] == 150
        assert "<invoke" not in cleaned
        assert "read_narrowpeak(f)" in cleaned

    def test_exact_znf445_end_turn_shape(self):
        """Regression: the exact Apr 15 ZNF445 failure shape is cleaned."""
        content = (
            "<think>Final verification.</think>\n"
            "<execute>print('ok')</execute>\n"
            + "<end_turn>\n" * 4700
        )
        cleaned, counts = _strip_runaway_tokens(content)
        assert counts["end_turn"] == 4700
        assert "<end_turn" not in cleaned
        assert "print('ok')" in cleaned

    def test_exact_pttg1_orphan_close_shape(self):
        """Regression: the exact Apr 15 PTTG1 failure shape is cleaned."""
        content = (
            "<execute>print('done')</execute>\n"
            + "</execute>\n" * 15000
        )
        cleaned, counts = _strip_runaway_tokens(content)
        assert counts["orphan_close"] == 15000
        # Legitimate balanced pair survives
        assert "<execute>print('done')</execute>" in cleaned
        assert cleaned.count("</execute>") == 1


