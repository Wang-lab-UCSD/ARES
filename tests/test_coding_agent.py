"""Tests for the coding agent."""

import pytest

from src.agents.coding_agent import CodingAgent


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
        """Test extracting from multiple code blocks."""
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
        assert "df = pd.read_csv" in code


class TestCodeValidation:
    """Tests for code validation."""

    @pytest.fixture
    def agent(self):
        """Create a coding agent with mock LLM."""
        class MockLLM:
            pass

        return CodingAgent(MockLLM())

    def test_valid_code(self, agent):
        """Test validation of valid code."""
        code = """
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
"""
        is_valid, error = agent.validate_code(code)
        assert is_valid
        assert error is None

    def test_empty_code(self, agent):
        """Test validation of empty code."""
        is_valid, error = agent.validate_code("")
        assert not is_valid
        assert "Empty code" in error

    def test_syntax_error(self, agent):
        """Test validation of code with syntax error."""
        code = "def foo(\n  print('hi')"  # Missing closing paren
        is_valid, error = agent.validate_code(code)

        assert not is_valid
        assert "Syntax error" in error

    def test_valid_subprocess_call(self, agent):
        """Test that subprocess calls are allowed."""
        code = """
import subprocess
subprocess.run(['bedtools', 'intersect', '-a', 'a.bed', '-b', 'b.bed'])
"""
        is_valid, error = agent.validate_code(code)
        assert is_valid
