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


