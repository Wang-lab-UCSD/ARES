# Experiment Design Generation Pipeline

## Project Context
Multi-agent pipeline for automated hypothesis generation and verification in bioinformatics.
User provides a finding (X predicts Y) + data paths, pipeline runs until convergence.

## Architecture
- `src/orchestrator.py` - Main loop coordinator
- `src/llm/` - LLM provider abstraction (OpenAI, Claude, Gemini)
- `src/agents/` - Hypothesis, Coding, Summary agents
- `src/execution/` - Jupyter kernel for code execution
- `src/memory/` - Conversation history management
- `src/prompts/` - Prompt templates

## Code Style
- Python 3.10+
- Type hints required (use `from __future__ import annotations`)
- Async/await for LLM calls
- Pydantic for config and data models
- Keep functions focused and small

## Running the Pipeline
```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Run pipeline
python -m src.main --config config/config.yaml --manifest examples/atf6_rest/data_manifest.yaml
```

## Testing
```bash
pytest tests/ -v
```

## Key Patterns
- All LLM calls go through `src/llm/base.py` interface
- Conversation memory stored in `src/memory/conversation.py`
- Structured logging to `logs/` directory
- JSON schema validation for LLM outputs
- Graceful shutdown on SIGINT (Ctrl+C)

## File Naming
- Use snake_case for all Python files
- Use descriptive names that indicate purpose
