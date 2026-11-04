import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import tempfile
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, mock_open


@pytest.mark.parametrize("blocked", [False, True])
def test_profiler_preserves_claude_analysis(blocked):
    from agents import profiler

    analysis = {"semantic_meanings": {"id": "identifier"}, "join_keys": ["id"]}
    text = json.dumps(analysis)
    content = [{"type": "text", "text": text}] if blocked else text
    result = {"messages": [
        MagicMock(content=json.dumps({"files": [], "datasets": {}})),
        MagicMock(content=content),
    ]}
    with patch.object(profiler, "_make_llm"), \
         patch.object(profiler, "create_agent") as agent_factory, \
         patch.object(profiler, "_make_profiler_tools", return_value=(MagicMock(), MagicMock())), \
         patch.object(profiler, "AuditLogger"), \
         patch.object(profiler, "AgentTrace"), \
         patch.object(profiler.pd, "Timestamp"), \
         patch("builtins.open", mock_open()), \
         patch.object(profiler.json, "dump") as dump:
        agent_factory.return_value.invoke.return_value = result
        profiler.profile_multiple_datasets([], "test-run", "Profile data")

    assert dump.call_args.args[0]["analysis"] == analysis


class TestProfiler:
    def test_profile_dataset_creates_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.profiler.PROFILES_DIR", tmp_path)

        # Create a test CSV
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [10.0, 20.0, None]})
        df.to_csv(csv_path, index=False)

        # Mock LLM call
        mock_response = MagicMock()
        mock_response.content = '{"id": "unique identifier", "name": "entity name", "value": "numeric measurement"}'

        with patch("agents.profiler._make_llm") as mock_llm_factory:
            mock_llm_factory.return_value.invoke.return_value = mock_response

            from agents.profiler import profile_dataset
            result = profile_dataset(str(csv_path), "test-run")

        assert Path(result).exists()
        with open(result) as f:
            profile = json.load(f)
        assert profile["shape"]["rows"] == 3
        assert profile["shape"]["columns"] == 3
        assert "id" in profile["columns"]

    def test_profile_multiple_datasets(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.profiler.PROFILES_DIR", tmp_path)

        # Create test CSVs
        csv1 = tmp_path / "sales.csv"
        csv2 = tmp_path / "products.csv"
        pd.DataFrame({"product_id": [1, 2], "revenue": [100, 200]}).to_csv(csv1, index=False)
        pd.DataFrame({"product_id": [1, 2], "name": ["A", "B"]}).to_csv(csv2, index=False)

        mock_response = MagicMock()
        mock_response.content = '```json\n{"semantic_meanings": {}, "join_keys": ["product_id"], "quality_notes": "ok"}\n```'

        with patch("agents.profiler._make_llm") as mock_llm_factory:
            mock_llm_factory.return_value.invoke.return_value = mock_response

            from agents.profiler import profile_multiple_datasets
            result = profile_multiple_datasets([str(csv1), str(csv2)], "test-run")

        assert Path(result).exists()
        with open(result) as f:
            profile = json.load(f)
        assert len(profile["datasets"]) == 2
