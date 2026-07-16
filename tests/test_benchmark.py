from __future__ import annotations

from hermes_plugins.nemotron_embed_test.benchmark import run_compaction_benchmark
from hermes_plugins.nemotron_embed_test.settings import NemotronSettings


class _BenchmarkClient:
    def embed(self, input_type, texts):
        if input_type == "query":
            return [[1.0, 0.0] if "region" in text.casefold() else [0.0, 1.0] for text in texts]
        return [[1.0, 0.0] if "us-central1" in text else [0.0, 1.0] for text in texts]


def test_compaction_benchmark_proves_reduction_retention_and_recovery(tmp_path):
    settings = NemotronSettings.from_mapping(
        {
            "protect_first_turns": 1,
            "protect_last_turns": 4,
            "target_ratio": 0.35,
        },
        hermes_home=tmp_path,
    )

    result = run_compaction_benchmark(settings, client=_BenchmarkClient())

    assert result["passed"] is True
    assert result["token_reduction_ratio"] > 0
    assert result["output_turns"] < result["input_turns"]
    assert result["relevant_turn_retained"] is True
    assert result["archive_recovery_succeeded"] is True
    assert result["atomic_turns_valid"] is True
