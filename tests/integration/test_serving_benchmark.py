import json

from riskforge.core.contracts import LifeSavingRule
from riskforge.serving.artifact import sha256_file
from riskforge.serving.benchmark import benchmark_artifact
from tests.integration.test_onnx_runtime import _write_dynamic_test_model


def test_benchmark_tool_runs_against_real_onnx_runtime(tmp_path) -> None:
    model_path = tmp_path / "benchmark.onnx"
    manifest_path = tmp_path / "manifest.json"
    _write_dynamic_test_model(model_path)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_sha256": sha256_file(model_path),
                "backbone": "riskforge-benchmark-test",
                "max_sequence_length": 8,
                "quantization": "NONE",
                "temperature": 1.0,
                "input_names": ["input_ids", "attention_mask"],
                "output_names": ["sif_logits", "iogp_logits"],
                "iogp_rule_order": [rule.value for rule in LifeSavingRule],
            }
        ),
        encoding="utf-8",
    )

    report = benchmark_artifact(
        model_path,
        manifest_path,
        iterations=1,
        concurrent_iterations=1,
        concurrency=2,
        sequence_length=8,
        max_queue_delay_ms=1.0,
    )

    assert [item["batch_size"] for item in report["warm_batches"]] == [1, 8, 16, 32]
    assert report["concurrent"]["request_count"] == 2
    assert report["cold"]["load_and_first_inference_ms"] >= 0.0
    assert report["target_latency_ms"] == 35.0
    assert report["process_memory"]["peak_rss_bytes"] > 0
    assert report["artifact"]["quantization"] == "NONE"
    assert report["benchmark_parameters"]["sequence_length"] == 8
    assert report["environment"]["onnxruntime_version"]
    assert report["generated_at"]
    assert report["measurement_scope"] == "pretokenized_onnx_execution_and_postprocessing"
    assert report["acceptance"]["int8_artifact_passed"] is False
    assert report["acceptance"]["passed"] is False
