import json
from datetime import UTC, datetime

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from riskforge.core.contracts import AssetType, IncidentNormalizedRecord, LifeSavingRule
from riskforge.serving.artifact import sha256_file
from riskforge.serving.engine import ONNXInferenceEngine


def _write_dynamic_test_model(path) -> None:
    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, ["batch", "sequence"]
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", "sequence"]
    )
    sif_logits = helper.make_tensor_value_info(
        "sif_logits", TensorProto.FLOAT, ["batch", 1]
    )
    iogp_logits = helper.make_tensor_value_info(
        "iogp_logits", TensorProto.FLOAT, ["batch", 9]
    )

    nodes = [
        helper.make_node("Cast", ["input_ids"], ["float_ids"], to=TensorProto.FLOAT),
        helper.make_node(
            "ReduceMean", ["float_ids"], ["sif_logits"], axes=[1], keepdims=1
        ),
        helper.make_node("Shape", ["attention_mask"], ["input_shape"]),
        helper.make_node(
            "Constant",
            [],
            ["batch_index"],
            value=helper.make_tensor("batch_index_value", TensorProto.INT64, [1], [0]),
        ),
        helper.make_node("Gather", ["input_shape", "batch_index"], ["batch_size"], axis=0),
        helper.make_node(
            "Constant",
            [],
            ["rule_count"],
            value=helper.make_tensor("rule_count_value", TensorProto.INT64, [1], [9]),
        ),
        helper.make_node("Concat", ["batch_size", "rule_count"], ["rule_shape"], axis=0),
        helper.make_node(
            "ConstantOfShape",
            ["rule_shape"],
            ["iogp_logits"],
            value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0]),
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "riskforge-serving-integration",
        [input_ids, attention_mask],
        [sif_logits, iogp_logits],
    )
    model = helper.make_model(
        graph,
        producer_name="riskforge-tests",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _record(index: int) -> IncidentNormalizedRecord:
    return IncidentNormalizedRecord(
        log_id=f"ORT_{index:03d}",
        timestamp=datetime.now(UTC),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Routine inspection completed.",
        spans=[],
    )


def test_real_onnx_runtime_cpu_session_and_dynamic_batching(tmp_path) -> None:
    model_path = tmp_path / "dynamic_multitask.onnx"
    manifest_path = tmp_path / "manifest.json"
    _write_dynamic_test_model(model_path)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_sha256": sha256_file(model_path),
                "backbone": "riskforge-test-model",
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

    engine = ONNXInferenceEngine.from_artifact(
        model_path, manifest_path, max_batch_size=2
    )

    options = engine.session.get_session_options()
    assert options.intra_op_num_threads == 4
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert engine.session.get_providers() == ["CPUExecutionProvider"]

    records = [_record(index) for index in range(3)]
    input_ids = np.zeros((3, 8), dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    results = engine.infer_batch(records, input_ids, attention_mask)

    assert [result.log_id for result in results] == [record.log_id for record in records]
    assert all(result.raw_sif_p_score == 0.5 for result in results)
    assert all(result.latency_ms >= 0.0 for result in results)
