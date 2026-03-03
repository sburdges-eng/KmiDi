from scripts.sync_entities import _json_to_ts, _model_schema, _render_typescript


def test_json_to_ts_maps_nullable_anyof_to_union():
    node = {
        "anyOf": [
            {"type": "string"},
            {"type": "null"},
        ]
    }

    assert _json_to_ts("rule_to_break", node, {}) == "string | null"


def test_render_typescript_keeps_nullable_optional_string_fields():
    rendered = _render_typescript(_model_schema())

    assert "rule_to_break?: string | null;" in rendered
    assert "rule_justification?: string | null;" in rendered
