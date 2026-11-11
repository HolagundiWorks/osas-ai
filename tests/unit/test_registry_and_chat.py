from myna_ai.common.config import load_yaml, repo_root
from myna_ai.models.chat import render_olmo3_chat
from myna_ai.models.registry import load_registry


def test_training_configs_parse() -> None:
    root = repo_root()
    smoke = load_yaml(root / "configs" / "training" / "smoke_tiny_lora.yaml")
    qlora = load_yaml(root / "configs" / "training" / "olmo3_7b_qlora.yaml")
    assert smoke["model_key"] == "smoke-tiny"
    assert smoke["allow_cpu_smoke"] is True
    assert smoke["precision"] == "fp32"
    assert qlora["method"] == "qlora"
    assert qlora["allow_cpu_smoke"] is False


def test_registry_defaults_to_verified_olmo() -> None:
    registry = load_registry()
    spec = registry.get(registry.default_model)
    assert spec.hf_id == "allenai/Olmo-3-7B-Instruct"
    assert spec.transformers_class == "Olmo3ForCausalLM"
    assert spec.licence == "apache-2.0"
    assert spec.licence_verified is True
    assert spec.eos_token == "<|endoftext|>"
    assert "q_proj" in spec.lora_target_modules
    assert "down_proj" in spec.lora_target_modules
    assert registry.get("smoke-tiny").allow_cpu_smoke is True
    assert registry.get("qwen2.5-7b-instruct").licence_verified is False


def test_olmo_template_matches_jinja_default_system() -> None:
    rendered = render_olmo3_chat(
        [{"role": "user", "content": "Who would win?"}],
        add_generation_prompt=True,
    )
    assert rendered.text == (
        "<|im_start|>system\n"
        "You are a helpful function-calling AI assistant. "
        "You do not currently have access to any functions. <|im_end|>\n"
        "<|im_start|>user\n"
        "Who would win?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert "<functions></functions>" not in rendered.text


def test_assistant_span_covers_function_call_and_eos() -> None:
    messages = [
        {"role": "system", "content": "Stay concise."},
        {"role": "user", "content": "Select the grid."},
        {
            "role": "assistant",
            "content": "Requesting selection.",
            "tool_calls": [
                {"function": {"name": "cad.select_entities", "arguments": {"project_id": "p"}}}
            ],
        },
    ]
    rendered = render_olmo3_chat(messages, tools=[{"name": "cad.select_entities"}])
    start, end = rendered.assistant_spans[0]
    body = rendered.text[start:end]
    assert body.startswith("Requesting selection.")
    assert "<function_calls>cad.select_entities(project_id=\"p\")</function_calls>" in body
    assert body.endswith("<|endoftext|>")
    assert "Select the grid." not in body
