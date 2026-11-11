from myna_ai.common.logging import redact


def test_redact_hides_token(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_thisisatesttoken")
    text = redact("using hf_thisisatesttoken for the hub")
    assert "hf_thisisatesttoken" not in text
    assert "HF_TOKEN" in text or "REDACTED" in text
