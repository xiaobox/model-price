"""OpenRouter provider parser regressions."""

from datetime import UTC, datetime

from providers.openrouter import OpenRouterProvider


def _payload(
    model_id: str,
    name: str,
    input_modalities: list[str] | None = None,
    output_modalities: list[str] | None = None,
) -> dict:
    return {
        "id": model_id,
        "name": name,
        "pricing": {
            "prompt": "0.000008",
            "completion": "0.000015",
        },
        "input_modalities": input_modalities or [],
        "output_modalities": output_modalities or [],
        "context_length": 272000,
        "top_provider": {"max_completion_tokens": 128000},
    }


def test_gpt_image_family_forces_image_generation_output():
    provider = OpenRouterProvider()

    model = provider._parse_model(
        _payload(
            "openai/gpt-5.4-image-2",
            "GPT 5.4 Image 2",
            input_modalities=["text", "image"],
            output_modalities=["text"],
        ),
        datetime.now(UTC),
    )

    assert model is not None
    assert model.capabilities == ["image_generation", "vision"]
    assert "text" not in model.capabilities
    assert "reasoning" not in model.capabilities
    assert "tool_use" not in model.capabilities
    assert model.input_modalities == ["text", "image"]
    assert model.output_modalities == ["image"]


def test_gpt_5_chat_keeps_chat_capabilities():
    provider = OpenRouterProvider()

    model = provider._parse_model(
        _payload(
            "openai/gpt-5-chat-latest",
            "GPT 5 Chat Latest",
            input_modalities=["text", "image"],
            output_modalities=["text"],
        ),
        datetime.now(UTC),
    )

    assert model is not None
    assert "image_generation" not in model.capabilities
    assert {"text", "vision", "reasoning", "tool_use"}.issubset(model.capabilities)
    assert model.output_modalities == ["text"]
