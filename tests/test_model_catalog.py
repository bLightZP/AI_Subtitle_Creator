from ai_subtitle_creator.model_catalog import available_model_names, describe_model


def test_available_model_names_contains_common_models() -> None:
    names = available_model_names()

    assert "tiny" in names
    assert "small" in names


def test_describe_model_returns_text() -> None:
    assert describe_model("tiny")
    assert describe_model("custom/model")

