from ai_subtitle_creator.model_catalog import (
    import_model_to_cache,
    imported_model_names,
    local_model_names,
    resolve_model_reference,
    available_model_names,
    describe_model,
)


def test_available_model_names_contains_common_models() -> None:
    names = available_model_names()

    assert "tiny" in names
    assert "small" in names


def test_describe_model_returns_text() -> None:
    assert describe_model("tiny")
    assert describe_model("custom/model")


def test_import_model_file_copies_into_cache(tmp_path) -> None:
    source = tmp_path / "source" / "custom.bin"
    source.parent.mkdir()
    source.write_bytes(b"model")

    imported = import_model_to_cache(source, tmp_path / "cache")

    assert imported == tmp_path / "cache" / "imported" / "custom"
    assert (imported / "custom.bin").read_bytes() == b"model"
    assert imported_model_names(tmp_path / "cache") == ["local:custom"]
    assert resolve_model_reference("local:custom", tmp_path / "cache") == str(imported)


def test_import_model_marker_file_copies_parent_folder(tmp_path) -> None:
    source_dir = tmp_path / "ct2-model"
    source_dir.mkdir()
    (source_dir / "model.bin").write_bytes(b"weights")
    (source_dir / "config.json").write_text("{}", encoding="utf-8")

    imported = import_model_to_cache(source_dir / "model.bin", tmp_path / "cache")

    assert imported == tmp_path / "cache" / "imported" / "ct2-model"
    assert (imported / "model.bin").read_bytes() == b"weights"
    assert (imported / "config.json").read_text(encoding="utf-8") == "{}"


def test_local_model_names_returns_downloaded_and_imported(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "ai_subtitle_creator.model_catalog.is_model_downloaded",
        lambda model_name, _cache_dir: model_name in {"small", "medium"},
    )
    monkeypatch.setattr(
        "ai_subtitle_creator.model_catalog.imported_model_names",
        lambda _cache_dir: ["local:custom"],
    )

    names = local_model_names(tmp_path, ["tiny", "small", "medium"])

    assert names == ["small", "medium", "local:custom"]

