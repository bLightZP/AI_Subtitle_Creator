from ai_subtitle_creator.settings_store import GuiSettings, load_gui_settings, save_gui_settings


def test_settings_round_trip(tmp_path) -> None:
    settings_path = tmp_path / "settings.json"
    expected = GuiSettings(
        window_geometry="1200x720+30+40",
        default_model="small",
        selected_model="medium",
        device="cpu",
        compute_type="int8",
        language="en",
        task="translate",
        priority="Normal",
        cpu_threads="4",
        model_cache=r"C:\Models",
    )

    saved_path = save_gui_settings(expected, settings_path)
    loaded = load_gui_settings(saved_path)

    assert loaded == expected


def test_load_gui_settings_ignores_invalid_payload(tmp_path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"default_model": 123, "device": "cpu", "extra": "value"}', encoding="utf-8")

    loaded = load_gui_settings(settings_path)

    assert loaded.default_model is None
    assert loaded.device == "cpu"
    assert loaded.compute_type == "int8"
