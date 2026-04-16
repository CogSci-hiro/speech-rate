from rate.viz.theme import get_role_palette, resolve_theme_config


def test_default_theme_uses_semantic_paper_palette() -> None:
    theme = resolve_theme_config()

    assert theme["name"] == "paper"
    assert theme["colors"]["self"] == "#8C2D2D"
    assert theme["colors"]["other"] == "#1F4E79"
    assert theme["colors"]["shared"] == "#2F6B3B"


def test_role_palette_allows_overrides() -> None:
    palette = get_role_palette(
        {
            "colors": {
                "self": "#6E1F1F",
                "other": "#20496B",
                "shared": "#2C5E35",
            }
        }
    )

    assert palette == {
        "self": "#6E1F1F",
        "other": "#20496B",
        "shared": "#2C5E35",
    }
