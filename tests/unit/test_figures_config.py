from rate.cli.commands.figures_glm import _normalize_regressors


def test_normalize_regressors_canonicalizes_interactions() -> None:
    regressors = [
        "speech_rate",
        "speech_rate * speaker_role",
    ]

    assert _normalize_regressors(regressors) == [
        "speech_rate",
        "speech_rate__x__speaker_role",
    ]
