from rate.cli.main import build_arg_parser


def test_build_arg_parser_registers_mass_univariate_commands() -> None:
    parser = build_arg_parser()
    subparsers_action = next(action for action in parser._actions if getattr(action, "choices", None))
    commands = set(subparsers_action.choices)

    assert "dataset-index" in commands
    assert "build-events" in commands
    assert "preprocess-subject" in commands
    assert "epoch-subject" in commands
    assert "glm-subject" in commands
    assert "glm-group-stats" in commands
    assert "fooof-mass-subject" in commands
    assert "fooof-mass-group-stats" in commands
    assert "lmeeeg-subject" in commands
    assert "lmeeeg-prepare-trials" in commands
    assert "lmeeeg-group-stats" in commands
    assert "figures-glm" in commands
    assert "figures-fooof" in commands
