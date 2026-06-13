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
    assert "lmeeeg-subject" in commands
    assert "lmeeeg-group-stats" in commands
    assert "lmeeeg-nested-group-stats" in commands
    assert "lmeeeg-similarity" in commands
    assert "lmeeeg-variance-partition-fit" in commands
    assert "lmeeeg-variance-partition-group-stats" in commands
    assert "figures-behavioural" in commands
    assert "figures-glm" in commands
    assert "figures-lmeeeg" in commands
    assert "figures-lmeeeg-inference" in commands
    assert "figures-lmeeeg-main-results" in commands
    assert "figures-lmeeeg-variance-partition" in commands
    assert "figures-similarity" in commands
    assert "model-weights-glm" in commands
