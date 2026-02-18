# Project Snapshot

- Generated: `2026-02-18T23:33:34+01:00`

- Root: `/Users/hiro/PycharmProjects/speech-rate`

- Config: `MISSING`

## Environment

```json
{
  "python": "3.11.8 (v3.11.8:db85d51d3e, Feb  6 2024, 18:02:37) [Clang 13.0.0 (clang-1300.0.29.30)]",
  "executable": "/Users/hiro/PycharmProjects/speech-rate/.venv/bin/python",
  "platform": "macOS-14.4.1-arm64-arm-64bit",
  "cwd": "/Users/hiro/PycharmProjects/speech-rate"
}
```

## Git

- HEAD: `d380878450aba1aff5d01cc72234170d208107ef`

- Dirty files: `26`

```text
M Snakefile
 M cli/main.py
 M config/config.yaml
 M config/schema.json
 M src/rate/__init__.py
A  src/rate/__pycache__/__init__.cpython-311.pyc
A  src/rate/__pycache__/config.cpython-311.pyc
A  src/rate/__pycache__/manifest.cpython-311.pyc
A  src/rate/__pycache__/util.cpython-311.pyc
 M src/rate/config.py
 M src/rate/dataset.py
 M src/rate/util.py
A  tests/unit/__pycache__/test_config_and_manifest.cpython-311-pytest-9.0.2.pyc
?? .snakemake/
?? cli/__pycache__/
?? cli/commands/__pycache__/
?? cli/commands/dataset_index.py
?? cli/commands/events_table.py
?? pyproject.toml
?? src/rate/__pycache__/dataset.cpython-311.pyc
?? src/rate/events.py
?? src/speech_rate.egg-info/
?? tests/unit/__pycache__/test_dataset_index.cpython-311-pytest-9.0.2.pyc
?? tests/unit/__pycache__/test_events_table.cpython-311-pytest-9.0.2.pyc
?? tests/unit/test_events_table.py
?? workflow/rules/events.smk
```

## Snakemake Rules

- `dataset_index`
- `events_table`
- `tests`
- `validate_config`
- `write_manifest`

## Key Config Excerpts

_Config not found or invalid._

## Repository File Summary

Top 43 files by size (excluding .git/.venv/__pycache__/.snakemake):

| file | size |
|---|---:|
| `src/rate/dataset.py` | 14.8 KB |
| `src/rate/events.py` | 13.3 KB |
| `project_info.py` | 11.7 KB |
| `dev/FOOOF_plan.md` | 9.6 KB |
| `dev/DDC_plan.md` | 8.7 KB |
| `src/rate/util.py` | 5.8 KB |
| `dev/project_specs.md` | 5.4 KB |
| `dev/induced_TRF_plan.md` | 5.1 KB |
| `src/rate/config.py` | 4.9 KB |
| `config/schema.json` | 4.4 KB |
| `tests/unit/test_events_table.py` | 4.1 KB |
| `dev/TRF_plan.md` | 3.9 KB |
| `tests/unit/test_dataset_index.py` | 3.5 KB |
| `tests/unit/test_config_and_manifest.py` | 2.5 KB |
| `src/rate/manifest.py` | 2.5 KB |
| `cli/commands/events_table.py` | 1.8 KB |
| `cli/commands/validate_config.py` | 1.6 KB |
| `cli/commands/dataset_index.py` | 1.4 KB |
| `cli/commands/write_manifest.py` | 1.3 KB |
| `cli/main.py` | 1.1 KB |
| `workflow/rules/common.smk` | 822.0 B |
| `workflow/rules/events.smk` | 797.0 B |
| `cli/commands/run_tests.py` | 748.0 B |
| `.pytest_cache/v/cache/nodeids` | 701.0 B |
| `src/rate/__init__.py` | 591.0 B |
| `pyproject.toml` | 577.0 B |
| `workflow/rules/dataset.smk` | 544.0 B |
| `config/config.yaml` | 498.0 B |
| `src/speech_rate.egg-info/PKG-INFO` | 398.0 B |
| `src/speech_rate.egg-info/SOURCES.txt` | 306.0 B |
| `.pytest_cache/README.md` | 302.0 B |
| `.pytest_cache/CACHEDIR.TAG` | 191.0 B |
| `Snakefile` | 182.0 B |
| `src/speech_rate.egg-info/requires.txt` | 100.0 B |
| `pytest.ini` | 68.0 B |
| `workflow/rules/tests.smk` | 58.0 B |
| `.pytest_cache/v/cache/lastfailed` | 45.0 B |
| `.pytest_cache/.gitignore` | 37.0 B |
| `cli/__init__.py` | 36.0 B |
| `cli/commands/__init__.py` | 28.0 B |
| `.gitignore` | 10.0 B |
| `src/speech_rate.egg-info/top_level.txt` | 5.0 B |
| `src/speech_rate.egg-info/dependency_links.txt` | 1.0 B |

## Output Artifact Inventory

_Could not determine out_dir from config (expected io.out_dir or paths.out_dir)._


## pip freeze

```text
attrs==25.4.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
numpy==2.4.2
pandas==3.0.1
python-dateutil==2.9.0.post0
PyYAML==6.0.3
referencing==0.37.0
rpds-py==0.30.0
six==1.17.0
speech-rate==0.1.0
typing_extensions==4.15.0
```