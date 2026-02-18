rule tests:
    shell:
        "PYTHONPATH=src pytest -q"
