"""
Quickstart Example
==================

This example imports :mod:`rate` and creates a compact plot summarizing the number of
public API symbols exported by the package.
"""

import matplotlib.pyplot as plt

import rate

public_symbols = sorted(rate.__all__)
counts = [1] * len(public_symbols)

fig, ax = plt.subplots(figsize=(6, 2.75))
ax.bar(range(len(public_symbols)), counts, color="#4C72B0")
ax.set_title("rate public API surface")
ax.set_xlabel("Public symbol index")
ax.set_ylabel("Count")
ax.set_xticks([])
ax.set_ylim(0, 1.2)
ax.text(0.02, 0.92, f"{len(public_symbols)} public symbols", transform=ax.transAxes, va="top")
fig.tight_layout()

