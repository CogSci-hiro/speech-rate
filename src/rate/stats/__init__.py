from .config import ClusterMethod, StatsConfig

__all__ = [
    "ClusterMethod",
    "StatsConfig",
    "TwoLevelClusterResult",
    "run_two_level_kernel0",
]


def __getattr__(name: str):
    if name in {"TwoLevelClusterResult", "run_two_level_kernel0"}:
        from .two_level import TwoLevelClusterResult, run_two_level_kernel0

        exports = {
            "TwoLevelClusterResult": TwoLevelClusterResult,
            "run_two_level_kernel0": run_two_level_kernel0,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
