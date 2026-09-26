"""Reference (toy) plugins for the synthetic vertical slice. Registered via entry points."""

from .descriptors import ALL, register_all

__all__ = ["ALL", "register_all"]
