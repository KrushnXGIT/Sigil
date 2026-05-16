"""Action executor.

Dispatches ``ActionDispatch`` events into OS-level actions via a verb
registry. Exception-isolated per verb so a broken verb doesn't tear
down the daemon.

See ADR-0006 for the design rationale.
"""

from sigil.executor.dispatcher import Dispatcher
from sigil.executor.registry import DEFAULT_REGISTRY, Verb, make_registry

__all__ = ["DEFAULT_REGISTRY", "Dispatcher", "Verb", "make_registry"]
