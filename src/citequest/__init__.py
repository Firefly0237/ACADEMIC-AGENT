"""CiteQuest academic search and research workstation.

Current service entrypoints are web_v3.main and agents_v3.worker. Capability
packages ending in _v3 share the retained v2 storage, material and index
contracts; the suffix alone does not make a module obsolete. Package exports
remain lazy so each role can import only its own installed dependencies.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citequest.config.settings import Settings, get_settings

__all__ = ["Settings", "__version__", "get_settings"]

__version__ = "0.1.0"


def __getattr__(name: str) -> object:
    """PEP 562 lazy compatibility exports keep unrelated role imports independent."""
    if name in {"Settings", "get_settings"}:
        from citequest.config.settings import Settings, get_settings

        exports = {"Settings": Settings, "get_settings": get_settings}
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
