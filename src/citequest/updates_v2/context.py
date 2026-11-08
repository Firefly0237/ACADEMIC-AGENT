"""Trusted updater-only context, extending the unchanged S04 HTTP deadline."""

from dataclasses import dataclass

from citequest.materials_v2.reference_fetch import FetchContext


@dataclass
class WindowFetchContext(FetchContext):
    source_window_id: str | None = None
    source_sequence: int | None = None
