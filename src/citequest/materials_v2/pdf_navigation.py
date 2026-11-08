"""Inert current-document OpenAction destinations, never an action executor.

ISO 32000-1:2008 12.3.2.2/Table151 defines XYZ left/top/zoom. Adobe's
official Actions and Destinations guide confirms null retains the current
parameter and zero zoom has null semantics. We admit a deliberately bounded
subset: local page reference + exact Fit/FitB or five-element XYZ, optionally
wrapped in a GoTo dictionary with no action chain or remote/file attributes.
https://developer.adobe.com/acrobat-pdfl-sdk/guides/pdfmark/pdfmark-actions
"""

from __future__ import annotations

import math
from typing import Any


def safe_local_open_destination(value: Any, reader: Any, page_refs: set[tuple[int, int]]) -> bool:
    from pypdf.generic import (
        ArrayObject,
        DictionaryObject,
        FloatObject,
        IndirectObject,
        NameObject,
        NullObject,
        NumberObject,
    )

    if isinstance(value, IndirectObject):
        if value.pdf is not reader:
            return False
        value = value.get_object()
    if isinstance(value, DictionaryObject):
        if (
            not {"/S", "/D"} <= set(value) <= {"/S", "/D", "/Type"}
            or not isinstance(value["/S"], NameObject)
            or str(value["/S"]) != "/GoTo"
        ):
            return False
        if "/Type" in value and (not isinstance(value["/Type"], NameObject) or str(value["/Type"]) != "/Action"):
            return False
        value = value["/D"]
    if not isinstance(value, ArrayObject) or len(value) not in {2, 5}:
        return False
    if not isinstance(value[1], NameObject):
        return False
    page = value[0]
    if not isinstance(page, IndirectObject) or page.pdf is not reader or (page.idnum, page.generation) not in page_refs:
        return False
    if len(value) == 2:
        return str(value[1]) in {"/Fit", "/FitB"}
    if str(value[1]) != "/XYZ":
        return False
    for index, parameter in enumerate(value[2:]):
        if isinstance(parameter, NullObject):
            continue
        # Explicit types exclude booleans, strings, names, references, streams
        # and executable/action dictionaries. No parameter object is followed.
        if not isinstance(parameter, (NumberObject, FloatObject)):
            return False
        try:
            number = float(parameter)
        except (ValueError, OverflowError):
            return False
        if not math.isfinite(number) or (index == 2 and number < 0):
            return False
    return True
