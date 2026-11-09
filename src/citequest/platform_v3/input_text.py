"""Lossless validation of free-form request text before resource admission."""


def request_text(value: str, *, allow_empty: bool = False) -> str:
    if value == "" and allow_empty:
        return value
    if any((ord(char) < 32 and char not in "\t\n\r")
           or 0x7F <= ord(char) <= 0x9F or 0xD800 <= ord(char) <= 0xDFFF
           for char in value):
        raise ValueError("request text contains binary controls or invalid Unicode")
    if not any(char.isprintable() and not char.isspace() for char in value):
        raise ValueError("request text must contain visible text")
    return value
