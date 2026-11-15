"""PII de-identification: detect spans, swap them for placeholders, restore them later."""

import re
from typing import Any, Protocol

LABELS = [
    "person", "email", "phone number", "bank account number", "credit card number",
    "medical record number", "patient name", "medical condition", "address",
    "date of birth", "passport number", "national identification number",
]
TOKEN = re.compile(r"\[([A-Z_]+_\d+)\]")
MAX_TOKEN_LEN = 48  # longest tail we hold back while waiting for a closing "]"


class Detector(Protocol):
    def detect(self, text: str) -> list[dict[str, Any]]: ...


class GlinerDetector:
    def __init__(self, model_name: str, threshold: float) -> None:
        from gliner import GLiNER  # optional dependency: pip install -e '.[pii]'

        self.model, self.threshold = GLiNER.from_pretrained(model_name), threshold

    def detect(self, text: str) -> list[dict[str, Any]]:
        return self.model.predict_entities(text, LABELS, threshold=self.threshold)


class Vault:
    """Per-request mapping of placeholder <-> original value."""

    def __init__(self, detector: Detector) -> None:
        self.detector = detector
        self.tokens: dict[str, str] = {}  # placeholder -> original
        self._by_value: dict[tuple[str, str], str] = {}
        self._pending = ""  # streamed text that may hold the start of a placeholder

    def redact(self, text: str) -> str:
        for ent in sorted(self.detector.detect(text), key=lambda e: e["start"], reverse=True):
            key = (ent["label"], ent["text"])
            if key not in self._by_value:
                n = sum(1 for k in self._by_value if k[0] == ent["label"]) + 1
                self._by_value[key] = f"[{ent['label'].upper().replace(' ', '_')}_{n}]"
                self.tokens[self._by_value[key][1:-1]] = ent["text"]
            text = text[: ent["start"]] + self._by_value[key] + text[ent["end"] :]
        return text

    def restore(self, text: str) -> str:
        return TOKEN.sub(lambda m: self.tokens.get(m.group(1), m.group(0)), text)

    def feed(self, chunk: str) -> str:
        """Restore streamed text, holding back a possibly-split placeholder like `[PERS`."""
        text, self._pending = self._pending + chunk, ""
        cut = text.rfind("[")
        if cut != -1 and "]" not in text[cut:] and len(text) - cut < MAX_TOKEN_LEN:
            text, self._pending = text[:cut], text[cut:]
        return self.restore(text)

    def flush(self) -> str:
        text, self._pending = self._pending, ""
        return text
