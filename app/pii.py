"""PII de-identification: detect spans, swap them for placeholders, restore them later."""

import re
from typing import Any, Protocol

LABELS = [
    "person", "email", "phone number", "bank account number", "credit card number",
    "medical record number", "patient name", "medical condition", "address",
    "date of birth", "passport number", "national identification number",
]
# Labels that get a realistic fake value instead of a bracket token: label -> kind
FAKE_KIND = {"person": "name", "patient name": "name", "address": "address", "email": "email", "phone number": "phone"}
FIRST_NAMES = ["James", "Emma", "Oliver", "Charlotte", "William", "Amelia", "Henry", "Grace", "Thomas", "Olivia", "Daniel", "Hannah"]
LAST_NAMES = ["Wilson", "Turner", "Harris", "Walker", "Clarke", "Wright", "Hughes", "Foster", "Morgan", "Cooper", "Ward", "Bailey"]
CITIES = ["Springfield", "Riverton", "Fairview", "Lakewood", "Georgetown", "Clayton", "Ashland", "Milford", "Oakdale", "Brookfield", "Hudson", "Salem"]
REGIONS = ["Ohio", "Oregon", "Texas", "Iowa", "Utah", "Maine", "Idaho", "Kansas", "Nevada", "Vermont", "Montana", "Georgia"]
STREETS = ["412 Oak Avenue", "1580 Cedar Lane", "73 Willow Street", "906 Birch Road", "221 Elm Drive", "38 Pine Court",
           "1175 Aspen Way", "64 Linden Street", "850 Poplar Avenue", "17 Sycamore Road", "309 Walnut Lane", "92 Spruce Drive"]
STATES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
    "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming",
}
EMAILS = ["alex.morgan", "sam.taylor", "chris.lee", "pat.quinn", "jordan.blake", "casey.reed",
          "drew.parker", "taylor.hayes", "riley.stone", "jamie.fox", "avery.cole", "robin.shaw"]  # at example.com
STREET_WORD = re.compile(r"\b(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|way|court|ct)\b", re.I)
GAP = {"name": re.compile(r" {1,2}"), "address": re.compile(r"\s*,?\s*")}  # text allowed between spans we merge
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


def merge_adjacent(text: str, ents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join neighbouring spans of one kind ("Emily" "Johnson", "Austin" ", " "Texas") into a single entity."""
    out: list[dict[str, Any]] = []
    for e in sorted(ents, key=lambda e: e["start"]):
        prev, kind = (out[-1] if out else None), FAKE_KIND.get(e["label"])
        if prev and kind in GAP and prev["label"] == e["label"] and e["start"] >= prev["end"] \
                and GAP[kind].fullmatch(text[prev["end"] : e["start"]]):
            out[-1] = {**prev, "end": e["end"], "text": text[prev["start"] : e["end"]]}
        else:
            out.append(dict(e))
    return out


class Vault:
    """Per-request mapping of placeholder <-> original value.

    People, addresses, emails and phones get realistic fake values (so the model can use them naturally); every
    other label gets a bracket token like `[EMAIL_1]`. Both are swapped back in the reply.
    """

    def __init__(self, detector: Detector) -> None:
        self.detector = detector
        self.tokens: dict[str, str] = {}  # bracket placeholder -> original
        self.names: dict[str, str] = {}  # fake name/address (full or part) -> original (full or part)
        self._by_value: dict[tuple[str, str], str] = {}
        self._pending = ""  # streamed text that may hold the start of a placeholder or fake value
        self._used_words: set[str] = set()  # lower-cased words of fake values already handed out
        self._next: dict[str, int] = {}  # next unused pool index per kind
        self._counts: dict[str, int] = {}  # bracket tokens handed out per label

    def _fake(self, label: str, original: str, text: str) -> str | None:
        """Pick an unused fake shaped like `original`; None when the pool is exhausted."""
        words = re.findall(r"[^\s,]+", original)
        if not words:
            return None
        kind = FAKE_KIND[label]
        if kind in ("email", "phone"):
            return self._fake_contact(kind, original, text)
        if kind == "address":
            if original[:1].isdigit() or STREET_WORD.search(original):
                kind = "street"
            elif original in STATES:
                kind = "state"
        pool = {"name": FIRST_NAMES, "address": CITIES, "state": REGIONS, "street": STREETS}[kind]
        second = {"name": LAST_NAMES, "address": REGIONS}.get(kind)
        sep = ", " if kind == "address" and "," in original else " "
        seen = {w.lower() for w in re.findall(r"[A-Za-z]+", text)} | self._used_words
        while (i := self._next.get(kind, 0)) < len(pool):
            self._next[kind] = i + 1
            parts = [pool[i]] + ([second[i]] if second and len(words) > 1 else [])
            fake_words = {w.lower() for p in parts for w in re.findall(r"[A-Za-z]+", p)}
            if fake_words & seen:
                continue  # would clash with real text or another fake value
            fake = sep.join(parts)
            self._used_words |= fake_words
            self.names[fake] = original
            if len(parts) == 2 and len(words) == 2:  # the model may reply with only one half
                self.names.setdefault(parts[0], words[0])
                self.names.setdefault(parts[1], words[1])
            return fake
        return None

    def _fake_contact(self, kind: str, original: str, text: str) -> str | None:
        """Fake email (example.com) or phone (555-01xx, same punctuation as the original)."""
        while (i := self._next.get(kind, 0)) < (len(EMAILS) if kind == "email" else 100):
            self._next[kind] = i + 1
            if kind == "email":
                fake = f"{EMAILS[i]}@example.com"
            else:
                n = len(re.findall(r"\d", original))
                digits = f"20255501{i:02d}"  # 10-digit fictional number; 555-0100..0199 is reserved for fiction
                digits = ("1" + digits if n == 11 and original.lstrip("+ ").startswith("1") else digits if n >= 10 else digits[-n:])
                digits = digits.rjust(n, "0")[-n:]
                it = iter(digits)
                fake = re.sub(r"\d", lambda _: next(it), original)
            if fake in text or fake in self.names:
                continue
            self.names[fake] = original
            return fake
        return None

    def _placeholder(self, label: str, value: str, text: str) -> str:
        key = (label, value)
        if key not in self._by_value:
            fake = self._fake(label, value, text) if label in FAKE_KIND else None
            if fake is None:
                n = self._counts[label] = self._counts.get(label, 0) + 1
                fake = f"[{label.upper().replace(' ', '_')}_{n}]"
                self.tokens[fake[1:-1]] = value
            self._by_value[key] = fake
        return self._by_value[key]

    def redact(self, text: str) -> str:
        ents = merge_adjacent(text, self.detector.detect(text))  # sorted by position
        fakes = [self._placeholder(e["label"], e["text"], text) for e in ents]  # hand out in reading order
        for ent, fake in reversed(list(zip(ents, fakes))):  # splice right-to-left so offsets stay valid
            text = text[: ent["start"]] + fake + text[ent["end"] :]
        return text

    def _pattern(self) -> re.Pattern[str]:
        alts = [TOKEN.pattern]
        if self.names:
            names = "|".join(re.escape(n) for n in sorted(self.names, key=len, reverse=True))
            alts.append(rf"(?<![A-Za-z0-9])({names})(?![A-Za-z0-9])")
        return re.compile("|".join(alts))

    def restore(self, text: str) -> str:
        def sub(m: re.Match[str]) -> str:
            if m.group(1) is not None:
                return self.tokens.get(m.group(1), m.group(0))
            return self.names[m.group(2)]

        return self._pattern().sub(sub, text)

    def _hold_from(self, text: str) -> int:
        """Index where a trailing, possibly unfinished placeholder or fake value starts."""
        cut = len(text)
        i = text.rfind("[")
        if i != -1 and "]" not in text[i:] and len(text) - i < MAX_TOKEN_LEN:
            cut = i
        longest = max(map(len, self.names), default=0)
        for i in range(max(0, len(text) - longest), cut):
            tail = text[i:]
            if any(n.startswith(tail) for n in self.names):
                return i
        return cut

    def feed(self, chunk: str) -> str:
        """Restore streamed text, holding back a possibly-split placeholder like `[PERS` or `Ja`."""
        text, self._pending = self._pending + chunk, ""
        cut = self._hold_from(text)
        text, self._pending = text[:cut], text[cut:]
        return self.restore(text)

    def flush(self) -> str:
        text, self._pending = self._pending, ""
        return self.restore(text)
