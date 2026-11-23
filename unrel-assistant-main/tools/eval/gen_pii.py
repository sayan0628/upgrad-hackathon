"""Generate the KPI-02 PII span benchmark: bench_results/eval/pii_spans.jsonl.

Each line is one labelled span: {"id", "split", "type", "span", "text"} where `span` is the exact
substring of `text` that is PII (after obfuscation) and `type` is its span type. Deterministic (fixed seed).
Split: ~30% dev (tuning), ~70% test (report only). Run: python3 tools/eval/gen_pii.py
"""
from __future__ import annotations

import json
import re
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench_results" / "eval" / "pii_spans.jsonl"
SEED = 20241
N_PER_TYPE = 150  # 11 types x 150 = 1650 spans (design minimum ~1,500)

FIRST = ["Jane", "Omar", "Priya", "Liam", "Chen", "Maria", "Kwame", "Sofia", "Hiro", "Aisha", "Tom", "Elena"]
LAST = ["Doe", "Hassan", "Patel", "Okafor", "Wang", "Garcia", "Mensah", "Rossi", "Tanaka", "Khan", "Walsh", "Novak"]
DOMAINS = ["example.com", "acme.io", "mailbox.org", "corp-mail.net", "uni.edu", "contoso.co.uk"]
TITLES = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sir", "Dame", "Mx."]
STREETS = ["High Street", "Oak Avenue", "Maple Drive", "Rue de Rivoli", "Main St", "Kingsway", "Elm Court"]
CITIES = ["Springfield", "Leeds", "Austin", "Bristol", "Lyon", "Denver", "Manchester", "Portland"]
UK_POSTCODES = ["SW1A 1AA", "M1 1AE", "EH1 2NG", "BS1 4DJ", "LS1 6AQ"]
US_STATES = ["CA", "TX", "NY", "WA", "IL", "CO"]
TEMPLATES = [
    "My record says {s}, can you check it?",
    "Please update the file for {s} before Friday.",
    "The form lists {s} as the contact.",
    "Quick question: {s} is on the ticket, what is the status?",
    "Reference {s} in the claim notes.",
    "Ignore the header. Entry: {s}. Thanks.",
    "Forwarding this: {s}",
    "{s} needs a callback this week.",
]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
HOMOGLYPH = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "i": "і"}  # Cyrillic lookalikes
GREEK = {"o": "ο", "a": "α", "e": "ε"}
ZW = ["​", "‌", "⁠"]
FULLWIDTH = str.maketrans("0123456789", "０１２３４５６７８９")


def luhn_check_digit(body: str) -> str:
    s, alt = 0, True
    for ch in reversed(body):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        s += d
        alt = not alt
    return str((10 - s % 10) % 10)


def valid_card(rng: random.Random) -> str:
    prefix = rng.choice(["4", "51", "52", "55", "37", "6011"])
    body = prefix + "".join(rng.choice("0123456789") for _ in range(15 - len(prefix)))
    return body + luhn_check_digit(body)


def invalid_card(rng: random.Random) -> str:
    c = valid_card(rng)
    bad = c[:-1] + str((int(c[-1]) + rng.randint(1, 9)) % 10)
    return bad


def grouped(digits: str, rng: random.Random) -> str:
    sep = rng.choice([" ", "-", ""])
    if len(digits) == 16:
        return sep.join(digits[i:i + 4] for i in range(0, 16, 4)) if sep else digits
    return digits


def digits_of(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


# --- span generators: each returns the clean (un-obfuscated) surface form ---
def gen_email(rng):
    local = rng.choice(FIRST).lower() + rng.choice(["", ".", "_"]) + rng.choice(LAST).lower()
    return f"{local}{rng.randint(1, 99) if rng.random() < .3 else ''}@{rng.choice(DOMAINS)}"


def gen_phone_us(rng):
    area = f"{rng.randint(200, 989)}"
    style = rng.choice(["dash", "paren", "dot", "plain", "plus1"])
    mid, last = digits_of(3, rng), digits_of(4, rng)
    if style == "paren":
        return f"({area}) {mid}-{last}"
    if style == "dot":
        return f"{area}.{mid}.{last}"
    if style == "plain":
        return f"{area}{mid}{last}"
    if style == "plus1":
        return f"+1 {area}-{mid}-{last}"
    return f"{area}-{mid}-{last}"


def gen_phone_uk(rng):
    style = rng.choice(["mobile", "landline", "intl"])
    if style == "mobile":
        return f"07{rng.randint(700, 799)} {digits_of(3, rng)} {digits_of(3, rng)}"
    if style == "landline":
        return f"020 {digits_of(4, rng)} {digits_of(4, rng)}"
    return f"+44 7{rng.randint(700, 799)} {digits_of(6, rng)}"


def gen_phone_intl(rng):
    cc = rng.choice(["+33", "+49", "+91", "+61", "+81", "+52", "+55"])
    return f"{cc} {digits_of(rng.randint(2, 3), rng)} {digits_of(rng.randint(3, 4), rng)} {digits_of(4, rng)}"


def gen_ssn(rng):
    a, b, c = digits_of(3, rng), digits_of(2, rng), digits_of(4, rng)
    sep = rng.choice(["-", " ", ".", ""])
    return f"{a}{sep}{b}{sep}{c}"


def gen_card_valid(rng):
    c = valid_card(rng)
    return grouped(c, rng)


def gen_card_invalid(rng):
    c = invalid_card(rng)
    return grouped(c, rng)


def gen_iban(rng):
    cc = rng.choice(["GB", "DE", "FR", "ES", "NL"])
    body = digits_of(rng.choice([14, 18, 20]), rng)
    s = cc + digits_of(2, rng) + body
    return " ".join(s[i:i + 4] for i in range(0, len(s), 4)) if rng.random() < .6 else s


def gen_address(rng):
    if rng.random() < .5:
        return f"{rng.randint(1, 999)} {rng.choice(STREETS)}, {rng.choice(CITIES)} {rng.choice(US_STATES)} {rng.randint(10000, 99999)}"
    return f"{rng.randint(1, 99)} {rng.choice(STREETS)}, {rng.choice(CITIES)} {rng.choice(UK_POSTCODES)}"


def gen_person(rng):
    return f"{rng.choice(TITLES)} {rng.choice(FIRST)} {rng.choice(LAST)}"


def gen_dob(rng):
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1940, 2005)
    # a date is a date of birth through its cue ("DOB:", "born"); bare dates are not labelled as DOB spans
    cue = rng.choice(["DOB: ", "dob ", "date of birth: ", "born on ", "born "])
    if cue == "born " and rng.random() < .5:
        return f"born {rng.choice(['January', 'March', 'July', 'October'])} {d}, {y}"
    return cue + rng.choice([f"{d:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{d:02d}", f"{m:02d}/{d:02d}/{y}"])


# --- obfuscations: (name, fn). Each takes the clean span text and returns the obfuscated span ---
def ob_none(s, rng):
    return s


def ob_bracket_at(s, rng):
    return s.replace("@", rng.choice(["[at]", "(at)", " [at] ", "{at}"])).replace(".", rng.choice(["[dot]", "(dot)", " dot "]))


def ob_spaced(s, rng):
    # spaced digits: a space between adjacent digits only (labels and separators are left alone)
    return re.sub(r"(?<=\d)(?=\d)", " ", s)


def ob_homoglyph(s, rng):
    table = {**HOMOGLYPH, **GREEK} if rng.random() < .3 else HOMOGLYPH
    return "".join(table.get(c, c) if rng.random() < .8 else c for c in s)


def ob_zero_width(s, rng):
    return "".join(c + (rng.choice(ZW) if rng.random() < .35 and c.isalnum() else "") for c in s)


def ob_fullwidth(s, rng):
    return s.translate(FULLWIDTH)


def ob_digit_words(s, rng):
    return " ".join(WORDS[int(c)] if c.isdigit() else c for c in s)


def ob_mixed_case(s, rng):
    return "".join(c.upper() if rng.random() < .5 else c.lower() for c in s)


OBF = {"none": ob_none, "bracket_at": ob_bracket_at, "spaced": ob_spaced, "homoglyph": ob_homoglyph,
       "zero_width": ob_zero_width, "fullwidth": ob_fullwidth, "digit_words": ob_digit_words,
       "mixed_case": ob_mixed_case}
# which obfuscations are meaningful for each span type
APPLICABLE = {
    "email": ["none", "bracket_at", "homoglyph", "zero_width", "fullwidth", "mixed_case"],
    "phone_us": ["none", "spaced", "homoglyph", "zero_width", "fullwidth", "digit_words"],
    "phone_uk": ["none", "spaced", "zero_width", "fullwidth", "digit_words"],
    "phone_intl": ["none", "spaced", "zero_width", "fullwidth"],
    "ssn": ["none", "spaced", "zero_width", "fullwidth", "digit_words"],
    "card_valid": ["none", "spaced", "zero_width", "fullwidth", "digit_words"],
    "card_invalid": ["none", "spaced", "zero_width", "fullwidth"],
    "iban": ["none", "spaced", "zero_width", "fullwidth"],
    "address": ["none", "homoglyph", "zero_width", "mixed_case"],
    "person_title": ["none", "homoglyph", "zero_width", "mixed_case"],
    "dob": ["none", "spaced", "fullwidth", "digit_words"],
}
GENERATORS = {"email": gen_email, "phone_us": gen_phone_us, "phone_uk": gen_phone_uk, "phone_intl": gen_phone_intl,
              "ssn": gen_ssn, "card_valid": gen_card_valid, "card_invalid": gen_card_invalid, "iban": gen_iban,
              "address": gen_address, "person_title": gen_person, "dob": gen_dob}


def build() -> list[dict]:
    rng = random.Random(SEED)
    rows: list[dict] = []
    n = 0
    for typ, gen in GENERATORS.items():
        for i in range(N_PER_TYPE):
            clean = gen(rng)
            ob = rng.choice(APPLICABLE[typ])
            if ob == "none" and rng.random() < .5:
                ob = rng.choice(APPLICABLE[typ][1:] or ["none"])
            span = OBF[ob](clean, rng)
            if typ == "email" and ob == "bracket_at":
                pass
            text = rng.choice(TEMPLATES).format(s=span)
            rows.append({"id": n, "split": "", "type": typ, "obfuscation": ob, "span": span, "text": text})
            n += 1
    # deterministic 30/70 split: dev = every ~3rd-in-10 item, stratified by type via index
    rng2 = random.Random(SEED + 1)
    order = list(range(len(rows)))
    rng2.shuffle(order)
    cut = int(round(0.30 * len(rows)))
    dev = set(order[:cut])
    for i, r in enumerate(rows):
        r["split"] = "dev" if i in dev else "test"
    return rows


def main() -> int:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dev = sum(r["split"] == "dev" for r in rows)
    print(f"wrote {len(rows)} spans to {OUT} (dev={dev}, test={len(rows) - dev})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
