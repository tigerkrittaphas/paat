"""
Language registry for PAAT experiments.

96 languages that appear in both mC4 (allenai/c4) and FLORES+.
Keys are mC4/ISO 639-1 config codes; values provide iso_639_3 + iso_15924
for FLORES+ filtering and natural mC4 document counts for full-scale runs.

Full-scale allocation: uniform scale applied to MC4_NATURAL_COUNTS so the
natural resource distribution is preserved exactly.

Demo mode: DEMO_DOC_COUNT (5 000 docs) per language — enough to validate
the full pipeline end-to-end in under an hour.
"""

from __future__ import annotations

# mC4 config code -> (iso_639_3, iso_15924)
# iso_15924 is required to disambiguate FLORES+ rows for languages with
# multiple scripts (e.g. Chinese Hans vs Hant, Serbian Cyrl vs Latn).
LANG_REGISTRY: dict[str, tuple[str, str]] = {
    "af": ("afr", "Latn"),  # Afrikaans
    "sq": ("als", "Latn"),  # Albanian (Tosk)
    "am": ("amh", "Ethi"),  # Amharic
    "ar": ("arb", "Arab"),  # Arabic (Modern Standard)
    "az": ("azj", "Latn"),  # Azerbaijani (North)
    "be": ("bel", "Cyrl"),  # Belarusian
    "bn": ("ben", "Beng"),  # Bengali
    "bg": ("bul", "Cyrl"),  # Bulgarian
    "ca": ("cat", "Latn"),  # Catalan
    "ceb": ("ceb", "Latn"),  # Cebuano
    "cs": ("ces", "Latn"),  # Czech
    "zh": ("cmn", "Hans"),  # Chinese (Simplified Mandarin)
    "cy": ("cym", "Latn"),  # Welsh
    "da": ("dan", "Latn"),  # Danish
    "de": ("deu", "Latn"),  # German
    "et": ("ekk", "Latn"),  # Estonian (Standard)
    "el": ("ell", "Grek"),  # Greek (Modern)
    "en": ("eng", "Latn"),  # English
    "eo": ("epo", "Latn"),  # Esperanto
    "eu": ("eus", "Latn"),  # Basque
    "fil": ("fil", "Latn"),  # Filipino
    "fi": ("fin", "Latn"),  # Finnish
    "fr": ("fra", "Latn"),  # French
    "gd": ("gla", "Latn"),  # Scottish Gaelic
    "ga": ("gle", "Latn"),  # Irish
    "gl": ("glg", "Latn"),  # Galician
    "gu": ("guj", "Gujr"),  # Gujarati
    "ht": ("hat", "Latn"),  # Haitian Creole
    "ha": ("hau", "Latn"),  # Hausa
    "iw": ("heb", "Hebr"),  # Hebrew  (mC4 uses legacy 'iw')
    "hi": ("hin", "Deva"),  # Hindi
    "hu": ("hun", "Latn"),  # Hungarian
    "hy": ("hye", "Armn"),  # Armenian
    "ig": ("ibo", "Latn"),  # Igbo
    "id": ("ind", "Latn"),  # Indonesian
    "is": ("isl", "Latn"),  # Icelandic
    "it": ("ita", "Latn"),  # Italian
    "jv": ("jav", "Latn"),  # Javanese
    "ja": ("jpn", "Jpan"),  # Japanese
    "kn": ("kan", "Knda"),  # Kannada
    "ka": ("kat", "Geor"),  # Georgian
    "kk": ("kaz", "Cyrl"),  # Kazakh
    "mn": ("khk", "Cyrl"),  # Mongolian (Halh)
    "km": ("khm", "Khmr"),  # Khmer
    "ky": ("kir", "Cyrl"),  # Kyrgyz
    "ku": ("kmr", "Latn"),  # Kurdish (Kurmanji)
    "ko": ("kor", "Hang"),  # Korean
    "lo": ("lao", "Laoo"),  # Lao
    "lt": ("lit", "Latn"),  # Lithuanian
    "lb": ("ltz", "Latn"),  # Luxembourgish
    "lv": ("lvs", "Latn"),  # Latvian (Standard)
    "ml": ("mal", "Mlym"),  # Malayalam
    "mr": ("mar", "Deva"),  # Marathi
    "mk": ("mkd", "Cyrl"),  # Macedonian
    "mt": ("mlt", "Latn"),  # Maltese
    "mi": ("mri", "Latn"),  # Maori
    "my": ("mya", "Mymr"),  # Burmese
    "nl": ("nld", "Latn"),  # Dutch
    "no": ("nob", "Latn"),  # Norwegian (Bokmål)
    "ne": ("npi", "Deva"),  # Nepali
    "ny": ("nya", "Latn"),  # Nyanja (Chichewa)
    "pa": ("pan", "Guru"),  # Punjabi (Gurmukhi)
    "ps": ("pbt", "Arab"),  # Pashto (Southern)
    "fa": ("pes", "Arab"),  # Persian (Western)
    "mg": ("plt", "Latn"),  # Malagasy (Plateau)
    "pl": ("pol", "Latn"),  # Polish
    "pt": ("por", "Latn"),  # Portuguese
    "ro": ("ron", "Latn"),  # Romanian
    "ru": ("rus", "Cyrl"),  # Russian
    "si": ("sin", "Sinh"),  # Sinhala
    "sk": ("slk", "Latn"),  # Slovak
    "sl": ("slv", "Latn"),  # Slovenian
    "sm": ("smo", "Latn"),  # Samoan
    "sn": ("sna", "Latn"),  # Shona
    "sd": ("snd", "Arab"),  # Sindhi
    "so": ("som", "Latn"),  # Somali
    "st": ("sot", "Latn"),  # Southern Sotho
    "es": ("spa", "Latn"),  # Spanish
    "sr": ("srp", "Cyrl"),  # Serbian
    "su": ("sun", "Latn"),  # Sundanese
    "sv": ("swe", "Latn"),  # Swedish
    "sw": ("swh", "Latn"),  # Swahili
    "ta": ("tam", "Taml"),  # Tamil
    "te": ("tel", "Telu"),  # Telugu
    "tg": ("tgk", "Cyrl"),  # Tajik
    "th": ("tha", "Thai"),  # Thai
    "tr": ("tur", "Latn"),  # Turkish
    "uk": ("ukr", "Cyrl"),  # Ukrainian
    "ur": ("urd", "Arab"),  # Urdu
    "uz": ("uzn", "Latn"),  # Uzbek (Northern, Latin)
    "vi": ("vie", "Latn"),  # Vietnamese
    "xh": ("xho", "Latn"),  # Xhosa
    "yi": ("ydd", "Hebr"),  # Yiddish (Eastern)
    "yo": ("yor", "Latn"),  # Yoruba
    "ms": ("zsm", "Latn"),  # Malay (Standard)
    "zu": ("zul", "Latn"),  # Zulu
}

ALL_LANGUAGES: list[str] = sorted(LANG_REGISTRY.keys())

# Natural document counts available in allenai/c4 (queried 2026-04-16).
# Use these for full-scale runs — the uniform scale factor preserves the
# natural resource distribution across languages.
MC4_NATURAL_COUNTS: dict[str, int] = {
    "af":  1_827_033,
    "sq":  1_547_681,
    "am":    162_870,
    "ar":    992_103,
    "az":  1_571_171,
    "be":    893_238,
    "bn":    802_291,
    "bg":    974_762,
    "ca":  1_606_726,
    "ceb":   351_894,
    "cs":  1_438_356,
    "zh":  1_728_801,
    "cy":    968_418,
    "da":  1_341_651,
    "de":  1_460_155,
    "et":  1_412_607,
    "el":    983_815,
    "en":  2_207_027,
    "eo":    500_048,
    "eu":  1_555_887,
    "fil": 1_433_114,
    "fi":  1_444_241,
    "fr":  1_483_670,
    "gd":    322_404,
    "ga":    465_670,
    "gl":  2_686_253,
    "gu":    631_600,
    "ht":    269_174,
    "ha":    247_479,
    "iw":    839_200,
    "hi":    678_926,
    "hu":  1_332_426,
    "hy":    998_973,
    "ig":     92_909,
    "id":  1_310_620,
    "is":  1_234_287,
    "it":  1_613_729,
    "jv":    581_528,
    "ja":    526_672,
    "kn":    680_861,
    "ka":    654_591,
    "kk":    658_192,
    "mn":    818_659,
    "km":    705_946,
    "ky":    957_035,
    "ku":    298_389,
    "ko":    783_719,
    "lo":    141_776,
    "lt":  1_433_613,
    "lb":  2_740_336,
    "lv":  1_336_424,
    "ml":    645_468,
    "mr":    506_946,
    "mk":  1_017_747,
    "mt":    852_641,
    "mi":    101_169,
    "my":    511_300,
    "nl":  1_740_248,
    "no":  1_289_372,
    "ne":    802_128,
    "ny":    174_696,
    "pa":    363_399,
    "ps":    335_452,
    "fa":  1_034_284,
    "mg":    345_040,
    "pl":  1_458_339,
    "pt":  1_662_189,
    "ro":  1_234_608,
    "ru":    904_423,
    "si":    509_536,
    "sk":  1_491_436,
    "sl":  1_432_900,
    "sm":     98_467,
    "sn":    326_392,
    "sd":    652_216,
    "so":    893_012,
    "st":     66_837,
    "es":  1_306_068,
    "sr":    763_490,
    "su":    280_719,
    "sv":  1_429_810,
    "sw":    985_654,
    "ta":    565_961,
    "te":    668_185,
    "tg":    844_606,
    "th":    840_348,
    "tr":  1_596_044,
    "uk":    849_029,
    "ur":    879_259,
    "uz":    796_416,
    "vi":  1_202_854,
    "xh":     69_048,
    "yi":    143_708,
    "yo":     46_214,
    "ms":  1_360_782,
    "zu":    555_458,
}

# Demo mode: fixed doc count per language.
# 5 000 docs × 96 languages ≈ 960 K docs ≈ ~1 GB — validates the full
# pipeline quickly before committing to the full-scale download.
DEMO_DOC_COUNT: int = 5_000


def get_doc_counts(demo: bool = False) -> dict[str, int]:
    """Return per-language doc counts.

    Args:
        demo: If True, return DEMO_DOC_COUNT for every language (capped at
              the natural count so we never request more than exists).
              If False, return MC4_NATURAL_COUNTS.
    """
    if not demo:
        return dict(MC4_NATURAL_COUNTS)
    return {lang: min(DEMO_DOC_COUNT, MC4_NATURAL_COUNTS[lang]) for lang in ALL_LANGUAGES}
