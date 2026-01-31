"""Normalization helpers for CSV importers.

Provides small, utility functions to normalize numeric strings, parse fractions,
strip currency/thousands separators, and normalize unit strings.
"""
from decimal import Decimal
from fractions import Fraction
import unicodedata
import re
from typing import Optional


_UNICODE_FRACTIONS = {
    '½': '1/2', '¼': '1/4', '¾': '3/4', '⅓': '1/3', '⅔': '2/3',
    '⅛': '1/8', '⅜': '3/8', '⅝': '5/8', '⅞': '7/8'
}


def _replace_unicode_fractions(s: str) -> str:
    for uf, ascii_rep in _UNICODE_FRACTIONS.items():
        if uf in s:
            s = s.replace(uf, f" {ascii_rep}")
    return s


def strip_currency_and_separators(s: Optional[str]) -> Optional[str]:
    """Strip currency symbols and thousands separators from numeric strings.

    Examples: "$1,234.56" -> "1234.56"; "(1,234)" -> "-1234"
    """
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize('NFKC', s).strip()
    # handle parenthesis for negatives
    negative = False
    if s.startswith('(') and s.endswith(')'):
        negative = True
        s = s[1:-1]
    # remove currency symbols and spaces
    s = re.sub(r'[\$€£¥₩₹]', '', s)
    # remove common thousands separators (commas, thin spaces)
    s = s.replace(',', '').replace('\u202f', '').replace('\u00A0', '')
    s = s.strip()
    if negative:
        s = '-' + s
    return s


def parse_fractional_quantity(s: Optional[str]) -> Optional[float]:
    """Parse a quantity that might be a fraction, mixed fraction, or decimal.

    Returns a float or None if parsing fails.
    """
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if s == '':
        return None

    # replace unicode fraction glyphs
    s = _replace_unicode_fractions(s)

    # handle mixed numbers like "1 1/2"
    if re.match(r'^\d+\s+\d+/\d+$', s):
        parts = s.split()
        whole = int(parts[0])
        frac = Fraction(parts[1])
        return float(whole + frac)

    # simple fraction like "1/2"
    if re.match(r'^\d+/\d+$', s):
        return float(Fraction(s))

    # strip currency & thousands
    cleaned = strip_currency_and_separators(s)
    try:
        return float(Decimal(cleaned))
    except Exception:
        # last-ditch: try to remove any stray characters
        cleaned2 = re.sub(r'[^\d\.\-]', '', cleaned or '')
        try:
            return float(Decimal(cleaned2))
        except Exception:
            return None


def normalize_unit(u: Optional[str]) -> Optional[str]:
    """Normalize unit strings to a canonical short form.

    Examples: 'lbs' -> 'lb', 'ounces' -> 'oz', 'KG' -> 'kg'
    """
    if u is None:
        return None
    if not isinstance(u, str):
        u = str(u)
    u = unicodedata.normalize('NFKC', u).strip().lower()
    u = re.sub(r'[\s\.]+', '', u)
    # common normalizations
    mapping = {
        'lbs': 'lb', 'pounds': 'lb', 'pound': 'lb', 'lb.': 'lb', 'lbs.': 'lb',
        'ozs': 'oz', 'ounces': 'oz', 'ounce': 'oz',
        'kg': 'kg', 'g': 'g', 'gram': 'g', 'grams': 'g',
        'l': 'l', 'lt': 'l', 'liter': 'l', 'ml': 'ml'
    }
    return mapping.get(u, u)


def safe_float_from_string(s: Optional[str], default: float = 0.0) -> float:
    """Combine stripping and fraction parsing to yield a safe float."""
    val = parse_fractional_quantity(s)
    if val is None:
        return default
    return val
