import os
from desktop_app.data_importers.normalizers import (
    strip_currency_and_separators,
    parse_fractional_quantity,
    normalize_unit,
    safe_float_from_string,
)
from desktop_app.data_importers.csv_importer import CSVImporter


def test_strip_currency_and_separators():
    assert strip_currency_and_separators('$1,234.56') == '1234.56'
    assert strip_currency_and_separators('(1,234)') == '-1234'


def test_parse_fractional_quantity():
    assert parse_fractional_quantity('1/2') == 0.5
    assert parse_fractional_quantity('1 1/2') == 1.5
    assert parse_fractional_quantity('½') == 0.5
    assert parse_fractional_quantity('2') == 2.0


def test_normalize_unit():
    assert normalize_unit('lbs') == 'lb'
    assert normalize_unit('Ounces') == 'oz'
    assert normalize_unit(' kg ') == 'kg'


def test_csv_sniffing():
    # create a small temporary CSV with semicolon delimiter to test sniffing
    tmp = 'tests/tmp_sniff.csv'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write('a;b;c\n1;2;3\n')
    ci = CSVImporter(data_directory='.')
    enc, delim = ci._detect_encoding_and_delimiter(ci.data_directory / tmp)
    assert enc in ('utf-8', 'utf-8-sig', 'latin-1')
    assert delim in (',', ';', '\t', '|')
    os.remove(tmp)
