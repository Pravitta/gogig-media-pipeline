import pytest
from worker.checks.plate import _normalize_and_validate, _clean_hsrp_prefix


def test_standard_indian_plates():
    # Standard format: State + RTO + Series + Number
    res1 = _normalize_and_validate("MH12AB1234")
    assert res1 is not None
    assert res1["plate_text"] == "MH 12 AB 1234"
    assert res1["normalized_text"] == "MH12AB1234"
    assert res1["is_valid_format"] is True
    assert res1["format_valid"] is True

    # Single series letter: KA 05 N 9999
    res2 = _normalize_and_validate("KA05N9999")
    assert res2 is not None
    assert res2["plate_text"] == "KA 05 N 9999"
    assert res2["is_valid_format"] is True

    # 0 series letters: MH 12 1234
    res3 = _normalize_and_validate("MH121234")
    assert res3 is not None
    assert res3["plate_text"] == "MH 12 1234"
    assert res3["is_valid_format"] is True


def test_hsrp_prefix_cleaning():
    # HSRP "IND" prefix on license plate
    raw = "INDMH12AB1234"
    cleaned = _clean_hsrp_prefix(raw)
    assert cleaned == "MH12AB1234"

    res = _normalize_and_validate("INDMH12AB1234")
    assert res is not None
    assert res["plate_text"] == "MH 12 AB 1234"


def test_bharat_series_plates():
    # BH Series: 22 BH 1234 A
    res = _normalize_and_validate("22BH1234A")
    assert res is not None
    assert res["plate_text"] == "22 BH 1234 A"
    assert res["plate_format"] == "BHARAT_SERIES"
    assert res["is_valid_format"] is True


def test_character_confusion_corrections():
    # OCR misreading 0 as O, 1 as I, 8 as B in digit positions
    res = _normalize_and_validate("MH12AB12O4")
    assert res is not None
    assert res["plate_text"] == "MH 12 AB 1204"

    # OCR misreading O as 0, I as 1 in state position
    res = _normalize_and_validate("M012AB1234")
    assert res is not None
    assert res["plate_text"] == "MO 12 AB 1234" or res["plate_text"] == "MH 12 AB 1234"


def test_invalid_plates():
    assert _normalize_and_validate("INVALID123456789") is None
    assert _normalize_and_validate("123") is None
