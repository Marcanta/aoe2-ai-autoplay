import pytest

from aoe2_ai.perception.detectors import match_age, parse_clock, parse_fraction, parse_population, parse_resource
from aoe2_ai.perception.game_state import Age, Population


@pytest.mark.parametrize("text, expected", [("200", 200), ("0", 0), ("1250", 1250), ("12 50", 1250), ("999999", 999999)])
def test_parse_resource_valid(text, expected):
    assert parse_resource(text) == expected


@pytest.mark.parametrize("text", ["", " ", "abc", "1200000", "0200", "2O0", "-5", "12/5", "1,250"])
def test_parse_resource_rejects_garbage_instead_of_zero(text):
    assert parse_resource(text) is None


def test_zero_only_from_a_literal_zero():
    assert parse_resource("0") == 0
    assert parse_resource("") is None


@pytest.mark.parametrize("text, expected", [("4/5", Population(4, 5)), ("125/200", Population(125, 200)), ("12/10", Population(12, 10))])
def test_parse_population(text, expected):
    assert parse_population(text) == expected


@pytest.mark.parametrize("text", ["", "45", "4/", "/5", "4/0", "4/5000", "04/5", "4/5/6"])
def test_parse_population_rejects(text):
    assert parse_population(text) is None


def test_parse_fraction():
    assert parse_fraction("45/45") == (45, 45)
    assert parse_fraction("x/45") is None


@pytest.mark.parametrize("text, expected", [("00:12:34", 754), ("12:34", 754), ("1:02:03", 3723), ("0:00", 0)])
def test_parse_clock(text, expected):
    assert parse_clock(text) == expected


@pytest.mark.parametrize("text", ["", "12:3", "12:60", "61:00", "1234", "a:bc"])
def test_parse_clock_rejects(text):
    assert parse_clock(text) is None


@pytest.mark.parametrize(
    "text, age",
    [
        ("Âge sombre", Age.DARK),
        ("Age sombre", Age.DARK),
        ("Ave sombre", Age.DARK),
        ("Âge féodal", Age.FEUDAL),
        ("Age des chateaux", Age.CASTLE),
        ("Âge impérial", Age.IMPERIAL),
        ("Dark Age", Age.DARK),
        ("Castle Age", Age.CASTLE),
        ("Feudal Age 45%", Age.FEUDAL),
    ],
)
def test_match_age(text, age):
    result = match_age(text)
    assert result is not None and result[0] is age


@pytest.mark.parametrize("text", ["", "Age", "Multijoueur", "1234"])
def test_match_age_rejects(text):
    assert match_age(text) is None
