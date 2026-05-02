import pytest

from mentat_session_logger.llm import parse_json_response


def test_parse_json_response_valid() -> None:
    parsed = parse_json_response('{"primary_category": "IC_GAMEPLAY"}')
    assert parsed["primary_category"] == "IC_GAMEPLAY"


def test_parse_json_response_invalid() -> None:
    with pytest.raises(ValueError):
        parse_json_response("not json")
