from mentat_session_logger.classification import (
    _build_canon_delta,
    _build_table_diary,
    _parse_summary_sections,
)


def test_parse_summary_sections_handles_mixed_markdown_styles() -> None:
    summary = """
Here is a summary of the transcript chunk:

### Canon Facts
- First factual line
- none mentioned in this transcript chunk

**Rules/Meta**: Initiative reroll was allowed once.
"""
    sections = _parse_summary_sections(summary)
    assert sections["canon_facts"] == ["First factual line"]
    assert sections["rules_meta"] == ["Initiative reroll was allowed once."]


def test_build_canon_delta_uses_summary_fallback_and_dedupes() -> None:
    classifications = [
        ("chunk_001", {"canon_facts": []}),
        ("chunk_002", {"canon_facts": ["Known canon fact"]}),
    ]
    summary_by_chunk = {
        "chunk_001": {
            "canon_facts": [
                "Fallback canon fact",
                "none applicable",
            ]
        },
        "chunk_002": {
            "canon_facts": [
                "Known canon fact",
            ]
        },
    }

    output = _build_canon_delta(classifications, summary_by_chunk)
    assert "- Fallback canon fact" in output
    assert output.count("Known canon fact") == 1
    assert "none applicable" not in output


def test_build_table_diary_falls_back_to_summary_short_summary() -> None:
    classifications = [
        (
            "chunk_001",
            {
                "primary_category": "LOGISTICS",
                "include_in_table_diary": False,
                "start": "00:10:00",
                "end": "00:12:00",
                "summary": "",
            },
        )
    ]
    summary_by_chunk = {"chunk_001": {"short_summary": ["Scheduling and planning discussion"]}}

    output = _build_table_diary(classifications, summary_by_chunk)
    assert "- [00:10:00-00:12:00] LOGISTICS: Scheduling and planning discussion" in output
