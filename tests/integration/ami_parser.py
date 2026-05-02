"""
Utility that parses AMI Corpus words XML files into a mentat-style
diarized transcript (``[HH:MM:SS-HH:MM:SS] SPEAKER: text``).

Only the first ``max_seconds`` of audio are included to keep LLM tests fast.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


SPEAKERS = ("A", "B", "C", "D")
NITE_NS = "http://nite.sourceforge.net/"


@dataclass(order=True)
class WordToken:
    start: float
    end: float
    speaker: str
    text: str


def _seconds_to_hms(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _parse_words_xml(path: Path, speaker: str, max_seconds: float) -> list[WordToken]:
    """Return word tokens for one speaker, up to max_seconds."""
    tokens: list[WordToken] = []
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return tokens

    root = tree.getroot()
    for elem in root:
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag != "w":
            continue
        start_attr = elem.get(f"{{{NITE_NS}}}id", "")  # fallback
        start = elem.get("starttime")
        end = elem.get("endtime")
        text = (elem.text or "").strip()
        punc = elem.get("punc", "false").lower() == "true"

        if start is None or end is None or not text:
            continue
        start_f = float(start)
        if start_f > max_seconds:
            break
        if punc:
            # attach punctuation to previous token if possible
            if tokens:
                tokens[-1] = WordToken(
                    start=tokens[-1].start,
                    end=float(end),
                    speaker=tokens[-1].speaker,
                    text=tokens[-1].text + text,
                )
            continue
        tokens.append(WordToken(start=start_f, end=float(end), speaker=speaker, text=text))
    return tokens


def _group_into_turns(
    tokens: list[WordToken],
    gap_threshold: float = 1.0,
) -> list[tuple[str, float, float, str]]:
    """
    Group consecutive same-speaker tokens into utterance turns.
    Returns list of (speaker, start, end, text).
    A gap >= gap_threshold seconds between tokens of the same speaker
    forces a turn break.
    """
    if not tokens:
        return []

    turns: list[tuple[str, float, float, str]] = []
    cur_speaker = tokens[0].speaker
    cur_start = tokens[0].start
    cur_end = tokens[0].end
    cur_words: list[str] = [tokens[0].text]

    for tok in tokens[1:]:
        same_speaker = tok.speaker == cur_speaker
        small_gap = (tok.start - cur_end) < gap_threshold
        if same_speaker and small_gap:
            cur_end = tok.end
            cur_words.append(tok.text)
        else:
            turns.append((cur_speaker, cur_start, cur_end, " ".join(cur_words)))
            cur_speaker = tok.speaker
            cur_start = tok.start
            cur_end = tok.end
            cur_words = [tok.text]

    turns.append((cur_speaker, cur_start, cur_end, " ".join(cur_words)))
    return turns


def build_diarized_transcript(
    words_dir: Path,
    meeting_id: str,
    max_seconds: float = 600.0,
) -> str:
    """
    Parse all four speaker XML files for *meeting_id* and return a
    mentat-style diarized transcript string.

    Example output line::

        [00:00:32-00:00:42] Speaker_A: Okay. Good morning everybody.
    """
    all_tokens: list[WordToken] = []
    for speaker in SPEAKERS:
        xml_path = words_dir / f"{meeting_id}.{speaker}.words.xml"
        if not xml_path.exists():
            continue
        all_tokens.extend(_parse_words_xml(xml_path, speaker, max_seconds))

    if not all_tokens:
        return ""

    all_tokens.sort()
    turns = _group_into_turns(all_tokens)

    lines: list[str] = []
    for speaker, start, end, text in turns:
        h_start = _seconds_to_hms(start)
        h_end = _seconds_to_hms(end)
        lines.append(f"[{h_start}-{h_end}] Speaker_{speaker}: {text}")

    return "\n".join(lines) + "\n"
