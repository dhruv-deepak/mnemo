from datetime import UTC, datetime

import pytest

from benchmarks.locomo import f1_score, full_context, parse_session_time, questions, sessions

SAMPLE = {
    "sample_id": "conv-x",
    "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_2_date_time": "8:10 pm on 20 May, 2023",
        "session_2": [{"speaker": "Melanie", "dia_id": "D2:1", "text": "I ran a charity race!"}],
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel!"},
            {"speaker": "Melanie", "dia_id": "D1:2", "text": "Look", "blip_caption": "a sunset"},
        ],
    },
    "qa": [
        {"question": "When did Mel run?", "answer": "20 May 2023", "category": 2},
        {"question": "Trick?", "adversarial_answer": "x", "category": 5},
        {"question": "Year?", "answer": 2022, "category": 4},
    ],
}


def test_parse_session_time():
    assert parse_session_time("1:56 pm on 8 May, 2023") == datetime(2023, 5, 8, 13, 56, tzinfo=UTC)


def test_sessions_are_ordered_and_include_photo_captions():
    parsed = sessions(SAMPLE)
    assert [when.day for when, _ in parsed] == [8, 20]
    assert parsed[0][1][1] == {
        "role": "user",
        "name": "Melanie",
        "content": "Look [shares a photo: a sunset]",
    }
    assert "Session on 08 May 2023" in full_context(SAMPLE)


def test_questions_skip_adversarial_and_stringify_answers():
    qs = questions(SAMPLE)
    assert [q["qid"] for q in qs] == ["conv-x-0", "conv-x-2"]
    assert qs[1]["answer"] == "2022"


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        ("20 May 2023", "20 May 2023", 1.0),
        ("The race was on 20 May", "20 May 2023", 0.5),
        ("no idea", "20 May 2023", 0.0),
        ("", "x", 0.0),
    ],
)
def test_f1(prediction, gold, expected):
    assert f1_score(prediction, gold) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    ("message", "seconds"),
    [
        ("Please try again in 6m5.04s. Need more tokens?", 365.04),
        ("try again in 1h2m3s", 3723.0),
        ("try again in 45.5s", 45.5),
        ("no hint here", 600.0),
    ],
)
def test_retry_after_seconds(message, seconds):
    from benchmarks.locomo import retry_after_seconds

    assert retry_after_seconds(message) == pytest.approx(seconds)
