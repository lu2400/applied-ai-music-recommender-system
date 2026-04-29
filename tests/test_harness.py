"""
Evaluation harness for the music recommender.
Runs predefined scenarios and prints a pass/fail summary with confidence scores.

Usage:
    python -m tests.test_harness
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommender import load_songs, recommend_songs

SONGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")

# Each scenario defines input preferences, what to assert, and a minimum score.
SCENARIOS = [
    {
        "name": "Pop happy high energy -> top result is pop/happy",
        "prefs": {
            "genre": "pop", "mood": "happy", "energy": 0.85,
            "likes_acoustic": False, "preferred_decade": 2020,
            "preferred_mood_tags": [], "likes_clean": False,
            "likes_instrumental": False, "likes_live": False,
        },
        "expect_genre": "pop",
        "expect_mood": "happy",
        "min_score": 50,
    },
    {
        "name": "Lofi chill focused -> top result is lofi/chill",
        "prefs": {
            "genre": "lofi", "mood": "chill", "energy": 0.35,
            "likes_acoustic": True, "preferred_decade": 2020,
            "preferred_mood_tags": ["focused", "peaceful"], "likes_clean": True,
            "likes_instrumental": True, "likes_live": False,
        },
        "expect_genre": "lofi",
        "min_score": 60,
    },
    {
        "name": "Rock intense workout -> top result is rock/intense",
        "prefs": {
            "genre": "rock", "mood": "intense", "energy": 0.9,
            "likes_acoustic": False, "preferred_decade": 2010,
            "preferred_mood_tags": ["aggressive", "powerful"], "likes_clean": False,
            "likes_instrumental": False, "likes_live": False,
        },
        "expect_genre": "rock",
        "expect_mood": "intense",
        "min_score": 50,
    },
    {
        "name": "Jazz relaxed evening -> top result is jazz",
        "prefs": {
            "genre": "jazz", "mood": "relaxed", "energy": 0.3,
            "likes_acoustic": True, "preferred_decade": 2000,
            "preferred_mood_tags": ["warm", "cozy"], "likes_clean": True,
            "likes_instrumental": False, "likes_live": True,
        },
        "expect_genre": "jazz",
        "min_score": 40,
    },
    {
        "name": "Clean lyrics preference -> top result is not explicit",
        "prefs": {
            "genre": "pop", "mood": "happy", "energy": 0.8,
            "likes_acoustic": False, "preferred_decade": 2010,
            "preferred_mood_tags": [], "likes_clean": True,
            "likes_instrumental": False, "likes_live": False,
        },
        "expect_explicit": 0,
        "min_score": 0,
    },
    {
        "name": "Diversity penalty -> top 3 results have different artists",
        "prefs": {
            "genre": "pop", "mood": "happy", "energy": 0.8,
            "likes_acoustic": False, "preferred_decade": 2010,
            "preferred_mood_tags": [], "likes_clean": False,
            "likes_instrumental": False, "likes_live": False,
        },
        "expect_diverse_artists": True,
        "k": 3,
        "min_score": 0,
    },
]


def _check(scenario: dict, recs: list) -> tuple[bool, str]:
    """Return (passed, failure_reason) for a scenario result."""
    if not recs:
        return False, "no recommendations returned"

    top_song, top_score, _ = recs[0]

    if "expect_genre" in scenario and top_song.get("genre") != scenario["expect_genre"]:
        return False, f"expected genre={scenario['expect_genre']}, got {top_song.get('genre')!r}"

    if "expect_mood" in scenario and top_song.get("mood") != scenario["expect_mood"]:
        return False, f"expected mood={scenario['expect_mood']}, got {top_song.get('mood')!r}"

    if top_score < scenario.get("min_score", 0):
        return False, f"score {top_score:.1f} below minimum {scenario['min_score']}"

    if "expect_explicit" in scenario:
        if int(top_song.get("explicit", 0)) != scenario["expect_explicit"]:
            return False, (
                f"expected explicit={scenario['expect_explicit']}, "
                f"got {top_song.get('explicit')}"
            )

    if scenario.get("expect_diverse_artists") and len(recs) >= 3:
        artists = [r[0]["artist"] for r in recs[:3]]
        if len(set(artists)) < len(artists):
            return False, f"duplicate artists in top 3: {artists}"

    return True, ""


def run_harness() -> tuple[int, int]:
    songs = load_songs(SONGS_PATH)
    passed = 0
    rows = []

    for s in SCENARIOS:
        k = s.get("k", 5)
        recs = recommend_songs(s["prefs"], songs, k=k)
        ok, reason = _check(s, recs)

        top_song, top_score, _ = recs[0] if recs else ({}, 0, "")
        confidence = min(100, max(0, int(top_score)))

        if ok:
            passed += 1
        rows.append((ok, s["name"], top_song, confidence, reason))

    print()
    print("=" * 72)
    print("  MUSIC RECOMMENDER — EVALUATION HARNESS")
    print("=" * 72)
    for ok, name, song, conf, reason in rows:
        mark = "PASS" if ok else "FAIL"
        title = song.get("title", "—")
        artist = song.get("artist", "—")
        print(f"\n  [{mark}] {name}")
        print(f"       Top pick : {title} — {artist}")
        print(f"       Confidence: {conf}/100")
        if reason:
            print(f"       Failure   : {reason}")

    total = len(SCENARIOS)
    print()
    print("=" * 72)
    print(f"  Result: {passed}/{total} passed")
    print("=" * 72)
    print()
    return passed, total


if __name__ == "__main__":
    passed, total = run_harness()
    sys.exit(0 if passed == total else 1)
