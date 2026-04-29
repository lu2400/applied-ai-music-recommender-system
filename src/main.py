"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from tabulate import tabulate
from .recommender import load_songs, recommend_songs


def print_recommendations(label: str, user_prefs: dict, recommendations: list) -> None:
    print(f"\n{'=' * 100}")
    print(f"  {label}")

    prefs_parts = [
        f"genre={user_prefs['genre']}",
        f"mood={user_prefs['mood']}",
        f"energy={user_prefs['energy']}",
        f"acoustic={user_prefs.get('likes_acoustic', False)}",
    ]
    if "preferred_decade" in user_prefs:
        prefs_parts.append(f"era={user_prefs['preferred_decade']}")
    if user_prefs.get("preferred_mood_tags"):
        prefs_parts.append(f"tags=[{', '.join(user_prefs['preferred_mood_tags'])}]")
    for flag in ("likes_clean", "likes_instrumental", "likes_live"):
        if user_prefs.get(flag):
            prefs_parts.append(f"{flag}=True")
    print("  " + "  |  ".join(prefs_parts))
    print(f"{'=' * 100}")

    rows = []
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        rows.append([
            f"#{rank}",
            song["title"],
            song["artist"],
            song["genre"],
            f"{score:.1f}",
            song.get("popularity", "N/A"),
            song.get("release_decade", "N/A"),
            song.get("mood_tags", "").replace("|", ", "),
            explanation,
        ])

    headers = ["#", "Title", "Artist", "Genre", "Score", "Pop", "Era", "Tags", "Reasons"]
    print(
        tabulate(
            rows,
            headers=headers,
            tablefmt="grid",
            maxcolwidths=[None, 22, 18, 14, None, None, None, 24, 52],
        )
    )


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    profiles = [
        (
            "High-Energy Pop",
            {"genre": "pop", "mood": "happy", "energy": 0.9, "likes_acoustic": False},
        ),
        (
            "Chill Lofi Study Session",
            {"genre": "lofi", "mood": "chill", "energy": 0.35, "likes_acoustic": True},
        ),
        (
            "Intense Rock Workout",
            {"genre": "rock", "mood": "intense", "energy": 0.95, "likes_acoustic": False},
        ),
        (
            "EDGE: Ghost Genre (country not in catalog)",
            {"genre": "country", "mood": "happy", "energy": 0.7, "likes_acoustic": False},
        ),
        (
            "EDGE: Mood not in catalog (sad)",
            {"genre": "pop", "mood": "sad", "energy": 0.8, "likes_acoustic": False},
        ),
        (
            "EDGE: Contradictory (high energy + loves acoustic)",
            {"genre": "folk", "mood": "melancholic", "energy": 0.9, "likes_acoustic": True},
        ),
        (
            "Nostalgic 90s Listener (no explicit)",
            {
                "genre": "alternative rock",
                "mood": "melancholic",
                "energy": 0.55,
                "likes_acoustic": False,
                "preferred_decade": 1990,
                "preferred_mood_tags": ["nostalgic", "melancholic", "raw"],
                "likes_clean": True,
                "likes_instrumental": False,
                "likes_live": False,
            },
        ),
        (
            "Dreamy Instrumental Lofi Fan",
            {
                "genre": "lofi",
                "mood": "chill",
                "energy": 0.38,
                "likes_acoustic": True,
                "preferred_decade": 2020,
                "preferred_mood_tags": ["dreamy", "peaceful", "focused"],
                "likes_clean": False,
                "likes_instrumental": True,
                "likes_live": False,
            },
        ),
        (
            "Live Classic Rock Enthusiast",
            {
                "genre": "rock",
                "mood": "intense",
                "energy": 0.88,
                "likes_acoustic": False,
                "preferred_decade": 1960,
                "preferred_mood_tags": ["nostalgic", "psychedelic", "intense"],
                "likes_clean": False,
                "likes_instrumental": False,
                "likes_live": True,
            },
        ),
    ]

    for label, user_prefs in profiles:
        recommendations = recommend_songs(user_prefs, songs, k=5)
        print_recommendations(label, user_prefs, recommendations)


if __name__ == "__main__":
    main()
