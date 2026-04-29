from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class Song:
    """Represents a song with audio attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    popularity: int = 50
    release_decade: int = 2010
    mood_tags: str = ""
    explicit: int = 0
    instrumentalness: float = 0.0
    liveness: float = 0.1

@dataclass
class UserProfile:
    """Represents a user's music taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    preferred_decade: int = 2010
    preferred_mood_tags: List[str] = field(default_factory=list)
    likes_clean: bool = False
    likes_instrumental: bool = False
    likes_live: bool = False

class Recommender:
    """
    O-OP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Generate recommended songs with diversity penalty applied."""
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
            "preferred_decade": user.preferred_decade,
            "preferred_mood_tags": user.preferred_mood_tags,
            "likes_clean": user.likes_clean,
            "likes_instrumental": user.likes_instrumental,
            "likes_live": user.likes_live,
        }

        def song_to_dict(s: Song) -> Dict:
            return {
                "id": s.id, "title": s.title, "artist": s.artist,
                "genre": s.genre, "mood": s.mood, "energy": s.energy,
                "tempo_bpm": s.tempo_bpm, "valence": s.valence,
                "danceability": s.danceability, "acousticness": s.acousticness,
                "popularity": s.popularity, "release_decade": s.release_decade,
                "mood_tags": s.mood_tags, "explicit": s.explicit,
                "instrumentalness": s.instrumentalness, "liveness": s.liveness,
            }

        scored = [
            (song_to_dict(s), sc, reasons)
            for s in self.songs
            for sc, reasons in [score_song(user_prefs, song_to_dict(s))]
        ]

        diverse = _greedy_diverse_select(scored, k)
        id_to_song = {s.id: s for s in self.songs}
        return [id_to_song[entry[0]["id"]] for entry in diverse]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain why a song was recommended."""
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
            "preferred_decade": user.preferred_decade,
            "preferred_mood_tags": user.preferred_mood_tags,
            "likes_clean": user.likes_clean,
            "likes_instrumental": user.likes_instrumental,
            "likes_live": user.likes_live,
        }
        song_dict = {
            "id": song.id,
            "title": song.title,
            "artist": song.artist,
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
            "tempo_bpm": song.tempo_bpm,
            "valence": song.valence,
            "danceability": song.danceability,
            "acousticness": song.acousticness,
            "popularity": song.popularity,
            "release_decade": song.release_decade,
            "mood_tags": song.mood_tags,
            "explicit": song.explicit,
            "instrumentalness": song.instrumentalness,
            "liveness": song.liveness,
        }
        _, reasons = score_song(user_prefs, song_dict)
        return ", ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """Load and parse songs from a CSV file."""
    import csv

    int_fields = {"id", "tempo_bpm", "popularity", "release_decade", "explicit"}
    float_fields = {"energy", "valence", "danceability", "acousticness", "instrumentalness", "liveness"}

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field_name in int_fields:
                if field_name in row:
                    row[field_name] = int(row[field_name])
            for field_name in float_fields:
                if field_name in row:
                    row[field_name] = float(row[field_name])
            songs.append(row)
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score a song based on user preferences."""
    score = 0.0
    reasons = []

    # Genre exact match: +15 pts
    if song["genre"] == user_prefs.get("genre", ""):
        score += 15
        reasons.append("genre match (+15)")

    # Mood exact match: +40 pts
    mood_match = song["mood"] == user_prefs.get("mood", "")
    if mood_match:
        score += 40
        reasons.append("mood match (+40)")

    # Energy proximity: 0–40 pts
    energy_pts = (1 - abs(user_prefs.get("energy", 0.5) - song["energy"])) * 40
    score += energy_pts
    reasons.append(f"energy proximity (+{energy_pts:.1f})")

    # Attribute bonuses: up to +10 pts (only stack when mood already matches)
    if user_prefs.get("likes_acoustic") and song["acousticness"] > 0.7:
        score += 5
        reasons.append("acoustic match (+5)")

    if mood_match and song["valence"] > 0.7:
        score += 3
        reasons.append("high valence bonus (+3)")

    if mood_match and song["danceability"] > 0.7:
        score += 2
        reasons.append("high danceability bonus (+2)")

    # Rule 1: Popularity bonus (up to +12 pts)
    popularity = int(song.get("popularity", 50))
    if popularity >= 90:
        score += 12
        reasons.append("mainstream hit (+12)")
    elif popularity >= 75:
        score += 6
        reasons.append("popular track (+6)")
    elif popularity >= 50:
        score += 2
        reasons.append("moderate popularity (+2)")

    # Rule 2: Release decade proximity bonus (up to +15 pts)
    decade_gap = abs(int(song.get("release_decade", 2010)) - user_prefs.get("preferred_decade", 2010))
    if decade_gap == 0:
        score += 15
        reasons.append("era match (+15)")
    elif decade_gap == 10:
        score += 8
        reasons.append("near-era match (+8)")
    elif decade_gap == 20:
        score += 3
        reasons.append("adjacent-era match (+3)")

    # Rule 3: Mood tags matching (up to +21 pts, +7 per matching tag)
    song_tags = set(song.get("mood_tags", "").split("|")) if song.get("mood_tags", "") else set()
    user_tags = set(user_prefs.get("preferred_mood_tags", []))
    matching = song_tags & user_tags
    tag_pts = min(len(matching) * 7, 21)
    if tag_pts > 0:
        score += tag_pts
        reasons.append(f"mood tag match x{len(matching)} (+{tag_pts})")

    # Rule 4: Explicit content penalty (-20 pts)
    if user_prefs.get("likes_clean", False) and int(song.get("explicit", 0)) == 1:
        score -= 20
        reasons.append("explicit content penalty (-20)")

    # Rule 5: Instrumentalness bonus (up to +8 pts)
    instrumentalness = float(song.get("instrumentalness", 0.0))
    if user_prefs.get("likes_instrumental", False) and instrumentalness > 0.7:
        score += 8
        reasons.append("instrumental match (+8)")
    elif not user_prefs.get("likes_instrumental", False) and instrumentalness < 0.2:
        score += 4
        reasons.append("vocal track bonus (+4)")

    # Rule 6: Liveness bonus (up to +10 pts)
    liveness = float(song.get("liveness", 0.1))
    if user_prefs.get("likes_live", False) and liveness > 0.4:
        score += 10
        reasons.append("live feel bonus (+10)")
    elif not user_prefs.get("likes_live", False) and liveness < 0.15:
        score += 3
        reasons.append("studio polish bonus (+3)")

    return score, reasons

def _greedy_diverse_select(
    scored: List[Tuple[Dict, float, List[str]]],
    k: int,
    artist_penalty: int = 15,
    genre_penalty: int = 8,
) -> List[Tuple[Dict, float, str]]:
    """
    Greedy re-ranking with diversity penalties.

    After each selection the remaining candidates are re-scored:
      - artist_penalty pts deducted for every repeat of an already-selected artist
      - genre_penalty  pts deducted for every repeat of an already-selected genre

    This prevents the top-k from being dominated by one artist or genre.
    """
    selected: List[Tuple[Dict, float, str]] = []
    selected_artists: Dict[str, int] = defaultdict(int)
    selected_genres: Dict[str, int] = defaultdict(int)
    remaining = list(scored)        

    while len(selected) < k and remaining:
        adjusted = []
        for song, base_score, reasons in remaining:
            penalty = (
                selected_artists[song["artist"]] * artist_penalty
                + selected_genres[song["genre"]] * genre_penalty
            )
            adj_reasons = list(reasons)
            if penalty > 0:
                adj_reasons.append(f"diversity penalty (-{penalty})")
            adjusted.append((song, base_score - penalty, adj_reasons))

        adjusted.sort(key=lambda x: x[1], reverse=True)
        best_song, best_score, best_reasons = adjusted[0]

        selected.append((best_song, best_score, ", ".join(best_reasons)))
        remaining = [t for t in remaining if t[0]["id"] != best_song["id"]]
        selected_artists[best_song["artist"]] += 1
        selected_genres[best_song["genre"]] += 1

    return selected


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    artist_penalty: int = 15,
    genre_penalty: int = 8,
) -> List[Tuple[Dict, float, str]]:
    """Rank and return top k recommended songs with diversity penalty applied."""
    scored = [
        (song, score, reasons)
        for song in songs
        for score, reasons in [score_song(user_prefs, song)]
    ]
    return _greedy_diverse_select(scored, k, artist_penalty, genre_penalty)
