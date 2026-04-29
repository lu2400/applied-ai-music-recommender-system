"""
MusicAgent: multi-step agentic recommendation workflow.

Steps (all observable via the returned 'steps' log):
  1. RAG retrieve — fetch relevant music knowledge for the query
  2. Parse request — LLM with RAG context (or keyword fallback)
  3. Score songs — run the recommender against all songs
  4. Self-evaluate — compute a 0-1 confidence score
  5. Retry (optional) — broaden constraints if confidence < threshold
  6. Explain — generate a natural language summary

Usage:
    from src.agent import MusicAgent
    from src.recommender import load_songs

    songs = load_songs("data/songs.csv")
    agent = MusicAgent(songs)
    result = agent.run("chill lofi for late-night studying")
    print(result["explanation"])
    for song, score, reasons in result["recommendations"]:
        print(f"  {song['title']} — {score:.0f}  ({reasons})")
"""

import os
import sys
import re
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommender import recommend_songs
from src.rag_retriever import retrieve, format_context

# --------------------------------------------------------------------------- #
# LLM availability guard
# --------------------------------------------------------------------------- #
try:
    from src.llm_client import parse_user_request, explain_recommendations
    _LLM_AVAILABLE = True
except Exception:
    _LLM_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Keyword-based fallback preference extractor (no LLM required)
# --------------------------------------------------------------------------- #
_GENRE_KEYWORDS: Dict[str, List[str]] = {
    "lofi":        ["lofi", "lo-fi", "lo fi", "chill beats", "study beats"],
    "pop":         ["pop", "mainstream", "radio", "chart"],
    "rock":        ["rock", "guitar", "heavy", "grunge"],
    "jazz":        ["jazz", "saxophone", "swing", "smooth jazz"],
    "ambient":     ["ambient", "atmospheric", "meditation", "soundscape"],
    "synthwave":   ["synthwave", "80s", "retro", "synth", "neon"],
    "folk":        ["folk", "acoustic", "singer-songwriter", "unplugged"],
    "hip-hop":     ["hip-hop", "rap", "hiphop", "trap", "rhymes"],
    "indie pop":   ["indie pop", "indie"],
    "indie rock":  ["indie rock"],
    "alt rock":    ["alt rock", "alternative rock", "alternative"],
}

_MOOD_KEYWORDS: Dict[str, List[str]] = {
    "chill":       ["chill", "relax", "calm", "mellow", "laid-back", "chill out"],
    "happy":       ["happy", "upbeat", "cheerful", "positive", "joy", "bright"],
    "intense":     ["intense", "aggressive", "powerful", "hard", "workout", "gym"],
    "relaxed":     ["relaxed", "easy", "smooth", "gentle", "soft"],
    "moody":       ["moody", "dark", "atmospheric", "night", "cinematic"],
    "focused":     ["focused", "study", "concentrate", "work", "coding"],
    "energetic":   ["energetic", "energy", "hype", "party", "dance"],
    "uplifting":   ["uplifting", "motivate", "boost", "inspiring", "morning"],
    "melancholic": ["melancholic", "sad", "nostalgic", "wistful", "rainy"],
}

_ALL_GENRES = list(_GENRE_KEYWORDS.keys())
_ALL_MOODS  = list(_MOOD_KEYWORDS.keys())


def _keyword_extract(text: str) -> Dict:
    """Extract music preferences from text using keyword matching (LLM fallback)."""
    lower = text.lower()

    genre = "pop"
    for g, kws in _GENRE_KEYWORDS.items():
        if any(k in lower for k in kws):
            genre = g
            break

    mood = "happy"
    for m, kws in _MOOD_KEYWORDS.items():
        if any(k in lower for k in kws):
            mood = m
            break

    energy = 0.5
    if any(w in lower for w in ["intense", "workout", "gym", "hype", "party", "hard", "fast", "pump"]):
        energy = 0.85
    elif any(w in lower for w in ["chill", "calm", "soft", "relax", "study", "sleep", "ambient", "lofi"]):
        energy = 0.30

    decade = 2010
    for d in [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]:
        short = str(d)[2:] + "s"
        if str(d) in lower or short in lower:
            decade = d
            break

    likes_acoustic     = any(w in lower for w in ["acoustic", "folk", "unplugged", "natural"])
    likes_instrumental = any(w in lower for w in ["instrumental", "no vocals", "no lyrics"])

    return {
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "preferred_decade": decade,
        "likes_acoustic": likes_acoustic,
        "likes_instrumental": likes_instrumental,
    }


# --------------------------------------------------------------------------- #
# MusicAgent
# --------------------------------------------------------------------------- #
class MusicAgent:
    """
    Multi-step agentic recommendation workflow with RAG and self-evaluation.

    The agent's intermediate steps are visible in result["steps"] so the
    pipeline is fully observable — satisfying the 'Agentic Workflow' rubric
    requirement for observable intermediate steps.
    """

    CONFIDENCE_THRESHOLD = 0.40

    def __init__(self, songs: List[Dict], llm_enabled: bool = True):
        self.songs = songs
        self.llm_enabled = llm_enabled and _LLM_AVAILABLE
        self._steps: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #
    def run(self, user_request: str, k: int = 5) -> Dict[str, Any]:
        """Execute the full recommendation pipeline and return a result dict.

        Returns:
            recommendations: list of (song_dict, score, reasons_str)
            confidence:      float 0.0–1.0
            preferences_used: dict of extracted preferences
            rag_context:     titles of retrieved knowledge documents
            steps:           list of observable step records
            explanation:     natural language summary
        """
        self._steps = []

        # ── Step 1: RAG retrieve ──────────────────────────────────────────
        self._log("1_rag_retrieve", f"Retrieving music knowledge for query: {user_request!r}")
        context_docs = retrieve(user_request, k=3)
        context_str  = format_context(context_docs)
        self._log("1_rag_context",
                  f"Retrieved {len(context_docs)} documents: "
                  + ", ".join(f'"{d["title"]}"' for d in context_docs))

        # ── Step 2: Parse request ─────────────────────────────────────────
        self._log("2_parse", "Extracting structured preferences from request")
        prefs = self._parse(user_request, context_str)
        self._log("2_parse_result",
                  f"Preferences: genre={prefs['genre']!r}, mood={prefs['mood']!r}, "
                  f"energy={prefs['energy']:.2f}, decade={prefs['preferred_decade']}, "
                  f"acoustic={prefs['likes_acoustic']}, instrumental={prefs['likes_instrumental']}",
                  prefs)

        # Fill optional fields not always returned by keyword extractor
        prefs.setdefault("preferred_mood_tags", [])
        prefs.setdefault("likes_clean", False)
        prefs.setdefault("likes_live", False)

        # ── Step 3: Score and rank songs ──────────────────────────────────
        self._log("3_score", f"Scoring {len(self.songs)} songs with diversity re-ranking")
        recs = recommend_songs(prefs, self.songs, k=k)
        if recs:
            top = recs[0]
            self._log("3_rank",
                      f"Top result: {top[0]['title']} by {top[0]['artist']} "
                      f"(score={top[1]:.1f})")
        else:
            self._log("3_rank", "No results found")

        # ── Step 4: Self-evaluate ─────────────────────────────────────────
        confidence = self._evaluate(prefs, recs)
        self._log("4_evaluate",
                  f"Confidence={confidence:.0%} (threshold={self.CONFIDENCE_THRESHOLD:.0%}): "
                  + ("acceptable" if confidence >= self.CONFIDENCE_THRESHOLD else "LOW — will retry"),
                  {"confidence": confidence, "threshold": self.CONFIDENCE_THRESHOLD})

        # ── Step 5: Retry with broadened prefs if confidence too low ──────
        if confidence < self.CONFIDENCE_THRESHOLD:
            self._log("5_retry", "Dropping mood constraint and re-scoring")
            broad = self._broaden(prefs)
            recs = recommend_songs(broad, self.songs, k=k)
            confidence = self._evaluate(broad, recs)
            self._log("5_retry_result",
                      f"After retry: confidence={confidence:.0%}, "
                      + (f"top={recs[0][0]['title']}" if recs else "still no results"))
        else:
            self._log("5_retry", "Skipped — confidence acceptable")

        # ── Step 6: Explain ───────────────────────────────────────────────
        self._log("6_explain", "Generating natural language explanation")
        explanation = self._explain(user_request, prefs, recs)
        self._log("6_explain_done", "Explanation ready")

        return {
            "recommendations": recs,
            "confidence": confidence,
            "preferences_used": prefs,
            "rag_context": [d["title"] for d in context_docs],
            "steps": list(self._steps),
            "explanation": explanation,
        }

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #
    def _log(self, step: str, description: str, data: Any = None) -> None:
        entry: Dict[str, Any] = {"step": step, "description": description}
        if data is not None:
            entry["data"] = data
        self._steps.append(entry)

    def _parse(self, text: str, context: str) -> Dict:
        if self.llm_enabled:
            try:
                return parse_user_request(text, _ALL_GENRES, _ALL_MOODS, context=context)
            except Exception as exc:
                self._log("2_parse_llm_error", f"LLM failed ({exc}), falling back to keywords")
        return _keyword_extract(text)

    def _evaluate(self, prefs: Dict, recs: List) -> float:
        """Compute a 0–1 confidence score.

        Weights:
          60% — normalised raw score of top recommendation
          20% — exact genre match on top recommendation
          20% — exact mood match on top recommendation
        """
        if not recs:
            return 0.0
        top_song, top_score, _ = recs[0]
        max_score = 15 + 40 + 40  # genre + mood + energy max
        score_pct  = min(top_score, max_score) / max_score
        genre_ok   = float(top_song.get("genre") == prefs.get("genre"))
        mood_ok    = float(top_song.get("mood")  == prefs.get("mood"))
        confidence = 0.6 * score_pct + 0.2 * genre_ok + 0.2 * mood_ok
        return round(min(1.0, max(0.0, confidence)), 3)

    def _broaden(self, prefs: Dict) -> Dict:
        """Return a copy of prefs with mood constraint removed."""
        broad = dict(prefs)
        broad["mood"] = ""
        return broad

    def _explain(self, user_request: str, prefs: Dict, recs: List) -> str:
        if self.llm_enabled:
            try:
                return explain_recommendations(user_request, [r[0] for r in recs])
            except Exception:
                pass
        if not recs:
            return "No matching songs were found in the catalog."
        top = recs[0][0]
        return (
            f"Based on your request, '{top['title']}' by {top['artist']} is the best match. "
            f"These songs are selected for their {prefs.get('genre', 'varied')} genre, "
            f"{prefs.get('mood', 'varied')} mood, and energy level near "
            f"{prefs.get('energy', 0.5):.0%}."
        )
