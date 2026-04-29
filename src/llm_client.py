import json
import os

from groq import Groq

_MODEL = "llama-3.1-8b-instant"


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env")
    return Groq(api_key=api_key)


def _generate(prompt: str) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def parse_user_request(
    text: str, all_genres: list, all_moods: list, context: str = ""
) -> dict:
    """Use Llama (via Groq) to extract structured music preferences from a natural language request.

    context: optional RAG-retrieved knowledge to inject before the extraction prompt.
    """
    context_block = f"\n\n{context}\n" if context else ""
    prompt = f"""Extract music preferences from this request: "{text}"{context_block}
Available genres (choose the single closest match): {all_genres}
Available moods (choose the single closest match): {all_moods}

Return ONLY a valid JSON object with these exact fields, no markdown, no explanation:
{{
  "genre": "<one of the available genres>",
  "mood": "<one of the available moods>",
  "energy": <float 0.0-1.0, where 0.0 = calm/ambient and 1.0 = intense/high-energy>,
  "preferred_decade": <one of 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020>,
  "likes_acoustic": <true or false>,
  "likes_instrumental": <true or false>
}}"""

    raw = _generate(prompt)
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def explain_recommendations(user_request: str, songs: list) -> str:
    """Generate a concise natural language explanation of why these songs were recommended."""
    song_lines = "\n".join(
        f"- {s['title']} by {s['artist']} "
        f"({s['genre']}, {s['mood']}, energy={float(s.get('energy', 0.5)):.1f})"
        for s in songs[:5]
    )
    prompt = f"""A user asked for music like: "{user_request}"

The recommender retrieved these songs from its catalog:
{song_lines}

In 2-3 sentences, explain why these songs match what the user wanted. \
Be specific about musical qualities (genre, mood, energy level, era). Be concise."""

    return _generate(prompt)
