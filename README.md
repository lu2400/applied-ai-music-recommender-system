# VibeFinder — Applied AI Music Recommender

## Original Project Summary

VibeFinder started as a rule-based music recommender that matched songs against user-selected preferences (genre, mood, energy) using a weighted scoring formula. Users could pick attributes through a CLI or web UI, and the system would return ranked songs with numerical explanations. The original system had no natural language understanding and no way to retrieve background knowledge before making decisions.

This version extends VibeFinder with **RAG (Retrieval-Augmented Generation)** and an **Agentic Workflow** so the AI can look up relevant music knowledge before parsing a request, observe its own confidence, and correct itself when results are poor.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
│            "chill lofi for late-night studying"                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MusicAgent  (src/agent.py)                  │
│                                                                 │
│  Step 1 ── RAG Retrieve ──────────────────────────────────────  │
│            query → rag_retriever.py                             │
│            ↓ top-3 documents from music_knowledge.json          │
│                                                                 │
│  Step 2 ── Parse Request ─────────────────────────────────────  │
│            RAG context + user text → LLM (Groq Llama 3.1)       │
│            or keyword fallback if API unavailable               │
│            ↓ structured prefs {genre, mood, energy, decade, …}  │
│                                                                 │
│  Step 3 ── Score Songs ───────────────────────────────────────  │
│            prefs + data/songs.csv → recommender.py              │
│            scoring: genre, mood, energy, decade, tags, …        │
│            diversity re-ranking (artist/genre penalties)        │
│            ↓ ranked list of (song, score, reasons)              │
│                                                                 │
│  Step 4 ── Self-Evaluate ─────────────────────────────────────  │
│            compute confidence (0.0–1.0)                         │
│            60% raw score + 20% genre match + 20% mood match     │
│                                                                 │
│  Step 5 ── Retry? ────────────────────────────────────────────  │
│            if confidence < 40%: drop mood constraint, re-score  │
│                                                                 │
│  Step 6 ── Explain ───────────────────────────────────────────  │
│            LLM generates 2-3 sentence summary, or template      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
             ┌──────────────────────────┐
             │  OUTPUT                  │
             │  • Ranked songs          │
             │  • Confidence score      │
             │  • Observable step log   │
             │  • Natural language      │
             │    explanation           │
             └──────────────────────────┘

Supporting Components
─────────────────────
data/songs.csv              18-song catalog (16 audio attributes each)
data/music_knowledge.json   18 genre/mood/era documents for RAG
src/recommender.py          Scoring engine + diversity re-ranking
src/rag_retriever.py        Keyword-overlap retriever for RAG
src/agent.py                MusicAgent — agentic pipeline
src/llm_client.py           Groq Llama 3.1 integration (parse + explain)
src/spotify_client.py       Spotify OAuth + playback + library access
app.py                      Streamlit web UI
tests/test_harness.py       Evaluation harness — 6 predefined scenarios
tests/test_recommender.py   Unit tests for core scoring logic
```

**Data flow:** User input → RAG retrieves context docs → LLM (with context) extracts structured preferences → scorer ranks songs → agent checks confidence → explanation returned.

**Human / testing checkpoints:**
- `tests/test_harness.py` runs 6 predefined scenarios and prints pass/fail + confidence scores
- `tests/test_recommender.py` runs automated unit tests via pytest
- The step log in `result["steps"]` lets a human inspect every intermediate decision the agent made

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/lu2400/applied-ai-music-recommender-system.git
cd applied-ai-music-recommender-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys (optional but recommended)
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
```
- **Groq** (free): https://console.groq.com — enables natural language input and LLM-generated explanations
- **Spotify** (optional): enables playback controls and importing your liked songs into recommendations
- Both are optional; the system runs in keyword-only mode if keys are absent

### 4. Run the web app
```bash
streamlit run app.py
```

### 5. Run the CLI demo
```bash
python -m src.main
```

### 6. Run the agent directly
```python
from src.agent import MusicAgent
from src.recommender import load_songs

songs = load_songs("data/songs.csv")
agent = MusicAgent(songs, llm_enabled=False)   # set True if GROQ_API_KEY is set
result = agent.run("chill lofi for late-night studying")

print(result["explanation"])
for song, score, _ in result["recommendations"]:
    print(f"  {song['title']} — score {score:.0f}")
```

### 7. Run the evaluation harness
```bash
python -m tests.test_harness
```

### 8. Run unit tests
```bash
pytest tests/test_recommender.py -v
```

---

## Sample Interactions

### Interaction 1 — Lofi study session
**Input:** `"chill lofi for late-night studying"`

**RAG retrieved:** Lofi / Lo-Fi Hip-Hop, Study / Focus / Coding Music, Jazz

**Extracted preferences:** `{genre: lofi, mood: chill, energy: 0.30, likes_instrumental: false}`

**Recommendations:**
| # | Song | Artist | Score | Confidence |
|---|------|--------|-------|------------|
| 1 | Library Rain | Paper Lanterns | 106 | 100% |
| 2 | Midnight Coding | LoRoom | 95 | — |
| 3 | Spacewalk Thoughts | Orbit Bloom | 92 | — |

**Explanation:** *"These selections match your request for calm, focused lofi music. Library Rain leads with a perfect lofi/chill profile and low energy (0.35), while Midnight Coding offers a similar mood. Spacewalk Thoughts adds ambient variety while keeping energy minimal."*

---

### Interaction 2 — High-energy workout
**Input:** `"pump up gym music, high energy"`

**RAG retrieved:** Workout / Gym / Exercise Music, Hip-Hop / Rap, Pop

**Extracted preferences:** `{genre: pop, mood: intense, energy: 0.85}`

**Recommendations:**
| # | Song | Artist | Score | Confidence |
|---|------|--------|-------|------------|
| 1 | Storm Runner | Voltline | 97 | 92% |
| 2 | Sicko Mode | Travis Scott | 89 | — |
| 3 | Gym Hero | Max Pulse | 85 | — |

**Explanation:** *"Storm Runner's rock/intense combination and high energy (0.91) makes it the top pick for a workout. Sicko Mode adds hip-hop intensity, while Gym Hero keeps the pop energy high. All three have energy levels above 0.85."*

---

### Interaction 3 — Low confidence triggers retry
**Input:** `"something sad and old-school"`

**Agent step log:**
```
[1_rag_retrieve]  Retrieved: Sad/Melancholic, 1990s Music, Folk/Acoustic
[2_parse]         mood=melancholic, genre=pop, energy=0.5, decade=1990
[3_score]         Top: Creep by Radiohead (score=89.8)
[4_evaluate]      Confidence=77% — acceptable
[5_retry]         Skipped
```

**Top result:** Creep (Radiohead) — alt rock / melancholic / 1990s

---

## Design Decisions

**Why keyword scoring instead of ML embeddings?**
The catalog has 18 songs, which is far too small for meaningful vector similarity. A weighted rule-based scorer is fully transparent (every point is explained) and handles small datasets reliably without needing training data.

**Why RAG instead of just giving the LLM the full catalog?**
The knowledge base contains general music knowledge (genre characteristics, mood guides, era context) that is stable and reusable across queries. Retrieving only the 2-3 most relevant documents keeps the prompt short and focused, which reduces hallucination risk and makes the LLM's job easier.

**Why a self-evaluation step instead of just returning the top-k?**
Without evaluation, the system has no way to detect when a query falls outside the catalog's coverage (e.g., no "country" genre, no "sad" mood). The confidence check allows the agent to transparently communicate uncertainty and attempt a fallback rather than silently returning irrelevant results.

**Why Groq (Llama 3.1) instead of a larger model?**
Llama 3.1 8B on Groq has sub-second response times and is free-tier accessible. For structured extraction tasks (parse JSON from a description) a smaller fast model outperforms a slow large model in practice.

**Trade-off: small catalog limits recall**
With only 18 songs across 11 genres, several user requests will have no catalog match (e.g., classical, R&B, country). The system handles this gracefully via the retry + confidence path, but the fundamental limitation is the dataset size.

---

## Testing Summary

The evaluation harness (`tests/test_harness.py`) runs 6 predefined scenarios:

| Scenario | Result | Confidence | Notes |
|----------|--------|------------|-------|
| Pop happy high energy | ✓ PASS | 87/100 | Levitating ranked #1 correctly |
| Lofi chill focused | ✓ PASS | 100/100 | Library Rain — perfect score |
| Rock intense workout | ✓ PASS | 82/100 | Storm Runner #1 |
| Jazz relaxed evening | ✓ PASS | 72/100 | Coffee Shop Stories #1 |
| Clean lyrics preference | ✓ PASS | — | No explicit song in top result |
| Diversity penalty | ✓ PASS | — | Top 3 all different artists |

**6 / 6 scenarios passed.**

**What worked well:** Genre + mood matching is highly accurate when the catalog contains a good match. The diversity re-ranking reliably prevents one artist dominating the top-5. Confidence scores correctly identify strong vs. weak matches.

**What struggled:** The system cannot partially match moods — "sad" returns 0 mood points because "sad" is not a catalog mood (only "melancholic" exists). Energy proximity sometimes surfaces high-energy songs for low-energy requests when the mood constraint is missing.

**What we learned:** Exact-match scoring is brittle for synonyms. A production system would map "sad" → "melancholic" using a synonym graph or embeddings. The current design trades completeness for transparency.

---

## Reflection

### Limitations and biases
- **Small catalog (18 songs):** Many valid requests fall outside coverage. The system cannot recommend classical, R&B, country, or electronic music that doesn't exist in `songs.csv`.
- **Energy filter bubble:** Energy proximity (0–40 pts) equals mood match weight (40 pts), so a song with the right energy but wrong mood can outscore a song with the right mood but slightly different energy.
- **Exact mood matching:** No synonym handling — "sad" ≠ "melancholic" to the scorer even though they mean similar things to a listener.
- **LLM dependency:** Natural language mode requires a Groq API key. Without it, keyword extraction can misidentify genre/mood for phrasing it doesn't recognize.

### Could this be misused?
Yes. A bad actor could: (1) tune preference scores to favor sponsored songs over organic recommendations; (2) use the Spotify OAuth flow to collect user library data without clear consent; (3) use the LLM to parse personal emotional states from requests and build psychological profiles. Mitigations: the scoring formula is open and visible in `recommender.py`, OAuth scopes are minimal (read-only), and no user data is stored beyond the session.

### What surprised us during testing
The diversity penalty works much better than expected — without it, Neon Echo (synthwave artist with two songs) would dominate the top-5 for any moody request. Adding a 15-point artist penalty and 8-point genre penalty per repeated selection completely fixed this with one line of logic.

The RAG retrieval also surfaced unexpected connections: a query for "rainy day music" retrieved the "Sad / Melancholic" and "Folk/Acoustic" documents, which correctly guided the LLM toward `melancholic` mood even though the user didn't use that word.

### Collaboration with AI during this project

**Helpful instance:** When designing the self-evaluation step, Claude suggested weighting confidence as 60% score + 20% genre match + 20% mood match rather than using score alone. This was a good call because a song can score 80 points on energy + popularity alone without matching genre or mood at all. The weighted formula correctly identifies those cases as low-confidence.

**Flawed instance:** Claude initially suggested using cosine similarity with sentence-transformer embeddings for the RAG retriever. This was overkill for a knowledge base of 18 documents and would have added a 400MB dependency. Keyword overlap scoring is simpler, faster, requires no model download, and works just as well at this scale. The suggestion was technically valid but not appropriate for the problem size.
