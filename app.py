import html as _html
import json
import os
import pathlib
import secrets
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.recommender import load_songs, recommend_songs
from src.spotify_client import SpotifyClient
from src.llm_client import parse_user_request, explain_recommendations

st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* tighter card borders */
div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px;
}
/* shrink default image bottom margin inside cards */
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stImage"] {
    margin-bottom: 0;
}
</style>
""", unsafe_allow_html=True)


def _fmt_ms(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501")
SPOTIFY_ENABLED = bool(CLIENT_ID and CLIENT_SECRET)
LLM_ENABLED = bool(os.getenv("GROQ_API_KEY", ""))

_STATE_FILE = pathlib.Path(".oauth_state")


def _save_oauth_state(state: str) -> None:
    _STATE_FILE.write_text(json.dumps({"state": state, "ts": time.time()}))


def _load_oauth_state() -> str | None:
    if not _STATE_FILE.exists():
        return None
    try:
        data = json.loads(_STATE_FILE.read_text())
        if time.time() - data["ts"] < 600:
            return data["state"]
    except Exception:
        pass
    return None


def _clear_oauth_state() -> None:
    _STATE_FILE.unlink(missing_ok=True)


_GENRE_KEYWORDS = [
    ("lofi", "lofi"), ("lo-fi", "lofi"), ("lo fi", "lofi"),
    ("jazz", "jazz"), ("blues", "jazz"), ("soul", "jazz"), ("bossa", "jazz"),
    ("synthwave", "synthwave"), ("synth pop", "synthwave"), ("vapor", "synthwave"),
    ("darkwave", "synthwave"), ("chillwave", "synthwave"),
    ("ambient", "ambient"), ("new age", "ambient"), ("drone", "ambient"),
    ("classical", "ambient"), ("orchestral", "ambient"),
    ("metal", "rock"), ("punk", "rock"), ("grunge", "rock"),
    ("alternative", "rock"), ("indie rock", "rock"), ("hard rock", "rock"),
    ("rock", "rock"),
    ("folk", "folk"), ("acoustic", "folk"), ("singer-songwriter", "folk"),
    ("hip hop", "pop"), ("trap", "pop"), ("r&b", "pop"), ("soul", "pop"),
    ("dance", "pop"), ("edm", "pop"), ("electro", "pop"),
    ("indie", "pop"), ("pop", "pop"),
]


def _map_spotify_genre(spotify_genres: list) -> str:
    for genre_str in spotify_genres:
        g = genre_str.lower()
        for keyword, mapped in _GENRE_KEYWORDS:
            if keyword in g:
                return mapped
    return "pop"


def _spotify_track_to_dict(track: dict, artist_genres: list, uid: str) -> dict:
    """Convert a Spotify track object into the song dict format the recommender expects."""
    release_date = track.get("album", {}).get("release_date", "2010") or "2010"
    decade = (int(release_date[:4]) // 10) * 10
    artists = track.get("artists", [])
    return {
        "id": uid,
        "title": track["name"],
        "artist": ", ".join(a["name"] for a in artists),
        "genre": _map_spotify_genre(artist_genres),
        # Audio features were deprecated by Spotify in Nov 2024 — use neutral defaults.
        "mood": "",
        "energy": 0.5,
        "tempo_bpm": 120,
        "valence": 0.5,
        "danceability": 0.5,
        "acousticness": 0.5,
        "popularity": track.get("popularity", 50),
        "release_decade": decade,
        "mood_tags": "",
        "explicit": 1 if track.get("explicit") else 0,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "_source": "spotify",
        "_album_art": (track.get("album", {}).get("images") or [{}])[0].get("url", ""),
    }


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    if "spotify" not in st.session_state:
        st.session_state.spotify = (
            SpotifyClient(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)
            if SPOTIFY_ENABLED
            else None
        )
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "track_cache" not in st.session_state:
        st.session_state.track_cache: dict = {}
    if "spotify_library" not in st.session_state:
        # list of song dicts fetched from the user's Spotify library
        st.session_state.spotify_library: list = []
    if "library_loaded" not in st.session_state:
        st.session_state.library_loaded = False
    if "ai_prefs" not in st.session_state:
        st.session_state.ai_prefs = {}
    if "ai_query" not in st.session_state:
        st.session_state.ai_query = ""
    if "ai_explanation" not in st.session_state:
        st.session_state.ai_explanation = ""


_init_state()

@st.cache_data
def _load_catalogue():
    songs = load_songs("data/songs.csv")
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})
    tags = sorted(
        {tag for s in songs for tag in s.get("mood_tags", "").split("|") if tag}
    )
    return songs, genres, moods, tags


songs, ALL_GENRES, ALL_MOODS, ALL_TAGS = _load_catalogue()

params = st.query_params
if "code" in params and "state" in params:
    code = params["code"]
    returned_state = params["state"]

    expected = _load_oauth_state()
    _clear_oauth_state()

    sp_cb: SpotifyClient | None = st.session_state.spotify

    if sp_cb is None:
        st.query_params.clear()
        st.error("Spotify is not configured. Add credentials to .env and restart.")
    elif not expected:
        st.query_params.clear()
        st.warning("Auth session expired — please try connecting again.")
    elif returned_state != expected:
        st.query_params.clear()
        st.warning("State mismatch during OAuth — please try connecting again.")
    else:
        ok = sp_cb.exchange_code(code)
        st.query_params.clear()
        if ok:
            st.rerun()
        else:
            st.error(
                "Token exchange failed. Verify that your **Client Secret** and "
                "**Redirect URI** in the Spotify Dashboard exactly match your `.env`."
            )

# SIDEBAR
_DECADES = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

with st.sidebar:
    st.title("🎵 Your Preferences")

    if LLM_ENABLED:
        with st.expander("🤖 Describe what you want (AI)", expanded=bool(st.session_state.ai_prefs)):
            ai_input = st.text_input(
                "What are you in the mood for?",
                placeholder="e.g. something chill for late night studying",
                label_visibility="collapsed",
            )
            if st.button("Extract preferences with AI", use_container_width=True, disabled=not ai_input):
                with st.spinner("Thinking…"):
                    try:
                        prefs = parse_user_request(ai_input, ALL_GENRES, ALL_MOODS)
                        st.session_state.ai_prefs = prefs
                        st.session_state.ai_query = ai_input
                        st.session_state.ai_explanation = ""
                        st.rerun()
                    except Exception as exc:
                        st.error(f"AI error: {exc}")
            if st.session_state.ai_prefs:
                st.caption("✓ Preferences loaded from AI — adjust below if needed")
                if st.button("Clear AI preferences", use_container_width=True):
                    st.session_state.ai_prefs = {}
                    st.session_state.ai_query = ""
                    st.session_state.ai_explanation = ""
                    st.rerun()
        st.divider()

    _ai = st.session_state.ai_prefs
    _genre_default = _ai.get("genre", ALL_GENRES[0])
    _genre_idx = ALL_GENRES.index(_genre_default) if _genre_default in ALL_GENRES else 0
    _mood_default = _ai.get("mood", ALL_MOODS[0])
    _mood_idx = ALL_MOODS.index(_mood_default) if _mood_default in ALL_MOODS else 0
    _energy_default = float(_ai.get("energy", 0.7))
    _decade_default = _ai.get("preferred_decade", 2010)
    _decade_idx = _DECADES.index(_decade_default) if _decade_default in _DECADES else 6

    genre = st.selectbox("Genre", ALL_GENRES, index=_genre_idx)
    mood = st.selectbox("Mood", ALL_MOODS, index=_mood_idx)
    energy = st.slider(
        "Energy", 0.0, 1.0, _energy_default, step=0.05,
        help="0 = ambient/chill · 1 = intense/high-energy",
    )

    preferred_decade = st.selectbox(
        "Preferred Era",
        _DECADES,
        index=_decade_idx,
    )
    preferred_tags = st.multiselect("Mood Tags", ALL_TAGS)

    col_a, col_b = st.columns(2)
    with col_a:
        likes_acoustic = st.checkbox("Acoustic", value=bool(_ai.get("likes_acoustic", False)))
        likes_clean = st.checkbox("Clean lyrics")
    with col_b:
        likes_instrumental = st.checkbox("Instrumental", value=bool(_ai.get("likes_instrumental", False)))
        likes_live = st.checkbox("Live feel")

    num_recs = st.slider("How many?", 1, 10, 5)

    use_library = (
        SPOTIFY_ENABLED
        and st.session_state.spotify is not None
        and st.session_state.spotify.is_authenticated()
        and st.checkbox(
            "Include my Spotify library",
            help="Mix your liked songs into the recommendation pool",
        )
    )

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        catalog = list(songs)  # start with local CSV songs
        if use_library:
            if not st.session_state.library_loaded:
                with st.spinner("Loading your Spotify library…"):
                    _sp = st.session_state.spotify
                    raw_tracks = _sp.get_saved_tracks(limit=100)
                    if not raw_tracks:
                        st.warning(
                            "Could not load your library. "
                            "Please **Disconnect** and reconnect Spotify so the "
                            "new `user-library-read` permission is granted."
                        )
                    else:
                        # Batch-fetch artist genres (one call per 50 artists)
                        artist_ids = list({
                            a["id"]
                            for t in raw_tracks
                            for a in t.get("artists", [])
                            if a.get("id")
                        })
                        artists_by_id = {
                            a["id"]: a
                            for a in _sp.get_artists(artist_ids)
                            if a
                        }
                        lib = []
                        for idx, track in enumerate(raw_tracks):
                            genres = [
                                g
                                for a in track.get("artists", [])
                                for g in artists_by_id.get(a["id"], {}).get("genres", [])
                            ]
                            lib.append(
                                _spotify_track_to_dict(track, genres, f"sp_{idx}")
                            )
                        st.session_state.spotify_library = lib
                        st.session_state.library_loaded = True

            catalog = list(songs) + st.session_state.spotify_library

        st.session_state.recommendations = recommend_songs(
            {
                "genre": genre,
                "mood": mood,
                "energy": energy,
                "likes_acoustic": likes_acoustic,
                "preferred_decade": preferred_decade,
                "preferred_mood_tags": preferred_tags,
                "likes_clean": likes_clean,
                "likes_instrumental": likes_instrumental,
                "likes_live": likes_live,
            },
            catalog,
            k=num_recs,
        )
        # Generate AI explanation if an AI query was used
        if LLM_ENABLED and st.session_state.ai_query and st.session_state.recommendations:
            rec_songs = [r[0] for r in st.session_state.recommendations]
            with st.spinner("Generating AI summary…"):
                try:
                    st.session_state.ai_explanation = explain_recommendations(
                        st.session_state.ai_query, rec_songs
                    )
                except Exception:
                    st.session_state.ai_explanation = ""

    st.divider()
    st.subheader("Spotify")

    if not SPOTIFY_ENABLED:
        st.info("Add credentials to `.env` to enable Spotify features.")
    else:
        sp: SpotifyClient = st.session_state.spotify
        if sp.is_authenticated():
            st.success("Connected", icon="✅")
            if st.button("Disconnect", use_container_width=True):
                sp.logout()
                st.session_state.track_cache.clear()
                st.session_state.spotify_library.clear()
                st.session_state.library_loaded = False
                st.rerun()
        else:
            # Write state to disk on every render so the fresh session that
            # Spotify redirects back to can still verify it.
            state = secrets.token_urlsafe(16)
            _save_oauth_state(state)
            auth_url = sp.get_auth_url(state)
            st.markdown(
                f'<a href="{auth_url}" target="_self" style="'
                "display:block;text-align:center;background-color:#1db954;"
                "color:#fff;padding:0.5rem 1rem;border-radius:0.375rem;"
                'text-decoration:none;font-weight:600;font-size:0.875rem;">'
                "Connect to Spotify</a>",
                unsafe_allow_html=True,
            )

# MAIN — Now Playing
st.title("🎵 Music Recommender")

sp: SpotifyClient | None = st.session_state.spotify

if SPOTIFY_ENABLED and sp and sp.is_authenticated():
    playback = sp.get_current_playback()

    if playback and playback.get("item"):
        track      = playback["item"]
        is_playing = playback.get("is_playing", False)
        images     = track.get("album", {}).get("images", [])
        art_url    = images[0]["url"] if images else ""
        dur_ms     = max(track.get("duration_ms", 1), 1)
        prog_ms    = playback.get("progress_ms", 0)
        pct        = int(prog_ms / dur_ms * 100)

        # Extract and escape all strings before the f-string to avoid
        # the {{}} inside an expression being parsed as set({}) by Python.
        track_name  = _html.escape(track["name"])
        artists_str = _html.escape(", ".join(a["name"] for a in track["artists"]))
        album_name  = _html.escape((track.get("album") or {}).get("name", ""))
        status_dot   = "▶" if is_playing else "⏸"
        status_color = "#1db954" if is_playing else "#888"
        status_label = "Now Playing" if is_playing else "Paused"

        art_tag = (
            f'<img src="{art_url}" style="width:160px;height:160px;'
            'border-radius:10px;object-fit:cover;'
            'box-shadow:0 8px 32px rgba(0,0,0,0.5);flex-shrink:0;">'
            if art_url else
            '<div style="width:160px;height:160px;border-radius:10px;'
            'background:#2a2a2a;display:flex;align-items:center;'
            'justify-content:center;font-size:52px;flex-shrink:0;">🎵</div>'
        )

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
                    border-radius:16px;padding:24px 28px;
                    display:flex;align-items:center;gap:24px;">
          {art_tag}
          <div style="flex:1;min-width:0;">
            <div style="color:{status_color};font-size:10px;text-transform:uppercase;
                        letter-spacing:2px;font-weight:700;margin-bottom:8px;">
              {status_dot}&nbsp; {status_label}
            </div>
            <div style="color:#fff;font-size:22px;font-weight:700;line-height:1.2;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                        margin-bottom:4px;">
              {track_name}
            </div>
            <div style="color:#b3b3b3;font-size:14px;margin-bottom:2px;">
              {artists_str}
            </div>
            <div style="color:#535353;font-size:12px;margin-bottom:20px;">
              {album_name}
            </div>
            <div style="display:flex;align-items:center;gap:10px;">
              <span style="color:#b3b3b3;font-size:11px;font-variant-numeric:tabular-nums;
                           min-width:32px;">{_fmt_ms(prog_ms)}</span>
              <div style="flex:1;background:#404040;border-radius:3px;height:4px;">
                <div style="background:#1db954;height:4px;border-radius:3px;
                            width:{pct}%;"></div>
              </div>
              <span style="color:#b3b3b3;font-size:11px;font-variant-numeric:tabular-nums;
                           min-width:32px;text-align:right;">{_fmt_ms(dur_ms)}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Controls row — centred with spacers
        _, prev_c, play_c, next_c, ref_c, _ = st.columns([3, 1, 1, 1, 1, 3])
        with prev_c:
            if st.button("⏮", use_container_width=True, help="Previous"):
                sp.skip_previous(); st.rerun()
        with play_c:
            if st.button("⏸" if is_playing else "▶",
                         use_container_width=True,
                         help="Pause" if is_playing else "Play"):
                sp.pause() if is_playing else sp.play(); st.rerun()
        with next_c:
            if st.button("⏭", use_container_width=True, help="Next"):
                sp.skip_next(); st.rerun()
        with ref_c:
            if st.button("↻", use_container_width=True, help="Refresh"):
                st.rerun()

    elif playback is not None:
        st.info("Nothing playing. Start a track in Spotify then hit ↻.")
        if st.button("↻ Refresh"):
            st.rerun()
    else:
        st.warning("No active Spotify device found. Open Spotify on any device.")

    st.divider()

elif SPOTIFY_ENABLED:
    st.info("Connect Spotify in the sidebar to enable playback controls and album art.")
    st.divider()


st.subheader("Recommendations")

if not st.session_state.recommendations:
    st.markdown(
        "Set your preferences in the sidebar and click **Get Recommendations**."
    )
else:
    recs = st.session_state.recommendations

    if st.session_state.ai_explanation:
        st.info(f"**AI Summary:** {st.session_state.ai_explanation}")

    # For CSV songs, search Spotify for album art.
    # For library songs, art is already embedded in the song dict.
    if sp and sp.is_authenticated():
        csv_missing = [
            sd
            for sd, _, _ in recs
            if sd.get("_source") != "spotify"
            and sd["id"] not in st.session_state.track_cache
        ]
        if csv_missing:
            with st.spinner(f"Fetching album art for {len(csv_missing)} song(s)…"):
                for sd in csv_missing:
                    st.session_state.track_cache[sd["id"]] = sp.search_track(
                        sd["title"], sd["artist"]
                    )

    COLS = 3
    for row in [recs[i : i + COLS] for i in range(0, len(recs), COLS)]:
        cols = st.columns(COLS)
        for col, (song_dict, score, explanation) in zip(cols, row):
            from_spotify = song_dict.get("_source") == "spotify"

            # Resolve album art URL
            art_url = None
            if from_spotify:
                art_url = song_dict.get("_album_art") or None
            else:
                track_info = st.session_state.track_cache.get(song_dict["id"])
                if track_info:
                    imgs = track_info.get("album", {}).get("images", [])
                    if imgs:
                        # pick the smallest available (last in list) for thumbnail
                        art_url = imgs[-1]["url"] if len(imgs) >= 3 else imgs[0]["url"]

            with col:
                with st.container(border=True):
                    thumb_col, info_col = st.columns([1, 3])

                    with thumb_col:
                        if art_url:
                            st.image(art_url, width=64)
                        else:
                            st.markdown(
                                '<div style="width:64px;height:64px;border-radius:6px;'
                                'background:#f0f2f6;display:flex;align-items:center;'
                                'justify-content:center;font-size:26px;">🎵</div>',
                                unsafe_allow_html=True,
                            )

                    with info_col:
                        title = _html.escape(song_dict["title"])
                        artist = _html.escape(song_dict["artist"])
                        genre_label = "🎵 Spotify" if from_spotify else f"🎸 {song_dict['genre']}"
                        st.markdown(
                            f'<div style="font-weight:600;font-size:14px;line-height:1.3;'
                            f'margin-bottom:2px;">{title}</div>'
                            f'<div style="color:#666;font-size:12px;margin-bottom:6px;">{artist}</div>'
                            f'<div style="display:flex;gap:8px;align-items:center;">'
                            f'<span style="background:#f0f2f6;padding:2px 7px;border-radius:10px;'
                            f'font-size:11px;color:#444;">{genre_label}</span>'
                            f'<span style="color:#888;font-size:11px;">⭐ {score:.0f}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    with st.expander("Why this song?"):
                        if from_spotify:
                            st.caption(
                                "• From your Spotify library — scored on genre, "
                                "popularity, and era (audio features not available)"
                            )
                        for part in explanation.split(", "):
                            st.caption(f"• {part}")
