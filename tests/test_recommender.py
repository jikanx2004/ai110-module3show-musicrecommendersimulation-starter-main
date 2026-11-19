from src.recommender import Song, UserProfile, Recommender

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# --- Algorithm tests (see ALGORITHM.md) ---
from pathlib import Path

from src.recommender import load_songs, recommend_songs, score_song
from src.ranking import build_playlist, rank, similarity
from src.scoring import fold_tempo, gauss, genre_match, route

CSV = str(Path(__file__).resolve().parents[1] / "data" / "songs.csv")
POP_FAN = {"favorite_genre": "pop", "favorite_mood": "happy",
           "target_energy": 0.8, "likes_acoustic": False}


def test_gauss_peaks_at_zero_and_decreases():
    assert gauss(0, 1) == 1.0
    assert gauss(1, 1) == gauss(-1, 1)
    assert gauss(1, 1) > gauss(2, 1)


def test_genre_match_partial_tokens():
    assert genre_match("pop", "pop") == 1.0
    assert genre_match("indie pop", "pop") == 0.5
    assert genre_match("k-pop", "pop") == 0.5


def test_tempo_folds_double_time():
    assert fold_tempo(80, 160) == 0
    assert fold_tempo(160, 80) == 0


def test_tree_routes_to_expected_branch():
    assert route(POP_FAN).name == "HYPE"
    assert route({**POP_FAN, "target_energy": 0.35, "likes_acoustic": True}).name == "MELLOW"
    assert route({**POP_FAN, "target_energy": 0.40}).name == "FOCUS"
    assert route({**POP_FAN, "target_energy": 0.55}).name == "BALANCED"


def test_gates_demote_but_never_remove():
    songs = load_songs(CSV)
    calm = [s for s in songs if s["energy"] < 0.5]
    assert len(rank(POP_FAN, calm)) == len(calm)


def test_scores_stay_between_zero_and_one():
    for song in load_songs(CSV):
        score, reasons = score_song(POP_FAN, song)
        assert 0 <= score <= 1
        assert len(reasons) == 3


def test_ranking_breaks_exact_ties_by_id():
    song = {"title": "x", "artist": "a", "genre": "pop", "mood": "happy", "energy": 0.8,
            "tempo_bpm": 120, "valence": 0.8, "danceability": 0.5, "acousticness": 0.2}
    ranked = rank(POP_FAN, [{**song, "id": 2}, {**song, "id": 1}])
    assert [s["id"] for s, _, _ in ranked] == [1, 2]


def test_similarity_is_symmetric():
    songs = load_songs(CSV)
    for a in songs:
        for b in songs:
            assert abs(similarity(a, b) - similarity(b, a)) < 1e-12


def test_playlist_has_k_unique_songs_without_seed():
    songs = load_songs(CSV)
    seed = songs[0]
    picks = [s["id"] for s, _, _ in build_playlist(POP_FAN, songs, seed, k=5)]
    assert len(picks) == 5
    assert len(set(picks)) == 5
    assert seed["id"] not in picks


def test_recommend_songs_matches_main_contract():
    for song, score, explanation in recommend_songs(POP_FAN, load_songs(CSV), k=3):
        assert isinstance(song, dict) and isinstance(score, float)
        assert isinstance(explanation, str) and explanation
