"""
Ranking rule (Layer 2) and the similarity-driven playlist walk (Layer 3) of ALGORITHM.md.
"""
from typing import Dict, List, Tuple

from src.scoring import (
    fold_tempo,
    gauss,
    genre_match,
    genre_tier,
    mood_match,
    resolve_targets,
    score_components,
)

SCORE_BUCKET = 0.02


def rank_key(user: Dict, song: Dict, score: float, gates: int) -> Tuple:
    """
    Tie-break cascade, checked in order:
    gates passed, score bucket, genre tier, tempo distance, song id.
    """
    target_tempo, _ = resolve_targets(user)
    return (
        -gates,
        -round(score / SCORE_BUCKET),
        -genre_tier(user["favorite_genre"], song["genre"]),
        fold_tempo(song["tempo_bpm"], target_tempo),
        song["id"],
    )


def rank(user: Dict, songs: List[Dict]) -> List[Tuple[Dict, float, List[str]]]:
    """Scores every song and returns (song, score, reasons), best first. Nothing is dropped."""
    scored = []
    for song in songs:
        score, gates, reasons = score_components(user, song)
        scored.append((rank_key(user, song, score, gates), song, score, reasons))
    scored.sort(key=lambda item: item[0])
    return [(song, score, reasons) for _, song, score, reasons in scored]


def similarity(a: Dict, b: Dict) -> float:
    """Song-to-song similarity in [0, 1] over genre, mood, tempo and tone. Symmetric."""
    return (
        1.2 * genre_match(a["genre"], b["genre"])
        + 1.0 * mood_match(a["mood"], b["mood"])
        + 1.0 * gauss(fold_tempo(a["tempo_bpm"], b["tempo_bpm"]), 20)
        + 0.8 * gauss(abs(a["valence"] - b["valence"]), 0.25)
        + 0.6 * gauss(abs(a["energy"] - b["energy"]), 0.25)
        + 0.4 * gauss(abs(a["acousticness"] - b["acousticness"]), 0.30)
    ) / 5.0


def build_playlist(
    user: Dict,
    songs: List[Dict],
    seed: Dict,
    k: int = 5,
    alpha: float = 0.55,
    beta: float = 0.45,
    gamma: float = 0.08,
) -> List[Tuple[Dict, float, float]]:
    """
    Autoplay walk from `seed`. Each pick maximizes
        alpha * similarity(previous pick, c) + beta * taste score(c) - gamma * plays by c's artist
    over the top 3k songs by taste score, with ties broken by ranking order.
    Returns (song, similarity to the previous song, taste score) for each pick.
    """
    ranked = rank(user, songs)
    position = {song["id"]: i for i, (song, _, _) in enumerate(ranked)}
    taste = {song["id"]: score for song, score, _ in ranked}
    pool = [song for song, _, _ in ranked[: min(len(ranked), 3 * k)]]

    played_ids = {seed["id"]}
    played_artists = [seed["artist"]]
    current = seed
    playlist = []

    while len(playlist) < k:
        candidates = [song for song in pool if song["id"] not in played_ids]
        if not candidates:
            if len(pool) == len(ranked):
                break
            pool = [song for song, _, _ in ranked]
            continue

        def utility(song: Dict) -> float:
            return (
                alpha * similarity(current, song)
                + beta * taste[song["id"]]
                - gamma * played_artists.count(song["artist"])
            )

        pick = max(candidates, key=lambda song: (utility(song), -position[song["id"]]))
        playlist.append((pick, similarity(current, pick), taste[pick["id"]]))
        played_ids.add(pick["id"])
        played_artists.append(pick["artist"])
        current = pick

    return playlist
