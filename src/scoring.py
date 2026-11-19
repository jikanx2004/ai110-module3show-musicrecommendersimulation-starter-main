"""
Scoring rule (Layer 1 of ALGORITHM.md).

A gate tree picks a weight profile for the user, then each song gets a weighted
score in [0, 1] and a count of how many of that profile's gates it passes.
Users and songs are plain dicts (TASTE_PROFILE keys and songs.csv columns).
"""
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


def gauss(delta: float, sigma: float) -> float:
    """Gaussian closeness: 1.0 when delta is 0, falling smoothly toward 0."""
    return math.exp(-(delta ** 2) / (2 * sigma ** 2))


def fold_tempo(bpm_a: float, bpm_b: float) -> float:
    """Tempo distance that treats half-time and double-time as close (symmetric)."""
    return min(abs(bpm_a - bpm_b), abs(2 * bpm_a - bpm_b) / 2, abs(bpm_a - 2 * bpm_b) / 2)


def _symmetric(pairs: Dict[Tuple[str, str], float]) -> Dict[frozenset, float]:
    return {frozenset(pair): value for pair, value in pairs.items()}


ADJ_GENRE = _symmetric({
    ("lofi", "ambient"): 0.6,
    ("lofi", "jazz"): 0.4,
    ("pop", "synthwave"): 0.3,
    ("rock", "synthwave"): 0.3,
    ("k-pop", "j-pop"): 0.8,
    ("rock", "metal"): 0.6,
    ("rap", "r&b"): 0.5,
    ("americana", "jazz"): 0.3,
    ("ambient", "classical"): 0.5,
    ("reggae", "r&b"): 0.3,
})

ADJ_MOOD = _symmetric({
    ("chill", "relaxed"): 0.9,
    ("chill", "focused"): 0.7,
    ("relaxed", "focused"): 0.6,
    ("moody", "intense"): 0.5,
    ("moody", "chill"): 0.4,
    ("happy", "intense"): 0.4,
    ("happy", "chill"): 0.3,
    ("happy", "euphoric"): 0.8,
    ("happy", "playful"): 0.8,
    ("happy", "uplifting"): 0.8,
    ("euphoric", "uplifting"): 0.7,
    ("intense", "angry"): 0.6,
    ("intense", "triumphant"): 0.6,
    ("sad", "nostalgic"): 0.6,
    ("sad", "moody"): 0.5,
    ("peaceful", "relaxed"): 0.8,
    ("peaceful", "chill"): 0.7,
    ("romantic", "relaxed"): 0.5,
    ("nostalgic", "relaxed"): 0.4,
})
MOOD_FLOOR = 0.1

MOOD_VALENCE = {
    "happy": 0.85, "relaxed": 0.70, "focused": 0.60, "chill": 0.60,
    "intense": 0.45, "moody": 0.35, "euphoric": 0.90, "playful": 0.85,
    "uplifting": 0.80, "triumphant": 0.70, "romantic": 0.65, "peaceful": 0.55,
    "nostalgic": 0.50, "angry": 0.35, "sad": 0.25,
}
DEFAULT_VALENCE = 0.60


def _genre_tokens(genre: str) -> set:
    return {token for token in re.split(r"[\s\-]+", genre) if token}


def genre_match(a: str, b: str) -> float:
    """1.0 for the same genre, else the larger of token Jaccard and the adjacency table."""
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return 1.0
    tokens_a, tokens_b = _genre_tokens(a), _genre_tokens(b)
    union = tokens_a | tokens_b
    jaccard = len(tokens_a & tokens_b) / len(union) if union else 0.0
    return max(jaccard, ADJ_GENRE.get(frozenset((a, b)), 0.0))


def genre_tier(a: str, b: str) -> int:
    """2 = exact genre, 1 = related genre, 0 = unrelated."""
    match = genre_match(a, b)
    if match == 1.0:
        return 2
    return 1 if match > 0 else 0


def mood_match(a: str, b: str) -> float:
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return 1.0
    return ADJ_MOOD.get(frozenset((a, b)), MOOD_FLOOR)


@dataclass(frozen=True)
class Branch:
    """One leaf of the gate tree: feature weights plus gates as (song field, op, threshold)."""
    name: str
    weights: Dict[str, float]
    gates: Tuple[Tuple[str, str, float], ...]


HYPE = Branch(
    "HYPE",
    {"genre": 1.0, "mood": 1.2, "energy": 2.0, "tempo": 1.5, "valence": 0.8, "acoustic": 0.5},
    (("energy", ">=", 0.55), ("tempo_bpm", ">=", 100)),
)
MELLOW = Branch(
    "MELLOW",
    {"genre": 0.8, "mood": 1.5, "energy": 1.2, "tempo": 0.8, "valence": 1.0, "acoustic": 2.0},
    (("acousticness", ">=", 0.50), ("energy", "<=", 0.55)),
)
FOCUS = Branch(
    "FOCUS",
    {"genre": 0.8, "mood": 1.8, "energy": 1.2, "tempo": 1.0, "valence": 1.2, "acoustic": 0.6},
    (("energy", "<=", 0.60), ("tempo_bpm", "<=", 110)),
)
BALANCED = Branch(
    "BALANCED",
    {"genre": 1.5, "mood": 1.5, "energy": 1.0, "tempo": 0.8, "valence": 1.0, "acoustic": 0.6},
    (),
)


def route(user: Dict) -> Branch:
    """The gate tree: picks the branch from target energy, then acoustic preference."""
    energy = user["target_energy"]
    if energy >= 0.65:
        return HYPE
    if energy <= 0.45:
        return MELLOW if user.get("likes_acoustic", False) else FOCUS
    return BALANCED


def resolve_targets(user: Dict) -> Tuple[float, float]:
    """Target tempo and valence, derived from energy and mood when the profile leaves them out."""
    tempo = user.get("target_tempo_bpm")
    if tempo is None:
        tempo = 60 + 100 * user["target_energy"]
    valence = user.get("target_valence")
    if valence is None:
        valence = MOOD_VALENCE.get(user["favorite_mood"].lower(), DEFAULT_VALENCE)
    return tempo, valence


def feature_matches(user: Dict, song: Dict) -> Dict[str, float]:
    """The six feature-match values, each in [0, 1]."""
    target_tempo, target_valence = resolve_targets(user)
    if user.get("likes_acoustic", False):
        acoustic = song["acousticness"]
    else:
        acoustic = 1 - song["acousticness"]
    return {
        "genre": genre_match(user["favorite_genre"], song["genre"]),
        "mood": mood_match(user["favorite_mood"], song["mood"]),
        "energy": gauss(abs(song["energy"] - user["target_energy"]), 0.25),
        "tempo": gauss(fold_tempo(song["tempo_bpm"], target_tempo), 20),
        "valence": gauss(abs(song["valence"] - target_valence), 0.30),
        "acoustic": acoustic,
    }


def gates_passed(branch: Branch, song: Dict) -> int:
    passed = 0
    for field, op, threshold in branch.gates:
        value = song[field]
        if (op == ">=" and value >= threshold) or (op == "<=" and value <= threshold):
            passed += 1
    return passed


def _reason(feature: str, match: float, user: Dict, song: Dict) -> str:
    """One plain-language sentence for a feature's contribution."""
    target_tempo, target_valence = resolve_targets(user)
    if feature == "genre":
        if match == 1.0:
            return f"same genre ({song['genre']})"
        if match > 0:
            return f"{song['genre']} is related to {user['favorite_genre']}"
        return f"different genre ({song['genre']})"
    if feature == "mood":
        if match == 1.0:
            return f"same mood ({song['mood']})"
        if match >= 0.5:
            return f"{song['mood']} mood is close to {user['favorite_mood']}"
        return f"{song['mood']} mood differs from {user['favorite_mood']}"
    if feature == "energy":
        return f"energy {song['energy']:.2f} vs your {user['target_energy']:.2f}"
    if feature == "tempo":
        folded = fold_tempo(song["tempo_bpm"], target_tempo) < abs(song["tempo_bpm"] - target_tempo)
        note = " (counting half/double time)" if folded else ""
        return f"{song['tempo_bpm']:.0f} BPM vs your {target_tempo:.0f} BPM{note}"
    if feature == "valence":
        return f"brightness {song['valence']:.2f} vs your {target_valence:.2f}"
    if user.get("likes_acoustic", False):
        return f"acoustic sound (acousticness {song['acousticness']:.2f})"
    return f"produced, non-acoustic sound (acousticness {song['acousticness']:.2f})"


def score_components(user: Dict, song: Dict) -> Tuple[float, int, List[str]]:
    """
    Returns (score in [0, 1], gates passed, top-3 reasons).
    score = sum(w_i * f_i) / sum(w_i), with weights from the user's branch.
    """
    branch = route(user)
    matches = feature_matches(user, song)
    score = sum(branch.weights[f] * m for f, m in matches.items()) / sum(branch.weights.values())
    top = sorted(matches, key=lambda f: branch.weights[f] * matches[f], reverse=True)[:3]
    reasons = [_reason(f, matches[f], user, song) for f in top]
    return score, gates_passed(branch, song), reasons
