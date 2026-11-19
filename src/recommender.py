import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

from src.ranking import build_playlist, rank
from src.scoring import route, score_components

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
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

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    Tempo and valence targets are optional; when None they are derived from energy and mood.
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_tempo_bpm: Optional[float] = None
    target_valence: Optional[float] = None

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        by_id = {song.id: song for song in self.songs}
        ranked = rank(asdict(user), [asdict(song) for song in self.songs])
        return [by_id[song["id"]] for song, _, _ in ranked[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        prefs = asdict(user)
        score, gates, reasons = score_components(prefs, asdict(song))
        branch = route(prefs)
        gate_note = f", passes {gates}/{len(branch.gates)} gates" if branch.gates else ""
        return f"Score {score:.2f} ({branch.name} profile{gate_note}): " + "; ".join(reasons)

    def playlist(self, user: UserProfile, seed: Song, k: int = 5) -> List[Song]:
        """Autoplay queue starting after `seed`, each song chosen to follow the one before it."""
        by_id = {song.id: song for song in self.songs}
        walk = build_playlist(asdict(user), [asdict(song) for song in self.songs], asdict(seed), k)
        return [by_id[song["id"]] for song, _, _ in walk]

FLOAT_FIELDS = ("energy", "tempo_bpm", "valence", "danceability", "acousticness")

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["id"] = int(row["id"])
            for field in FLOAT_FIELDS:
                row[field] = float(row[field])
            songs.append(row)
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    score, _, reasons = score_components(user_prefs, song)
    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    return [(song, score, "; ".join(reasons)) for song, score, reasons in rank(user_prefs, songs)[:k]]
