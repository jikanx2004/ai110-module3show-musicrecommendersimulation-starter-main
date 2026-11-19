"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from pathlib import Path

from src.ranking import build_playlist
from src.recommender import load_songs, recommend_songs

SONGS_CSV = Path(__file__).resolve().parents[1] / "data" / "songs.csv"


# Taste profile: the target value for each song feature the scoring rule compares against.
TASTE_PROFILE = {
    "favorite_genre": "pop",      # exact match = 1.0, partial ("indie pop", "k-pop") = 0.5
    "favorite_mood": "happy",     # close moods (euphoric, playful, uplifting) get partial credit
    "target_energy": 0.80,        # 0.0 = calm, 1.0 = intense; also picks the scoring branch
    "target_tempo_bpm": 124,      # half-time and double-time count as close
    "target_valence": 0.82,       # emotional brightness: 0.0 = dark, 1.0 = bright
    "likes_acoustic": False,      # False = prefer electronic / produced sound
}

# Contrast profiles for testing different branches of the scoring tree.
CHILL_ACOUSTIC = {
    "favorite_genre": "lofi",
    "favorite_mood": "chill",
    "target_energy": 0.35,
    "likes_acoustic": True,
}

LATE_NIGHT_FOCUS = {
    "favorite_genre": "synthwave",
    "favorite_mood": "focused",
    "target_energy": 0.40,
    "likes_acoustic": False,
}


def main() -> None:
    songs = load_songs(str(SONGS_CSV))

    user_prefs = TASTE_PROFILE

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()

    seed = next(song for song in songs if song["title"] == "Gym Hero")
    print(f"Autoplay after {seed['title']}:\n")
    for i, (song, sim, taste) in enumerate(build_playlist(user_prefs, songs, seed, k=5), 1):
        print(f"{i}. {song['title']} ({song['genre']}) - similarity {sim:.2f}, taste {taste:.2f}")


if __name__ == "__main__":
    main()
