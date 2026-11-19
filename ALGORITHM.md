# Algorithm Recipe: Scoring Rule, Ranking Rule, and Unified Next-Song Algorithm

The recommender has three layers. Each layer does one job:

1. **Scoring rule**: how well does one song fit the user's taste profile? This produces a candidate pool.
2. **Ranking rule**: given the scores, what is the final order? A decision-tree tie-break cascade.
3. **Unified algorithm**: the next song is chosen for *similarity to what just played* (tone, genre,
   BPM). It is picked **from the pool the scoring rule selected**, **ordered by the ranking rule**.

Both the scoring rule and the ranking rule are decision trees. A **gate tree** routes the user into a
weight profile, and a **cascade tree** orders the songs.

**Gates never remove songs.** Gates are the *first tier of the ranking rule*. Every song that passes
its gates ranks above every song that fails them, but nothing is dropped. This keeps
`tests/test_recommender.py` green (it asks for `k=2` from a 2-song list and expects 2 results), and a
real recommender should never return an empty page.

---

## The Taste Profile

Defined in `src/main.py`. It holds one target value per feature the scoring rule uses:

```python
TASTE_PROFILE = {
    "favorite_genre":   "pop",    # f_genre   (token Jaccard + adjacency)
    "favorite_mood":    "happy",  # f_mood    (mood adjacency table)
    "target_energy":    0.80,     # f_energy  (gauss, σ = 0.25); also picks the tree branch
    "target_tempo_bpm": 124,      # f_tempo   (octave-folded gauss, σ = 20 BPM)
    "target_valence":   0.82,     # f_valence (gauss, σ = 0.30), the "tone" target
    "likes_acoustic":   False,    # f_acoustic = 1 − acousticness
}
```

- Key names match the `UserProfile` dataclass, plus two optional fields (`target_tempo_bpm`,
  `target_valence`) to be added after its existing four. If either is missing, it is derived (see 1b).
- This profile routes to the **HYPE** branch (`0.80 ≥ 0.65`).
- `danceability` is left out on purpose because the scoring rule doesn't use it.

Contrast profiles: `CHILL_ACOUSTIC` (lofi / chill / 0.35 / acoustic) → **MELLOW**, and
`LATE_NIGHT_FOCUS` (synthwave / focused / 0.40 / not acoustic) → **FOCUS**.

---

## Layer 1: The Scoring Rule

### 1a. One shared math primitive

```
gauss(Δ, σ) = exp( −Δ² / (2σ²) )          → 1.0 at Δ=0, decays smoothly, always in (0, 1]
```

### 1b. Six feature-match functions, each in [0, 1]

| Feature | Formula | Constant |
|---|---|---|
| `f_genre` | `max( jaccard(tokens(a), tokens(b)), ADJ_GENRE[a][b] )` | — |
| `f_mood` | `ADJ_MOOD[a][b]` (symmetric, diagonal = 1.0, floor 0.1) | — |
| `f_energy` | `gauss(\|e_song − e_target\|, σ)` | σ = 0.25 |
| `f_tempo` | `gauss(fold(b_song, b_target), σ)` | σ = 20 BPM |
| `f_valence` | `gauss(\|v_song − v_target\|, σ)` | σ = 0.30 |
| `f_acoustic` | `acousticness` if `likes_acoustic` else `1 − acousticness` | — |

**Genre token Jaccard.** Tokens are split on spaces **and hyphens**:
`|{indie,pop} ∩ {pop}| / |{indie,pop} ∪ {pop}| = 1/2 = 0.5`. `"k-pop"` and `"j-pop"` also score 0.5
against `"pop"`. An exact match scores 1.0.

**`ADJ_GENRE`** (symmetric): `lofi↔ambient 0.6`, `lofi↔jazz 0.4`, `pop↔synthwave 0.3`,
`rock↔synthwave 0.3`, `k-pop↔j-pop 0.8`, `rock↔metal 0.6`, `rap↔r&b 0.5`, `americana↔jazz 0.3`,
`ambient↔classical 0.5`, `reggae↔r&b 0.3`. Every other pair is 0.0.

**`ADJ_MOOD`** (symmetric): `chill↔relaxed 0.9`, `chill↔focused 0.7`, `relaxed↔focused 0.6`,
`moody↔intense 0.5`, `moody↔chill 0.4`, `happy↔intense 0.4`, `happy↔chill 0.3`, `happy↔euphoric 0.8`,
`happy↔playful 0.8`, `happy↔uplifting 0.8`, `euphoric↔uplifting 0.7`, `intense↔angry 0.6`,
`intense↔triumphant 0.6`, `sad↔nostalgic 0.6`, `sad↔moody 0.5`, `peaceful↔relaxed 0.8`,
`peaceful↔chill 0.7`, `romantic↔relaxed 0.5`, `nostalgic↔relaxed 0.4`. Every other pair is 0.1.

**`MOOD_VALENCE`**: happy .85, relaxed .70, focused .60, chill .60, intense .45, moody .35,
euphoric .90, playful .85, uplifting .80, triumphant .70, romantic .65, peaceful .55, nostalgic .50,
angry .35, sad .25.

**Octave folding.** Half-time and double-time count as close in tempo:

```
fold(b_s, b_t) = min( |b_s − b_t|, |2·b_s − b_t|, |b_s/2 − b_t| )
```

**Derived targets.** These are used when the profile leaves them out:

```
target_tempo_bpm = 60 + 100 · target_energy        # 0.0→60, 1.0→160 BPM
target_valence   = MOOD_VALENCE[favorite_mood]
```

### 1c. The gate tree (selects weights + gates)

```
root: target_energy >= 0.65 ?
├─ yes → HYPE
│        w = {genre 1.0, mood 1.2, energy 2.0, tempo 1.5, valence 0.8, acoustic 0.5}
│        gates: energy >= 0.55 , tempo_bpm >= 100
└─ no  → target_energy <= 0.45 ?
         ├─ yes → likes_acoustic ?
         │        ├─ yes → MELLOW
         │        │        w = {genre 0.8, mood 1.5, energy 1.2, tempo 0.8, valence 1.0, acoustic 2.0}
         │        │        gates: acousticness >= 0.50 , energy <= 0.55
         │        └─ no  → FOCUS
         │                 w = {genre 0.8, mood 1.8, energy 1.2, tempo 1.0, valence 1.2, acoustic 0.6}
         │                 gates: energy <= 0.60 , tempo_bpm <= 110
         └─ no  → BALANCED
                  w = {genre 1.5, mood 1.5, energy 1.0, tempo 0.8, valence 1.0, acoustic 0.6}
                  gates: (none)
```

### 1d. The score

```
S(song | user) = Σ wᵢ · fᵢ  /  Σ wᵢ            ∈ [0, 1]
```

Dividing by the weight sum keeps `S` in `[0, 1]` for every branch, so scores stay comparable.
Scoring returns `(S, gates_passed, reasons)`, where `reasons` are the top 3 contributing terms
written as plain sentences. Those sentences become the explanation shown to the user.

---

## Layer 2: The Ranking Rule (tie-break cascade tree)

This is a deterministic total order. Each level is only checked when the level above it is tied:

```
1. gates_passed        (desc)   ← the gate tree's verdict; this is the tier
2. round(S / 0.02)     (desc)   ← score, quantized so near-ties fall through to musical tie-breaks
3. genre tier          (desc)   ← exact 2 > partial 1 > none 0
4. fold(Δbpm)          (asc)    ← closer tempo wins
5. song id             (asc)    ← determinism
```

The score is rounded in step 2 so that tiny float differences don't decide the order. Without the
rounding, steps 3–5 would never be used. Artist diversity is *not* part of this key, because it
depends on which songs were already picked. It is handled in Layer 3.

---

## Layer 3: The Unified Algorithm (similarity-driven playlist walk)

### 3a. Song-to-song similarity: tone, genre, BPM

```
Sim(a, b) = [ 1.2·f_genre(a,b)
            + 1.0·f_mood(a,b)
            + 1.0·gauss(fold(bpm_a, bpm_b), 20)
            + 0.8·gauss(|val_a − val_b|, 0.25)      ┐
            + 0.6·gauss(|eng_a − eng_b|, 0.25)      ├ "tone"
            + 0.4·gauss(|aco_a − aco_b|, 0.30) ] / 5.0    ┘
```

Every term is symmetric, so `Sim(a, b) == Sim(b, a)`.

### 3b. The walk

```
pool    = top-N songs by the SCORING RULE, N = min(len(songs), 3k)
played  = {seed}
current = seed

for step in 1..k:
    for each candidate c in pool \ played:
        U(c) = α·Sim(current, c) + β·S(c | user) − γ·artist_count(c.artist, played)
    pick = argmax U, ties broken by the LAYER 2 RANKING RULE
    append pick ; played.add(pick) ; current = pick      ← compares against the PREVIOUS pick, not the seed
```

`α = 0.55`, `β = 0.45` (`α + β = 1`), `γ = 0.08`. `α = 0` gives pure taste matching (a fixed top-k).
`α = 1` gives pure flow, which drifts away from the profile. The `γ` term stops the same artist
from playing back to back. If the pool runs out before `k` picks, it widens to the full catalog.

---

## Where it goes in code

- `src/scoring.py`: `gauss`, `fold_tempo`, `genre_match`, `mood_match`, the tables, `BRANCHES`,
  `route(user)`, `score_components(user, song)`.
- `src/ranking.py`: `rank_key`, `rank`, `similarity`, `next_song`, `build_playlist`.
- `src/recommender.py`: fill the five stubs (`load_songs`, `score_song`, `recommend_songs`,
  `Recommender.recommend`, `Recommender.explain_recommendation`), add the two optional
  `UserProfile` fields, and add `Recommender.playlist`.
- `src/main.py`: change the import to `from src.recommender import ...` and make the CSV path
  independent of the current directory, so `python -m src.main` works.
