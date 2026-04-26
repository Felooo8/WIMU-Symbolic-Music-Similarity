import json
import math
import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "features"))
sys.path.insert(0, str(REPO_ROOT / "similarity"))


class _Note:
    def __init__(self, time: int, pitch: int, duration: int, velocity: int):
        self.start = time
        self.pitch = pitch
        self.duration = duration
        self.velocity = velocity


class _Track:
    def __init__(self, program: int = 0):
        self.program = program
        self.notes = []


class _Music:
    def __init__(self, resolution: int = 24, tracks=None):
        self.resolution = resolution
        self.tracks = tracks or []


def _iter_pitches(music):
    for track in music.tracks:
        for note in track.notes:
            yield note.pitch


def _iter_durations(music):
    for track in music.tracks:
        for note in track.notes:
            yield note.duration


def _save(path, music):
    payload = {
        "resolution": music.resolution,
        "tracks": [
            {
                "program": track.program,
                "notes": [
                    {
                        "time": note.start,
                        "pitch": note.pitch,
                        "duration": note.duration,
                        "velocity": note.velocity,
                    }
                    for note in track.notes
                ],
            }
            for track in music.tracks
        ],
    }
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def _load(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tracks = []
    for track_data in payload["tracks"]:
        track = _Track(program=track_data["program"])
        track.notes = [
            _Note(
                time=note_data["time"],
                pitch=note_data["pitch"],
                duration=note_data["duration"],
                velocity=note_data["velocity"],
            )
            for note_data in track_data["notes"]
        ]
        tracks.append(track)
    return _Music(resolution=payload["resolution"], tracks=tracks)


def _pitch_class_entropy(music):
    counts = np.zeros(12)
    for pitch in _iter_pitches(music):
        counts[pitch % 12] += 1
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0


def _pitch_entropy(music):
    pitches = list(_iter_pitches(music))
    if not pitches:
        return 0.0
    _, counts = np.unique(pitches, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def _pitch_range(music):
    pitches = list(_iter_pitches(music))
    return int(max(pitches) - min(pitches)) if pitches else 0


def _scale_consistency(music):
    return 0.75 if list(_iter_pitches(music)) else 0.0


def _polyphony(music):
    return 1.0 if list(_iter_pitches(music)) else 0.0


def _empty_beat_rate(music):
    return 0.0 if list(_iter_pitches(music)) else 1.0


def _groove_consistency(music, measure_resolution=1):
    return 1.0 / (measure_resolution + 1)


muspy_stub = types.SimpleNamespace(
    Music=_Music,
    Track=_Track,
    Note=_Note,
    save=_save,
    load=_load,
    pitch_class_entropy=_pitch_class_entropy,
    pitch_entropy=_pitch_entropy,
    pitch_range=_pitch_range,
    scale_consistency=_scale_consistency,
    polyphony=_polyphony,
    empty_beat_rate=_empty_beat_rate,
    groove_consistency=_groove_consistency,
)


def _jensen_shannon(p, q, base=2):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    if base:
        js /= math.log(base)
    return float(math.sqrt(js))


scipy_distance_stub = types.SimpleNamespace(jensenshannon=_jensen_shannon)
scipy_spatial_stub = types.SimpleNamespace(distance=scipy_distance_stub)
scipy_stub = types.SimpleNamespace(spatial=scipy_spatial_stub)

sys.modules.setdefault("muspy", muspy_stub)
sys.modules.setdefault("scipy", scipy_stub)
sys.modules.setdefault("scipy.spatial", scipy_spatial_stub)
sys.modules.setdefault("scipy.spatial.distance", scipy_distance_stub)
