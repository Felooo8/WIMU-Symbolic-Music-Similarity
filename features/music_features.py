import os
import muspy
import numpy as np
from muspy import Music
import json
import logging
import yaml


def calc_pitch_class(music: Music):
    pitch_class = np.zeros(12)

    for track in music.tracks:
        for note in track.notes:
            pitch_class[note.pitch % 12] += 1

    return pitch_class


def calc_intervals(music: Music):
    intervals = np.zeros(49)

    for track in music.tracks:
        sorted_notes = sorted(track.notes, key=lambda note: note.start)

        for i in range(len(sorted_notes) - 1):
            interval = sorted_notes[i + 1].pitch - sorted_notes[i].pitch

            if -24 <= interval <= 24:
                idx = interval + 24
                intervals[idx] += 1

    return intervals


def durations(music: Music):
    durations = []

    for track in music.tracks:
        for note in track.notes:
            durations.append(note.duration)

    return np.array(durations)


class MusicFeatures:
    def __init__(self, measure_resolution: int = 1):
        self.measure_resolution = measure_resolution

        self.genre = "Unknown"

        self.pitch_class_entropy = 0.0
        self.pitch_class = np.zeros(12)
        self.pitch_entropy = 0.0
        self.pitch_range = 0
        self.intervals = np.zeros(49)
        self.durations = np.array([])
        self.scale_consistency = 0.0
        self.polyphony = 0.0
        self.empty_beat_rate = 0.0
        self.groove_consistency = 0.0

    def calc(self, file_path: str):
        try:
            music = muspy.load_json(file_path)

            with open(file_path, "r") as f:
                meta = json.load(f).get("metadata", {})

            self.genre = meta.get("genre") or "Unknown"

            self.pitch_class_entropy = muspy.pitch_class_entropy(music)
            self.pitch_class = calc_pitch_class(music)
            self.pitch_entropy = muspy.pitch_entropy(music)
            self.pitch_range = muspy.pitch_range(music)
            self.intervals = calc_intervals(music)
            self.durations = durations(music)
            self.scale_consistency = muspy.scale_consistency(music)
            self.polyphony = muspy.polyphony(music)
            self.empty_beat_rate = muspy.empty_beat_rate(music)
            self.groove_consistency = muspy.groove_consistency(
                music, self.measure_resolution
            )

            return (
                self.pitch_class_entropy,
                self.pitch_class,
                self.pitch_entropy,
                self.pitch_range,
                self.scale_consistency,
                self.polyphony,
                self.empty_beat_rate,
                self.groove_consistency
            )

        except MemoryError as e:
            logging.error(f"File={file_path} skipped (too big): {e}")
            return None

        except Exception as e:
            logging.error(f"File={file_path} skipped: {e}")
            return None

    def get_as_txt(self):
        return (
            f"features:\n"
            f"\tPitch class entropy: {self.pitch_class_entropy}\n"
            f"\tPitch entropy: {self.pitch_entropy}\n"
            f"\tPitch class: {self.pitch_class}\n"
            f"\tPitch range: {self.pitch_range}\n"
            f"\tScale consistency: {self.scale_consistency}\n"
            f"\tPolyphony: {self.polyphony}\n"
            f"\tEmpty beat rate: {self.empty_beat_rate}\n"
            f"\tGroove consistency: {self.groove_consistency} "
            f"(measure_resolution={self.measure_resolution})"
        )

    def to_json(self):
        return {
            "genre": self.genre,
            "pitch_class_entropy": self.pitch_class_entropy,
            "pitch_entropy": self.pitch_entropy,
            "pitch_range": self.pitch_range,
            "scale_consistency": self.scale_consistency,
            "polyphony": self.polyphony,
            "empty_beat_rate": self.empty_beat_rate,
            "groove_consistency": self.groove_consistency
        }