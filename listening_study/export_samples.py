#!/usr/bin/env python3

from __future__ import annotations
import csv
import random
import shutil
import subprocess
from pathlib import Path

import muspy

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_DIR = REPO_ROOT / "results" / "listening_pairs"

SOUNDFONT = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
RANDOM_SEED = 42
EXAMPLES_PER_SIDE = 2

PAIRS = [
    ("maestro_v3", "music_net"),
    ("maestro_v3", "jsb_chorales"),
    ("nes_mdb", "maestro_v3"),
    ("nes_mdb", "jsb_chorales"),
    ("lakh_midi_rock", "lakh_midi_metal"),
    ("lakh_midi_pop", "lakh_midi_rock"),
]
def check_bin(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Brak programu w PATH: {name}")

def check_inputs() -> None:
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"Brak katalogu: {PROCESSED_DIR}")
    if not Path(SOUNDFONT).exists():
        raise FileNotFoundError(f"Brak soundfontu: {SOUNDFONT}")

def list_jsons(dataset: str) -> list[Path]:
    d = PROCESSED_DIR / dataset
    if not d.exists():
        raise FileNotFoundError(f"Brak datasetu: {d}")
    files = sorted(d.glob("*.json"))
    if not files:
        raise RuntimeError(f"Brak plików JSON w: {d}")
    return files

def midi_from_json(json_path: Path, midi_path: Path) -> None:
    music = muspy.load_json(json_path)

    try:
        muspy.write_midi(midi_path, music)
    except ValueError as e:
        if "invalid key" in str(e).lower():
            music.key_signatures = []
            muspy.write_midi(midi_path, music)
        else:
            raise

def wav_from_midi(midi_path: Path, wav_path: Path) -> None:
    cmd = [
        "fluidsynth",
        "-ni",
        SOUNDFONT,
        str(midi_path),
        "-F",
        str(wav_path),
        "-T",
        "wav",
        "-r",
        "44100",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def mp3_from_wav(wav_path: Path, mp3_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-vn",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-b:a",
        "192k",
        str(mp3_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def convert_one(json_path: Path, out_prefix: Path) -> Path:
    midi_path = out_prefix.with_suffix(".mid")
    wav_path = out_prefix.with_suffix(".wav")
    mp3_path = out_prefix.with_suffix(".mp3")

    midi_from_json(json_path, midi_path)
    wav_from_midi(midi_path, wav_path)
    mp3_from_wav(wav_path, mp3_path)

    if midi_path.exists():
        midi_path.unlink()
    if wav_path.exists():
        wav_path.unlink()

    return mp3_path

def main() -> None:
    random.seed(RANDOM_SEED)
    check_bin("fluidsynth")
    check_bin("ffmpeg")
    check_inputs()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "manifest.csv"

    rows = []

    for idx, (left_ds, right_ds) in enumerate(PAIRS, start=1):
        left_files = list_jsons(left_ds)
        right_files = list_jsons(right_ds)

        left_pick = random.sample(left_files, k=min(EXAMPLES_PER_SIDE, len(left_files)))
        right_pick = random.sample(right_files, k=min(EXAMPLES_PER_SIDE, len(right_files)))

        pair_dir = OUTPUT_DIR / f"pair_{idx:02d}_{left_ds}_vs_{right_ds}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        for j, src in enumerate(left_pick, start=1):
            mp3 = convert_one(src, pair_dir / f"A_{j:02d}_{src.stem}")
            rows.append({
                "pair_id": idx,
                "pair_name": f"{left_ds}_vs_{right_ds}",
                "side": "A",
                "dataset": left_ds,
                "source_json": str(src.relative_to(REPO_ROOT)),
                "output_mp3": str(mp3.relative_to(REPO_ROOT)),
            })

        for j, src in enumerate(right_pick, start=1):
            mp3 = convert_one(src, pair_dir / f"B_{j:02d}_{src.stem}")
            rows.append({
                "pair_id": idx,
                "pair_name": f"{left_ds}_vs_{right_ds}",
                "side": "B",
                "dataset": right_ds,
                "source_json": str(src.relative_to(REPO_ROOT)),
                "output_mp3": str(mp3.relative_to(REPO_ROOT)),
            })

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair_id", "pair_name", "side", "dataset", "source_json", "output_mp3"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Gotowe: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()