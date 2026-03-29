import os
import json

from pathlib import Path
from itertools import combinations
from jsd import calc_jsd


def main():
    savepath = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "distributions"))
        
    if not os.path.exists(savepath):
        print(f"\n[DISTRIBUTIONS] file does not exist for savepath={savepath}")
        return

    path = os.path.join(savepath, "distributions.json")

    if os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8") as f:
            datasets = json.load(f)
    else:
        print(f"\n[DISTRIBUTIONS] cannot read, path={path}")

    print(f"\n[DISTRIBUTIONS] file read for path={path}\n")

    names = list(combinations(datasets.keys(), 2))
    distributions = list(combinations(datasets.values(), 2))

    for name_pair, dis_pair in zip(names, distributions):
        print(f"\n[SIMILARITY] pitch class")
        if dis_pair[0]["pitch_class"] is not None and dis_pair[1]["pitch_class"] is not None:
            pitch_class_jsd = calc_jsd(name_pair[0],  name_pair[1], dis_pair[0]["pitch_class"], dis_pair[1]["pitch_class"])

        print(f"\n[SIMILARITY] intervals")
        if dis_pair[0]["interval"] is not None and dis_pair[1]["interval"] is not None:
            interval_jsd = calc_jsd(name_pair[0],  name_pair[1], dis_pair[0]["interval"], dis_pair[1]["interval"])


if __name__ == "__main__":
    main()