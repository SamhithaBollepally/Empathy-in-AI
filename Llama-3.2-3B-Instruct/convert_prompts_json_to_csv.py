#%%
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable
import re


def _write_csv(rows: Iterable[dict[str, Any]], fieldnames: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)


#%%
def convert_prompts_set1_json_to_csv(in_path: Path, out_path: Path) -> None:
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object at top-level in {in_path}")

    def rows() -> Iterable[dict[str, Any]]:
        for emotion in sorted(data.keys()):
            prompts = data[emotion]
            if not isinstance(prompts, list):
                raise TypeError(f"Expected list for emotion '{emotion}' in {in_path}")
            for idx, prompt in enumerate(prompts):
                yield {"emotion": emotion, "idx": idx, "prompt": prompt}

    _write_csv(rows(), ["emotion", "idx", "prompt"], out_path)


#%%
_NEWLINE_RE = re.compile(r"\r\n|\r|\n")
_RATINGS_TRAIL_RE = re.compile(r",\d\|\d\|")


def _clean_text_cell(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    return " ".join(_NEWLINE_RE.sub(" ", text).split())


def _clean_first_utterance(text: Any) -> Any:
    if not isinstance(text, str):
        return text

    first_line = text.splitlines()[0] if text else text
    m = _RATINGS_TRAIL_RE.search(first_line)
    if m:
        first_line = first_line[: m.start()]

    return " ".join(first_line.split())


#%%
def convert_prompts_utterances_set2_json_to_csv(
    in_path: Path,
    out_path: Path,
    *,
    sanitize: bool = False,
) -> None:
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object at top-level in {in_path}")

    extra_keys: set[str] = set()
    for emotion, items in data.items():
        if not isinstance(items, list):
            raise TypeError(f"Expected list for emotion '{emotion}' in {in_path}")
        for item in items:
            if isinstance(item, dict):
                extra_keys.update(k for k in item.keys() if k not in {"prompt", "first_utterance"})

    fieldnames = ["emotion", "idx", "prompt", "first_utterance"] + sorted(extra_keys)

    def rows() -> Iterable[dict[str, Any]]:
        for emotion in sorted(data.keys()):
            items = data[emotion]
            for idx, item in enumerate(items):
                if isinstance(item, dict):
                    row = {"emotion": emotion, "idx": idx, **item}
                else:
                    row = {"emotion": emotion, "idx": idx, "prompt": None, "first_utterance": item}

                if sanitize:
                    row["prompt"] = _clean_text_cell(row.get("prompt"))
                    row["first_utterance"] = _clean_first_utterance(row.get("first_utterance"))

                yield row

    _write_csv(rows(), fieldnames, out_path)


#%%
def main() -> int:
    parser = argparse.ArgumentParser(description="Convert prompt JSON files to CSV.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing prompts_set1.json and prompts_utterances_set2.json",
    )
    parser.add_argument(
        "--sanitize-set2",
        action="store_true",
        help="Write an additional prompts_utterances_set2_sanitized.csv for Excel-friendly import.",
    )
    args = parser.parse_args()

    base_dir: Path = args.base_dir

    set1_in = base_dir / "prompts_set1.json"
    set2_in = base_dir / "prompts_utterances_set2.json"

    set1_out = base_dir / "prompts_set1.csv"
    set2_out = base_dir / "prompts_utterances_set2.csv"

    convert_prompts_set1_json_to_csv(set1_in, set1_out)
    convert_prompts_utterances_set2_json_to_csv(set2_in, set2_out)

    print(f"Wrote: {set1_out}")
    print(f"Wrote: {set2_out}")

    if args.sanitize_set2:
        set2_out_sanitized = base_dir / "prompts_utterances_set2_sanitized.csv"
        convert_prompts_utterances_set2_json_to_csv(set2_in, set2_out_sanitized, sanitize=True)
        print(f"Wrote: {set2_out_sanitized}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
