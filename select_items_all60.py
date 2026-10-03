"""Select ALL 60 items, tagged by group, for the Anthropic pressure runs.

Writes three files:

  items_all.json       all 60, each carrying a `group` field  <- run this one
  items_matched.json   the 38 where the model gave the same correct answer
                       with and without the image (the clean comparison)
  items_grounded.json  the 4 where it was correct only with the image

Running the full 60 answers the objection that the matched subset is, by
construction, where the image mattered least — so a null result there is
partly baked into the selection. The other groups each test something the
matched set cannot: `grounded` is where the image genuinely mattered,
`wrong_both` shows whether incorrect answers are defended as stubbornly as
correct ones, and `unstable` items already move with no pressure at all.

Selection uses the 2025 real-vs-blank accuracy pattern, which is
model-specific: aggregated_statistics.csv is Claude Sonnet 4.5, so the judge
in run_pressure_anthropic.py should be that same model.

    python3 select_items_all60.py --stats data/aggregated_statistics.csv \
        --questions data/subset_with_images.xlsx --images data/images
"""

import argparse
import csv
import json
from pathlib import Path

# Column names in subset_with_images.xlsx (sheet SSM_Q_ITA).
XLSX_COLS = {
    "id": "questionID",
    "text": "question_text",
    "image": "picture_link",
    "correct": "correct_option",
    "domain": "domain",
    "subdomain": "subdomain",
}


def load_questions(path):
    """Read the question bank. Accepts the .xlsx directly, or a JSON list."""
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xlsm"):
        import pandas as pd

        df = pd.read_excel(p)
        missing = [v for v in XLSX_COLS.values() if v not in df.columns]
        if missing:
            raise SystemExit(f"{p} is missing expected column(s): {missing}")

        out = {}
        for _, r in df.iterrows():
            qid = str(r[XLSX_COLS["id"]]).strip()
            img = r[XLSX_COLS["image"]]
            out[qid] = {
                "vignette": str(r[XLSX_COLS["text"]]).strip(),
                "options": {L: str(r[f"option_{L.lower()}"]).strip()
                            for L in "ABCDE" if f"option_{L.lower()}" in df.columns},
                "correct": str(r[XLSX_COLS["correct"]]).strip().upper(),
                "image_file": (str(img).strip() if isinstance(img, str) else None),
                "domain": r.get(XLSX_COLS["domain"]),
            }
        return out

    payload = json.loads(p.read_text(encoding="utf-8"))
    qs = payload if isinstance(payload, list) else payload.get("questions", [])
    out = {}
    for q in qs:
        qid = str(q.get("question_id") or q.get("item_id") or q.get("id") or "").strip()
        if not qid:
            continue
        out[qid] = {
            "vignette": q.get("vignette") or q.get("question") or q.get("question_text", ""),
            "options": q.get("options") or q.get("choices") or {},
            "correct": str(q.get("correct") or q.get("correct_option") or "").upper(),
            "image_file": q.get("image_file") or q.get("picture_link") or q.get("image"),
            "domain": q.get("domain"),
        }
    return out


# Filename conventions seen in the wild, tried in order against the image
# directory. The 2025 export uses image_<questionID>.png (one file per
# question, zero-padded); the spreadsheet's picture_link column uses a
# different and inconsistent scheme (dropped zeros, shared files, mixed
# case). Resolve against the disk rather than trusting either.
def resolve_image(item_id, picture_link, img_dir):
    """-> (path_or_None, how_it_was_found)"""
    if not img_dir.exists():
        return None, "no image dir"

    on_disk = {p.name: p for p in img_dir.iterdir() if p.is_file()}
    lower = {n.lower(): n for n in on_disk}

    candidates = [f"image_{item_id}.png", f"image_{item_id}.PNG"]
    if picture_link:
        candidates.append(picture_link)
    candidates += [f"{item_id}.png", f"{item_id}.PNG"]

    for cand in candidates:
        if cand in on_disk:
            return on_disk[cand], "exact"
        if cand.lower() in lower:
            return on_disk[lower[cand.lower()]], "case-insensitive"
    return None, "not found"


def load_stats(path):
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def classify(rows, reps, threshold):
    """Split items into the four groups that matter."""
    groups = {"matched": [], "grounded": [], "wrong_both": [], "unstable": []}
    for r in rows:
        real = int(r["real_correct_count"])
        fake = int(r["fake_correct_count"])
        hi, lo = threshold, reps - threshold

        if real >= hi and fake >= hi:
            groups["matched"].append(r)          # same correct answer either way
        elif real >= hi and fake <= lo:
            groups["grounded"].append(r)         # needs the image — contrast set
        elif real <= lo and fake <= lo:
            groups["wrong_both"].append(r)       # uninformative
        else:
            groups["unstable"].append(r)         # varies across repetitions
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--questions", help="subset_with_images.xlsx (or a JSON equivalent)")
    ap.add_argument("--images", default="data/images",
                    help="directory holding the PNGs named by picture_link")
    ap.add_argument("--out", default="items_matched.json")
    ap.add_argument("--out-grounded", default="items_grounded.json",
                    help="the contrast set: items that genuinely need the image")
    ap.add_argument("--out-all", default="items_all.json",
                    help="every item, tagged with its group. Running the full set "
                         "answers the objection that the matched subset is, by "
                         "construction, where the image mattered least.")
    ap.add_argument("--threshold", type=int, default=10,
                    help="correct_count required in BOTH conditions (default 10 = unanimous)")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    rows = load_stats(args.stats)
    groups = classify(rows, args.reps, args.threshold)

    print(f"{len(rows)} items, {args.reps} repetitions, threshold {args.threshold}\n")
    for name, label in [
        ("matched", "same correct answer with and without the image"),
        ("grounded", "correct only with the image (contrast set)"),
        ("wrong_both", "wrong in both conditions (uninformative)"),
        ("unstable", "varies across repetitions"),
    ]:
        print(f"  {name:11s} {len(groups[name]):3d}   {label}")

    if args.threshold == args.reps and groups["unstable"]:
        print(f"\n  note: {len(groups['unstable'])} items excluded for repetition "
              f"instability. These wiggle with no pressure applied at all — the "
              f"mechanical floor. Worth reporting, and --threshold 8 would keep "
              f"{len(classify(rows, args.reps, 8)['matched']) - len(groups['matched'])} more.")

    n_matched = len(groups["matched"])
    if n_matched < 20:
        print(f"\n  WARNING: only {n_matched} matched items. Underpowered for a "
              f"per-model wiggle rate; consider a larger item pool.")

    # --- join with question content ---------------------------------------
    if not args.questions:
        ids = [r["question_id"] for r in groups["matched"]]
        Path(args.out).write_text(json.dumps(ids, indent=2), encoding="utf-8")
        print(f"\nwrote {len(ids)} ids -> {args.out}")
        print("re-run with --questions to emit full items for run_pressure.py")
        return

    by_id = load_questions(args.questions)

    missing = []
    all_items = []
    for name, out_path in [("matched", args.out), ("grounded", args.out_grounded),
                           ("wrong_both", None), ("unstable", None)]:
        items = []
        unresolved = []
        for r in groups[name]:
            qid = r["question_id"]
            q = by_id.get(qid)
            if q is None:
                missing.append(qid)
                continue
            resolved, how = resolve_image(qid, q["image_file"], Path(args.images))
            if resolved is None:
                unresolved.append(qid)
            items.append({
                "item_id": qid,
                "group": name,
                "vignette": q["vignette"],
                "options": q["options"],
                "correct": r["correct_answer"],
                # Taken from picture_link, NOT derived from the id: 25 of 60
                # rows have a filename that does not match their questionID
                # (inconsistent zero-padding, and 6 images shared by two
                # questions each).
                "image_file": q["image_file"],
                "image_path": str(resolved) if resolved else None,
                "domain": q.get("domain"),
            })
        all_items.extend(items)
        if out_path is None:
            continue
        Path(out_path).write_text(json.dumps(items, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        found = sum(1 for i in items if i["image_path"])
        print(f"\nwrote {len(items)} items -> {out_path}   ({found} with a resolved image)")
        if unresolved:
            print(f"  WARNING: no image found for {len(unresolved)} item(s): "
                  f"{unresolved[:6]}{'...' if len(unresolved) > 6 else ''}")
            print(f"  looked in {Path(args.images).resolve()}")

    if missing:
        print(f"\n  WARNING: {len(missing)} ids in the stats file had no matching "
              f"question: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    all_items.sort(key=lambda i: i["item_id"])
    Path(args.out_all).write_text(json.dumps(all_items, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    found = sum(1 for i in all_items if i["image_path"])
    from collections import Counter as _C
    dist = dict(_C(i["group"] for i in all_items))
    print(f"\nwrote {len(all_items)} items -> {args.out_all}   "
          f"({found} with a resolved image)")
    print(f"  groups: {dist}")
    print("  each item carries a 'group' field, so the matched subset stays")
    print("  separable at analysis time.")

    # Shared images: 6 pictures serve two questions each. Harmless for the
    # real/blank comparison, but any future mismatched-image arm must not
    # pair two questions that share a picture — the "wrong" image would be
    # the right one.
    from collections import Counter
    shared = Counter(q["image_file"] for q in by_id.values() if q["image_file"])
    dupes = {k: v for k, v in shared.items() if v > 1}
    if dupes:
        print(f"\n  note: {len(dupes)} image(s) shared by more than one question "
              f"({', '.join(sorted(dupes)[:4])}). Fine here; exclude these pairs "
              f"if you ever add a mismatched-image condition.")

    # Sanity: the answer key in the two files must agree.
    clashes = [r["question_id"] for r in groups["matched"]
               if (q := by_id.get(r["question_id"]))
               and q.get("correct") and q["correct"] != r["correct_answer"].upper()]
    if clashes:
        print(f"\n  WARNING: correct_answer disagrees between files for "
              f"{len(clashes)} item(s): {clashes[:5]}")


if __name__ == "__main__":
    main()