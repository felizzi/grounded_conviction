#!/usr/bin/env python3
"""Verify every item's resolved image_path exists, before spending on a run.
 
    python3 check_images.py --items items_matched.json
 
Checks `image_path` — the path select_items.py resolved against the disk —
not `image_file`, which is the spreadsheet's own (inconsistent) name for the
picture and frequently does not match any real file.
"""
import argparse
import json
from pathlib import Path
 
ap = argparse.ArgumentParser()
ap.add_argument("--items", default="items_matched.json")
a = ap.parse_args()
 
items = json.loads(Path(a.items).read_text(encoding="utf-8"))
 
ok, missing, unset = [], [], []
for it in items:
    p = it.get("image_path")
    if not p:
        unset.append(it["item_id"])
    elif Path(p).exists():
        ok.append(it["item_id"])
    else:
        missing.append((it["item_id"], p))
 
print(f"{len(items)} items in {a.items}")
print(f"  image present    {len(ok)}")
print(f"  image_path unset {len(unset)}")
print(f"  file not found   {len(missing)}")
 
if unset:
    print(f"\nno image_path recorded: {unset[:10]}")
if missing:
    print("\nimage_path recorded but file absent:")
    for iid, p in missing[:10]:
        print(f"  {iid}  ->  {p}")
    print("\nre-run select_items.py with the correct --images directory.")
 
if not missing and not unset:
    print("\nall images resolve. safe to run the pressure loop.")
    sample = items[0]
    print(f"  e.g. {sample['item_id']}: spreadsheet said {sample['image_file']!r}, "
          f"using {Path(sample['image_path']).name!r}")