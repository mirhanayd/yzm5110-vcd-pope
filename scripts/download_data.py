#!/usr/bin/env python
"""Download POPE question files and the COCO val2014 images they reference.

POPE files are taken from the official VCD repository at a pinned commit (they are the
POPE splits of Li et al., 2023, as used by VCD). Images are fetched one by one from the
COCO website (CC BY 4.0 annotations; images under their Flickr licenses, see
https://cocodataset.org/#termsofuse), so only the ~1000 images needed are downloaded
instead of the 6 GB val2014 archive.

    python scripts/download_data.py                 # COCO + A-OKVQA POPE files and images
    python scripts/download_data.py --no-images     # question files only
"""
import argparse
import concurrent.futures as cf
import json
import os
import urllib.request

VCD_COMMIT = "d6568ff81b8fd306a49e630df44f2db5c2300191"
RAW = f"https://raw.githubusercontent.com/DAMO-NLP-SG/VCD/{VCD_COMMIT}/experiments/data/POPE"
COCO = "http://images.cocodataset.org/val2014"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return False
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["coco", "aokvqa"])
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    images = set()
    for ds in a.datasets:
        os.makedirs(os.path.join(ROOT, "data", "pope", ds), exist_ok=True)
        for split in ("random", "popular", "adversarial"):
            name = f"{ds}_pope_{split}.json"
            dst = os.path.join(ROOT, "data", "pope", ds, name)
            fetch(f"{RAW}/{ds}/{name}", dst)
            images |= {json.loads(l)["image"] for l in open(dst) if l.strip()}
            print("ok", dst)
    if a.no_images:
        return
    if "gqa" in a.datasets:
        print("GQA images are not downloaded by this script (see README).")
        images = {i for i in images if i.startswith("COCO_")}
    img_dir = os.path.join(ROOT, "data", "images", "coco_val2014")
    os.makedirs(img_dir, exist_ok=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch, f"{COCO}/{im}", os.path.join(img_dir, im)): im for im in sorted(images)}
        n = 0
        for f in cf.as_completed(futs):
            f.result()
            n += 1
            if n % 200 == 0:
                print(f"images {n}/{len(images)}")
    print(f"{len(images)} images in {img_dir}")


if __name__ == "__main__":
    main()
