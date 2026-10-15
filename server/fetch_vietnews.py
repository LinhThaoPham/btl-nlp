"""Download a small, pinned VietNews subset for local educational experiments."""
import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

REPO = "ThanhChinhBK/vietnews"
ROOT = Path(__file__).resolve().parents[1]

def main():
    response = requests.get(f"https://api.github.com/repos/{REPO}/commits/master", timeout=30)
    response.raise_for_status()
    commit = response.json()["sha"]
    tasks = [("dev", "val_tokenized", i) for i in range(1, 11)] + [("test", "test_tokenized", i) for i in range(1, 21)]
    def fetch(task):
        split, folder, index = task
        filename = f"{index:06d}.txt.seg"
        source = f"https://github.com/{REPO}/blob/{commit}/data/{folder}/{filename}"
        url = f"https://raw.githubusercontent.com/{REPO}/{commit}/data/{folder}/{filename}"
        r = requests.get(url, timeout=45)
        r.raise_for_status()
        raw = r.content.decode("utf-8-sig")
        if len(raw.split("\n\n")) < 3:
            raise ValueError(f"Unexpected format: {source}")
        return {"id": f"{split}-{index:06d}", "split": split, "originalSplit": folder,
                "source": source, "raw": raw, "sha256": hashlib.sha256(r.content).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        items = list(pool.map(fetch, tasks))
    result = {"schemaVersion": 1, "repository": f"https://github.com/{REPO}", "commit": commit,
              "downloadedAt": datetime.now(timezone.utc).isoformat(),
              "sampling": "First 10 validation and first 20 test filenames; small convenience sample, not the full benchmark.",
              "items": items}
    target = ROOT / "public" / "data" / "vietnews.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(items)} articles ({target.stat().st_size} bytes), commit {commit}", flush=True)

if __name__ == "__main__":
    main()
