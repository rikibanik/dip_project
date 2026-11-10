
#!/usr/bin/env python3

import hashlib
import json
import shutil
import ssl
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "datasets" / "insurance" / "raw"
EXTRACTED = ROOT / "datasets" / "insurance" / "extracted"
MANIFEST = ROOT / "datasets" / "insurance" / "download_manifest.json"

SOURCES = [
    {
        "name": "hdfc_ergo_filled_health_claim_form",
        "url": "https://customer-portal-assets.hdfcergo.com/assets/v2/docs/default-source/documents/health_suraksha_claim_form-(sample-claim-form)-889956076655.pdf",
        "filename": "hdfc_ergo_filled_health_claim_form.pdf",
        "zip": False,
    },
    {
        "name": "paramount_tpa_sample_filled_claim_form",
        "url": "https://www.paramounttpa.com/Home/ClaimForms/Paramount/Sample%20filled%20Claim%20Form.pdf",
        "filename": "paramount_sample_filled_claim_form.pdf",
        "zip": False,
    },
    {
        "name": "gritworks_insurance_sample",
        "url": "https://gritworks.ai/datasets/gritworks-insurance-sample.zip",
        "filename": "gritworks_insurance_sample.zip",
        "zip": True,
    },
    {
        "name": "strata_insurance_corpus_source",
        "url": "https://codeload.github.com/NikolaiSachok/strata-insurance-corpus/zip/refs/heads/main",
        "filename": "strata_insurance_corpus_source.zip",
        "zip": True,
    },
]

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(source):
    destination = RAW / source["filename"]

    if destination.exists() and destination.stat().st_size > 0:
        print(f"[SKIP] {destination.name} already exists")
        return {"name": source["name"], "status": "already_exists",
                "path": str(destination.relative_to(ROOT)),
                "sha256": sha256(destination)}

    request = urllib.request.Request(source["url"], headers=HEADERS)
    temporary = destination.with_suffix(destination.suffix + ".part")

    try:
        print(f"[DOWNLOAD] {source['name']}")
        with urllib.request.urlopen(
            request, context=SSL_CONTEXT, timeout=60
        ) as response:
            with open(temporary, "wb") as output:
                shutil.copyfileobj(response, output)

        if temporary.stat().st_size == 0:
            raise RuntimeError("Empty download")

        temporary.replace(destination)
        print(f"[OK] {destination.name} ({destination.stat().st_size:,} bytes)")

        return {
            "name": source["name"],
            "status": "downloaded",
            "path": str(destination.relative_to(ROOT)),
            "sha256": sha256(destination),
        }

    except Exception as error:
        temporary.unlink(missing_ok=True)
        print(f"[ERROR] {source['name']}: {error}")
        return {"name": source["name"], "status": "failed",
                "error": str(error)}


def extract(source):
    archive = RAW / source["filename"]
    target = EXTRACTED / source["name"]

    if not archive.exists():
        return {"name": source["name"], "status": "archive_missing"}

    try:
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
        print(f"[EXTRACTED] {source['filename']}")
        return {"name": source["name"], "status": "extracted"}
    except Exception as error:
        print(f"[EXTRACT ERROR] {error}")
        return {"name": source["name"], "status": "failed",
                "error": str(error)}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    EXTRACTED.mkdir(parents=True, exist_ok=True)

    downloads = [download(s) for s in SOURCES]
    extractions = [
        extract(s) for s in SOURCES if s["zip"]
    ]

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloads": downloads,
        "extractions": extractions,
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    successful = sum(
        x["status"] in ("downloaded", "already_exists")
        for x in downloads
    )

    print("\n========== SUMMARY ==========")
    print(f"Successful: {successful}/{len(SOURCES)}")
    print(f"Failed: {len(SOURCES) - successful}/{len(SOURCES)}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
