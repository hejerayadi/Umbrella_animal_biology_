"""Download the benchmark's images into the git-ignored `assets/` directory.

This is an acquisition helper and nothing else. It does not import, construct or
call `RecognitionAgent`, it computes no metric, and it is not the Phase 2 runner.

Design constraints, all deliberate:

* **Opt-in.** It refuses to run unless `RECOGNITION_EVAL_FETCH=1` is set, for the
  same reason the Sprint 3 smoke scripts do: no import, test run or stray
  invocation may reach the network on its own.
* **Deterministic.** Cases are processed in committed manifest order, each file
  is written to the exact `local_path` the manifest names, and an existing file
  is left alone. Running it twice changes nothing.
* **Bounded.** One attempt per file, a fixed timeout, and a fixed pause between
  requests so the source is not hammered. No retry loop.
* **Contained.** Every write is checked to land inside `assets/`. Nothing is
  written anywhere else, and no file outside `assets/` is read or modified.

Usage:

    RECOGNITION_EVAL_FETCH=1 python -m \\
        backend.agents.multimodal_recognition_agent.evaluation.fetch_assets
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
ASSETS = HERE / "assets"

OPT_IN_VAR = "RECOGNITION_EVAL_FETCH"
TIMEOUT_SECONDS = 60.0
PAUSE_SECONDS = 1.5
USER_AGENT = (
    "UmbrellaRecognitionSprint4/1.0 "
    "(bounded evaluation benchmark asset fetch; see evaluation/README.md)"
)


def _target_for(local_path: str) -> Path:
    """Resolve `local_path` under `assets/`, refusing anything that escapes it."""
    target = (HERE / local_path).resolve()
    if not str(target).startswith(str(ASSETS.resolve()) + os.sep):
        raise SystemExit(f"refusing to write outside assets/: {local_path!r}")
    return target


def main() -> int:
    if os.getenv(OPT_IN_VAR) != "1":
        print(
            f"Refusing to run: set {OPT_IN_VAR}=1 to download benchmark images.\n"
            "Nothing was fetched and nothing was written.",
            file=sys.stderr,
        )
        return 2

    with open(MANIFEST, encoding="utf-8") as handle:
        manifest = json.load(handle)

    ASSETS.mkdir(exist_ok=True)
    wanted = [c for c in manifest["cases"] if c.get("asset")]
    fetched = skipped = failed = 0

    for case in wanted:
        asset = case["asset"]
        target = _target_for(asset["local_path"])
        if target.exists():
            skipped += 1
            print(f"  skip    {case['case_id']}  (already present)")
            continue

        request = urllib.request.Request(asset["asset_url"], headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                payload = response.read()
        except Exception as exc:  # noqa: BLE001 - the reason is reported, not raised
            failed += 1
            # Only the exception type: a remote error body can echo the request.
            print(f"  FAILED  {case['case_id']}  ({type(exc).__name__})", file=sys.stderr)
            time.sleep(PAUSE_SECONDS)
            continue

        if len(payload) != asset["bytes"]:
            failed += 1
            print(
                f"  FAILED  {case['case_id']}  (size {len(payload)} != manifest "
                f"{asset['bytes']}; the source file changed - re-verify it before use)",
                file=sys.stderr,
            )
            time.sleep(PAUSE_SECONDS)
            continue

        target.write_bytes(payload)
        fetched += 1
        print(f"  ok      {case['case_id']}  -> {asset['local_path']}")
        time.sleep(PAUSE_SECONDS)

    print(
        f"\n{len(wanted)} cases with an image: "
        f"{fetched} fetched, {skipped} already present, {failed} failed."
    )
    if failed:
        print(
            "A failure here is a dataset finding, not something to work around: "
            "re-verify the source page and licence before replacing a case.",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
