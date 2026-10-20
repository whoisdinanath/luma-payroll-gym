"""Submit a saved model answer to the live sandbox and show its real reward."""
import argparse
import json
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--output", dest="input", type=Path, default=ROOT / "results/sandbox-demo",
                        help="Folder containing the saved model demo")
    args = parser.parse_args()
    directory = args.input
    meta = json.loads((directory / "demo.json").read_text())
    submission = json.loads((directory / "submission.json").read_text())
    # Use the original submission, including the untouched model answer and server-issued IDs.
    print(f"Submitting {meta['model']}'s saved answer to {meta['sandbox_url']}...", flush=True)
    # Repeating this exact submission is safe: the sandbox binds it to one run
    # and returns the saved result if the first submission already completed.
    for attempt, wait in enumerate((2, 5, 10, None), 1):
        try:
            response = httpx.post(meta["sandbox_url"] + "/submit", json=submission,
                                  headers={"User-Agent": "Mozilla/5.0 LumaDemo"}, timeout=120)
            break
        except (httpx.ConnectError, httpx.TimeoutException) as error:
            if wait is None:
                parser.exit(1, f"Could not reach the sandbox after {attempt} attempts: {error}\n"
                            f"Your saved answer is unchanged. Check your connection/DNS and rerun:\n"
                            f"uv run python scripts/demo_score.py --input {directory}\n")
            print(f"Connection unavailable; retrying in {wait}s ({attempt}/4)...", flush=True)
            time.sleep(wait)
    if response.is_error:
        parser.exit(1, f"Sandbox returned HTTP {response.status_code}: {response.text[:500]}\n"
                    "Your saved answer is unchanged. Runs expire after 24 hours.\n")
    result = response.json()
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    score = result["per_task"][0]
    print("\nLIVE SANDBOX RESULT")
    print(f"Reward: {score['score']:+.4f}")
    print(f"Passed: {score['passed']}")
    print(f"Status: {score['status']}")
    for name, part in score["breakdown"].items():
        print(f"  {name}: {part['score']:.4f} | rules: {', '.join(part.get('clauses', []))}")
    print(f"\nFull result saved: {directory / 'result.json'}")


if __name__ == "__main__":
    main()
