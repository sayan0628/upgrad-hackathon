"""Streaming paths over a real socket (design.md §1.2.12, §1.4.2). Runs tools/stream_check.py offline: a real uvicorn
server, the real local models and the insurance corpus, with a scripted, delayed LLM. Asserts delivery timing, not
just event order. Live OpenRouter verification is manual: `python tools/stream_check.py --live`."""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_streaming_paths_deliver_incrementally_over_http():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "stream_check.py")], capture_output=True, text=True,
                       timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert "6/6 streaming scenarios passed" in r.stdout
