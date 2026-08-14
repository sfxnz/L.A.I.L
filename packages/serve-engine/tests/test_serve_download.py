"""Live hf-download log frames (tqdm uses \\r, not newlines)."""

from app.services import serve as sv


def test_split_cli_frames_cr_and_newline():
    frames, rest = sv._split_cli_frames(b"Fetching 18 files:  12%|\rFetching 18 files:  40%|\nkept\npartial")
    assert frames == ["Fetching 18 files:  12%|", "Fetching 18 files:  40%|", "kept"]
    assert rest == b"partial"


def test_split_cli_frames_strips_ansi():
    frames, rest = sv._split_cli_frames(b"\x1b[32mmodel-00001\x1b[0m:  45%\r")
    assert frames == ["model-00001:  45%"]
    assert rest == b""


def test_download_pct_from_tqdm_and_fetching():
    assert sv._download_pct_from_line("Fetching 18 files:  42%|") == 42
    assert sv._download_pct_from_line("model-00003:  7%|▋        | 200M/2.8G") == 7
    assert sv._download_pct_from_line("no percent here") is None
