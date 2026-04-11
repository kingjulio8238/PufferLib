"""High-level Python API for PufferVerify math verification."""

from pufferlib.verify._binding import verify_single, verify_batch


def check(response: str, ground_truth: str, tolerance: float = 1e-5) -> float:
    """Verify a single math response. Returns 1.0 if correct, 0.0 if not."""
    result = verify_single(response, ground_truth, tolerance)
    return float(result.reward)


def check_batch(responses: list[str], ground_truths: list[str],
                tolerance: float = 1e-5) -> list[float]:
    """Verify a batch of math responses. Returns list of rewards (1.0 or 0.0).
    Uses bulk buffer API for minimal Python↔C overhead."""
    from pufferlib.verify._binding import verify_bulk
    return verify_bulk(responses, ground_truths, tolerance)


def extract_answer(text: str) -> str | None:
    """Extract the numerical answer from an LLM response as a string.
    Returns None if no number is found."""
    from pufferlib.verify._binding import extract_number as _extract
    ok, val = _extract(text)
    if not ok:
        return None
    # Return as clean string (integer if whole number, else float)
    if val == int(val) and abs(val) < 1e15:
        return str(int(val))
    return str(val)
