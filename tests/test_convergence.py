import numpy as np

from pyswarm.pso import pso


def test_minfunc_plateau_terminates_early() -> None:
    call_count = 0

    def flat_func(x: np.ndarray) -> float:
        nonlocal call_count
        call_count += 1
        return 0.0

    result = pso(flat_func, [-1, -1], [1, 1], maxiter=500, seed=0, patience=1)
    fopt = result[1]
    if not np.isclose(fopt, 0.0):
        msg = f"Expected fopt == 0.0, got {fopt}"
        raise AssertionError(msg)
    # With patience=1 the search stops after the first non-improving iteration.
    # Without the fix the algorithm exhausts all 500 * swarmsize evaluations.
    if call_count >= 500 * 100:
        msg = f"Expected early stop, got {call_count} calls"
        raise AssertionError(msg)


def test_patience_zero_runs_to_maxiter() -> None:
    call_count = 0

    def flat_func(x: np.ndarray) -> float:
        nonlocal call_count
        call_count += 1
        return 0.0

    pso(flat_func, [-1, -1], [1, 1], maxiter=10, swarmsize=5, seed=0)
    # 5 particles * (1 init eval + 10 loop iters) = 55 calls
    expected = 5 + 10 * 5
    if call_count != expected:
        msg = f"Expected {expected} calls, got {call_count}"
        raise AssertionError(msg)
