import numpy as np


def detection_events_to_measurements(
    detection_events: np.ndarray,
    rounds: int,
    num_stabilizers: int,
) -> np.ndarray:
    """
    Recover per-round stabilizer measurements from binary detection events.

    Detection events are the XOR of consecutive measurement rounds:
        D[r, s] = M[r, s] XOR M[r-1, s]   (with M[0, s] = 0)

    r rounds of detection events come from (r+1) rounds of measurements,
    since each detection event is a difference between two consecutive rounds.
    The initial round M[0] is all zeros (the expected state before any QEC round).

    Inverting via cumulative XOR recovers all (r+1) measurement rounds:
        M[0, s] = 0
        M[1, s] = D[1, s]
        M[r, s] = D[1, s] XOR D[2, s] XOR ... XOR D[r, s]

    For binary values, XOR = addition mod 2.

    Args:
        detection_events: np.array of shape (shots, num_detectors), dtype float32
                          Binary detection events (0.0 or 1.0).
        rounds:           Number of QEC rounds (= number of detection event rounds).
        num_stabilizers:  Number of stabilizers per round (d**2 - 1).

    Returns:
        measurements: np.array of shape (shots, (rounds + 1) * num_stabilizers), dtype float32
                      All (rounds + 1) measurement rounds including the initial zeros.
                      e.g. for d=3, r=3: shape (shots, 32).
    """
    shots = detection_events.shape[0]
    num_bulk = rounds * num_stabilizers

    bulk = detection_events[:, :num_bulk].reshape(shots, rounds, num_stabilizers)

    # Cumulative XOR recovers M[1] through M[rounds]
    recovered = np.cumsum(bulk.astype(np.int32), axis=1) % 2

    # Prepend M[0] = zeros (initial state before first round)
    m0 = np.zeros((shots, 1, num_stabilizers), dtype=np.int32)
    measurements = np.concatenate([m0, recovered], axis=1)

    return measurements.reshape(shots, (rounds + 1) * num_stabilizers).astype(np.float32)
