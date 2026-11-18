"""Runnable checks for Stage2 sampled/dense source windows and provenance."""
from copy import deepcopy

import numpy as np

from compile_stage2_contact_windows import apply
from merge_agile_contact_windows import merge


def check_raises(reference, windows, expected):
    try:
        apply(reference, windows)
    except ValueError as error:
        assert expected in str(error), f"Unexpected error: {error}"
    else:
        raise AssertionError(f"Expected failure containing {expected}")


def main():
    frames = np.arange(20, 84, 4)
    count = len(frames)
    bottle = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    bottle[:, 0, 3] = np.linspace(0., 3., count)
    pad = bottle.copy()
    pad[:, 1, 3] += .07
    grip = np.full((count, 2), .03)
    reference = dict(src_frames=frames, T_world_object=bottle, T_world_pad=pad,
                     T_world_pad_retarget=pad, grip=grip, grip_retarget=grip,
                     base_reference_se2=np.c_[bottle[:, 0, 3], np.zeros((count, 2))],
                     object_ids=np.full(count, "bottle0"), hand_observed=np.ones(count, bool),
                     object_observed=np.ones(count, bool))
    local = np.eye(4)
    local[:3, 3] = [0., .04, .02]
    windows = [dict(window_id=f"event_{i}", status="accepted", consumable=True, object_id="bottle0",
                    source_frames=frames[i:i+5], contact_evidence_frames=frames[i+2:i+3],
                    grip=[.017, .017], T_object_pad=local) for i in (2, 9)]
    output, audit = apply(reference, windows)
    assert audit["source_contiguous"] is False
    assert audit["physical_replay_verified"] is False
    changed = np.flatnonzero(np.any(output["T_world_pad"] != pad, axis=(1,2)))
    assert changed.tolist() == [3, 4, 5, 10, 11, 12]
    assert np.array_equal(output["base_reference_se2"], reference["base_reference_se2"])
    assert np.array_equal(output["T_world_object"], bottle)
    outside = np.setdiff1d(np.arange(count), changed)
    assert np.array_equal(output["T_world_pad"][outside], pad[outside])
    assert np.array_equal(output["grip"][outside], grip[outside])
    assert np.allclose(np.linalg.solve(bottle[4], output["T_world_pad"][4]), local)
    assert np.allclose(output["T_world_pad"] @ output["T_pad_object"], bottle)

    unlabeled = deepcopy(windows)
    unlabeled[1]["status"] = "proposed"
    preview, audit = apply(reference, unlabeled)
    assert audit["status"] == "contact_review_partial"
    assert audit["windows_pending"] == [unlabeled[1]["window_id"]]
    assert np.array_equal(preview["T_world_pad"][9:], pad[9:])
    unlabeled[0]["status"] = "proposed"
    assert apply(reference, unlabeled)[0] is None
    unlabeled[1]["status"] = "accepted"
    del unlabeled[1]["consumable"]
    assert apply(reference, unlabeled)[0] is None
    unlabeled[1]["consumable"] = False
    assert apply(reference, unlabeled)[0] is None
    unlabeled[1]["event_type"] = "carry_continuation"
    unlabeled[0]["status"] = "accepted"
    assert apply(reference, unlabeled)[1]["status"] == "contact_targets_compiled"
    wrong_id = deepcopy(windows)
    wrong_id[0]["object_id"] = "bottle1"
    check_raises(reference, wrong_id, "object identity")
    wrong_id[0]["object_id"] = "bottle0"
    wrong_id[0]["contact_evidence_frames"] = [frames[4] + 1]
    check_raises(reference, wrong_id, "exact source frames")
    reference["hand_observed"][4] = False
    check_raises(reference, windows, "missing observations")
    try:
        merge(reference, [])
    except ValueError as error:
        assert "contiguous" in str(error)
    else:
        raise AssertionError("Native merge accepted a sampled timeline without opt-in")
    print("Stage2 passed: two windows; frozen object/base/outside; review block; identity/observation guards.")


if __name__ == "__main__":
    main()
