"""Per-object/support release authorization for continuous video replay."""

from dataclasses import dataclass


@dataclass
class ReleaseGate:
    """Call once per video frame with accumulated support impulse."""

    stable_frames: int = 3
    state: object = None
    width: float = 0.0
    state_edge: bool = False
    pending: bool = False
    authorized: bool = False
    supported_frames: int = 0
    support_id: object = None
    last_frame: int = -1
    withdrawn: bool = False

    def update(self, frame, state, width, support_id, support_impulse,
               support_threshold, bilateral, withdrawal_clear):
        if frame != self.last_frame + 1:
            self.supported_frames = 0
        self.last_frame = frame
        self.state_edge |= self.state is not None and state != self.state
        opening = width > self.width
        if not self.pending and self.state_edge and opening and not self.withdrawn:
            self.pending = True
            self.state_edge = False
            self.support_id = support_id
            self.supported_frames = 0
        if self.pending and not self.authorized and width < self.width:
            self.pending = False
            self.state_edge = False
            self.supported_frames = 0
        if self.pending and not self.authorized:
            supported = (support_id == self.support_id and bilateral
                         and support_impulse >= support_threshold)
            self.supported_frames = self.supported_frames + 1 if supported else 0
            self.authorized = self.supported_frames >= self.stable_frames
        release = self.authorized
        if self.authorized and withdrawal_clear:
            self.pending = self.authorized = False
            self.state_edge = False
            self.withdrawn = True
            self.supported_frames = 0
        elif self.withdrawn and width < self.width:
            self.withdrawn = False
        self.state, self.width = state, width
        return release


class CycleGates:
    """Maintain independent release history for each object/support pair."""

    def __init__(self, stable_frames=3):
        self.stable_frames = stable_frames
        self.gates = {}

    def update(self, object_id, support_id, frame, state, width,
               support_impulse, support_threshold, bilateral, withdrawal_clear):
        key = (object_id, support_id)
        gate = self.gates.setdefault(key, ReleaseGate(self.stable_frames))
        return gate.update(frame, state, width, support_id, support_impulse,
                           support_threshold, bilateral, withdrawal_clear)


class PairReleaseAdapter:
    """Bind pair rows to physical support-contact summaries.

    Pair fields: grip_states[N], grip[N,2], object_ids[N], support_ids[N].
    Contact maps use (object_id, support_id) keys.
    """

    def __init__(self, targets, stable_frames=3):
        for field in ("grip_states", "grip", "object_ids", "support_ids"):
            if field not in targets:
                raise ValueError("pair is missing " + field)
        n = len(targets["grip_states"])
        if any(len(targets[field]) != n for field in
               ("grip", "object_ids", "support_ids")):
            raise ValueError("pair cycle fields must have equal frame counts")
        self.targets = targets
        self.gates = CycleGates(stable_frames)

    def update(self, frame, support_impulses, support_thresholds, bilateral,
               withdrawal_clear):
        object_id = self.targets["object_ids"][frame].item()
        support_id = self.targets["support_ids"][frame].item()
        if support_id in ("", None):
            return False
        key = (object_id, support_id)
        gate = self.gates.gates.get(key)
        if frame and (gate is None or gate.last_frame != frame-1):
            fresh = ReleaseGate(self.gates.stable_frames)
            if object_id == self.targets["object_ids"][frame-1].item():
                fresh.state = self.targets["grip_states"][frame-1].item()
                fresh.width = float(max(self.targets["grip"][frame-1]))
            self.gates.gates[key] = fresh
        return self.gates.update(
            object_id, support_id, frame, self.targets["grip_states"][frame].item(),
            float(max(self.targets["grip"][frame])),
            support_impulses.get(key, 0.0), support_thresholds[key],
            bilateral.get(key, False), withdrawal_clear)


def self_check():
    gate = ReleaseGate()
    assert not gate.update(0, 1, .01, "table", 0., 1., False, False)
    assert not gate.update(1, 2, .01, "table", .8, 1., True, False)
    assert not gate.update(2, 2, .02, "table", .8, 1., True, False)
    assert not gate.update(3, 2, .02, "table", 1.1, 1., True, False)
    assert not gate.update(4, 2, .02, "table", 1.1, 1., True, False)
    assert gate.update(5, 2, .02, "table", 1.1, 1., True, False)

    gate = ReleaseGate()
    assert not gate.update(0, 1, .01, "table", 0., 1., False, False)
    assert not gate.update(1, 2, .02, "table", .8, 1., True, False)
    assert not gate.update(2, 2, .02, "table", 1.1, 1., True, False)
    assert not gate.update(3, 2, .02, "table", 1.1, 1., True, False)
    assert gate.update(4, 2, .02, "table", 1.1, 1., True, False)

    gate = ReleaseGate()
    assert not gate.update(0, 1, .01, "table", 0., 1., False, False)
    assert not gate.update(1, 2, .02, "table", 1.2, 1., True, False)
    assert not gate.update(2, 2, .02, "chair", 1.2, 1., True, False)
    assert not gate.update(3, 2, .02, "chair", 1.2, 1., True, False)
    assert not gate.update(4, 2, .02, "chair", 1.2, 1., True, False)
    assert not gate.update(5, 2, .02, "chair", 1.2, 1., True, False)
    assert not gate.update(6, 2, .02, "table", 1.2, 1., True, False)
    assert not gate.update(7, 2, .02, "table", 1.2, 1., True, False)
    assert gate.update(8, 2, .02, "table", 1.2, 1., True, False)
    assert gate.update(9, 2, .02, "table", 0., 1., False, True)
    assert not gate.update(10, 2, .01, "table", 0., 1., False, True)
    assert not gate.update(11, 3, .02, "table", 0., 1., False, False)
    assert not gate.update(12, 3, .02, "table", 1.2, 1., True, False)
    assert not gate.update(13, 3, .02, "table", 1.2, 1., True, False)
    assert gate.update(14, 3, .02, "table", 1.2, 1., True, False)

    import numpy as np
    pair = dict(grip_states=np.array([2, 3, 3, 3, 3]),
                grip=np.array([[.01,.01], [.02,.02], [.02,.02], [.02,.02], [.02,.02]]),
                object_ids=np.array(["object"]*5),
                support_ids=np.array(["initial", "destination", "destination", "destination", "initial"]))
    adapter = PairReleaseAdapter(pair)
    for row, expected in enumerate([False, False, False, True, False]):
        pair_key = ("object", pair["support_ids"][row])
        assert adapter.update(row, {pair_key: 1.2}, {pair_key: 1.}, {pair_key: True}, False) == expected


if __name__ == "__main__":
    self_check()
    print("cycle gate self-check passed")

