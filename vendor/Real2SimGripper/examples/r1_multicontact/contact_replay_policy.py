"""Replay priorities and rigid transport targets from a real 250 Hz probe."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class ContactPlan:
    pad: np.ndarray
    holder: np.ndarray
    measured_anchor: np.ndarray
    command_anchor: np.ndarray
    source_windows: list
    aperture: np.ndarray


def contact_plan(source_frames, target_pad, target_object, width, observed,
                 actual_pad, actual_object, bilateral_steps, support_steps, step_counts):
    """Lock only intervals demonstrated in the pilot, from sustained lift to opening intent."""
    pad = target_pad.copy()
    n = len(pad)
    holder = np.zeros(n, bool)
    measured = np.full_like(pad, np.nan)
    command = measured.copy()
    windows = []
    aperture = np.array(width, copy=True)
    bilateral = observed & (bilateral_steps >= np.ceil(.6 * step_counts))
    detached = bilateral & (support_steps <= 1e-5)
    i = 0
    while i < n:
        if not detached[i]:
            i += 1
            continue
        j = i
        while j < n and detached[j]:
            j += 1
        # 48 ms of consecutive simultaneous contacts excludes one-frame grazes.
        if step_counts[i:j].sum() < 12:
            i = j
            continue
        # Look back through the same closed acquisition, bounded by physical
        # contacts, to identify the station that completed the approach.
        approach = i
        while approach and bilateral[approach-1]:
            approach -= 1
        if not np.any(support_steps[approach:i] > 1e-5) and not np.any(support_steps[:approach] > 1e-5):
            i = j
            continue
        # End on a new source opening, or the first loss of the probed contact
        # after landing. A partial carry can still be planned and scored.
        k = i
        latched_width = float(width[i])
        while k < n and bilateral[k] and width[k] <= latched_width + .002:
            latched_width = min(latched_width, width[k])
            k += 1
        if k <= i:
            i = j
            continue
        # Capture a physically confirmed acquisition. Fit the matching
        # command-space transform to avoid counting actuator lag twice.
        a = i + np.searchsorted(np.cumsum(step_counts[i:min(j, k)]), 12)
        a = min(a, j-1, k-1)
        command_grasp = np.linalg.solve(target_object[a], target_pad[a])
        real_grasp = np.linalg.solve(actual_object[a], actual_pad[a])
        anchor_targets = target_object[i:k] @ command_grasp
        elapsed = np.cumsum(step_counts[i:k])/250
        fraction = np.clip(elapsed / .12, 0., 1.)
        blend = fraction*fraction*(3-2*fraction)
        rows = np.arange(i,k)
        moving = ((np.linalg.norm(anchor_targets[:,:3,3]-pad[rows,:3,3], axis=1) > 1e-7)
                  | (np.linalg.norm(anchor_targets[:,:3,:3]-pad[rows,:3,:3], axis=(1,2)) > 1e-7))
        # Keep already rigid source commands bit-identical. Reprojecting
        # rotations at numerical noise scale can change a redundant IK branch.
        rows = rows[moving]
        if len(rows):
            locked, weight = anchor_targets[moving], blend[moving]
            pad[rows,:3,3] += weight[:,None]*(locked[:,:3,3]-pad[rows,:3,3])
            start, end = Rotation.from_matrix(pad[rows,:3,:3]), Rotation.from_matrix(locked[:,:3,:3])
            pad[rows,:3,:3] = (start * Rotation.from_rotvec(weight[:,None]*(start.inv()*end).as_rotvec())).as_matrix()
        measured[i:k] = real_grasp
        command[i:k] = command_grasp
        holder[i:k] = True
        aperture[i:k] = min(width[i], width[a])
        release_intent = k if k < n and width[k] > latched_width + .002 else None
        windows.append(dict(acquired=int(source_frames[approach]), lifted=int(source_frames[i]),
                            anchor=int(source_frames[a]), last_confirmed=int(source_frames[k-1]),
                            release_intent=None if release_intent is None else int(source_frames[release_intent])))
        i = max(j, k)
    return ContactPlan(pad, holder, measured, command, windows, aperture)


def lead_stations(base_ref, source_frames, plans, fps, source_events=()):
    """Move along the existing station route earlier for verified contact events."""
    t = np.arange(len(source_frames), dtype=float)
    advanced = t.copy()
    events = set(source_events)
    for plan in plans:
        for win in plan.source_windows:
            events.update(f for f in (win["acquired"], win["release_intent"]) if f is not None)
    for frame in sorted(events):
        k = int(np.searchsorted(source_frames, frame))
        lead = min(k//2, max(1, round(.20*fps)))
        if not lead:
            continue
        # Monotone time warp: same route and yaw, earlier arrival.
        mask = (t >= k-2*lead) & (t <= k)
        progress = (t[mask]-(k-2*lead))/(2*lead)
        bump = 4.*lead*progress*(1.-progress)
        advanced[mask] = np.maximum(advanced[mask], np.minimum(k, t[mask]+bump))
    advanced = np.maximum.accumulate(advanced)
    return np.column_stack([np.interp(advanced, t, axis) for axis in base_ref.T]), advanced


def protected_transport(source_frames, plans, fps, baseline, progressive, step_limits):
    """Apply the new solution in carry interiors while retaining contact anchors."""
    mixed = baseline.copy()
    ramp = max(2, round(.5*fps))
    windows = plans[0].source_windows
    for window in windows:
        if window["release_intent"] is None:
            continue
        start = int(np.searchsorted(source_frames, window["lifted"])) + round(.2*fps)
        end = int(np.searchsorted(source_frames, window["release_intent"])) - round(.9*fps)
        if end-start <= 2*ramp:
            continue
        rows = np.arange(start+1, end)
        w = np.clip(np.minimum(rows-start,end-rows)/ramp, 0., 1.)
        w = w*w*(3-2*w)
        targets = baseline[rows]+w[:,None]*(progressive[rows]-baseline[rows])
        for dst, target in zip(rows, targets):
            reach = (end-dst)*step_limits
            low = np.maximum(mixed[dst-1]-step_limits, baseline[end]-reach)
            high = np.minimum(mixed[dst-1]+step_limits, baseline[end]+reach)
            mixed[dst] = np.clip(target, low, high)
    return mixed


if __name__ == "__main__":
    frames = np.arange(200, 225)
    obj = np.tile(np.eye(4), (len(frames), 1, 1))
    pad = obj.copy()
    pad[:, 2, 3] = .04 + np.arange(len(frames))*.001
    steps = np.full(len(frames), 5)
    bilateral = np.r_[np.zeros(4), np.full(14, 5), np.zeros(7)]
    supported = np.r_[np.ones(6), np.zeros(9), np.ones(10)]
    widths = np.r_[np.full(4,.04), np.full(13,.015), np.full(8,.04)]
    plan = contact_plan(frames, pad, obj, widths, np.ones(len(frames), bool),
                        pad, obj, bilateral, supported, steps)
    assert plan.source_windows[0]["acquired"] == 204 and plan.source_windows[0]["release_intent"] == 217
    assert np.all(plan.holder[6:17]) and not np.any(plan.holder[:6])
    assert np.all(plan.command_anchor[6:17] == plan.command_anchor[6])
    assert np.allclose(plan.pad[11:17], obj[11:17] @ plan.command_anchor[6])
    empty = contact_plan(frames, pad, obj, widths, np.ones(len(frames), bool),
                         pad, obj, bilateral*0, supported, steps)
    assert not empty.source_windows and np.array_equal(empty.pad, pad)
    route = np.c_[frames*.01, frames*0, frames*0]
    led, clock = lead_stations(route, frames, [plan], 30.)
    assert np.all(np.diff(clock) >= 0) and np.all(led[:,0] >= route[:,0]-1e-12)
    assert clock[13] >= 16. and np.array_equal(led[-1], route[-1])

    rigid = obj.copy()
    rigid[:,2,3] = .06
    fixed = contact_plan(frames+731, rigid, obj, widths, np.ones(len(frames), bool),
                         rigid, obj, bilateral, supported, steps)
    assert np.array_equal(fixed.pad, rigid) and np.array_equal(fixed.holder, plan.holder)
    assert fixed.source_windows[0]["release_intent"] == 217+731
    vals = (pad, obj, widths, np.ones(len(frames), bool), pad, obj, bilateral, supported, steps)
    repeated = contact_plan(np.arange(2*len(frames)), *(np.concatenate([v,v]) for v in vals))
    assert len(repeated.source_windows) == 2
    # A blocked open command during the same airborne event is not a new grasp.
    suspended = contact_plan(frames, pad, obj, widths, bilateral > 0, pad, obj, bilateral,
                             np.r_[np.ones(6), np.zeros(19)], steps)
    assert len(suspended.source_windows) == 1
    t = np.arange(60)
    example = ContactPlan(None,None,None,None,[dict(lifted=0,release_intent=59)],None)
    base = np.column_stack([.005*t,0.*t])
    proposal = base.copy()
    proposal[:,1] = .2
    joined = protected_transport(t, [example], 20., base, proposal, np.array([.01,.012]))
    assert np.any(joined[:,1] > .1)
    assert np.array_equal(joined[:5],base[:5]) and np.array_equal(joined[41:],base[41:])
    assert np.all(np.abs(np.diff(joined,axis=0)) <= [.01000000001,.01200000001])
    print("contact lock, airborne release, regrasp, rigid targets, station lead and protected transport passed")
