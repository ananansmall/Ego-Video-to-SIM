# R1 continuous multi-contact replay

Run the separate generic runtime without altering the episode-specific EP5 replay:

```bash
python3 examples/r1_multicontact/replay_multicontact.py PACKAGE OUTPUT
```

The package uses `paired_targets.npz` fields `src_frames`, `video_fps`, `grip_states`, `grip[N,2]`, `T_world_pad[N,4,4]`, `T_world_object[N,4,4]`, `object_ids[N]`, `support_ids[N]`, and optionally `base_reference_se2[N,3]`. `geometry.json` maps IDs through `objects[id]` (`mesh`, `mass_kg`, optional `collision_mesh` and `texture`) and `supports[id]` (`center`, `half_extents`, optional `yaw`). This runner currently accepts one tracked object across multiple named supports.

Supply optional `T_world_pad_left[N,4,4]` in the same world frame and source-frame order. `left_pad_valid[N]` masks missing left poses. `grip_left[N,2]` supplies the left finger joint targets on valid rows. One WBC solve chooses planar base, four torso joints and both arms. It minimizes inter-row motion and change in motion. Target continuity and actuator limits apply before joint commands are interpolated over 250 Hz substeps. The right wrist target remains required and the release gate observes the right fingers and the tracked object.

`base_reference_weight` (default 3) weights deviation from the supplied station path. It differs from the continuity cost on actual planar displacement. `ik_orientation_weight` (default 4) weights pad orientation in the solve. An optional `ik_initial_arm_seed_rad` can use the first source retarget state, ordered as six right-arm joints, four torso joints and, with valid left targets, six left-arm joints. The runtime records the base command and actual base pose, joint commands and readback, plus left wrist measurements when active.


Render the result as a 2x2 simulation video plus a separate wrist FPV video; add the original source clip to create the left-source/right-4-view comparison with the existing renderer:

python3 examples/r1_multicontact/render_multicontact.py PACKAGE OUTPUT [--source ORIGINAL_VIDEO]

The runtime aggregates object/support vertical contact impulses and bilateral finger contact over each video frame. Opening is held until the active support exceeds the configured fraction of object weight for three consecutive video frames; `--support-fraction` and `--stable-support-frames` override package/default values. The trace records support impulses, bilateral contact, executed grip, release authorization, and relative pose error.

`diagnostic-fixture/` is an explicitly scaled, synthetic controller fixture based on the accepted EP5 flask asset. Its current recorded smoke does not authorize a release and is not a passing multi-cycle proof. No result is claimed for the new source video.

`replay_multicontact.py PACKAGE OUTPUT --speed 0.5` is a half-speed replay. It adds physics substeps at the existing 250 Hz and keeps the ordered source frame IDs and geometric pose targets. The trace includes `source_fps`, `playback_fps`, and `sim_time_s`. The separate wrist video uses `playback_fps`; the source comparison displays corresponding source frames at the original rate.

## Contact-aware replay

```bash
python examples/r1_multicontact/replay_multicontact.py PACKAGE OUTPUT --contact-policy
python examples/r1_multicontact/audit_contact_replay.py PACKAGE OUTPUT
python examples/r1_multicontact/render_multicontact.py PACKAGE OUTPUT --source ORIGINAL_VIDEO
```

The replay keeps the real 250 Hz pilot under `OUTPUT/contact_probe/`. Continuous, simultaneously loaded fingers and loss of support identify a lift. On that verified interval the replay keeps one object-relative pad transform and aperture. The measured grasp anchor and the corresponding command-space anchor are both saved so the audit can distinguish actuator tracking from object slip. Already rigid commands stay bit-identical. If both source wrists exist, the physically holding hand leads the simultaneous whole-body solve while an observed free arm retains a tracking objective. When both hold, both objectives retain full priority.

The reference clock can advance along the supplied SE(2) route at source aperture cues and physically verified contact events. It follows the original waypoints. Physical station arrival is measured separately from reference clock advance. The package SE(2) reference is absolute world XY with yaw relative to R1_BASE_Q, just as the original solver uses relative yaw. The approach can open before a sustained grip is acquired. After acquisition the release gate requires actual opposing finger contact and load on the designated support. If a loaded hand would leave the seated pose before authorization, the runner holds that base and arm command until the object is independently supported and rejoins with step limits. Left targets remain optional and report zero covered rows when unavailable.

A support can name a metric world-space `collision_mesh` and optional `visual_mesh`. Support IDs are the names assigned to their separate static physics actors. A metric background `scene_visual` can carry the reconstructed appearance. Without a support collision mesh the runner uses the declared slab. A background mesh supplies visuals only.

## Progressive base and torso motion

Add `--progressive-motion` to a `--contact-policy` run. The probe is unchanged. In the alternate WBC solve, corrections to the base route, changes in actual base velocity and excursions of cumulative torso link pitch have a quadratic plus quartic cost. The runner uses the standard plan at acquisition and seating, and blends within verified carry windows with bounded planar, yaw and joint increments. Existing position and torso bounds still apply. `optimized_motion_rows` reports the actual number of replaced plan rows. Review wrist tracking, sustained contact, destination support and release as well as the measured base and torso metrics before selecting a run. The flag remains opt-in.
