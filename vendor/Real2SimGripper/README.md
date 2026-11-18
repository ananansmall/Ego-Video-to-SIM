# Real2SimGripper source snapshot

Source revision 50ee6bfe2983bb3bdafe34eb5b1ef599a92f3827.

The snapshot provides Python and shell source, the R1 tracking and simulation code, RoboTwin-derived environments, configurations, and example runners. Start with [scripts/large_range_track.py](scripts/large_range_track.py), [scripts/pipeline.py](scripts/pipeline.py), or the [script guide](scripts/README.md). The Chairbottle acceptance snapshot includes its [archived replay runner](examples/r1_multicontact/runs/pixel-umi-ear-20261118/chair_joint_seat/recovery/runner/event_replay.py).

Supply R1 assets, input scenes and target packages, SAPIEN and RoboTwin dependencies, and the upstream reconstruction models separately for execution. Set example paths under /path/to/ to suit your local environment. [SOURCE_EXPORT.json](SOURCE_EXPORT.json) lists exclusions and files with path or host substitutions.

Preserve the [RoboTwin MIT license notice](LICENSE-RoboTwin) when redistributing the derived environment code. See also the [Ego-Video-to-SIM license](LICENSE).
