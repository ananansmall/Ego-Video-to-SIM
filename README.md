# Ego-centric-Video-to-Simulation: ReplicateAnyScene Module

> **Based on**: [ReplicateAnyScene](https://github.com/xiac20/ReplicateAnyScene) - Zero-Shot Video-to-3D Composition via Textual-Visual-Spatial Alignment
>
> **Paper**: [arXiv:2604.10789](https://arxiv.org/abs/2604.10789)
>
> **Original Authors**: Mingyu Dong, Chong Xia, Mingyuan Jia, Weichen Lyu, Long Xu, Zheng Zhu, Yueqi Duan (Tsinghua University & Zhejiang University)

This repository contains our customized version of ReplicateAnyScene, adapted for the **Ego-Video-to-SIM** pipeline — transforming egocentric hand-object interaction videos into robot simulation environments.

## What's Changed from Upstream

| Feature | Description |
|---|---|
| `mainv2.py` | Enhanced pipeline with stage-based GLB naming, scene graph relation inference, and configurable Stage5 methods |
| `object_tracking/` | Complete object tracking pipeline: GLB-video alignment, point tracking, grasp control, trajectory refinement, and SAPIEN simulation |
| `tools/infer_relations_scene_graph.py` | SimRecon-style scene graph relation inference using VLM |
| `tools/run_post_pipeline.py` | Post-processing pipeline with configurable Stage5 methods |
| `assess/` | Visual, geometric, and textual assessment metrics |
| `docs/` | Detailed technical documentation for each pipeline stage |

## Pipeline Overview

```
Input Video
    │
    ├── Stage 1: VGGT 3D Prediction + SAM3 Segmentation
    ├── Stage 2: Object Deduplication + Instance Generation
    ├── Stage 3: Spatial Refinement + Scene Composition
    ├── Stage 4: Coordinate Alignment (optional)
    └── Stage 5: Relation Inference + Geometry Refinement (optional)
```

## Quick Start

```bash
# Basic usage
python mainv2.py --input_video ./assets/video.mp4 --output_path ./outputs/scene

# With scene graph relation inference
python mainv2.py --input_video ./assets/video.mp4 --output_path ./outputs/scene --stage5_method scene_graph

# Object tracking pipeline
python -m object_tracking.run_pipeline --glb_path ./outputs/scene/final_scene.glb --video_path ./assets/video.mp4
```

## Project Structure

```
├── mainv2.py                          # Main pipeline (enhanced)
├── src/                               # Core modules
│   ├── models.py                      # Model management
│   ├── vggt_predict.py                # VGGT 3D prediction
│   ├── object_segmentation.py         # SAM3 segmentation
│   ├── instance_generation.py         # 3D asset generation
│   └── sp_refinement.py               # Spatial refinement
├── object_tracking/                   # Object tracking & simulation
│   ├── run_pipeline.py                # Tracking pipeline
│   ├── simulation/                    # SAPIEN simulation
│   └── grasp_controller.py           # Grasp control
├── tools/                             # Post-processing tools
├── assess/                            # Quality assessment
└── docs/                              # Technical documentation
```

## Documentation

- [Technical Documentation](TECHNICAL_DOCUMENTATION.md) - Deep technical analysis
- [mainv2 Technical Doc](docs/pipeline/mainv2_technical_doc.md) - mainv2 pipeline details
- [Coordinate & Alignment](docs/design/coordinate_and_alignment.md) - Coordinate system alignment
- [Object Tracking Usage](object_tracking/USAGE.md) - Tracking pipeline guide

## Acknowledgement

This project is built upon [ReplicateAnyScene](https://github.com/xiac20/ReplicateAnyScene). We thank the original authors for their excellent work.

Related projects:
- [SimRecon](https://github.com/xiac20/SimRecon), [SAM3](https://github.com/facebookresearch/sam3), [SAM3D](https://github.com/facebookresearch/sam-3d-objects), [VGGT](https://github.com/facebookresearch/vggt), [MASt3R](https://github.com/naver/mast3r)

## Citation

If you use this work, please cite the original paper:

```bibtex
@misc{dong2026replicateanyscenezeroshotvideoto3dcomposition,
      title={ReplicateAnyScene: Zero-Shot Video-to-3D Composition via Textual-Visual-Spatial Alignment},
      author={Mingyu Dong and Chong Xia and Mingyuan Jia and Weichen Lyu and Long Xu and Zheng Zhu and Yueqi Duan},
      year={2026},
      eprint={2604.10789},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.10789},
}
```
