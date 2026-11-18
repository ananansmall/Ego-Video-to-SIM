from pathlib import Path
import cb_large_range_sim as cb
folder=Path('/path/to/project_data/mano_ws/EgoFound3R/outputs/pickplace_456483158')
out=Path.cwd()
cb.LR_RESULTS_DIR=str(out)
cb.LR_DEMO_DIR=str(out)
cb.main(targets=str(folder/'ef3r_right_targets.npz'), name='456483158_reference',
        object_npz=str(folder/'object_instance.npz'), scene_cloud=str(folder/'pointcloud/scene_cloud.npz'),
        object_physics=True, w_ori=1.0, base_interp=True, clock_mode='source')
