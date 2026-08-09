"""
Quick verification: after fixing duplicate _render_to_sapien, check if transforms work.
"""
import numpy as np
import trimesh

p = np.load('/home/an/data/hawor/121_C5_CellPhone_161deg/output/alignment/transform_params.npz')
s = float(p['scale_ratio'])
R_h2g = p['R_hand_to_glb']
t_h2g = p['t_hand_to_glb']
Rx_hand = p['Rx_hand']
R_c2w_h = p['R_c2w_hawor0']
t_c2w_h = p['t_c2w_hawor0']

# R_AXIS from common.py
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

d = np.load('/home/an/data/hawor/121_C5_CellPhone_161deg/reconstruction/hawor_results_0_600.npz', allow_pickle=True)
j_slam = d['pred_trans'][1][56]  # right hand, frame 56

# render_quick.py reference
R_hand = R_h2g @ Rx_hand
j_glb = s * R_hand @ j_slam + t_h2g
cam_glb = s * R_h2w_h + t_c2w_h

# 002 fix (what _render_to_sapien should produce with global params)
j_sapien = R_AXIS @ (s * R_h2g @ Rx_hand @ j_slam + t_h2g)
cam_sapien = R_AXIS @ (s * R_h2g @ t_c2w_h + t_h2g)

# GLB phone
glb = trimesh.load('/home/an/data/ras/121_C5_CellPhone_161deg_vggt_omega/final_scene.glb')
phone = glb.geometry['geometry_1'].vertices
phone_glb = phone.mean(0)
phone_sapien = R_AXIS @ phone_glb

print("="*70)
print("VERIFICATION: 002 vs render_quick.py")
print("="*70)

print(f"\nHand in GLB:   {j_glb}")
print(f"Hand in SAPIEN: {j_sapien}")
print(f"Cam  in GLB:   {cam_glb}")
print(f"Cam  in SAPIEN: {cam_sapien}")
print(f"Phone in GLB:   {phone_glb}")
print(f"Phone in SAPIEN: {phone_sapien}")

print(f"\nDistances (GLB):")
print(f"  Hand-phone: {np.linalg.norm(j_glb - phone_glb):.4f}m")
print(f"  Cam-phone:  {np.linalg.norm(cam_glb - phone_glb):.4f}m")
print(f"  Hand-cam:   {np.linalg.norm(j_glb - cam_glb):.4f}m")

print(f"\nDistances (SAPIEN):")
print(f"  Hand-phone: {np.linalg.norm(j_sapien - phone_sapien):.4f}m")
print(f"  Cam-phone:  {np.linalg.norm(cam_sapien - phone_sapien):.4f}m")
print(f"  Hand-cam:   {np.linalg.norm(j_sapien - cam_sapien):.4f}m")

print(f"\nMatch (pure rotation preserves distances):")
print(f"  Hand-phone: {np.isclose(np.linalg.norm(j_sapien - phone_sapien), np.linalg.norm(j_glb - phone_glb))}")
print(f"  Cam-phone:  {np.isclose(np.linalg.norm(cam_sapien - phone_sapien), np.linalg.norm(cam_glb - phone_glb))}")
print(f"  Hand-cam:   {np.isclose(np.linalg.norm(j_sapien - cam_sapien), np.linalg.norm(j_glb - cam_glb))}")
