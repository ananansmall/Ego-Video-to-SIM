import sapien
import numpy as np

obj_path = "ReplicateAnyScene/output_v2/2chair-bottle_vggt_omega/urdf/bottle_0/mesh/bottle_0.obj"

scene = sapien.Scene()
scene.set_ambient_light([0.3, 0.3, 0.3])
scene.add_directional_light([0, -1, -1], [1.0, 1.0, 1.0])

ground_mat = sapien.SapienRenderer().create_material()
ground_mat.base_color = [0.3, 0.3, 0.3, 1]
scene.add_ground(0, render_material=ground_mat)

# Load OBJ — no material, no vertex_color shader, just the raw OBJ
actor = scene.create_actor_builder().add_visual_from_file(
    obj_path, sapien.Pose(p=[0, 0.2, 0], q=[1, 0, 0, 0])
).build()
print(f"Actor: {actor.name}")

cam = scene.add_camera("cam", width=640, height=480, fovy=np.deg2rad(45), near=0.01, far=1000)
# Look at origin from (1.5, 1.5, 1.5), looking toward (0, 0.2, 0)
forward = np.array([0, 0.2, 0], dtype=np.float64) - np.array([1.5, 1.5, 1.5], dtype=np.float64)
forward = forward / np.linalg.norm(forward)
right = np.cross(forward, np.array([0, 1, 0], dtype=np.float64))
right = right / np.linalg.norm(right)
up = np.cross(right, forward)
rot = np.column_stack([right, up, -forward])
from scipy.spatial.transform import Rotation as R
xyzw = R.from_matrix(rot).as_quat()
quat = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
cam.set_entity_pose(sapien.Pose(p=[1.5, 1.5, 1.5], q=quat))

scene.update_render()
cam.take_picture()

color = cam.get_picture("Color")
print(f"Color shape: {color.shape}")
print(f"Color min={color.min():.4f}, max={color.max():.4f}, mean={color.mean():.4f}")

# Count colored vs gray pixels
rgb = color[:, :, :3]
gray_mask = (np.abs(rgb[:, :, 0] - rgb[:, :, 1]) < 0.01) & (np.abs(rgb[:, :, 1] - rgb[:, :, 2]) < 0.01)
non_gray = (~gray_mask).sum()
print(f"Non-gray (colorful) pixels: {non_gray} / {rgb.shape[0]*rgb.shape[1]}")

# Save debug image
import cv2
rgb_uint8 = (np.clip(color[:,:,:3], 0, 1) * 255).astype(np.uint8)
cv2.imwrite("tools/test_sapien_color_debug.png", cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
print("Saved: tools/test_sapien_color_debug.png")

# Also check if OBJ vertex colors appear in any picture type
for name in cam.get_picture_names():
    try:
        img = cam.get_picture(name)
        print(f"  {name}: shape={img.shape}, min={img.min():.4f}, max={img.max():.4f}")
    except Exception as e:
        print(f"  {name}: ERROR {e}")
