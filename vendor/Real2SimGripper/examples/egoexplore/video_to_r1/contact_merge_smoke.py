import numpy as np
from scipy.spatial.transform import Rotation
from merge_agile_contact_windows import merge

frames=np.arange(10, 20, dtype=int)
n=len(frames)
pad=np.broadcast_to(np.eye(4),(n,4,4)).copy()
pad[:,:3,:3]=Rotation.from_euler('xyz',[.33,.41,.07]).as_matrix()
pad[:,0,3]=np.arange(n)/10
obj=pad.copy()
obj[:,2,3]=.1
gripper=pad.copy()
gripper[:,1,3]=.02
relative=np.linalg.inv(pad)@obj
g_relative=np.linalg.inv(gripper)@obj
grip=np.full((n,2),.03)
ids=np.array(['item']*n)
baseline=dict(src_frames=frames,T_world_pad_retarget=pad,T_world_pad=pad,
              T_world_object=obj,grip_retarget=grip,grip=grip,object_ids=ids,
              T_world_gripper=gripper,T_pad_object=relative,T_gripper_object=g_relative)
patches=[]
for i in (0,5):
  p=pad[i:i+5].copy()
  p[:,1,3]+=.006
  patches.append((dict(window_id=str(i),object_id='item',first_frame=frames[i],last_frame=frames[i+4],
                      contact_evidence_frames=[frames[i+2]]),
                  dict(source_frames=frames[i:i+5],object_id=np.array('item'),T_world_pad=p,
                       T_world_object=obj[i:i+5],grip=grip[i:i+5],weight=np.array([0.,.5,1.,.5,0.]))))
new,s=merge(baseline,patches)
outside=np.array([0,4,5,9])
for k in ('T_world_pad','T_world_object','grip','T_world_gripper','T_pad_object','T_gripper_object'):
  assert np.array_equal(baseline[k][outside],new[k][outside]), k
assert np.allclose(new['T_world_pad']@new['T_pad_object'],new['T_world_object'])
assert np.allclose(new['T_world_gripper']@new['T_gripper_object'],new['T_world_object'])
assert [entry['changed_source_frames'] for entry in s]==[[11,12,13],[16,17,18]]
try:
  merge(baseline,[patches[0],patches[0]])
except ValueError as e:
  assert 'overlapping' in str(e)
else:
  raise AssertionError('overlap accepted')
print('Two source windows, consistent dependent transforms, exact outside equality, overlap rejected')
