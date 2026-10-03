"""damping_benchmark.py — test8 阻尼对比实验 (通过临时补丁运行)"""
import subprocess, os, numpy as np

SRC = '/home/an/robot_world_ws/src/dex-retargeting/example/combination/05_gripper_test.py'
PY = '/home/an/miniconda3/envs/dex/bin/python3'
OUT = '/tmp/damping_bench'
os.makedirs(OUT, exist_ok=True)
K = 3000.0
DAMPINGS = [50, 80, 110, 150, 200, 220, 300]

header = '{:>5} {:>6} {:>10} {:>13} {:>8} {:>7} {:>7} {:>8} {:>8}'.format(
    'D', 'zeta', 'min_j1_mm', 'stable_j1_mm', 'settle_f', 'osc_mm', 'j1_std', 'lift_cm', 'behavior')
print(header)
print('-' * 85)

with open(SRC) as f:
    orig = f.read()

for D in DAMPINGS:
    patched = orig.replace(
        'GRIPPER_K = 3000.0;   GRIPPER_D = 300.0',
        'GRIPPER_K = {};   GRIPPER_D = {}'.format(K, D)
    )
    p = '{}/test8_d{}.py'.format(OUT, D)
    with open(p, 'w') as f:
        f.write(patched)

    env = os.environ.copy()
    r = subprocess.run(
        [PY, p, '--test', '8', '--num-frames', '800', '--output-dir', '{}/d_{}'.format(OUT, D)],
        capture_output=True, text=True, timeout=300,
        cwd='/home/an/robot_world_ws/src/dex-retargeting/example/combination', env=env
    )

    zeta = D / (2 * np.sqrt(K))
    log_path = '{}/d_{}/test8_frame_log.log'.format(OUT, D)
    if not os.path.exists(log_path):
        print('{:>5} {:>6}  FAILED'.format(D, zeta))
        continue

    lines = [l for l in open(log_path).readlines() if not l.startswith('#') and l.strip()]
    j1_arr, cz_arr = [], []
    for l in lines:
        parts = l.strip().split()
        if len(parts) < 25:
            continue
        j1_arr.append(float(parts[9]))
        cz_arr.append(float(parts[14]))

    if len(j1_arr) < 10:
        print('{:>5} {:>6}  PARSE_FAIL'.format(D, zeta))
        continue

    j1 = np.array(j1_arr)
    cz = np.array(cz_arr)
    min_j1 = float(np.min(j1))
    stable_j1 = float(j1[-10:].mean())
    settle = 700
    for i in range(200, len(j1)):
        if abs(j1[i] - stable_j1) < 1.0:
            settle = i
            break
    osc = float(np.max(j1) - min_j1)
    j1_std = float(np.std(j1[400:560]))
    lift = float(np.max(cz) - 0.0125) * 100
    label = 'under' if zeta < 0.95 else ('crit' if zeta < 1.05 else 'over')
    print('{:>5} {:>6} {:>10} {:>13} {:>8} {:>7} {:>7} {:>8} {:>8}'.format(
        D, zeta, min_j1, stable_j1, settle, osc, j1_std, lift, label))

with open(SRC, 'w') as f:
    f.write(orig)

print()
print('zeta = D / (2*sqrt(K*mass))  K=3000  mass~1  crit~110')
print('under:欠阻尼  crit:临界  over:过阻尼')
