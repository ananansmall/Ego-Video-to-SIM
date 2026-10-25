# FP 跟踪效果可视化

FoundationPose 6DoF 跟踪在 `output_v2/7_vggt_omega` 上的批量评测产物。

## 打开方式

**本地文件直接打开**（推荐，无需服务器）：

```
test/fp_vis/index.html
```

VSCode / IDE 里如果内置预览打不开（会注入 Vite 代理层拦截静态页），
用系统浏览器打开，或起个静态服务：

```bash
cd test/fp_vis && python -m http.server 8767
# http://127.0.0.1:8767/index.html
```

## 图例

| 颜色 | 含义 |
|---|---|
| 绿色轮廓 | FoundationPose 位姿下 mesh 的面片光栅化投影 |
| 红色轮廓 | SAM3 真值 mask |
| 橙色点 | 历史轨迹尾迹（最近 40 帧质心）|

两者重合得好 = 跟得准。

## 结果

| 实例 | mesh 尺寸 | register IoU | 全帧 IoU | 逐帧 step 中位 |
|---|---|---|---|---|
| plate_inst0 | 250×251×25 mm | 0.965 | 0.887 | 15 mm |
| toy_inst2 | 83×121×54 mm | 0.824 | 0.146 | 28 mm |
| toy_inst0 | 90×112×79 mm | 0.752 | 0.653 | 37 mm |
| block_inst2 | 67×68×35 mm | 0.696 | 0.467 | 38 mm |
| toy_inst1 | 69×99×83 mm | 0.687 | 0.318 | 53 mm |
| block_inst0 | 82×87×21 mm | 0.649 | 0.263 | 26 mm |
| block_inst1 | 93×77×50 mm | 0.644 | 0.162 | 38 mm |
| toy_inst3 | 18×37×36 mm | 0.397 | 0.107 | 21 mm |

## 结论

1. **register（首帧位姿）全部准确**：IoU 0.40~0.97，8/8 mask 自动匹配成功。
2. **track（逐帧跟踪）会累积抖动**：静止物体上路径长达 3~7m，
   而真实运动只有 0.002~0.34m（ICP 实测），逐帧 step 中位 15~53mm。
3. toy_inst3IoU 低是因为物体本身只有 18mm，不是跟踪失败。

## 目录结构

```
test/fp_vis/
├── index.html              索引页
├── <instance>/
│   ├── small.gif           压缩动画 (1.4MB, 38帧)
│   └── traj.png            轨迹图(XZ 轨迹 / 各轴 / 逐帧 step)
└── json/
    ├── <instance>.json              逐帧 IoU 报告
    └── <instance>_mask_select.json  register IoU 选 mask 的候选打分
```

## 复现命令

```bash
S=output_v2/7_vggt_omega
# 1. 选 mask(对每个候选 _pre_dedup mask 跑 register, 取 IoU 最高)
# 2. 准备数据(去重后实例 + original_mesh 施加 T + 选中 mask)
# 3. 跟踪
# 4. 评测 + 可视化
python tools/analysis/fp_batch_eval.py --scene_dir $S --gpu 1
python tools/analysis/fp_overlay_batch.py --scene_dir $S --stride 3
python tools/analysis/fp_build_index.py --scene_dir $S
```