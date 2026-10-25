# 手工场景 JSON (感知失败时的 curated 类别表)

VLM 检测不准 (过分割/类别错) 的场景, 用这里的 JSON 通过 `--scene_json`
跳过 Stage 1 类别推断, 直接指定类别与关系。已验证: wxD (wx3.json)、
3chair (3chair_v2.json) 均由此路径跑通。

## 重跑命令模板
```
CUDA_VISIBLE_DEVICES=<gpu> python mainv3.py \
  --input_video <原视频路径, 见各场景旧日志首行> \
  --vggt_model vggt_omega --max_frames 100 --max_frames_stage1 10 \
  --enable_stage5 \
  --scene_json scene_jsons/<场景>.json \
  --output_path output_v3/<场景>-curated
```

## 当前覆盖的问题场景
- 112_C5_CellPhone: 蓝方桌+白抽屉柜+手持瓶 (旧跑把柜门切成 cabinet×6)
- 121_C5_CellPhone: 书桌+电话/碗/杯/盒/剪刀 (旧跑 table×2 过分割)
- 151_C7_Plate: 厨房桌+碗×N+剪刀+瓶+凳 (旧跑 29 实例, 报纸贴墙被当成物体×4)
- 002_C1_Bowl: 卧室床+枕头×N+毯子+帽子+玩具 (旧跑 11 实例互相穿模)
