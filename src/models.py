from vggt.models.vggt import VGGT
import os
import torch
import gc
# transformers 只在加载 VLM 时才用到。做成惰性导入, 否则
# hand_removal (hawor 环境, 只用 VGGT + RAFT) 会因为 import 链上缺
# transformers 而整个跑不起来 —— 实测 hawor 环境有 torchvision 0.14 + RAFT
# 权重, 是跑手部去除的正确环境, 但不该被 VLM 的依赖卡住。
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except Exception:  # ModuleNotFoundError 等
    AutoModelForImageTextToText = None
    AutoProcessor = None
from omegaconf import OmegaConf
from hydra.utils import instantiate, get_method
from scipy.spatial import Delaunay
# SAM3 只在真正做分割时才用到, 同样惰性导入。它依赖 torch>=2.0
# (sam3/model/decoder.py 用了 torch.nn.attention.sdpa_kernel), 而
# hand_removal 的 hawor 环境是 torch 1.13 + torchvision 0.14 (带 RAFT 权重),
# 那个环境只跑 VGGT + 光流, 不需要 SAM3 —— 顶层导入会让它整个起不来。
try:
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
    from sam3.model.sam3_image_processor import Sam3Processor
except Exception:  # ModuleNotFoundError / torch 版本不兼容等
    build_sam3_image_model = None
    build_sam3_video_predictor = None
    Sam3Processor = None
# from src.sam3d_inference import Inference

def unload_model(model):
    if model is None:
        return None
    # Move to CPU before deleting to make CUDA memory reclaim more reliable.
    if hasattr(model, "to"):
        try:
            model.to("cpu")
        except Exception:
            pass
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    return None

def load_vggt_model():
    model_path = os.path.join('./models/VGGT')
    return VGGT.from_pretrained(model_path)

def load_sam3_image_model():
    sam3_model = build_sam3_image_model(bpe_path='./sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz', checkpoint_path='./models/SAM3/sam3.pt')
    processor = Sam3Processor(sam3_model, confidence_threshold=0.5)
    return processor

def load_sam3_video_model():
    video_predictor = build_sam3_video_predictor(checkpoint_path='./models/SAM3/sam3.pt')
    return video_predictor


def load_vggt_omega_model(checkpoint_path="/mnt/data/lza/models/vggt_omega/vggt_omega_1b_512.pt"):
    from src.vggt_omega_predict import load_vggt_omega_model as _load
    return _load(checkpoint_path)


def load_vggt4d_model(checkpoint_path="/mnt/data_8THDD/lza/workspace/robot_world_ws/src/VGGT4D/ckpts/model_tracker_fixed_e20.pt"):
    from src.vggt4d_predict import load_vggt4d_model as _load
    return _load(checkpoint_path)