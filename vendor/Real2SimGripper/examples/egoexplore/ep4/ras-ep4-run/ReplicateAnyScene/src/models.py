from vggt.models.vggt import VGGT
import os
import torch
import gc
from transformers import AutoModelForImageTextToText, AutoProcessor
from omegaconf import OmegaConf
from hydra.utils import instantiate, get_method
from scipy.spatial import Delaunay
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor
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
    sam3_model = build_sam3_image_model(bpe_path='/path/to/project_data/mano_ws/do-as-i-do/reconstruction/modules/sam3/assets/bpe_simple_vocab_16e6.txt.gz', checkpoint_path='./models/SAM3/sam3.pt')
    processor = Sam3Processor(sam3_model, confidence_threshold=0.5)
    return processor

def load_sam3_video_model():
    video_predictor = build_sam3_video_predictor(checkpoint_path='./models/SAM3/sam3.pt')
    return video_predictor


def load_vggt_omega_model(checkpoint_path="/path/to/project_data/mano_ws/vggt-omega/checkpoints/VGGT-Omega-1B-512/vggt_omega_1b_512.pt"):
    from src.vggt_omega_predict import load_vggt_omega_model as _load
    return _load(checkpoint_path)


def load_vggt4d_model(checkpoint_path="/mnt/data_8THDD/lza/workspace/robot_world_ws/src/VGGT4D/ckpts/model_tracker_fixed_e20.pt"):
    from src.vggt4d_predict import load_vggt4d_model as _load
    return _load(checkpoint_path)