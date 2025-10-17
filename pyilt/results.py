import sys
sys.path.append(".")  # Make sure your repo root is in sys.path
import os
import json
os.chdir(r"C:\Users\dakoo\OpenILT_editable")  # change to repo root
repo_root = r"C:\Users\dakoo\OpenILT_editable"
sys.path.insert(0, repo_root)
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pycommon.utils as common
from pylitho.exact import LithoSim  # or pylitho.simple if you want simpler model
# Import modules from SimpleILT/OpenILT
from pyilt.initializer import PlainInit, PixelInit
from pycommon.glp1 import Design
import torch.nn.functional as F

def IOU0(glp_file, mask_path):
    sizeX, sizeY = 1024, 1024
    downscale = 1  # Same as `down` in SimpleILT
    offsetX, offsetY = 0, 0
    
    # ---- Step 1: Load GLP design ----
    design = Design(glp_file, down=downscale)
    
    # Optional: center / crop to a specific tile (like SimpleILT does)
    design.center(sizeX, sizeY, offsetX, offsetY)
    
    # ---- Step 2: Initialize target and parameters ----
    # Use PlainInit or PixelInit
    init = PixelInit()  # or PlainInit()
    target, params = init.run(design, sizeX, sizeY, offsetX, offsetY)
    
    # ---- Step 4: Check parameter range (PixelInit gives -1/+1) ----
    print("Target tensor shape:", target.shape)
    print("Parameter range:", params.min().item(), "to", params.max().item())

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    mask = torch.tensor(mask_img, dtype=torch.float32, device='cuda')  # or 'cpu'
    
    class SimpleCfg: 
        def __init__(self, config): 
            # Read the config from file or a given dict
            if isinstance(config, dict): 
                self._config = config
            elif isinstance(config, str): 
                self._config = common.parseConfig(config)
        def __getitem__(self, key): 
            return self._config[key]
    
    cfg   = SimpleCfg(r'C:\Users\dakoo\OpenILT_editable\config\lithoiccad13.txt')
    sim = LithoSim(cfg._config)
    aerial = sim.forward(mask)[0]
    # aerial_np = aerial.detach().cpu().numpy()  # convert to NumPy
    # --- 5. Threshold resist ---
    resist = (aerial > 0.3).float()
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    aerial_np = aerial.detach().cpu().numpy()
    resist_np = resist.detach().cpu().numpy()
    # --- Ensure same size ---
    if target.shape != resist.shape:
        # Resize resist to match target shape
        resist_resized = F.interpolate(
            resist.unsqueeze(0).unsqueeze(0),  # add batch & channel dims
            size=target.shape[-2:],            # match H,W
            mode='nearest'                     # binary-friendly resize
        ).squeeze()
    else:
        resist_resized = resist
    
    # --- Binarize ---
    target_bin = (target > 0.5).float()   #Actually, there is no need to binarize.
    resist_bin = (resist_resized > 0.5).float()
    
    # --- IoU computation ---
    intersection = torch.logical_and(target_bin.bool(), resist_bin.bool()).sum().item()
    union        = torch.logical_or(target_bin.bool(), resist_bin.bool()).sum().item()
    
    iou = intersection / union if union > 0 else 0.0
    print(f"IoU between target and resist: {iou:.3f}")
    
    target_np = target.cpu().numpy()
    resist_np = resist_resized.cpu().numpy()
    
    # --- Create RGB overlay image ---
    # Red = Target only, Green = Resist only, Yellow = Overlap
    overlay = np.zeros((*target_np.shape, 3))
    overlay[..., 0] = target_np          # Red channel = target
    overlay[..., 1] = resist_np          # Green channel = resist
    # Where both are 1, red+green = yellow (overlap)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(target_np, cmap="gray")
    plt.title("Input Target")
    
    plt.subplot(1, 3, 2)
    plt.imshow(resist_np, cmap="gray")
    plt.title("Target -> Inverse -> Litho")
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlap (Yellow = Match)")
    plt.tight_layout()
    plt.show()

def IOU1(target_path):
    target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    target = torch.tensor(target_img, dtype=torch.float32, device='cuda')  # or 'cpu'
    class SimpleCfg: 
        def __init__(self, config): 
            # Read the config from file or a given dict
            if isinstance(config, dict): 
                self._config = config
            elif isinstance(config, str): 
                self._config = common.parseConfig(config)
        def __getitem__(self, key): 
            return self._config[key]
    # --- Simulate lithography ---
    cfg = SimpleCfg(r'C:\Users\dakoo\OpenILT_editable\config\lithoiccad13.txt')
    sim = LithoSim(cfg._config)
    aerial_target = sim.forward(target)[0]
    
    # --- Threshold to get resist pattern ---
    result = (aerial_target > 0.3).float()
    
    # --- Resize if necessary ---
    if result.shape != target.shape:
        result_resized = F.interpolate(
            result.unsqueeze(0).unsqueeze(0),
            size=target.shape[-2:],
            mode='nearest'
        ).squeeze()
    else:
        result_resized = result
    
    # --- Compute IoU ---
    target_bin = (target > 0.5).float()
    result_bin = (result_resized > 0.5).float()
    
    intersection = torch.logical_and(target_bin.bool(), result_bin.bool()).sum().item()
    union        = torch.logical_or(target_bin.bool(), result_bin.bool()).sum().item()
    iou = intersection / union if union > 0 else 0.0
    print(f"IoU between target and resist: {iou:.3f}")
    
    # --- Convert to NumPy for visualization ---
    target_np = target_bin.cpu().numpy()
    result_np = result_bin.cpu().numpy()
    
    # --- Create RGB overlap visualization ---
    # Red = Target only, Green = Resist only, Yellow = Overlap
    overlay = np.zeros((*target_np.shape, 3))
    overlay[..., 0] = target_np        # Red channel = target
    overlay[..., 1] = result_np        # Green channel = resist
    # Where both are 1 → yellow (R+G)
    
    # --- Plot ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(target_np, cmap="gray")
    plt.title("Target (Mask)")
    
    plt.subplot(1, 3, 2)
    plt.imshow(result_np, cmap="gray")
    plt.title("Target -> Litho")
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Overlap (IoU = {iou:.3f})\nYellow = Match")
    plt.tight_layout()
    plt.show()
