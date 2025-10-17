import sys
import os
os.chdir(r"C:\Users\dakoo\OpenILT_editable") #idk what to do abt this
repo_root = r"C:\Users\dakoo\OpenILT_editable"
sys.path.insert(0, repo_root)
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

from pycommon.settings import *
import pycommon.utils as common
import pycommon.glp as glp
import pycommon.glp1 as glp1
import pylitho.simple as lithosim
# import pylitho.exact as lithosim
from PIL import Image
import pyilt.initializer as initializer
import pyilt.evaluation as evaluation
import matplotlib.pyplot as plt

device = 'cuda' 

#OpenILT test dataset
# testimg_path = r"C:\Users\dakoo\OpenILT_editable\tmp\MOSAIC_test1.png"
# target_folder = r"C:\Users\dakoo\claude fcn\lithodata\MetalSet\target"   
# output_folder = r"C:\Users\dakoo\OpenILT_editable\outputs\Inverse"
# os.makedirs(output_folder, exist_ok=True)  # make sure folder exists


#My dataset
testimg_path = r"C:\Users\dakoo\claude fcn\lithodata\MetalSet\target\cell0.png" #User input here
# target_folder = r"C:\Users\dakoo\claude fcn\lithodata\MetalSet\target"   
output_folder = r"C:\Users\dakoo\OpenILT_editable\outputs\Inverse"
os.makedirs(output_folder, exist_ok=True)  # make sure folder exists

target_img = cv2.imread(testimg_path, cv2.IMREAD_GRAYSCALE)
target_img = target_img.astype(np.float32) / 255.0

target = torch.tensor(target_img, dtype=torch.float32, device='cuda')  # or 'cpu'
params = torch.zeros_like(target, dtype=torch.float32, device='cuda')  # initial mask

class SimpleCfg: 
    def __init__(self, config): 
        # Read the config from file or a given dict
        if isinstance(config, dict): 
            self._config = config
        elif isinstance(config, str): 
            self._config = common.parseConfig(config)
        required = ["Iterations", "TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize", 
                    "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in required: 
            assert key in self._config, f"[SimpleILT]: Cannot find the config {key}."
        intfields = ["Iterations", "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in intfields: 
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize"]
        for key in floatfields: 
            self._config[key] = float(self._config[key])
    
    def __getitem__(self, key): 
        return self._config[key]
        
class SimpleILT: 
    def __init__(self, config=SimpleCfg("./config/simpleilt2048.txt"), lithosim=lithosim.LithoSim("./config/lithosimple.txt"), device=DEVICE, multigpu=False): 
        super(SimpleILT, self).__init__()
        self._config = config
        self._device = device
        # Lithosim
        self._lithosim = lithosim.to(DEVICE)
        if multigpu: 
            self._lithosim = nn.DataParallel(self._lithosim)
        # Filter
        self._filter = torch.zeros([self._config["TileSizeX"], self._config["TileSizeY"]], dtype=REALTYPE, device=self._device)
        self._filter[self._config["OffsetX"]:self._config["OffsetX"]+self._config["ILTSizeX"], \
                     self._config["OffsetY"]:self._config["OffsetY"]+self._config["ILTSizeY"]] = 1
    
    def solve(self, target, params, curv=None, verbose=0): 
        # Initialize
        if not isinstance(target, torch.Tensor): 
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        if not isinstance(params, torch.Tensor): 
            params = torch.tensor(params, dtype=REALTYPE, device=self._device)
        backup = params
        params = params.clone().detach().requires_grad_(True)

        # Optimizer 
        opt = optim.SGD([params], lr=self._config["StepSize"])
        # opt = optim.Adam([params], lr=self._config["StepSize"])

        # Optimization process
        lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
        bestParams = None
        bestMask = None
        for idx in range(self._config["Iterations"]): 
            mask = torch.sigmoid(self._config["SigmoidSteepness"] * params) * self._filter
            mask += torch.sigmoid(self._config["SigmoidSteepness"] * backup) * (1.0 - self._filter)
            printedNom, printedMax, printedMin = self._lithosim(mask)
            l2loss = func.mse_loss(printedNom, target, reduction="sum")
            pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(printedMin, target, reduction="sum")
            pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
            pvband = torch.sum((printedMax >= self._config["TargetDensity"]) != (printedMin >= self._config["TargetDensity"]))
            loss = l2loss + self._config["WeightPVBL2"] * pvbl2 + self._config["WeightPVBand"] * pvbloss
            if not curv is None: 
                kernelCurv = torch.tensor([[-1.0/16, 5.0/16, -1.0/16], [5.0/16, -1.0, 5.0/16], [-1.0/16, 5.0/16, -1.0/16]], dtype=REALTYPE, device=DEVICE)
                curvature = func.conv2d(mask[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
                losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
                loss += curv * losscurv
            if verbose == 1: 
                print(f"[Iteration {idx}]: L2 = {l2loss.item():.0f}; PVBand: {pvband.item():.0f}")

            if bestParams is None or bestMask is None or loss.item() < lossMin: 
                lossMin, l2Min, pvbMin = loss.item(), l2loss.item(), pvband.item()
                bestParams = params.detach().clone()
                bestMask = mask.detach().clone()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        return l2Min, pvbMin, bestParams, bestMask

def save_mask_image(mask, output_dir="outputs", filename="mask.png"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path for the file
    filepath = os.path.join(output_dir, filename)
    
    # Save the mask using matplotlib
    plt.figure(figsize=(6,6))
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory
    
    return filepath


#For non-ILT datasets.
def serial0(img_pth, output_folder="outputs"): 
    SCALE = 1
    l2s, pvbs, epes, shots, runtimes = [], [], [], [], []

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    cfg   = SimpleCfg("./config/simpleilt2048.txt")
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    solver = SimpleILT(cfg, litho)

    # --- Design and initialization ---
    design = glp1.Design(img_pth, down=SCALE)
    design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
    target, params = initializer.PixelInit().run(design, cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
    
    # --- Solve ILT ---
    begin = time.time()
    l2, pvb, bestParams, bestMask = solver.solve(target, params, curv=None)
    runtime = time.time() - begin
    
    # --- Evaluation ---
    ref = glp1.Design(testimg_path, down=1)
    ref.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
    target, params = initializer.PixelInit().run(ref, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
    l2, pvb, epe, shot = evaluation.evaluate(bestMask, target, litho, scale=SCALE, shots=True)

    # --- Save best mask as matplotlib image ---
    mask_np = (bestMask.detach().cpu().numpy() * 255).astype(np.uint8)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_filename = f"cell_{timestamp}.png"
    output_path = os.path.join(output_folder, unique_filename)

    plt.figure(figsize=(6,6))
    plt.imshow(mask_np, cmap='gray')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # print(f"[Cell0]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}; SolveTime: {runtime:.2f}s")


    # --- Show the best mask using matplotlib ---
    plt.figure(figsize=(6,6))
    plt.imshow(bestMask.detach().cpu().numpy(), cmap='gray')
    plt.title("Inverse Mask (Best Result)")
    plt.axis('off')
    plt.show()

    # --- Store metrics ---
    l2s.append(l2)
    pvbs.append(pvb)
    epes.append(epe)
    shots.append(shot)
    runtimes.append(runtime)

    print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; Shot {np.mean(shots):.1f}; SolveTime {np.mean(runtimes):.2f}s")
    
    # Return path so the agent can access the image
    return output_path


def serial1(img_pth): 
    SCALE = 1
    l2s = []
    pvbs = []
    epes = []
    shots = []
    runtimes = []
    cfg   = SimpleCfg("./config/simpleilt2048.txt")
    litho = lithosim.LithoSim("./config/lithosimple.txt")

    solver = SimpleILT(cfg, litho)

    design = glp1.Design(img_pth, down=SCALE)
    design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
    target, params = initializer.PixelInit().run(design, cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
    
    begin = time.time()
    l2, pvb, bestParams, bestMask = solver.solve(target, params, curv=None)
    runtime = time.time() - begin
    
    ref = glp1.Design(testimg_path, down=1)
    ref.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
    target, params = initializer.PixelInit().run(ref, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
    l2, pvb, epe, shot = evaluation.evaluate(bestMask, target, litho, scale=SCALE, shots=True)

    # --- Show the best mask using matplotlib ---
    plt.figure(figsize=(6,6))
    plt.imshow(bestMask.detach().cpu().numpy(), cmap='gray')
    plt.title("Inverse Mask (Best Result)")
    plt.axis('off')
    plt.show()

    l2s.append(l2)
    pvbs.append(pvb)
    epes.append(epe)
    shots.append(shot)
    runtimes.append(runtime)

    print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; Shot {np.mean(shots):.1f}; SolveTime {np.mean(runtimes):.2f}s")


