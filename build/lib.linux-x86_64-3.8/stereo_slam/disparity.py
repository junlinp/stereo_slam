import cv2
import numpy as np
def compute_disparity_sgbm(left_img, right_img):
    matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = matcher.compute(left_img, right_img).astype(np.float32) / 16
    disparity[disparity < 0] = 0 # mask invalid disparities
    return disparity


from omegaconf import OmegaConf
from FoundationStereo.core.foundation_stereo import *
from FoundationStereo.core.utils.utils import InputPadder

cfg = OmegaConf.load("/mnt/nas/share-all/junlinp/pretrained_models/foundation_stereo/cfg.yaml")
if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
args = OmegaConf.create(cfg)
model = FoundationStereo(args)
# ckpt = torch.load("/mnt/nas/share-all/junlinp/pretrained_models/foundation_stereo/model_best_bp2.pth")
# model.load_state_dict(ckpt['model'])
# model.cuda()
# model.eval()

def foundation_stereo_disparity(image0, image1, scale=0.25):
    original_H, original_W = image0.shape[:2]
    image0 = cv2.resize(image0, None, fx=scale, fy=scale)
    image1 = cv2.resize(image1, None, fx=scale, fy=scale)
    H, W = image0.shape[:2]
    image0_original = image0.copy()

    image0 = (torch.as_tensor(image0) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
    image1 = (torch.as_tensor(image1) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()

    padder = InputPadder(image0.shape, divis_by=32, force_square=False)

    image0, image1 = padder.pad(image0, image1)

    # print(f"image0 : {image0.shape} and image1 : {image1.shape}") 

    with torch.cuda.amp.autocast(True):
        disp = model.forward(image0, image1, iters=args.valid_iters, test_mode=True)
    disp = padder.unpad(disp.float())
    down_disp = disp.cpu().detach().numpy().reshape(H, W)
    return cv2.resize(down_disp, (original_W, original_H), interpolation=cv2.INTER_CUBIC) / scale