from transformers import AutoImageProcessor, AutoModel
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
import cv2
import torch
import einops

matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

# === Helper: Convert OpenCV to PIL ===
def cv2_to_tensor(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = einops.rearrange(img, 'h w c -> c h w')
    return torch.tensor(img / 255.0, dtype=torch.float).unsqueeze(0).cuda()

extractor = SuperPoint(max_num_keypoints=4096).eval().cuda()  # load the extractor

def extract_keypoints(image):
    feats = extractor.extract(cv2_to_tensor(image))
    return feats

def match_keypoints(feats0, feats1):
    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    keypoints0 = feats0['keypoints'][matches[..., 0]]
    keypoints1 = feats1['keypoints'][matches[..., 1]]
    return matches.cpu().numpy(),keypoints0.cpu().numpy(), keypoints1.cpu().numpy()

def match_keypoints_original(image0, image1):
    feats0 = extractor.extract(cv2_to_tensor(image0))
    feats1 = extractor.extract(cv2_to_tensor(image1))
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01['matches'] # indices with shape (K,2)
    keypoints0 = feats0['keypoints'][matches[..., 0]] # coordinates in imag
    keypoints1 = feats1['keypoints'][matches[..., 1]] # coordinates in imag
    return keypoints0.cpu().numpy(), keypoints1.cpu().numpy()

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").eval()
def get_embeddings(image):
    image = processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**image)
        embeddings = outputs.last_hidden_state[:, 0, :]
    # (1, 768)
    return embeddings.cpu().numpy()


