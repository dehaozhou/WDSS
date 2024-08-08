import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pywt
from tqdm import tqdm
import logging
import torch
import torchvision
from skimage import img_as_float
from skimage.segmentation import find_boundaries
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
gpu_index = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

directory_name = r'/home/SSRS-main/SAM_RS/Postdam/img'
save_dir_sgb = r"/home//SSRS-main/SAM_RS/Postdam/sam2_sgb"
save_dir_sgo = r"/home/SSRS-main/SAM_RS/Postdam/sam2_sgo"
output_folder = r"/home/SSRS-main/SAM_RS/Postdam/sam2_3sgo"

w1, w2 = 0.05, 0.95
wavelets = ['haar', 'sym2']
mode = 'symmetric'
os.makedirs(save_dir_sgb, exist_ok=True)
os.makedirs(save_dir_sgo, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

device = 'cuda'

# SAM2 model setup
sam2_checkpoint = "/data5/open_code/pretrain/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

#原版
mask_generator = SAM2AutomaticMaskGenerator(sam2)


# mask_generator = SAM2AutomaticMaskGenerator(
#     model=sam2,
#     points_per_side=16,
#     points_per_batch=128,
#     pred_iou_thresh=0.9,
#     stability_score_thresh=0.92,
#     stability_score_offset=0.7,
#     crop_n_layers=1,
#     box_nms_thresh=0.53,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=25.0,
#     use_m2m=True,
# )



def redundant_function_one(image, mask_generator):
    return process_large_image(image, mask_generator)

def redundant_function_two(image, mask_generator):
    return process_sub_images(image, mask_generator)

def SAMAug(tI, mask_generator):
    masks = mask_generator.generate(tI)
    if len(masks) == 0:
        return None, None
    tI = img_as_float(tI)
    BoundaryPrior = np.zeros((tI.shape[0], tI.shape[1]))
    BoundaryPrior_output = np.zeros((tI.shape[0], tI.shape[1]))
    Objects_first_few = np.zeros((tI.shape[0], tI.shape[1]))
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    idx = 1
    for ann in sorted_anns:
        if ann['area'] < 50:
            continue
        if idx == 100:
            break
        m = ann['segmentation']
        color_mask = idx
        Objects_first_few[m] = color_mask
        idx = idx + 1
    for maskindex in range(len(masks)):
        thismask = masks[maskindex]['segmentation']
        mask_ = np.zeros((thismask.shape))
        mask_[np.where(thismask == True)] = 1
        BoundaryPrior = BoundaryPrior + find_boundaries(mask_, mode='thick')
    BoundaryPrior[np.where(BoundaryPrior > 0)] = 1
    BoundaryPrior_index = np.where(BoundaryPrior > 0)
    Objects_first_few[BoundaryPrior_index] = 0
    BoundaryPrior_output[np.where(BoundaryPrior > 0)] = 255
    BoundaryPrior_output = BoundaryPrior_output.astype(np.uint8)
    return BoundaryPrior_output, Objects_first_few

def process_large_image(image, mask_generator):
    BoundaryPrior_output, Objects_first_few = SAMAug(image, mask_generator)
    return BoundaryPrior_output, Objects_first_few

def process_sub_images(image, mask_generator):
    h, w = image.shape[:2]
    h_half, w_half = h // 2, w // 2
    sub_images = [
        image[:h_half, :w_half],
        image[:h_half, w_half:],
        image[h_half:, :w_half],
        image[h_half:, w_half:]
    ]
    sub_results = [SAMAug(sub_img, mask_generator) for sub_img in sub_images]
    BoundaryPrior_combined = np.zeros((h, w), dtype=np.uint8)
    Objects_first_few_combined = np.zeros((h, w), dtype=np.uint8)
    for i, (BoundaryPrior, Objects_first_few) in enumerate(sub_results):
        if BoundaryPrior is None or Objects_first_few is None:
            continue
        if i == 0:
            BoundaryPrior_combined[:h_half, :w_half] = BoundaryPrior
            Objects_first_few_combined[:h_half, :w_half] = Objects_first_few
        elif i == 1:
            BoundaryPrior_combined[:h_half, w_half:] = BoundaryPrior
            Objects_first_few_combined[:h_half, w_half:] = Objects_first_few
        elif i == 2:
            BoundaryPrior_combined[h_half:, :w_half] = BoundaryPrior
            Objects_first_few_combined[h_half:, :w_half] = Objects_first_few
        elif i == 3:
            BoundaryPrior_combined[h_half:, w_half:] = BoundaryPrior
            Objects_first_few_combined[h_half:, w_half:] = Objects_first_few
    return BoundaryPrior_combined, Objects_first_few_combined

def replace_with_original(large_result, sub_result):
    large_boundary, large_objects = large_result
    sub_boundary, sub_objects = sub_result
    combined_boundary = np.where(large_boundary > 0, large_boundary, sub_boundary)
    combined_objects = np.where(large_objects > 0, large_objects, sub_objects)
    return combined_boundary, combined_objects

def wavelet_transform_and_merge(coeffs_dict1, coeffs_dict2):
    merged_coeffs_dict = {}
    for wavelet in wavelets:
        level1 = coeffs_dict1[wavelet]
        level2 = coeffs_dict2[wavelet]
        merged_levels = []
        for i in range(len(level1)):
            merged_levels.append([[l1 * w1 + l2 * w2 for l1, l2 in zip(row1, row2)] for row1, row2 in zip(level1[i], level2[i])])
        merged_coeffs_dict[wavelet] = merged_levels
    final_images = {}
    for wavelet in wavelets:
        final_image = pywt.waverec2(merged_coeffs_dict[wavelet], wavelet, mode=mode)
        final_images[wavelet] = final_image
    return final_images

def process_wavelet(image_path1, img_name1):
    image1 = Image.open(image_path1).convert('L')
    image_array1 = np.array(image1)
    img_name2 = os.path.splitext(img_name1)[0] + '.png'
    image_path2 = os.path.join(save_dir_sgo, img_name2)
    if os.path.exists(image_path2):
        image2 = Image.open(image_path2).convert('L')
        image_array2 = np.array(image2)
        coeffs_dict1 = {}
        coeffs_dict2 = {}
        for wavelet in wavelets:
            coeffs1_single = pywt.wavedec2(image_array1, wavelet, mode=mode)
            coeffs2_single = pywt.wavedec2(image_array2, wavelet, mode=mode)
            coeffs1_single_float = [[c.astype(np.float32) for c in level] for level in coeffs1_single]
            coeffs2_single_float = [[c.astype(np.float32) for c in level] for level in coeffs2_single]
            coeffs_dict1[wavelet] = coeffs1_single_float
            coeffs_dict2[wavelet] = coeffs2_single_float
        final_images = wavelet_transform_and_merge(coeffs_dict1, coeffs_dict2)
        final_image = final_images['haar'] * 0.5 + final_images['sym2'] * 0.5
        if os.path.exists(image_path2):
            comparison_img = Image.open(image_path2).convert('L')
            comparison_array = np.array(comparison_img)
            alpha = 0.05
            threshold = 100
            gamma = 1.2
            combined_image = np.where(
                comparison_array > 5,
                np.clip((1 - (comparison_array / threshold) ** gamma) * comparison_array + alpha * final_image, 0, 255),
                comparison_array
            )
            output_image_path = os.path.join(output_folder, f"{os.path.splitext(img_name1)[0]}.png")
            combined_image_pil = Image.fromarray(combined_image.astype('uint8'))
            combined_image_pil.save(output_image_path)

def process_images_helper(img_input):
    img_name = os.path.splitext(img_input)[0]
    image_path1 = os.path.join(directory_name, img_input)
    image = np.array(Image.open(image_path1))
    large_result = redundant_function_one(image, mask_generator)
    sub_result = redundant_function_two(image, mask_generator)
    if large_result[0] is not None and large_result[1] is not None:
        combined_boundary, combined_objects = replace_with_original(large_result, sub_result)
        Image.fromarray(combined_boundary).save(os.path.join(save_dir_sgb, f"{img_name}.png"))
        Image.fromarray(combined_objects.astype(np.uint8)).save(os.path.join(save_dir_sgo, f"{img_name}.png"))
    process_wavelet(image_path1, img_input)

def combined_processing():
    img_list = sorted([f for f in os.listdir(directory_name) if f.endswith('.png') or f.endswith('.tif')])
    for img_input in tqdm(img_list, desc="Combined Processing"):
        process_images_helper(img_input)

if __name__ == '__main__':
    combined_processing()
