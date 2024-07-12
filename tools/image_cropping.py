import os
import tifffile
import numpy as np
import cv2
import numpy as np
import os
import tifffile
import numpy as np
import os

def normalize(img):
    min_val = img.min()
    max_val = img.max()
    x = 255.0 * (img - min_val) / (max_val - min_val)
    return x

def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

split_width = 512
split_height = 512
overlap = 0.2

path = ".."
output_path = r'..'
list_ids  = ['..','..']

os.makedirs(output_path, exist_ok=True)

#########################     tif   #######################################
# filelist = os.listdir(path)
# for file in filelist:
#     if file.endswith('.tif'):
#         full_file_path = os.path.join(path, file)
#
#         # Extract ID directly from filename
#         file_base = os.path.splitext(file)[0]  # Remove extension
#         base_id = ''.join(filter(str.isdigit, file_base))  # Extract numeric characters
#         if base_id not in list_ids:
#             print(f"Warning: Could not determine base_id for file {file}. Skipping.")
#             continue
#         print("Processing:", full_file_path)
#
#         try:
#             img = tifffile.imread(full_file_path)
#             # Simplified handling for demonstration; adjust based on actual image properties
#             if img.ndim == 3 and img.shape[2] > 3:
#                 img = img[:, :, :3]
#             elif img.ndim == 2:
#                 img = img[:, :, np.newaxis]
#
#             img_h, img_w = img.shape[:2]
#             X_points = start_points(img_w, split_width, overlap)
#             Y_points = start_points(img_h, split_height, overlap)
#
#             count = 0
#             for i in Y_points:
#                 for j in X_points:
#                     split = img[i:i + split_height, j:j + split_width]
#                     if split.dtype.kind == 'f':
#                         split = (split - split.min()) * (1 / (split.max() - split.min()) * 255).astype(np.uint8)
#
#                     # Save split with the extracted base_id in naming
#                     output_filename = os.path.join(output_path, f"{base_id}_{count}.tif")  # Adding zero padding
#                     tifffile.imwrite(output_filename, split)
#                     print(f"Saved split to {output_filename}")
#                     count += 1
#         except Exception as e:
#             print(f"Error processing {full_file_path}: {e}")
#
#         print("Processing complete.")
#

#########################     png   #######################################

filelist = os.listdir(path)
for file in filelist:
    if file.endswith('.png'):
        full_file_path = os.path.join(path, file)
        # Extract numeric characters from file name to determine base_id
        base_id = ''.join(filter(str.isdigit, file))
        print("Processing:", full_file_path)

        try:
            img = cv2.imread(full_file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Error reading image {full_file_path}")
                continue

            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]

            img_h, img_w = img.shape[:2]
            X_points = start_points(img_w, split_width, overlap)
            Y_points = start_points(img_h, split_height, overlap)

            count = 0
            for i in Y_points:
                for j in X_points:
                    split = img[i:i + split_height, j:j + split_width]

                    output_filename = os.path.join(output_path, f"{base_id}_{count}.png")
                    cv2.imwrite(output_filename, split)
                    print(f"Saved split to {output_filename}")
                    count += 1
        except Exception as e:
            print(f"Error processing {full_file_path}: {e}")

print("Processing complete.")