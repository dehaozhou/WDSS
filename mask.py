import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random, time
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils_mask import *
from torch.autograd import Variable
from IPython.display import clear_output
from model.FTUNetFormer_1 import ft_unetformer as FTUNetFormer
from skimage.color import rgb2gray
from tqdm import tqdm
import tifffile
import numpy as np
from scipy.special import softmax
from scipy.stats import mode
from skimage.transform import resize
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
# from skimage.morphology import remove_small_objects
from skimage import io
from skimage import morphology
import numpy as np
from scipy.special import softmax
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
from skimage.measure import regionprops
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from scipy.special import softmax
import numpy as np
import torch
from tqdm import tqdm
import tifffile
import numpy as np
from skimage import io
from tqdm import tqdm
from skimage.transform import resize
import albumentations as A
from albumentations.pytorch import ToTensorV2


def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()


try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

if MODEL == 'FTUNetformer':
    net = FTUNetFormer(num_classes=N_CLASSES).cuda()
params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)

# Load the datasets
print("training : ", len(train_ids))
print("testing : ", len(test_ids))
train_set = ISPRS_dataset(train_ids, cache=True)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

base_lr = 0.005
LBABDA_BDY = 0.1
LBABDA_OBJ = 1
print("LBABDA_BDY: ", LBABDA_BDY)
print("LBABDA_OBJ: ", LBABDA_OBJ)
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)

# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30,31, 37,38], gamma=0.33)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30,34,44], gamma=0.1)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25,35,45], gamma=0.1)


# # 生成伪标签
def test1(net, test_ids, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):

    def apply_adaptive_threshold(pred, n_classes=N_CLASSES, threshold_multiplier=3.5):
        pred_softmax = softmax(pred, axis=-1)
        adaptive_mask = np.zeros_like(pred_softmax)
        for c in range(n_classes):
            avg_confidence = np.mean(pred_softmax[:, :, c])
            adaptive_threshold = avg_confidence * threshold_multiplier
            adaptive_mask[:, :, c] = (pred_softmax[:, :, c] >= adaptive_threshold).astype(int)
        return adaptive_mask
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    with torch.no_grad():
        for img, id_ in tqdm(zip(test_images, test_ids), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                         leave=False)):
                # Build the tensor and do the inference steps remain unchanged...
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda())

                # Do the inference
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del outs
            pred = apply_adaptive_threshold(pred)
            img = convert_to_color(np.argmax(pred, axis=-1))
            img_id = id_.split('/')[-1].split('.')[0]
            io.imsave('./Result/vaihingen/0.25/1/lable_PL/' + img_id + '.tif', img)


#对训练集覆盖掩码
def test2(net, test_ids, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):

    def apply_adaptive_threshold(pred, n_classes=N_CLASSES, threshold_multiplier=2.5):
        pred_softmax = softmax(pred, axis=-1)
        adaptive_mask = np.zeros_like(pred_softmax)
        for c in range(n_classes):
            avg_confidence = np.mean(pred_softmax[:, :, c])
            adaptive_threshold = avg_confidence * threshold_multiplier
            adaptive_mask[:, :, c] = (pred_softmax[:, :, c] >= adaptive_threshold).astype(int)
        return adaptive_mask
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    with torch.no_grad():
        for img_rgb, id_ in tqdm(zip(test_images, test_ids), total=len(test_ids), leave=False):
            # print(f"Original Image {id_} shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
            pred = np.zeros(img_rgb.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img_rgb, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img_rgb, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img_rgb[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda())

                # Do the inference
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del outs
            pred = apply_adaptive_threshold(pred)
            white_mask_indices = np.argwhere(pred.sum(axis=-1) == 0)
            white_mask = np.zeros_like(img_rgb, dtype=np.uint8)
            for pos in white_mask_indices:
                white_mask[pos[0], pos[1], :] = 255
            masked_img = img_rgb * (255 - white_mask) + white_mask
            masked_img_clipped = np.clip(masked_img, 0, 255).astype(np.uint8)
            output_path = './Result/vaihingen/0.25/1/img_PL/' + id_ + '.png'
            io.imsave(output_path, masked_img_clipped, plugin='tifffile')


if MODE == 'Test':
    if DATASET == 'Potsdam':
        net.load_state_dict(torch.load('./Result/potsdam/0.25/1/Potsdam_PL_gen_model'))
    elif DATASET == 'Loveda':
        net.load_state_dict(torch.load('./Result/loveda/0.25/1/Loveda_PL_gen_model'))
    elif DATASET == 'Vaihingen':
        net.load_state_dict(torch.load('./Result/vaihingen/0.25/1/Vaihingen_PL_gen_model'))

    net.eval()
    test1(net, test_ids, stride=32)
    test2(net, test_ids, stride=32)
