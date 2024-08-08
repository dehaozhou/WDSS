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
from utils_2 import *
from torch.autograd import Variable
from IPython.display import clear_output
from model.FTUNetFormer import ft_unetformer as FTUNetFormer
from skimage.color import rgb2gray
from tqdm import tqdm
import tifffile


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
print("training : ", len(train_ids)+len(PL_ids))
print("testing : ", len(test_ids))

# train_set = ISPRS_dataset1(train_ids, cache=True)
train_set = ISPRS_dataset1(train_ids, cache=False)

# train_set_PL = ISPRS_dataset2(PL_ids, cache=True)
train_set_PL = ISPRS_dataset2(PL_ids, cache=False)

train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

train_loader_PL = torch.utils.data.DataLoader(train_set_PL,batch_size=BATCH_SIZE)

base_lr = 0.005
LBABDA_BDY = 0.1
LBABDA_OBJ = 1
print("LBABDA_BDY: ", LBABDA_BDY)
print("LBABDA_OBJ: ", LBABDA_OBJ)
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        params += [{'params':[value],'lr': base_lr}]
    else:
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)

#vaihingen
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [15,25,35], gamma=0.1)
#potsdam
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [33,40,48,54], gamma=0.33)
#loveda
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20,30,40], gamma=0.33)

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    if DATASET == 'Loveda':
        eroded_labels = ((np.asarray(io.imread(ERODED_FOLDER.format(id)), dtype='int64') - 1) for id in test_ids)
    else:
        eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    # Switch the network to inference mode
    with torch.no_grad():
        for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))
            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                # Do the inference
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()

    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train_and_test(net, optimizer, epochs, scheduler=scheduler, weights=WEIGHTS, save_epoch=1, start_testing_epoch=5):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.30
    criterionb = BoundaryLoss()
    criteriono = ObjectLoss()
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()

        # Combine train_loader and train_loader_PL
        combined_loader = zip(train_loader, train_loader_PL)

        for batch_idx, ((data, boundary, object, target), (data_PL, boundary_PL, object_PL, target_PL)) in enumerate(
                combined_loader):
            # Train with labeled data
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            loss_ce = loss_calc(output, target, weights)
            loss_boundary = criterionb(output, boundary)
            loss_object = criteriono(output, object)

            # Train with pseudo-labeled data
            data_PL, target_PL = Variable(data_PL.cuda()), Variable(target_PL.cuda())
            optimizer.zero_grad()
            output_PL = net(data_PL)
            loss_ce_PL = loss_calc(output_PL, target_PL, weights)
            loss_boundary_PL = criterionb(output_PL, boundary_PL)
            loss_object_PL = criteriono(output_PL, object_PL)

            loss = loss_ce*0.6 + loss_boundary * 0.06 + loss_object * 0.6 + loss_ce_PL*0.4 + loss_boundary_PL*0.04 + loss_object_PL *0.4

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Record losses
            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 1 == 0:
                clear_output()
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                pred_PL = np.argmax(output_PL.data.cpu().numpy()[0], axis=0)
                gt_PL = target_PL.data.cpu().numpy()[0]
                print(
                    'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss_ce: {:.6f}\tLoss_boundary: {:.6f}\tLoss_object: {:.6f}\tLoss: {:.6f}\tAccuracy: {}\tLoss_ce_PL: {:.6f}\tLoss_boundary_PL: {:.6f}\tLoss_object_PL: {:.6f}\tAccuracy_PL: {}'.format(
                        e, epochs, batch_idx, len(train_loader),
                        100. * batch_idx / len(train_loader), loss_ce.data, loss_boundary.data, loss_object.data,
                        loss.data, accuracy(pred, gt),loss_ce_PL.data,loss_boundary_PL.data,loss_object_PL.data,accuracy(pred_PL, gt_PL)))

            iter_ += 1

            del (data, target, loss,data_PL, target_PL)

        if e % save_epoch == 0 and e >= start_testing_epoch:
            # We validate with the largest possible stride for faster computing
            net.eval()
            MIoU = test(net, test_ids, all=False, stride=Stride_Size)
            net.train()
            if MIoU > MIoU_best:
                if DATASET == 'Vaihingen':
                    torch.save(net.state_dict(), './Result/vaihingen/0.25/2/WDSS_epoch{}_{}'.format(e, MIoU))
                elif DATASET == 'Loveda':
                    torch.save(net.state_dict(), './Result/loveda/0.25/2/WDSS_epoch{}_{}'.format(e, MIoU))
                elif DATASET == 'Potsdam':
                    torch.save(net.state_dict(), './Result/potsdam/0.25/2/WDSS_epoch{}_{}'.format(e, MIoU))
                MIoU_best = MIoU

# Usage example:
if MODE == 'Train':
    train_and_test(net, optimizer, 70, scheduler)
elif MODE == 'Test':
    # Perform testing here after training
    if DATASET == 'Potsdam':
        net.load_state_dict(torch.load('..')) # Replace YOUR_MODEL with actual model path
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=128)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./test/po/{}.png'.format(id_), img)

    elif DATASET == 'Loveda':
        net.load_state_dict(torch.load('..')) # Replace with actual model path
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=128)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./test/lo/{}.png'.format(id_), img)

    elif DATASET == 'Vaihingen':
        net.load_state_dict(torch.load('..')) # Replace YOUR_MODEL with actual model path
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./test/va/{}.png'.format(id_), img)
