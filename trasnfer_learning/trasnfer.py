# %% [code]
# Download the cat/folder to flower name map dict.

import json

# Data option 2 - Download train + validation + testing data
!wget -cq https://github.com/udacity/pytorch_challenge/raw/master/cat_to_name.json
!wget -cq https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
!rm -rf flower_data
!mkdir flower_data
!tar -xzf "flower_data.tar.gz" --directory flower_data

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

from IPython.display import HTML, display

import random
import time
import datetime
import math
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models, utils
from torch.utils.data.sampler import SubsetRandomSampler

import PIL

print("PIL.PILLOW_VERSION =", PIL.PILLOW_VERSION)

# check if CUDA is available
cuda_is_available = torch.cuda.is_available()

if cuda_is_available:
    print('CUDA is not available.')
else:
    print('CUDA is available!')

print(torch.__version__)

print("torch.backends.cudnn.version() =",torch.backends.cudnn.version())
print("torch.backends.cudnn.enabled =",torch.backends.cudnn.enabled)

# _init_fn for dataloader workers when not running the above 'fixed seed' cell.
def _init_fn(worker_id):
   pass

SMALL_SIZE = 12
MEDIUM_SIZE = 30
BIGGER_SIZE = 36

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

# Class balance check
import os

class_sample_counts = {}

for c in range(1, 103):
    l = class_sample_counts.get(c, 0)
    class_sample_counts[c] = l + len(os.listdir(train_dir + '/' + str(c)))

# Plot showing the class imbalance
plt.figure(figsize=(22, 5))
plt.bar(range(len(class_sample_counts)), list(class_sample_counts.values()), align='center')
plt.xticks(range(len(class_sample_counts)), list(class_sample_counts.keys()))
plt.xticks(rotation=80)
plt.title('Samples per Class')
plt.show()

batch_size = 32

data_transforms = {'train': transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                               ]),
                   'valid': transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                               ])}

image_datasets = {'train': ImageFolderWithPaths(train_dir, transform=data_transforms['train']),
                  'valid': ImageFolderWithPaths(valid_dir, transform=data_transforms['valid'])}

dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=5, worker_init_fn=_init_fn),
               'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=True, num_workers=5, worker_init_fn=_init_fn)}

# Process test data (if available).

data_transforms['test'] = transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                               ])

image_datasets['test'] = ImageFolderWithPaths(test_dir, transform=data_transforms['test'])

dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True, num_workers=5, worker_init_fn=_init_fn)

dataloader = dataloaders['train']  # Calc mean and std dev on valid dataset.

num_batches = len(dataloader)
num_images = len(dataloader.dataset)

print("num_images =", num_images)

sum_R = 0.0
sum_G = 0.0
sum_B = 0.0

# MEAN
for batch_idx, (images, labels, paths) in enumerate(dataloader):
    for image in images:
        numpy_image = image.numpy()

        sum_R += np.mean(numpy_image[0, :, :])
        sum_G += np.mean(numpy_image[1, :, :])
        sum_B += np.mean(numpy_image[2, :, :])


mean_R = sum_R / num_images
mean_G = sum_G / num_images
mean_B = sum_B / num_images

print("mean_R =", mean_R)
print("mean_G =", mean_G)
print("mean_B =", mean_B)

variance_sum_R = 0.0
variance_sum_G = 0.0
variance_sum_B = 0.0

# STD
for batch_idx, (images, labels, paths) in enumerate(dataloader):
    for image in images:
        numpy_image = image.numpy()

        variance_sum_R += np.mean(np.square(numpy_image[0, :, :] - mean_R))
        variance_sum_G += np.mean(np.square(numpy_image[1, :, :] - mean_G))
        variance_sum_B += np.mean(np.square(numpy_image[2, :, :] - mean_B))


std_R = math.sqrt(variance_sum_R / num_images)
std_G = math.sqrt(variance_sum_G / num_images)
std_B = math.sqrt(variance_sum_B / num_images)

print("std_R =", std_R)
print("std_G =", std_G)
print("std_B =", std_B)

# norm_mean = [0.485, 0.456, 0.406]
# norm_std = [0.229, 0.224, 0.225]

norm_mean = [0.5178361839861569, 0.4106749456881299, 0.32864167836880803]
norm_std = [0.2972239085211309, 0.24976049135203868, 0.28533308036347665]

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses

    # Counts per label
    for item in images:
        count[item[1]] += 1

    weight_per_class = [0.] * nclasses

    # Total number of images.
    N = float(sum(count))

    # super-sample the smaller classes.
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])

    weight = [0] * len(images)

    # Calculate a weight per image.
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


# Using the image datasets and the trainforms, define the dataloaders
batch_size = 32

# Transforms for the training and validation sets
data_transforms = {'train': transforms.Compose([# transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                                # transforms.CenterCrop(224),
                                                #===

                                                # transforms.RandomResizedCrop(224,
                                                #                              # scale=(0.75, 1.0), ratio=(1.0, 1.0),
                                                #                              interpolation=PIL.Image.BILINEAR),
                                                transforms.RandomAffine(45, translate=(0.4, 0.4), scale=(0.75, 1.5), shear=None, resample=PIL.Image.BILINEAR, fillcolor=0),
                                                transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                                transforms.CenterCrop(224),

                                                # transforms.RandomHorizontalFlip(),
                                                # transforms.RandomVerticalFlip(),
                                                # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.0, hue=0.1),
                                                # transforms.ColorJitter(brightness=0.025, contrast=0.0, saturation=0.0, hue=0.025),
                                                # transforms.RandomGrayscale(p=0.3),
                                                # transforms.RandomRotation(45.0, resample=PIL.Image.BILINEAR),
                                                #===
                                                # transforms.Grayscale(num_output_channels=3),
                                                transforms.ToTensor(),
                                                transforms.Normalize(norm_mean, norm_std),
                                               ]),
                   'valid': transforms.Compose([# transforms.RandomRotation(45, resample=PIL.Image.BILINEAR),
                                                transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                                transforms.CenterCrop(224),
                                                # transforms.RandomHorizontalFlip(),
                                                #===
                                                #===
                                                # transforms.Grayscale(num_output_channels=3),
                                                transforms.ToTensor(),
                                                transforms.Normalize(norm_mean, norm_std),
                                               ])}


# Load the datasets with ImageFolder
image_datasets = {'train': ImageFolderWithPaths(train_dir, transform=data_transforms['train']),
                  'valid': ImageFolderWithPaths(valid_dir, transform=data_transforms['valid'])}

weights = make_weights_for_balanced_classes(image_datasets['train'].imgs, len(image_datasets['train'].classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))


dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, sampler = sampler, num_workers=5, worker_init_fn=_init_fn),
               'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=True, num_workers=5, worker_init_fn=_init_fn)}

data_transforms['test'] = transforms.Compose([# transforms.RandomRotation(45, resample=PIL.Image.BILINEAR),
                                              transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                              transforms.CenterCrop(224),
                                              # transforms.RandomHorizontalFlip(),
                                              #===
                                              #===
                                              # transforms.Grayscale(num_output_channels=3),
                                              transforms.ToTensor(),
                                              transforms.Normalize(norm_mean, norm_std),
                                             ])

# Load the datasets with ImageFolder
image_datasets['test'] = ImageFolderWithPaths(test_dir, transform=data_transforms['test'])

dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True, num_workers=5, worker_init_fn=_init_fn)


def imshow_numpy(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))

    ax.grid(False)

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array(norm_mean)
    std = np.array(norm_std)
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

print("len(dataloaders['train'].dataset) =", len(dataloaders['train'].dataset))
print("len(dataloaders['valid'].dataset) =", len(dataloaders['valid'].dataset))

print("len(dataloaders['test'].dataset) =", len(dataloaders['test'].dataset))

images, labels, paths = next(iter(dataloaders['train']))
grid_images = utils.make_grid(images)
print(paths)
imshow_numpy(grid_images.numpy())


images, labels, paths = next(iter(dataloaders['valid']))
grid_images = utils.make_grid(images)
print(paths)
imshow_numpy(grid_images.numpy())

images, labels, paths = next(iter(dataloaders['test']))
grid_images = utils.make_grid(images)
print(paths)
imshow_numpy(grid_images.numpy())

def progress_bar(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))


valid_loss_min_A = np.Inf
valid_acc_max_A = 0

valid_loss_min_B = np.Inf
valid_acc_max_B = 0

train_losses, valid_losses = [], []

class FFClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x


def save_checkpoint(checkpoint_path, model):
    checkpoint = {
        "class_to_idx": model.class_to_idx,
        "idx_to_class": model.idx_to_class,
        "cat_to_name": model.cat_to_name,
        "state_dict": model.state_dict()
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model = models.resnet152(pretrained=False)

    for param in model.parameters():
       param.requires_grad = False

    # Put the classifier on the pretrained network
    model.fc = FFClassifier(2048, 102)

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint["class_to_idx"]
    model.idx_to_class = checkpoint["idx_to_class"]
    model.cat_to_name = checkpoint["cat_to_name"]

    return model

# Load pretrained model.
model = models.resnet152(pretrained=True)
model_requires_grad_params = []

# Freeze parameters so we don't backprop through them
for param in model.parameters():
  if param.requires_grad == True:
      # param.requires_grad = False
      model_requires_grad_params.append(param)

model.fc = FFClassifier(512, 102)

if cuda_is_available:
    model.cuda()

num_epochs = 20

# specify loss function (categorical cross-entropy)
criterion = nn.NLLLoss()

# specify optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer = optim.Adam(
    [
        {'params': model.conv1.parameters(),  'lr': 0.000001},
        {'params': model.layer1.parameters(), 'lr': 0.000001},
        {'params': model.layer2.parameters(), 'lr': 0.00001},
        {'params': model.layer3.parameters(), 'lr': 0.00001},
        {'params': model.layer4.parameters(), 'lr': 0.0001},
        {'params': model.fc.parameters(),     'lr': 0.001}
    ], lr=0.0, weight_decay=0.001)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10, verbose=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for idx, param_group in enumerate(optimizer.param_groups):
    print(idx, param_group['lr'])

def train(n_epochs = 10, manual_lrs = None, run_schedular = True):
    # Need access to some global notebook variables...
    global valid_loss_min_A
    global valid_acc_max_A
    global valid_loss_min_B
    global valid_acc_max_B
    global train_losses
    global valid_losses

    # Update the param group learning rates if supplied.
    if manual_lrs is not None:
        for idx, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = manual_lrs[idx]

    for epoch in range(1, n_epochs+1):
        print("Epoch {}".format(epoch))

        if run_schedular:
            scheduler.step()

        max_lr = 0.0  # max current lr over param groups.

        for param_group in optimizer.param_groups:
            if param_group['lr'] > max_lr:
              max_lr = param_group['lr']

        print("max_lr = {}".format(max_lr))

        # print("LRs", scheduler.get_lr())

        train_loss_sum = 0.0
        valid_loss_sum = 0.0
        train_correct_count = 0.0
        valid_correct_count = 0.0


        ##############################################
        # Choose the training and validation datasets.

        train_dataloader = dataloaders['train']
        # train_dataloader = super_train_dataloader

        valid_dataloader = dataloaders['valid']
        # valid_dataloader = dataloaders['test']
        ##############################################


        ###################
        # train the model #
        ###################
        print("Training...")

        training_start_time = time.time()
        train_display = display(progress_bar(0, 100), display_id=True)
        model.train()

        num_batches = math.ceil(len(train_dataloader.dataset) / batch_size)

        for batch_idx, (images, labels, paths) in enumerate(train_dataloader):
            # labels are the integer indexes of the class/folder names.

            if cuda_is_available:
                images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(images)  # batch_size x 102
                loss = criterion(outputs, labels)  # Average loss value over batch.
                loss.backward()
                optimizer.step()

            train_loss_sum += loss.item() * images.size(0)

            _, predicted_labels = torch.max(outputs, -1)
            train_correct_count += (predicted_labels == labels).double().sum().item()

            progress = (batch_idx+1) * 100.0 / num_batches
            train_display.update(progress_bar(progress, 100))


        ######################
        # validate the model #
        ######################
        print("Validating...")

        validation_start_time = time.time()
        valid_display = display(progress_bar(0, 100), display_id=True)
        model.eval()

        num_batches = math.ceil(len(valid_dataloader.dataset) / batch_size)

        for batch_idx, (images, labels, paths) in enumerate(valid_dataloader):
            if cuda_is_available:
                images, labels = images.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                outputs = model(images)  # batch_size x 102
                loss = criterion(outputs, labels)  # Average loss value over batch.

            valid_loss_sum += loss.item() * images.size(0)

            _, predicted_labels = torch.max(outputs, -1)
            valid_correct_count += (predicted_labels == labels).double().sum().item()

            progress = (batch_idx+1) * 100.0 / num_batches
            valid_display.update(progress_bar(progress, 100))


        epoch_end_time = time.time()

        ############################
        # calculate average losses #
        ############################
        train_loss = train_loss_sum / len(train_dataloader.dataset)
        valid_loss = valid_loss_sum / len(valid_dataloader.dataset)
        train_acc = train_correct_count / len(train_dataloader.dataset)
        valid_acc = valid_correct_count / len(valid_dataloader.dataset)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)


        #if run_schedular:
        #    scheduler.step(valid_loss)


        print('Training Loss={:.6f}  Training Accuracy={:.6f}  Duration={:.2f}'.format(train_loss,
                                                                                       train_acc,
                                                                                       validation_start_time - training_start_time))

        print('Validation Loss={:.6f}  Validation Accuracy={:.6f}  Duration={:.2f}'.format(valid_loss,
                                                                                           valid_acc,
                                                                                           epoch_end_time - validation_start_time))


        if (valid_loss < valid_loss_min_A) or ((valid_loss == valid_loss_min_A) and (valid_acc >= valid_acc_max_A)):
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min_A, valid_loss))
            torch.save(model.state_dict(), 'state_dict_best_valid_loss.pt')
            valid_loss_min_A = valid_loss
            valid_acc_max_A = valid_acc


        if (valid_acc > valid_acc_max_B) or ((valid_acc == valid_acc_max_B) and (valid_loss <= valid_loss_min_B)):
            print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_acc_max_B, valid_acc))
            torch.save(model.state_dict(), 'state_dict_best_valid_acc.pt')
            valid_loss_min_B = valid_loss
            valid_acc_max_B = valid_acc


        print()

# Freeze the earlier layers of the pre-trained network.
for param in model_requires_grad_params:
  param.requires_grad = False

# Set smaller learning rates for the earlier layers of the model.
manual_lrs = [0.000001,
              0.000001,
              0.00001,
              0.00001,
              0.0001,
              0.001]

train(20, manual_lrs, False)

# %% [code]
# Load the model that resulted in the lowest error to maybe or the highest validation accuract

model.load_state_dict(torch.load('state_dict_best_valid_loss.pt'))  # B
# model.load_state_dict(torch.load('state_dict_best_valid_acc.pt'))  # C

# %% [code]
# UN-Freeze the earlier layers of the pre-trained network!
for param in model_requires_grad_params:
  param.requires_grad = True

# Set smaller learning rates for the earlier layers of the model.
manual_lrs = [0.000001,
              0.000001,
              0.00001,
              0.00001,
              0.0001,
              0.001]

train(num_epochs, manual_lrs, True)

# %% [code]
# Load the model that resulted in the lowest error to maybe or the highest validation accuract

model.load_state_dict(torch.load('state_dict_best_valid_loss.pt'))  # B
# model.load_state_dict(torch.load('state_dict_best_valid_acc.pt'))  # C

# %% [code]
idx_to_class = {v: k for k, v in image_datasets['train'].class_to_idx.items()}

# %% [code]
# idx is the same as the label.
# class is the folder name ['1'...'102']

model.class_to_idx = image_datasets['train'].class_to_idx
model.idx_to_class = idx_to_class
model.cat_to_name = cat_to_name

# Save the checkpoint
save_checkpoint('model_checkpoint.pt', model)

# %% [code]
test_loss_sum = 0.0
test_correct_count = 0.0

##################
# test the model #
##################
testing_start_time = time.time()
print("Testing...")
test_display = display(progress_bar(0, 100), display_id=True)

model.eval()

if cuda_is_available:
    model.cuda()

num_batches = math.ceil(len(dataloaders['test'].dataset) / batch_size)

for batch_idx, (images, labels, paths) in enumerate(dataloaders['test']):

    if cuda_is_available:
        images, labels = images.cuda(), labels.cuda()

    with torch.set_grad_enabled(False):
        outputs = model(images)  # batch_size x 102
        loss = criterion(outputs, labels)  # Average loss value over batch.

    test_loss_sum += loss.item() * images.size(0)

    _, predicted_labels = torch.max(outputs, -1)
    test_correct_count += (predicted_labels == labels).double().sum().item()

    for i in range(len(labels.data)):
        if labels[i] != predicted_labels[i]:
            print(paths[i], " truth =", idx_to_class[labels[i].item()]," predicted =", idx_to_class[predicted_labels[i].item()])

    progress = (batch_idx+1) * 100.0 / num_batches
    test_display.update(progress_bar(progress, 100))


testing_end_time = time.time()

############################
# calculate average losses #
############################
test_loss = test_loss_sum / len(dataloaders['test'].dataset)
test_acc = test_correct_count / len(dataloaders['test'].dataset)

print('Testing Loss={:.6f}  Testing Accuracy={:.6f}  Duration={:.2f}'.format(test_loss,
                                                                             test_acc,
                                                                             testing_end_time - testing_start_time))

# %% [code]

torch.save(model.state_dict(), "model_flowers.pt")

# %% [code]
!ls -aoih
