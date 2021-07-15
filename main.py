import random
import torch
import torch.nn as nn
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
import os
from preproc import RemoveRectangle

from torchvision import datasets, transforms

import numpy as np

from PIL import Image

from eval import eval_f1

class ImagesDataset:
    """
    Class for loading the images in a dataset. This differs from torch's ImagesFolder
    in that it preloads and transforms all of the images in main memory, instead of
    accessing them from storage. This is feasible given the small size of the dataset
    (a few k's of images) and the availability of enough main memory. For 224x224 images,
    the tensor requires 224*224*3*8 ~= 1.14 MB. For 20k images (i.e. approximately the
    dataset size), that's about 22 GB of RAM (+ some overhead memory). Larger datasets
    would not be easily stored in main memory


    Parameters:
        - path: the root path of the dataset (should contain a folder for each class)
        - classes: a list of classes (i.e. the name of the folders). The order provided here
                   will be used to encode the y's (e.g. ["R","O"] maps R to 0, O to 1)
        - transform: transformations to be applied to each image, after loading
        - limits: the number of images that should be taken from each directory, for each class
                  (e.g. { "R": 500, "O": 750 }). If not specified, ImagesDataset will read all
                  available images
    """
    def __init__(self, path, classes, transform, limits={}):  
        files = []
        for c in classes:
            # TODO: avoid hardcoding the extension here
            # and consider other possible extensions (e.g. png, jpeg)
            all_files = glob(os.path.join(path, c, "*.jpg"))

            # the list of images is shuffled, so that if limits is not empty, 
            # we select a random subset of images (avoids problems when there is
            # some naming convention among the images)
            random.shuffle(all_files)
            limit = limits.get(c, len(all_files))
            all_files = all_files[:limit]
            files.append((c, all_files))
            
        
        self.y = []
        self.X = []
        self.filenames = []
        
        for class_name, img_list in files:
            with tqdm(img_list) as bar:
                for i in bar:
                    # Each image is read and transformed, then stored (as a torch tensor) 
                    # in X
                    self.X.append(transform(Image.open(i).convert("RGB")))
                    self.filenames.append(i)
            # Since the loading of the images occurs class-wise, we can simply add
            # a bunch of identical labels to y
            self.y.extend([classes.index(class_name)] * len(img_list))
           
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        # Also returning the filename along with X (image) and y (label), 
        # it will be useful when making predictions on the test set (to know
        # which image is which)
        return self.X[i], self.y[i], self.filenames[i]

class Model(nn.Module):
    def __init__(self, model="resnext", hidden_head_size=1024):
        super().__init__()
        # resnet18, resnet152 and vgg16 are some models I tested before going with resnext, to assess
        # the degree to which different pretrained model affect the performance (spoiler: they impact
        # the results *a lot*


        if model == "resnet18": # ~ 0.8 accuracy
            # self.model_out_size stores the output of the last layer of the pretrained model (w/o head)
            # it is then used to define the input size of our Linear head)
            # TODO: we can probably figure out the output size from self.model without explicitely 
            # defining it -- however for the time being this will work :)
            self.model_out_size = 512
            # the line below takes a model (in this case resnet18), discards the last layer ([:-1])
            # and appends a flatten layer to make sure we avoid any weird output shape
            self.model = nn.Sequential(*(list(models.resnet18().children())[:-1] + [nn.Flatten()]))

        elif model == "resnet152": # ~ 0.7 accuracy
            self.model_out_size = 2048
            self.model = nn.Sequential(*(list(models.resnet152().children())[:-1] + [nn.Flatten()]))
        
        elif model == "resnext": # 0.9+ accuracy
            self.model_out_size = 2048
            # I tried the following two ResNeXt models (see here for more details https://arxiv.org/abs/1611.05431)
            # Since I did not find any particular difference in terms of performance for the
            # two models, I ultimately used resnext50 as it has a lower memory footprint (i.e. less weights)
            # thus allowing for a larger batch size

            # resnext50_32x4d: batch size: 1024
            # resnext101_32x8d: batch size: 512

            model = torch.hub.load('pytorch/vision:v0.9.0', 'resnext50_32x4d', pretrained=True)
            self.model = nn.Sequential(*(list(model.children())[:-1] + [nn.Flatten()]))
            
        elif model == "vgg16":
            self.model_out_size = 4096
            m = list(models.vgg16().children())
            m[-1] = nn.Sequential(*list(m[-1])[:2])
            self.model = nn.Sequential(m[0], m[1], nn.Flatten(), m[2])

        for layer in self.model.parameters():
            layer.requires_grad = False
            
        self.hid_linear = nn.Linear(in_features=self.model_out_size, out_features=hidden_head_size)
        self.out_linear = nn.Linear(in_features=hidden_head_size, out_features=1)
    
    def forward(self, x):
        # Model architecture (ASCII art edition)
        # { input } -> [ ResNeXt ] -> [ linear ] -> [ ReLU ] -> [ linear ] -> [ sigmoid ] -> { binary output }
        x = self.model(x)
        x = torch.relu(self.hid_linear(x))
        x = self.out_linear(x)
        return torch.sigmoid(x).flatten()


tr_list = transforms.Compose([
    RemoveRectangle(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# dataset = ImagesDataset("DATASET/TRAIN", ["R","O"], transform=tr_list, limits={ "R": 9999, "O": 12565 })
dataset = ImagesDataset("/home/dauin_user/f.giobergia/datasets/dataset/TRAIN", ["R","O"], transform=tr_list)
ds_len = len(dataset)

train_test_split = False

if train_test_split:
    train_ndx = np.random.choice(ds_len, size=int(0.8 * ds_len), replace=False)
    test_ndx = np.array(list(set(range(ds_len)) - set(train_ndx)))

    train_set = Subset(dataset, train_ndx)
    test_set = Subset(dataset, test_ndx)

    dl_train = DataLoader(train_set, batch_size=1024, pin_memory=True, shuffle=True)
    dl_test = DataLoader(test_set, batch_size=512)

else:
    dl_train = DataLoader(dataset, batch_size=1024, pin_memory=True, shuffle=True)
#     ds_test = ImagesDataset("/home/dauin_user/f.giobergia/datasets/dataset/TEST", [""], transform=tr_list)
    ds_test = ImagesDataset("dataset_stage_2/TEST", [""], transform=tr_list)
    dl_test = DataLoader(ds_test, batch_size=512)

model = Model()
model.to(device)


opt = optim.Adam(model.parameters())
loss_func = nn.BCELoss()

for i in range(100):
    print(i)
    with tqdm(dl_train) as bar:
        cumul_loss = 0
        for j, (X, y, _) in enumerate(bar):

            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()

            y_pred = model(X)
            loss = loss_func(y_pred, y.to(torch.float32))

            loss.backward()
            opt.step()
            
            cumul_loss += loss.item()

            bar.set_postfix(loss=cumul_loss / (j+1))
    
