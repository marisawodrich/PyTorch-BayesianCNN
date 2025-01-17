import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import glob
import cv2
import config_bayesian as cfg

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample, label

class CustomLocalDataset(Dataset):
    def __init__(self, path, transform, source):
        self.imgs_path = path
        self.transform = transform
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for class_path in file_list:
            if source == 0:	
                class_name = class_path.split("/")[-1] # for work computer (linux)
            elif source == 1:
                class_name = class_path.split("\\")[-1] # for laptop (windows)
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
            for img_path in glob.glob(class_path + "/*.jpeg"):
                self.data.append([img_path, class_name])
        print("found data in ", str(path), " : ", len(self.data))
        self.class_map = {"Normal" : 0, "Benign": 1, "Malignant": 2}
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32)
        img = img / 255.0 
        img = self.transform(img)
        img = transforms.ToTensor()(np.array(img))
        label = self.class_map[class_name]
        sample = img

        return sample, label
    
class OOD_Dataset(Dataset):
    def __init__(self, path, transform):
        self.imgs_path = path
        self.transform = transform
        self.data = []
        class_name = "Normal" # TODO: what should we put here??

        for img_path in glob.glob(self.imgs_path + "/*.jpg"):
            self.data.append([img_path, class_name])
        for img_path in glob.glob(self.imgs_path + "/*.jpeg"):
            self.data.append([img_path, class_name])    
       
        print("found data in ", str(path), " : ", len(self.data))
        self.class_map = {"Normal" : 2, "Benign": 0, "Malignant": 1}
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32)
        img = img / 255.0 
        # normalize with mean and std of the image
        #img = (img - np.mean(img)) / np.std(img) # new
        img = self.transform(img)
        img = transforms.ToTensor()(np.array(img))
        label = self.class_map[class_name]
        sample = img

        return sample, label
    
def extract_classes(dataset, classes):
    idx = torch.zeros_like(dataset.targets, dtype=torch.bool)
    for target in classes:
        idx = idx | (dataset.targets==target)

    data, targets = dataset.data[idx], dataset.targets[idx]
    return data, targets


def getDataset(dataset):
    imgsize = cfg.imgsize
    transform_split_mnist = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.Resize((32, 32)),
        transforms.Resize((imgsize, imgsize)),
        transforms.ToTensor(),
        ])

    transform_mnist = transforms.Compose([
        #transforms.Resize((32, 32)),
        transforms.Resize((imgsize, imgsize)),
        transforms.ToTensor(),
        ])

    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    transform_pocus = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((imgsize, imgsize)), # 180x180 is the originally used size of the input images
        ])  

    transform_pocus_aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((imgsize, imgsize)), # 180x180 is the originally used size of the input images
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=(-1.0,1.0,-1.0,1.0)), # verical and horizontal shift range 0.1, shear range 0.1
        transforms.RandomResizedCrop(imgsize, scale=(0.8, 1.0), ratio=(1.0, 1.0)), # crop 128x128, scale 0.8-1.0, ratio 1-1 (for square crop)
        transforms.RandomHorizontalFlip(),
        ])
    
    if(dataset == 'CIFAR10'):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
        num_classes = 10
        inputs=3

    elif(dataset == 'CIFAR100'):
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_cifar)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_cifar)
        num_classes = 100
        inputs = 3
        
    elif(dataset == 'MNIST'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
        #num_classes = 10
        num_classes = 3 # for POCUS
        inputs = 1
        print('length of MNIST test set: ', len(testset))

    elif(dataset == 'SplitMNIST-2.1'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [0, 1, 2, 3, 4])
        test_data, test_targets = extract_classes(testset, [0, 1, 2, 3, 4])

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 5
        inputs = 1

    elif(dataset == 'SplitMNIST-2.2'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [5, 6, 7, 8, 9])
        test_data, test_targets = extract_classes(testset, [5, 6, 7, 8, 9])
        train_targets -= 5 # Mapping target 5-9 to 0-4
        test_targets -= 5 # Hence, add 5 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 5
        inputs = 1

    elif(dataset == 'SplitMNIST-5.1'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [0, 1])
        test_data, test_targets = extract_classes(testset, [0, 1])

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    elif(dataset == 'SplitMNIST-5.2'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [2, 3])
        test_data, test_targets = extract_classes(testset, [2, 3])
        train_targets -= 2 # Mapping target 2-3 to 0-1
        test_targets -= 2 # Hence, add 2 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    elif(dataset == 'SplitMNIST-5.3'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [4, 5])
        test_data, test_targets = extract_classes(testset, [4, 5])
        train_targets -= 4 # Mapping target 4-5 to 0-1
        test_targets -= 4 # Hence, add 4 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    elif(dataset == 'SplitMNIST-5.4'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [6, 7])
        test_data, test_targets = extract_classes(testset, [6, 7])
        train_targets -= 6 # Mapping target 6-7 to 0-1
        test_targets -= 6 # Hence, add 6 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    elif(dataset == 'SplitMNIST-5.5'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [8, 9])
        test_data, test_targets = extract_classes(testset, [8, 9])
        train_targets -= 8 # Mapping target 8-9 to 0-1
        test_targets -= 8 # Hence, add 8 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        inputs = 1

    elif(dataset == 'POCUS'):
        # we have different data sets (only POCUS, US and US+POCUS)
        set_type = 'US+POCUS' # choose fro POCUS or US+POCUS

        source = cfg.source # 0 for work computer, 1 for laptop

        datasetversion = cfg.datasetversion # 'new' for new data set, 'old' for old data set

        loc_pc = '/home/marisa/Documents/Thesis'
        loc_lap = 'C:/Users/maris/Documents/Thesis'
        locs = [loc_pc, loc_lap]
        loc = locs[source]

        if set_type == 'POCUS':
            train_dir  = f'{loc}/Data/POCUS/Train/' 
            val_dir = f'{loc}/Data/POCUS/Test/'
        elif set_type == 'US+POCUS':
            if datasetversion == 'new':
                train_dir  = f'{loc}/Data/Data_Dec_1/POCUS_and_US/Train/'
                val_dir = f'{loc}/Data/Data_Dec_1/POCUS_and_US/Test/'
            elif datasetversion == 'old':
                train_dir  = f'{loc}/Data/POCUS_and_US/Train/' # old training set
                val_dir = f'{loc}/Data/DATA_FOR_ISBI/GE_Test/'
            else:
                print('Error: datasetversion not recognized')
        else: 
            print('Error: set_type not recognized')
        
        if cfg.augmentation:
            tr = transform_pocus_aug
        else:
            tr = transform_pocus

        trainset = CustomLocalDataset(train_dir, transform=tr, source=source)
        testset = CustomLocalDataset(val_dir, transform=tr, source=source)
        num_classes = 3
        inputs = 1

    elif(dataset == 'OOD'):

        source = cfg.source # 0 for work computer, 1 for laptop

        loc_pc = '/home/marisa/Documents/Thesis/Data/OOD-examples/'
        loc_lap = 'C:/Users/maris/Documents/Thesis/Data/OOD-examples/'
        locs = [loc_pc, loc_lap]
        test_dir = locs[source]

        if cfg.augmentation:
            tr = transform_pocus_aug
        else:
            tr = transform_pocus

        trainset = OOD_Dataset(test_dir, transform=tr) # we don't use this
        print('length of train set: ', len(trainset))
        testset = OOD_Dataset(test_dir, transform=tr)
        num_classes = 3
        inputs = 1

    elif(dataset == 'OOD_CCA_US'):

        source = cfg.source # 0 for work computer, 1 for laptop

        loc_pc = '/home/marisa/Documents/Thesis/Data/DATA_FOR_ISBI/CCA_US/OOD/'
        loc_lap = 'C:/Users/maris/Documents/Thesis/Data/DATA_FOR_ISBI/CCA_US/OOD/'
        locs = [loc_pc, loc_lap]
        test_dir = locs[source]

        if cfg.augmentation:
            tr = transform_pocus_aug
        else:
            tr = transform_pocus

        trainset = OOD_Dataset(test_dir, transform=tr) # we don't use this
        print('length of train set: ', len(trainset))
        testset = OOD_Dataset(test_dir, transform=tr)
        num_classes = 3
        inputs = 1

    elif(dataset == 'OOD_combo'):
        loc = '/home/marisa/Documents/Thesis/Data/DATA_FOR_ISBI/Combo/'
        source = cfg.source
         # print the files in the test_dir
        if cfg.augmentation:
            tr = transform_pocus_aug
        else:
            tr = transform_pocus
        trainset = CustomLocalDataset(loc, transform=tr, source=source) # we don't use this
        testset = CustomLocalDataset(loc, transform=tr, source=source)
        print('length of OOD set: ', len(testset))
        num_classes = 3
        inputs = 1

    else:
        raise ValueError("Undefined dataset. Please specify which data set you want to use.")

    return trainset, testset, inputs, num_classes


def getDataloader(trainset, testset, valid_size, batch_size, num_workers):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader, valid_loader, test_loader
