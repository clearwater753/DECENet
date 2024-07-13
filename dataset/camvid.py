import os.path as osp
import numpy as np
import random
import cv2
from torch.utils import data
import pickle
import torch
from torch import Tensor
import torchvision.transforms.functional as TF 
from typing import Tuple, List, Union, Tuple, Optional
from torchvision import io
import math
"""
CamVid is a road scene understanding dataset with 367 training images and 233 testing images of day and dusk scenes. 
The challenge is to segment 11 classes such as road, building, cars, pedestrians, signs, poles, side-walk etc. We 
resize images to 360x480 pixels for training and testing.
"""
class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if mask.ndim == 2:
            assert img.shape[1:] == mask.shape
        else:
            assert img.shape[1:] == mask.shape[1:]

        for transform in self.transforms:
            img, mask = transform(img, mask)

        return img, mask
class RandomRotation:
    def __init__(self, degrees: float = 10.0, p: float = 0.2, seg_fill: int = 0, expand: bool = False) -> None:
        """Rotate the image by a random angle between -angle and angle with probability p

        Args:
            p: probability
            angle: rotation angle value in degrees, counter-clockwise.
            expand: Optional expansion flag. 
                    If true, expands the output image to make it large enough to hold the entire rotated image.
                    If false or omitted, make the output image the same size as the input image. 
                    Note that the expand flag assumes rotation around the center and no translation.
        """
        self.p = p
        self.angle = degrees
        self.expand = expand
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        random_angle = random.random() * 2 * self.angle - self.angle
        if random.random() < self.p:
            img = TF.rotate(img, random_angle, TF.InterpolationMode.BILINEAR, self.expand, fill=0)
            mask = TF.rotate(mask, random_angle, TF.InterpolationMode.NEAREST, self.expand, fill=self.seg_fill)
        return img, mask
class Normalize:
    def __init__(self, mean: list = (0.485, 0.456, 0.406), std: list = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        img = img.float()
        img /= 255
        img = TF.normalize(img, self.mean, self.std)
        return img, mask

class Resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size. 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]

        # scale the image 
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

        # make the image divisible by stride
        alignH, alignW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        img = TF.resize(img, (alignH, alignW), TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (alignH, alignW), TF.InterpolationMode.NEAREST)
        return img, mask 
class RandomResizedCrop:
    def __init__(self, size: Union[int, Tuple[int], List[int]], scale: Tuple[float, float] = (0.5, 2.0), seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        """
        self.size = size
        self.scale = scale
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        # get the scale
        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        # ratio = random.uniform(min(self.scale), max(self.scale))
        scale = int(tH*ratio), int(tW*4*ratio)

        # scale the image 
        scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        # nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

        # random crop
        margin_h = max(img.shape[1] - tH, 0)
        margin_w = max(img.shape[2] - tW, 0)
        y1 = random.randint(0, margin_h+1)
        x1 = random.randint(0, margin_w+1)
        y2 = y1 + tH
        x2 = x1 + tW
        img = img[:, y1:y2, x1:x2]
        mask = mask[:, y1:y2, x1:x2]

        # pad the image
        if img.shape[1:] != self.size:
            padding = [0, 0, tW - img.shape[2], tH - img.shape[1]]
            img = TF.pad(img, padding, fill=0)
            mask = TF.pad(mask, padding, fill=self.seg_fill)
        return img, mask 
class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            return TF.hflip(img), TF.hflip(mask)
        return img, mask


class CamVidDataSet(data.Dataset):
    """ 
       CamVidDataSet is employed to load train set
       Args:
        root: the CamVid dataset path, 
        list_path: camvid_train_list.txt, include partial path

    """
    PALETTE = torch.tensor([[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128], [0, 0, 192], [128, 128, 0], [192, 128, 128], [64, 64, 128],[64, 0, 128], [64, 64, 0], [0, 128, 192]])
    def __init__(self, root='/tmp/cyl/DABNet-master/dataset/camvid/', list_path='/tmp/cyl/DABNet-master/dataset/camvid/camvid_train_list.txt', max_iters=None, crop_size=(360, 360),
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=-1):
    
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.size=[360,480]
        self.transform=Compose([
        RandomHorizontalFlip(p=0.5),
        #RandomRotation(degrees=10, p=0.2, seg_fill=seg_fill),
        #RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),
        #RandomRotation(degrees=10, p=0.2, seg_fill=0),
        RandomResizedCrop(self.size, scale=(0.5, 2.0), seg_fill=0),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            # print(img_file)
            label_file = osp.join(self.root, name.split()[1])
            
            image_name = name.strip().split()[0].strip().split('/', 1)[1].split('.')[0]
            # print(label_file)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": image_name
            })

        print("length of train set: ", len(self.files))
        #print(self.files["img"])
        #print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        #print(index)
        datafiles = self.files[index]
        #a=datafiles["img"].replace('.png', '_L.png')
        #print('********************',a)
        #print(datafiles)
        #print("#################################",len(datafiles["img"]))
        #image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        #print("###########################",image)
        #label = cv2.imread(datafiles["label"].replace('.png', '_L.png'), cv2.IMREAD_COLOR)
        #print("###########################",label)
        #label = cv2.imread(datafiles["label"], cv2.IMREAD_COLOR)
        image = io.read_image(datafiles["img"])
        #label = io.read_image(datafiles["label"].replace('.png', '_L.png'))
        label = io.read_image(datafiles["label"])
        
        
        
        size = image.shape
        #if self.scale:
            #scale = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]  # random resize between 0.5 and 2
            #f_scale = scale[random.randint(0, 5)]
            #print('############################',datafiles["img"],type(image))
            #print(f_scale)
            #f_scale = 0.5 + random.randint(0, 15) / 10.0  #random resize between 0.5 and 2
            
            #image = cv2.resize(image, None, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            
            #image = cv2.resize(image, (int(nW), int(nH)), interpolation=cv2.INTER_LINEAR)
            #print('##################################################',datafiles["label"],type(label))
            
            #label = cv2.resize(label, None, None,fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
            
            #label = cv2.resize(label,(int(nW), int(nH)), interpolation=cv2.INTER_NEAREST)


        # scale the image 
        
        
        name = datafiles["name"]
        image, label = self.transform(image, label)

        #image = np.asarray(image, np.float32)

        #image -= self.mean
        # image = image.astype(np.float32) / 255.0
        #image = image[:, :, ::-1]  # change to RGB
        #img_h, img_w,a = label.shape
        #pad_h = max(self.crop_h - img_h, 0)
        #pad_w = max(self.crop_w - img_w, 0)
        #if pad_h > 0 or pad_w > 0:
        #    img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
        #                                 pad_w, cv2.BORDER_CONSTANT,
        #                                 value=(0.0, 0.0, 0.0))
        #    label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
        #                                   pad_w, cv2.BORDER_CONSTANT,
        #                                   value=(self.ignore_label,))
        #else:
        #    img_pad, label_pad = image, label

        #img_h, img_w,a = label_pad.shape
        #h_off = random.randint(0, img_h - self.crop_h)
        #w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        #image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        #label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        #print('%%%%%%%%%%%%%%%%%%%%%%%',label)
        
        #image = image.transpose((2, 0, 1))  # NHWC -> NCHW

        #if self.is_mirror:
        #    flip = np.random.choice(2) * 2 - 1
        #    image = image[:, :, ::flip]
        #    label = label[:, ::flip]
        #print('##############################',image.size)
        label=self.encode(label).long()-1
        ###################print(torch.max(label),torch.min(label))
        #print(datafiles["img"],label.unique())
        #print(type(label))
        #print('****************',label)
        image=np.array(image)
        label=np.array(label)
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)
        return image.copy(), label.copy(), np.array(size), name
        
    def encode(self, label: Tensor) -> Tensor:
        #label=torch.from_numpy(label)
        #label=np.transpose(label,(1,2,0))
        #label = label.permute(1, 2, 0)
        label = label.permute(1, 2, 0)
        #print('######',label.shape)
        mask = torch.zeros(label.shape[:-1])
        #print('######',mask.shape)

        for index, color in enumerate(self.PALETTE):
            #label=torch.from_numpy(label)
            #print(label)
            #print(color)
            bool_mask = torch.eq(label.float(), color.float())
            #print(type(bool_mask))
            class_map = torch.all(bool_mask, dim=-1)
            mask[class_map] = index+1
            
        #print('##############################',(mask-1).unique())
        return mask

class CamVidValDataSet(data.Dataset):
    """ 
       CamVidValDataSet is employed to load val set
       Args:
        root: the CamVid dataset path, 
        list_path: camvid_val_list.txt, include partial path

    """
    PALETTE = torch.tensor([[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128], [0, 0, 192], [128, 128, 0], [192, 128, 128], [64, 64, 128],[64, 0, 128], [64, 64, 0], [0, 128, 192]])
    def __init__(self, root='', list_path='',
                 f_scale=1, mean=(128, 128, 128), ignore_label=-1):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.mean = mean
        self.f_scale = f_scale
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.transform=Compose([
        #RandomHorizontalFlip(p=0.5),
        #RandomRotation(degrees=10, p=0.2, seg_fill=seg_fill),
        #RandomResizedCrop([360,480], scale=(0.5, 2.0), seg_fill=0),
        Resize([360,480]),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            # print(img_file)
            label_file = osp.join(self.root, name.split()[1])
            # print(label_file)
            image_name = name.strip().split()[0].strip().split('/', 1)[1].split('.')[0]
            # print("image_name:  ",image_name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": image_name
            })

        print("length of Validation set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        #image_1 = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        #label_1 = cv2.imread(datafiles["label"].replace('.png', '_L.png'), cv2.IMREAD_COLOR)
        
        image = io.read_image(datafiles["img"])
        label = io.read_image(datafiles["label"].replace('.png', '_L.png'))
        
        #print(type(image))
        size = image.shape
        
        name = datafiles["name"]
        image, label = self.transform(image, label)
        #print(type(image))
        
        #if self.f_scale != 1:
        #    print('############################',type(image))
        #    image = cv2.resize(image, None, fx=self.f_scale, fy=self.f_scale, interpolation=cv2.INTER_LINEAR)
            # label = cv2.resize(label, None, fx=self.f_scale, fy=self.f_scale, interpolation = cv2.INTER_NEAREST)

        #image = np.asarray(image, np.float32)

        #image -= self.mean
        # image = image.astype(np.float32) / 255.0
        #image = image[:, :, ::-1]  # revert to RGB
        #image = image.transpose((2, 0, 1))  # HWC -> CHW
        #print(image.shape)
        #image = image.permute(2, 0, 1)
        #print(image.shape)
        label=self.encode(label).long()-1
        #print(datafiles["img"],label.unique())
        
        
        #print(type(label))
        #print('****************',label)
        #label=self.encode(label).long()
        image=np.array(image)
        label=np.array(label)
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)
        
        # print('image.shape:',image.shape)
        return image.copy(), label.copy(), np.array(size), name
        
    def encode(self, label: Tensor) -> Tensor:
        #label=torch.from_numpy(label)
        #label=np.transpose(label,(1,2,0))
        #label = label.permute(1, 2, 0)
        label = label.permute(1, 2, 0)
        #print(label)
        #print('*********************',label.shape)
        mask = torch.zeros(label.shape[:-1])
        #print(label.shape)
        for index, color in enumerate(self.PALETTE):
            #label=torch.from_numpy(label)
            #print(type(label))
            #print(type(color))
            
            bool_mask = torch.eq(label.float(), color.float())
            #bool_mask = torch.eq(torch.tensor(label).float(), color.float())
            #print(type(bool_mask))
            class_map = torch.all(bool_mask, dim=-1)
            mask[class_map] = index + 1
        #print(mask)
        return mask








class CamVidTestDataSet(data.Dataset):
    """ 
       CamVidTestDataSet is employed to load test set
       Args:
        root: the CamVid dataset path, 
        list_path: camvid_test_list.txt, include partial path

    """

    def __init__(self, root='', list_path='',
                 mean=(128, 128, 128), ignore_label=11):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            # print(img_file)
            image_name = name.strip().split()[0].strip().split('/', 1)[1].split('.')[0]
            # print(image_name)
            self.files.append({
                "img": img_file,
                "name": image_name
            })
        print("lenth of test set ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        name = datafiles["name"]

        image = np.asarray(image, np.float32)

        size = image.shape
        image -= self.mean
        # image = image.astype(np.float32) / 255.0
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))  # HWC -> CHW

        return image.copy(), np.array(size), name


class CamVidTrainInform:
    """ To get statistical information about the train set, such as mean, std, class distribution.
        The class is employed for tackle class imbalance.
    """

    def __init__(self, data_dir='', classes=11, train_set_file="",
                 inform_data_file="", normVal=1.10):
        """
        Args:
           data_dir: directory where the dataset is kept
           classes: number of classes in the dataset
           inform_data_file: location where cached file has to be stored
           normVal: normalization value, as defined in ERFNet paper
        """
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.train_set_file = train_set_file
        self.inform_data_file = inform_data_file

    def compute_class_weights(self, histogram):
        """to compute the class weights
        Args:
            histogram: distribution of class samples
        """
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readWholeTrainSet(self, fileName, train_flag=True):
        """to read the whole train set of current dataset.
        Args:
        fileName: train set file that stores the image locations
        trainStg: if processing training or validation data
        
        return: 0 if successful
        """
        global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        min_val_al = 0
        max_val_al = 0
        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            # with open(fileName, 'r') as textFile:
            for line in textFile:
                # we expect the text file to contain the data in following format
                # <RGB Image> <Label Image>
                line_arr = line.split()
                img_file = ((self.data_dir).strip() + '/' + line_arr[0].strip()).strip()
                label_file = ((self.data_dir).strip() + '/' + line_arr[1].strip()).strip()

                label_img = cv2.imread(label_file, 0)
                unique_values = np.unique(label_img)
                max_val = max(unique_values)
                min_val = min(unique_values)

                max_val_al = max(max_val, max_val_al)
                min_val_al = min(min_val, min_val_al)

                if train_flag == True:
                    hist = np.histogram(label_img, self.classes, [0, self.classes - 1])
                    global_hist += hist[0]

                    rgb_img = cv2.imread(img_file)
                    self.mean[0] += np.mean(rgb_img[:, :, 0])
                    self.mean[1] += np.mean(rgb_img[:, :, 1])
                    self.mean[2] += np.mean(rgb_img[:, :, 2])

                    self.std[0] += np.std(rgb_img[:, :, 0])
                    self.std[1] += np.std(rgb_img[:, :, 1])
                    self.std[2] += np.std(rgb_img[:, :, 2])

                else:
                    print("we can only collect statistical information of train set, please check")

                if max_val > (self.classes - 1) or min_val < 0:
                    print('Labels can take value between 0 and number of classes.')
                    print('Some problem with labels. Please check. label_set:', unique_values)
                    print('Label Image ID: ' + label_file)
                no_files += 1

        # divide the mean and std values by the sample space size
        self.mean /= no_files
        self.std /= no_files

        # compute the class imbalance information
        self.compute_class_weights(global_hist)
        return 0

    def collectDataAndSave(self):
        """ To collect statistical information of train set and then save it.
        The file train.txt should be inside the data directory.
        """
        print('Processing training data')
        return_val = self.readWholeTrainSet(fileName=self.train_set_file)

        print('Pickling data')
        if return_val == 0:
            data_dict = dict()
            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            pickle.dump(data_dict, open(self.inform_data_file, "wb"))
            return data_dict
        return None
