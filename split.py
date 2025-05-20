import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

trainval_percent    = 0.9
train_percent       = 0.9

VOCdevkit_path      = 'VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size",tv)
    print("traub suze",tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    print('')
    print("Check datasets format, this may take a while.")

    classes_nums        = np.zeros([256], np.int)
    for i in tqdm(list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("The label image `%s` was not found; please verify that the file exists at the specified path and that its extension is `.png`."%(png_file_name))
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("The label image `%s` has shape `%s`, which is neither a grayscale nor an 8-bit color image. Please verify the dataset format carefully."%(name, str(np.shape(png))))
            print("Label images should be in grayscale or 8-bit color format, with each pixel’s value directly encoding the class to which that pixel belongs."%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            

    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("The label masks contain only pixel values 0 and 255, indicating an incorrect data format.")
        print("For the binary classification task, the label maps should be modified so that background pixels are assigned a value of 0 and target (foreground) pixels are assigned a value of 1.")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("The annotation masks contain only background pixels, indicating an incorrect data format. Please carefully verify the dataset’s formatting.")

    print("Images in the `JPEGImages` directory should use the `.jpg` extension, while those in the `SegmentationClass` directory should use the `.png` extension.")
