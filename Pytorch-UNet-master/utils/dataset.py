from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import SimpleITK as sitk

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, truth_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.truth_dir = truth_dir
        self.scale = scale
        self.img_dataset,  self.truth_dataset = load_dataset(self.imgs_dir,self.truth_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.img_dataset)

    def pad(array, reference, offsets):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
        result = np.zeros(reference)
        # Create a list of slices from offset to offset + shape in each dimension
        insertHere = [slice(offset[dim], offset[dim] + array.shape[dim]) for dim in range(a.ndim)]
        # Insert the array in the result at the specified offsets
        result[insertHere] = a
        return result

    #covert to input shape as 256*256
    def preprocess(img_3d):
        img_shape = img_3d.shape

        if img_shape[1]>256:
            gap = (img_shape[1]-256)//2
            img_3d = img_3d[:,gap:gap+256,:]

        if img_shape[2]>256:
            gap = (img_shape[1]-256)//2
            img_3d = img_3d[:,:,gap:gap+256]
        #just in case the img_shape changed
        img_shape = img_3d.shape
        offsets = [0, 256-img_shape[1], 256-img_shape[2]]
        if offsets[1] != 0 or offsets[2]!=0:
            img_3d = pad(img_3d, (img_shape[0],256,256),offsets)

        return img_3d

    def load_dataset(imgs_dir, truth_dir, slice_ranges=[60,150]):
        input_imgs = glob(imgs_dir/*/*/*nii.gz)
        truth_imgs = glob(truth_dir/*/*/*nii.gz)
        depth = slice_ranges[1] - slice_ranges[0]
        img_arrays = np.array([], dtype=np.int64).reshape(depth,256,256)
        truth_arrays = np.array([], dtype=np.int64).reshape(depth,256,256)

        for i, file in enumerate(input_imgs):
            print(f'loading image file {file}')
            print(f'loading truth file {truth_imgs[i]}')
            image_3d = sitk.ReadImage(file)
            image_3d = sitk.GetArrayFromImage(image_3d)
            image_3d_s = image_3d[slice_ranges]
            image_3d_s = preprocess(image_3d_s)
            img_arrays = np.vstack([img_arrays,image_3d_s])

            truth_3d = sitk.ReadImage(file)
            truth_3d = sitk.GetArrayFromImage(truth_3d)
            truth_3d_s = truth_3d[slice_ranges]
            truth_3d_s = preprocess(truth_3d[slice_ranges])
            truth_arrays = np.vstack([truth_arrays,truth_3d_s])

        return img_arrays,truth_arrays

    def __getitem__(self, i):
        idx = self.ids[i]
        img = self.img_dataset[idx]  
        mask = self.truth_dataset[idx]
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, truth_dir, scale=1):
        super().__init__(imgs_dir, truth_dir, scale, mask_suffix='_mask')
