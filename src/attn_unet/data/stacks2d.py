import os
import glob
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import torch.nn.functional as functional

'''
Dataset loader to generate a sequence of 2d images from the MSD Pancreas

- in resize: can create a nearest neighbor version for the labels (more realistic)
- can possibly add data augmentations (flip. rotate, crop and resize?), but i think for PoC it's enough
'''


def clip_norm_HU(data, low_HU=-150, high_HU=250, epsilon=1e-8):
    """
    clip and normalize CT in HU range
    using a VT database, and caring about internal organs we can exclude air, bones (above 300HU)
    and lungs (SAT ends around -115, lungs are below that)
    https://radiopaedia.org/articles/hounsfield-unit
    """
    data = np.clip(data, low_HU, high_HU)
    data = (data - low_HU) / (high_HU - low_HU + epsilon)
    return data


def binarize_labels(arr):
    # MSD Pancreas labels: 0 = background, 1 = organ, 2 = tumor (not relevant for Spleen task)
    # Unite the anatomy labels so we get a segmentation map
    return (arr > 0).astype(np.uint8)

def normalize_img(volume, clip_percent=(1.0, 99.0), eps=1e-6):
    volume_data = volume.astype(np.float32)
    lo, hi = np.percentile(volume_data, clip_percent)

    # for normalization issues:
    if hi <= lo:
        lo, hi = volume_data.min(), volume_data.max()

    # clip data
    volume_data = np.clip(volume_data, lo, hi)

    # normalize
    volume_data = (volume_data-lo) / max(hi - lo, eps)

    return volume_data

class Stacks2D(Dataset):
    """
    Take PyTorch dataset and create a stack of 2D slices
    """
    def __init__(self,
                 img_dir, label_dir,
                 img_size=224, k_slices=1,  # k=1 -> 2D slice, k>1 -> stacks of 2D slices
                 split='train', split_ratio=(0.8, 0.1, 0.1), seed=3360033,
                 augment=False):

        super().__init__()
        self._cache = {}
        self.k = k_slices
        self.size = img_size
        self.split = split
        self.augment = augment and (split == "train") # allow only for train set

        # Get images (NifTI format) and labels
        imgs = sorted(glob.glob(os.path.join(img_dir, '*.nii*')))
        labels = [os.path.join(label_dir, os.path.basename(image)) for image in imgs]
        # Check for missing labels
        assert all(os.path.exists(lab) for lab in labels), "Missing labels"

        # Shuffle and split at volume level
        rng = (np.random.RandomState(seed))
        idx = rng.permutation(len(imgs))

        # Get ratios
        r_tr, r_va, r_test = split_ratio

        # Get rounded counts, and make sure none are empty
        n_tr = int(round(len(imgs) * r_tr))
        n_va = int(round(len(imgs) * r_va))
        n_te = max(len(imgs) - n_tr - n_va, 0)

        # Make sure non empty sets for non zero ratios
        if len(imgs):
            if r_tr > 0 and n_tr == 0:
                n_tr = 1
            if r_va > 0 and n_va == 0:
                n_va = 1

        # Make sure we don't take too many volumes:
        overflow = max(n_tr + n_va - len(imgs), 0)
        if overflow:
            remove = min(overflow, n_va)
            n_va -= remove
            overflow -= remove
            if overflow > 0:  # still need to remove
                n_tr = max(n_tr - n_va, 0)
        n_te = max(len(imgs) - n_tr - n_va, 0)

        # selection indexes
        i_tr = idx[:n_tr]
        i_va = idx[n_tr : n_tr + n_va]
        i_te = idx[n_tr + n_va : n_tr + n_va + n_te]

        selection = {"train": i_tr, "val": i_va, "test": i_te}[split]

        # Define self.items to hold a list of (volume, label) tuples
        self.items = [(imgs[i], labels[i]) for i in selection]

        # Store all slices self.slice_index as (path_img, path_lab, z) list
        self.slice_index = []
        for item_volume, item_label in self.items:
            volume_z_dim = nib.load(item_volume).shape[-1]

            for z_idx in range(volume_z_dim):
                self.slice_index.append((item_volume, item_label, z_idx))

        if self.split == 'train': # sample the training slice to not overwhelm training
            stride = 2  # todo try 2 or 3
            self.slice_index = self.slice_index[::stride]

        print(f"{split}: {len(self.slice_index)} slices")

    def resize(self, arr, is_label=False):
        """
        Resize 2D slices (intensity or label data) using pytorch interpolate
        for images: use bilinear mode
        for labels: use nearest neighbor (keep it binary)
        :param arr: 2D intensity or label data [Height, Width]
        """
        # Convert to PyTorch tensor and add new axes in the beginning since PyTorch expects 4D [1, 1, H, W]
        tensor = torch.from_numpy(arr)[None, None, ...]

        # Interpolate the resized tensor [1, 1, self.size, self.size] (align corners only for bilinear)
        if is_label:
            tensor = functional.interpolate(tensor, size=(self.size, self.size), mode="nearest")
        else:
            tensor = functional.interpolate(tensor, size=(self.size, self.size), mode="bilinear", align_corners=False)

        # unwrap tensor back to numpy [size, size]
        resized_arr = tensor[0, 0].numpy()
        return resized_arr

    def random_augmentation(self, volume, label):
        '''
        :param volume: [k, H, W]
        :param label: [1, H, W] for the center slice, values 0 or 1
        '''

        rng = np.random
        k, H, W = volume.shape

        # Flip width 50:50
        if rng.rand() < 0.5:
            volume = torch.flip(volume, dims=[2])
            label = torch.flip(label, dims=[2])

        # Intensity noise 20:80
        if rng.rand() < 0.2:
            bias_int = rng.uniform(-0.05, 0.05)
            gain_int = rng.uniform(0.9, 1.1)
            volume = torch.clamp(volume * gain_int + bias_int, min=0.0, max=1.0)

        # todo add rotation/blur?

        return volume, label

    def __len__(self):
        """
        Return number of slices
        """
        return len(self.slice_index)

    def __getitem__(self, i):
        """
        Override the getitem method in the dataset
        Get stack of length k from slice with index i
        :param i:
        :return: volume_stack [k, H, W], slice_label [1, H, W] (only for chosen slice)
        """
        img_p, lab_p, z = self.slice_index[i]

        # Use nib proxies (no full volume in RAM)
        img_proxy = nib.load(img_p).dataobj
        lab_proxy = nib.load(lab_p).dataobj

        # gather z indices for the stack (clamped at edges)
        if self.k == 1:
            idxs = [z]
        else:
            half = self.k // 2
            Z = img_proxy.shape[-1]
            idxs = [int(np.clip(z + d, 0, Z - 1)) for d in range(-half, half + 1)]

        # build image stack slice-by-slice, normalize each slice
        stack = []
        for zi in idxs:
            slc = np.array(img_proxy[..., zi], dtype=np.float32)  # read one 2D slice
            slc = normalize_img(slc)  # per-slice percentile norm
            slc = self.resize(slc, is_label=False)
            stack.append(slc)
        stack = np.stack(stack, axis=0)  # [k,H,W]

        # label for the central slice only
        y2d = np.array(lab_proxy[..., z], dtype=np.uint8)
        y2d = (y2d > 0).astype(np.float32)
        y2d = self.resize(y2d, is_label=True)
        y2d = (y2d > 0.5).astype(np.float32)  # ensure {0,1}

        X = torch.from_numpy(stack).float()  # [k,H,W]
        y = torch.from_numpy(y2d)[None].float()  # [1,H,W]

        if self.augment:
            X, y = self.random_augmentation(X, y)

        return X, y
        # path_img, path_label, z = self.slice_index[i]
        #
        # # Load intensity and clip and normalize intensity
        # if path_img not in self._cache:
        #     self._cache[path_img] = normalize_img(nib.load(path_img).get_fdata().astype(np.float32))
        #
        # # Load labels and merge MSD maps to binary
        # if path_label not in self._cache:
        #     self._cache[path_label] = binarize_labels(nib.load(path_label).get_fdata().astype(np.uint8))
        #
        # intensity = self._cache[path_img]
        # label = self._cache[path_label]
        #
        # if self.k == 1:
        #     # only resize is needed, by make sure it's 3D in the end
        #     volume_stack = self.resize(intensity[..., z], is_label=False)[None, ...]         # [1,H,W]
        #
        # else:  # Generate stack of length k
        #     # slices list: if k=5 we need [z-2, z-1, z, z+1, z+2], clip to range [0, image_z_length]
        #     half_stack = self.k // 2
        #     min_z, max_z = 0, intensity.shape[2] - 1
        #     slices_list = [np.clip(z + d, min_z, max_z) for d in range(-half_stack, half_stack + 1)]
        #
        #     # Resize and stack the slices
        #     volume_stack = np.stack([self.resize(intensity[..., zi], False) for zi in slices_list], 0)
        #
        # # Resize label map for the chosen z slice
        # slice_label = self.resize(label[..., z], is_label=True).astype(np.float32)  # [H, W], center slice labels
        # slice_label = (slice_label > 0.5).astype(np.float32)
        #
        # # Convert to equal dim tensors
        # volume_stack = torch.from_numpy(volume_stack).float()  # [k, H, W]
        # slice_label = torch.from_numpy(slice_label)[None, ...].float()  # [1, H, W]
        #
        # # Augment
        # if self.augment:
        #     volume_stack, slice_label = self.random_augmentation(volume_stack, slice_label)
        #     volume_stack = volume_stack.float().contiguous()  # [k,H,W]
        #     slice_label = (slice_label > 0.5).float().contiguous()  # [1,H,W] with {0,1}
        #
        # return volume_stack, slice_label
