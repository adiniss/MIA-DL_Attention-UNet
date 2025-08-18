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


class Stacks2D(Dataset):
    """
    Take PyTorch dataset and create a stack of 2D slices
    """
    def __init__(self,
                 img_dir, label_dir,
                 img_size=256, k_slices=1,  # k=1 -> 2D slice, k>1 -> stacks of 2D slices
                 split='train', split_ratio=(0.8, 0.1, 0.1), seed=3360033):

        self.k = k_slices
        self.size = img_size

        # Get images (NifTI format) and labels
        imgs = sorted(glob.glob(os.path.join(img_dir, '*.nii*')))
        labels = [os.path.join(label_dir, os.path.basename(image)) for image in imgs]
        # Check for missing labels
        assert all(os.path.exists(lab) for lab in labels), "Missing labels"

        # Shuffle at volume level
        rng = (np.random.RandomState(seed))
        idx = rng.permutation(len(imgs))
        n_tr, n_va = int(len(idx) * split_ratio[0]), int(len(idx) * split_ratio[1])

        if split == 'train':
            selection = idx[:n_tr]
        elif split == 'val':
            selection = idx[n_tr:n_tr + n_va]
        else:
            selection = idx[n_tr + n_va:]

        # Define self.items to hold a list of (volume, label) tuples
        self.items = [(imgs[i], labels[i]) for i in selection]

        # Find what z values are relevant (not blank background) - what slices should we use
        # Store in self.slice_index as (path_img, path_lab, z) list
        self.slice_index = []
        for item_volume, item_label in self.items:
            voxel_volume = nib.load(item_volume).get_fdata()

            for z_idx in range(voxel_volume.shape[2]):
                self.slice_index.append((item_volume, item_label, z_idx))

            # previous version to include only non empty slices
            # # Get data using nibabel
            # intensity_data = nib.load(item_volume).get_fdata().astype(np.float32)
            # label_data = nib.load(item_label).get_fdata().astype(np.uint8)
            #
            # # Go over the z axis and check for informational labels
            # for z in range(intensity_data.shape[2]):
            #     if label_data[..., z].sum() > 0:
            #         # If non-zero, append the specific
            #         self.slice_index.append((item_volume, item_label, z))
        # print(f"{split}: {len(self.slice_index)} slices")
        print(f"{split}: {len(self.slice_index)} slices (all possible slices)")

    def resize(self, arr, is_label=False):
        """
        Resize 2D slices (intensity or label data) using pytorch interpolate
        for images: use bilinear mode
        for labels: use nearest neighbor (keep it binary)
        :param arr: 2D intensity or label data [Height, Width]
        """
        # Convert to PyTorch tensor and add new axes in the beginning since PyTorch expects 4D [1, 1, H, W]
        tensor = torch.from_numpy(arr)[None, None, ...]

        if is_label:
            intr_mode = "nearest"
        else:
            intr_mode = "bilinear"

        # Interpolate the resized tensor [1, 1, self.size, self.size] (align corners only for bilinear)
        if is_label:
            tensor = functional.interpolate(tensor, size=(self.size, self.size), mode="nearest")
        else:
            tensor = functional.interpolate(tensor, size=(self.size, self.size), mode="bilinear", align_corners=False)

        # unwrap tensor back to numpy [size, size]
        resized_arr = tensor[0, 0].numpy()
        return resized_arr

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
        path_img, path_label, z = self.slice_index[i]

        # Load labels and merge MSD maps to binary
        label = nib.load(path_label).get_fdata().astype(np.uint8)
        label = binarize_labels(label)

        # Load intensity and clip and normalize intensity
        intensity = nib.load(path_img).get_fdata().astype(np.float32)
        intensity = clip_norm_HU(intensity)

        if self.k == 1:
            # only resize is needed, by make sure it's 3D in the end
            volume_stack = self.resize(intensity[..., z], is_label=False)[None, ...]         # [1,H,W]

        else:  # Generate stack of length k
            # slices list: if k=5 we need [z-2, z-1, z, z+1, z+2], clip to range [0, image_z_length]
            half_stack = self.k // 2
            min_z, max_z = 0, intensity.shape[2] - 1
            slices_list = [np.clip(z + d, min_z, max_z) for d in range(-half_stack, half_stack + 1)]

            # Resize and stack the slices
            volume_stack = np.stack([self.resize(intensity[..., zi], False) for zi in slices_list], 0)

        # Resize label map for the chosen z slice
        slice_label = self.resize(label[..., z], is_label=True).astype(np.float32)  # [H, W], center slice labels
        slice_label = (slice_label > 0.5).astype(np.float32)

        # Convert to equal dim tensors
        volume_stack = torch.from_numpy(volume_stack).float()  # [k, H, W]
        slice_label = torch.from_numpy(slice_label)[None, ...].float()  # [1, H, W]

        #todo when adding data augmentation for train set
        #if getattr(self, "split", None) == 'train':
            # volume_stack, slice_label = self.augment(volume_stack, slice_label)

        return volume_stack, slice_label
