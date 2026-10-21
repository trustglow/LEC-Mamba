import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class ComplexImageCubeDataset(Dataset):
    def __init__(self, data, data2, gt, windowSize=5, removeZeroLabels=True):
        self.windowSize = windowSize
        self.margin = (windowSize - 1) // 2
        self.removeZeroLabels = removeZeroLabels

        # Zero-pad the first input cube.
        self.padded_data = np.pad(
            data,
            pad_width=((self.margin, self.margin), (self.margin, self.margin), (0, 0)),
            mode='constant',
            constant_values=0
        )
        
        # Zero-pad the second input cube.
        self.padded_data2 = np.pad(
            data2,
            pad_width=((self.margin, self.margin), (self.margin, self.margin), (0, 0)),
            mode='constant',
            constant_values=0
        )

        # Build the list of labeled pixel coordinates.
        self.label_coords = []
        for r in range(gt.shape[0]):
            for c in range(gt.shape[1]):
                if not removeZeroLabels or gt[r, c] > 0:
                    self.label_coords.append((r, c, gt[r, c]))

        # Store labels for operations such as stratified sampling.
        self.labels = np.array([label for _, _, label in self.label_coords])
        if removeZeroLabels:
            self.labels -= 1

    def __len__(self):
        return len(self.label_coords)

    def __getitem__(self, idx):
        r, c, orig_label = self.label_coords[idx]

        pr, pc = r + self.margin, c + self.margin

        # Extract patches from both input cubes.
        patch = self.padded_data[
            pr - self.margin: pr + self.margin + 1,
            pc - self.margin: pc + self.margin + 1
        ]
        patch2 = self.padded_data2[
            pr - self.margin: pr + self.margin + 1,
            pc - self.margin: pc + self.margin + 1
        ]

        label = orig_label
        if self.removeZeroLabels:
            label -= 1

        # Convert to tensors and change the layout from [H, W, C] to [C, H, W].
        patch = torch.FloatTensor(patch).permute(2, 0, 1)
        patch2 = torch.FloatTensor(patch2).permute(2, 0, 1)

        return (patch, patch2), torch.tensor(label, dtype=torch.long)

def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True  # Speed up host-to-device data transfer.
    )   
    
class ComplexImageCubeDatasetIDX(Dataset):
    def __init__(self, data, data2, gt, windowSize=13, removeZeroLabels=True):
        """
        Generate paired image-cube patches and labels on the fly.
        
        Args:
            data: First input cube with shape [H, W, C1].
            data2: Second input cube with shape [H, W, C2].
            gt: Ground-truth label map with shape [H, W].
            windowSize: Odd spatial window size used to extract patches.
            removeZeroLabels: Whether to omit samples with label 0.
        """
        self.windowSize = windowSize
        self.margin = (windowSize - 1) // 2
        self.removeZeroLabels = removeZeroLabels
        self.gt=gt
        # Zero-pad the first input cube.
        self.padded_data = np.pad(
            data,
            pad_width=((self.margin, self.margin), (self.margin, self.margin), (0, 0)),
            mode='constant',
            constant_values=0
        )

        # Zero-pad the second input cube.
        self.padded_data2 = np.pad(
            data2,
            pad_width=((self.margin, self.margin), (self.margin, self.margin), (0, 0)),
            mode='constant',
            constant_values=0
        )

        # Precompute the coordinates and labels of valid samples.
        self.indices = []
        self.labels = []
        for r in range(gt.shape[0]):
            for c in range(gt.shape[1]):
                label = gt[r, c]
                if removeZeroLabels and label == 0:
                    continue
                self.indices.append((r, c))
                self.labels.append(label)

        # Convert labels to a NumPy array for stratified sampling and related operations.
        self.labels_arr = np.array(self.labels)
        if removeZeroLabels:
            self.labels_arr -= 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        r, c = self.indices[idx]
        label = self.gt[r, c]

        pr, pc = r + self.margin, c + self.margin

        # Extract a patch from the first input cube.
        patch1 = self.padded_data[
            pr - self.margin : pr + self.margin + 1,
            pc - self.margin : pc + self.margin + 1
        ]

        # Extract a patch from the second input cube.
        patch2 = self.padded_data2[
            pr - self.margin : pr + self.margin + 1,
            pc - self.margin : pc + self.margin + 1
        ]

        # Convert to tensors and change the layout from [H, W, C] to [C, H, W].
        patch1_tensor = torch.FloatTensor(patch1).permute(2, 0, 1)
        patch2_tensor = torch.FloatTensor(patch2).permute(2, 0, 1)

        # Convert labels to zero-based indices when background labels are removed.
        if self.removeZeroLabels:
            label -= 1

        label_tensor = torch.tensor(label, dtype=torch.long)

        # Return the paired patches and label.
        return (patch1_tensor, patch2_tensor), label_tensor

def Standardize_data(X, eps=1e-8):
    new_X = np.zeros(X.shape, dtype=X.dtype)
    _, _, c = X.shape
    for i in range(c):
        channel = X[:, :, i]
        mean = np.mean(channel)
        std = np.std(channel)
        new_X[:, :, i] = (channel - mean) / (std + eps)  # Add epsilon to avoid division by zero.
        
    return new_X

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]),dtype=('float16'))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createComplexImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=('float16'))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def createComplexImageCubesMINI(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=('float32'))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin,2):
        for c in range(margin, zeroPaddedX.shape[1] - margin,2):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def createComplexImageCubesIDX(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    
    gtIndices = np.zeros((X.shape[0] * X.shape[1], 2), dtype=int)
    
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype='float16')
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            
            gtIndices[patchIndex] = [r - margin, c - margin]
            
            patchIndex += 1
    
    if removeZeroLabels:
        valid_indices = patchesLabels > 0
        patchesData = patchesData[valid_indices, :, :, :]
        patchesLabels = patchesLabels[valid_indices]
        patchesLabels -= 1
        gtIndices = gtIndices[valid_indices]
    
    return patchesData, patchesLabels, gtIndices
