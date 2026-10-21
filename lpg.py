import torch
import torch.nn as nn


class PA(nn.Module):
    """
    Position-attention gate used by the LPG block.
    
    A depthwise 3 x 3 convolution predicts a spatial gate, which is passed
    through a sigmoid and multiplied element-wise with the input features.
    
    Args:
        dim: Number of input feature channels.
    """
    def __init__(self, dim):
        super().__init__()
        # Depthwise convolution for extracting spatial responses.
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        # Sigmoid activation for producing the spatial gate.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Apply the spatial gate.
        
        Args:
            x: Input feature tensor with shape [B, C, H, W].
                
        Returns:
            The gated feature tensor with the same shape as the input.
        """
        # Predict a spatial gate and apply it element-wise to the input.
        return x * self.sigmoid(self.pa_conv(x))


class pre_conv(nn.Module):
    """
    Convolutional feature-extraction path used by the LPG block.
    
    The path applies 2D convolutions and adds a projected 3D-convolution
    residual to the extracted features.
    
    Args:
        in_chans: Number of input feature channels.
    """
    def __init__(self, in_chans):
        super().__init__()

        # GELU activation.
        self.act = nn.GELU()
        # Pointwise convolution for feature transformation.
        self.conv1 = nn.Conv2d(in_chans, in_chans, kernel_size=1, bias=False)
        # Batch normalization after the pointwise convolution.
        self.bn1 = nn.BatchNorm2d(in_chans)
        # Spatial convolution for local feature extraction.
        self.conv2 = nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1, bias=False)
        # Batch normalization after the spatial convolution.
        self.bn2 = nn.BatchNorm2d(in_chans)

        # 3D convolution used to project the residual branch.
        self.proj = nn.Conv3d(1, 1, kernel_size=1, bias=False)
        # Retained as a registered legacy layer; it is not used in forward.
        self.bn4 = nn.BatchNorm3d(1)

    def forward(self, x):
        """
        Extract convolutional features and add the residual projection.
        
        Args:
            x: Input feature tensor with shape [B, C, H, W].
                
        Returns:
            The processed feature tensor with the same shape as the input.
        """
        # Preserve the input for the residual connection.
        x_1 = x

        # Apply the pointwise convolution.
        x0 = self.conv1(x)
        # Apply the spatial convolution after activation and normalization.
        x1 = self.conv2(self.act(self.bn1(x0)))
        # Normalize and activate the extracted features.
        x = self.act(self.bn2(x1))
        
        # Add a singleton dimension to convert [B, C, H, W] to [B, 1, C, H, W].
        x_1 = x_1.unsqueeze(dim=1)
        # Apply the 3D convolution and activation.
        x_1 = self.act(self.proj(x_1))
        # Remove the singleton dimension to restore the 4D layout.
        x_1 = x_1.squeeze(dim=1)

        # Add the projected residual to the extracted features.
        x = x + x_1
        return x


class LPG(nn.Module):
    """
    Local-Pointwise Gating (LPG) module.
    
    The module retains the original convolutional feature-extraction path,
    followed by a depthwise spatial gate.
    
    Args:
        in_chans: Number of input feature channels.
    """
    def __init__(self, in_chans):
        super().__init__()
        # Convolutional feature-extraction path.
        self.pre_conv = pre_conv(in_chans)
        # Depthwise spatial gate.
        self.postion = PA(in_chans)

    def forward(self, x):
        """
        Apply convolutional feature extraction followed by spatial gating.
        
        Args:
            x: Input feature tensor with shape [B, C, H, W].
                
        Returns:
            The gated feature tensor with the same shape as the input.
        """
        # Extract features using the convolutional path.
        x = self.pre_conv(x)
        # Refine the features with the spatial gate.
        x = self.postion(x)
        return x


# Backward-compatible class alias.
SSIB = LPG
