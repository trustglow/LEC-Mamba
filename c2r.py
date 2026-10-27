"""
Center-Contrast Refinement (C2R), derived from the adaptive high-pass filtering
strategy used in Frequency-aware Feature Fusion for Dense Image Prediction.

Core idea:
    The original AHPF targets dense prediction and enhances boundaries across
    the entire feature map. For patch-based classification, however, the
    center pixel is the target and the surrounding pixels provide context.
    Unmodified global enhancement can amplify irrelevant boundary responses
    and distract the Transformer from the center target.

    C2R uses two complementary mechanisms:
    1. Center-guided dynamic filter generation: filter weights are predicted
       from the difference between each location and the center feature.
    2. Spatial Gaussian decay: a 2D Gaussian mask weights the high-frequency
       residual according to its distance from the patch center.

Fusion:
    X_tilde = X + G(i,j) * AHPF_center(X, |X - X_center|)
    where G(i,j) = exp(-((i-x_c)^2 + (j-y_c)^2) / (2*sigma^2)).

Reference:
    Frequency-aware Feature Fusion for Dense Image Prediction (TPAMI, 2024),
    which introduced the original AHPF formulation. C2R adapts the filtering
    process with center-aware guidance for patch classification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hamming2D(M: int, N: int) -> np.ndarray:
    """
    Generate a 2D Hamming window for filter regularization.

    Args:
        M: Number of rows in the window.
        N: Number of columns in the window.

    Returns:
        A 2D Hamming window with shape [M, N].
    """
    # Generate 1D Hamming windows for both axes.
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    # Form the 2D window as the outer product of the 1D windows.
    hamming_2d = np.outer(hamming_x, hamming_y)
    return hamming_2d


def normal_init(module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0) -> None:
    """
    Initialize module parameters with a normal distribution.

    Args:
        module: Neural network module to initialize.
        mean: Mean of the normal distribution (default: 0).
        std: Standard deviation of the normal distribution (default: 1).
        bias: Constant used to initialize the bias (default: 0).

    Returns:
        None.
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module: nn.Module, gain: float = 1, bias: float = 0,
                distribution: str = 'uniform') -> None:
    """
    Initialize module parameters with Xavier initialization.

    Args:
        module: Neural network module to initialize.
        gain: Scaling factor (default: 1).
        bias: Constant used to initialize the bias (default: 0).
        distribution: Initialization distribution, either 'uniform' or
            'normal' (default: 'uniform').

    Returns:
        None.

    Raises:
        AssertionError: If ``distribution`` is neither 'uniform' nor 'normal'.
    """
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class C2R(nn.Module):
    """
    Center-Contrast Refinement (C2R) module.

    C2R adapts the filter-inversion process of the original AHPF by adding
    center-guided filter prediction and spatial Gaussian weighting. Filter
    weights are predicted from center-to-location feature differences, and
    the high-frequency residual is weighted by a Gaussian mask centered on
    the target pixel.

    Args:
        in_channels: Number of input feature channels.
        kernel_size: High-pass filter size (default: 3, i.e., 3 x 3).
        encoder_kernel: Kernel size of the filter-weight predictor (default: 3).
        hamming_window: Whether to apply Hamming-window regularization
            (default: True).
        center: Center-pixel coordinates as (row, column) (default: (6, 6)).
        sigma_init: Initial standard deviation of the Gaussian mask
            (default: 2.0).
        learnable_sigma: Whether the Gaussian scale is learnable (default: True).
    """

    def __init__(self, in_channels: int, kernel_size: int = 3,
                 encoder_kernel: int = 3, hamming_window: bool = True,
                 center: tuple = (6, 6), sigma_init: float = 2.0,
                 learnable_sigma: bool = True):
        super(C2R, self).__init__()
        self.kernel_size = kernel_size  # High-pass filter kernel size.
        self.encoder_kernel = encoder_kernel  # Filter-weight predictor kernel size.
        self.hamming_window = hamming_window  # Whether Hamming regularization is enabled.
        self.center = center  # Center-pixel coordinates (row, column).

        # Center-guided dynamic filter generation.
        # Project the center-to-location feature differences into a weight space.
        self.diff_proj = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Predict a local filter at each spatial position from the projected differences.
        # The output has kernel_size squared channels, one for each filter coefficient.
        self.weight_encoder = nn.Conv2d(
            in_channels=in_channels,
            out_channels=kernel_size ** 2,
            kernel_size=encoder_kernel,
            padding=int((encoder_kernel - 1) // 2)
        )

        # Register the Hamming window as a non-trainable buffer.
        if self.hamming_window:
            self.register_buffer(
                'hamming',
                torch.FloatTensor(hamming2D(kernel_size, kernel_size))[None, None]
            )
        else:
            self.register_buffer('hamming', torch.FloatTensor([1.0]))

        # Learn the spatial decay range through the Gaussian scale parameter.
        if learnable_sigma:
            # Log-parameterization keeps sigma positive.
            self.log_sigma = nn.Parameter(torch.tensor(np.log(sigma_init), dtype=torch.float32))
        else:
            self.register_buffer('log_sigma', torch.tensor(np.log(sigma_init), dtype=torch.float32))

        # Initialize the filter-weight predictor.
        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize the module weights.

        The filter-weight predictor is initialized with a small standard
        deviation so the initial filter is close to uniform and high-frequency
        enhancement remains mild. ``diff_proj`` uses the default initialization.

        Returns:
            None.
        """
        normal_init(self.weight_encoder, std=0.001)

    def compute_gaussian_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        Compute a 2D Gaussian decay mask.

        The mask peaks at the center pixel and decays toward the patch edges.
        It weights the high-frequency residual according to the distance from
        the center.

        Args:
            height: Feature-map height.
            width: Feature-map width.
            device: Device on which to construct the mask.

        Returns:
            Gaussian decay mask with shape [1, 1, height, width].
        """
        # Recover the positive Gaussian scale from its log-parameterization.
        sigma = torch.exp(self.log_sigma)

        # Generate the spatial coordinate grid.
        y_coords = torch.arange(height, device=device, dtype=torch.float32)
        x_coords = torch.arange(width, device=device, dtype=torch.float32)
        # The coordinate grids have shape [height, width].
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Compute the squared distance from each location to the center.
        center_y, center_x = self.center
        dist_sq = (y_grid - center_y) ** 2 + (x_grid - center_x) ** 2

        # Apply Gaussian decay around the center pixel.
        gaussian_mask = torch.exp(-dist_sq / (2 * sigma ** 2))

        # Reshape to [1, 1, H, W] for broadcasting.
        gaussian_mask = gaussian_mask.unsqueeze(0).unsqueeze(0)

        return gaussian_mask

    def kernel_normalizer(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Normalize predicted filter weights to obtain a low-pass filter.

        Softmax is applied across the local-kernel coefficients so the weights
        are positive and sum to one. Hamming-window weighting is then applied
        and the result is renormalized.

        Args:
            mask: Predicted filter weights with shape [B, kernel_size^2, H, W].

        Returns:
            Normalized low-pass filters with shape [B, kernel_size^2, H, W].
        """
        n, mask_c, h, w = mask.size()
        # Reshape channels to [B, 1, kernel_size^2, H, W].
        mask = mask.view(n, 1, -1, h, w)
        # Normalize the local filter coefficients with Softmax.
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # Reshape to [B, 1, H, W, kernel_size, kernel_size].
        mask = mask.view(n, 1, self.kernel_size, self.kernel_size, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, self.kernel_size, self.kernel_size)
        # Apply Hamming-window regularization.
        mask = mask * self.hamming
        # Renormalize so each local filter sums to one.
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        # Restore the shape [B, kernel_size^2, H, W].
        mask = mask.view(n, 1, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def dynamic_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        Apply the predicted local filter at every spatial position.

        ``F.unfold`` extracts local neighborhoods, which are multiplied by the
        predicted filter weights and summed to perform adaptive filtering.

        Args:
            x: Input feature map with shape [B, C, H, W].
            kernel: Predicted filter weights with shape [B, kernel_size^2, H, W].

        Returns:
            Filtered feature map with shape [B, C, H, W].
        """
        b, c, h, w = x.shape
        pad = self.kernel_size // 2
        # Use reflection padding to handle feature-map boundaries.
        pad_x = F.pad(x, pad=[pad, pad, pad, pad], mode='reflect')
        # Extract local neighborhoods: [B, C * kernel_size^2, H * W].
        unfold_x = F.unfold(pad_x, kernel_size=(self.kernel_size, self.kernel_size),
                            stride=1, padding=0)
        # Reshape neighborhoods to [B, C, kernel_size^2, H, W].
        unfold_x = unfold_x.reshape(b, c, self.kernel_size * self.kernel_size, h, w)
        # Reshape filters to [B, 1, kernel_size^2, H, W].
        kernel = kernel.reshape(b, 1, self.kernel_size * self.kernel_size, h, w)
        # Apply the filters and sum over the local-kernel dimension.
        res = (unfold_x * kernel).sum(dim=2)
        return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run center-guided detail refinement.

        The sequence of operations is: extract the center feature, compute
        center-to-location differences, predict and normalize local filters,
        apply dynamic low-pass filtering, obtain the high-pass residual, and
        weight that residual with a center-focused Gaussian mask.

        Args:
            x: Input feature map with shape [B, C, H, W].

        Returns:
            Center-refined feature map with the same shape [B, C, H, W].
        """
        b, c, h, w = x.shape

        # Extract the center-pixel feature [B, C, 1, 1].
        center_y, center_x = self.center
        x_center = x[:, :, center_y:center_y + 1, center_x:center_x + 1]

        # Compute center-to-location differences [B, C, H, W].
        # The center has zero difference from itself; other values reflect feature dissimilarity.
        diff = torch.abs(x - x_center)

        # Project the differences to improve their representation.
        diff_feat = self.diff_proj(diff)

        # Predict local filter weights from the projected differences.
        weights = self.weight_encoder(diff_feat)

        # Normalize the weights to form low-pass filters.
        low_pass_kernel = self.kernel_normalizer(weights)

        # Apply dynamic low-pass filtering to the input features.
        low_pass_result = self.dynamic_filter(x, low_pass_kernel)

        # Obtain the high-pass residual as the input minus its low-pass response.
        high_pass_result = x - low_pass_result

        # Generate the center-focused Gaussian decay mask [1, 1, H, W].
        gaussian_mask = self.compute_gaussian_mask(h, w, x.device)

        # Inject the weighted detail residual; locations nearer the center receive stronger weighting.
        output = x + gaussian_mask * high_pass_result

        return output


# Backward-compatible aliases for the original class names.
C_AHPF = C2R
AHPF = C2R


if __name__ == '__main__':
    """Run a shape and Gaussian-mask smoke test for C2R."""
    print('C2R module smoke test started')
    batch_size = 32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create a C2R instance with 128 input channels and center coordinate (6, 6).
    c2r = C2R(in_channels=128, center=(6, 6)).to(device)

    # Create a dummy 13 x 13 patch tensor.
    dummy_input = torch.randn(batch_size, 128, 13, 13).to(device)

    # Run the forward pass.
    output = c2r(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Check the Gaussian-mask shape and values.
    gaussian_mask = c2r.compute_gaussian_mask(13, 13, device)
    print(f"Gaussian mask shape: {gaussian_mask.shape}")
    print(f"Gaussian mask center value: {gaussian_mask[0, 0, 6, 6].item():.4f}")
    print(f"Gaussian mask corner value: {gaussian_mask[0, 0, 0, 0].item():.4f}")

    # Check the current Gaussian scale.
    print(f"Initial sigma: {torch.exp(c2r.log_sigma).item():.4f}")

    # Verify the backward-compatible alias.
    ahpf_compat = AHPF(in_channels=128).to(device)
    output_compat = ahpf_compat(dummy_input)
    print(f"Compatibility-alias output shape: {output_compat.shape}")

    print('C2R module smoke test completed')
