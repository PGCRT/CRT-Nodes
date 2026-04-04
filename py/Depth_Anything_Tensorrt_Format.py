import torch
import numpy as np


class DepthAnythingTensorrtFormat:
    """
    Converts Depth Anything TensorRT output to proper ComfyUI image format.
    Handles various tensor shapes and converts single-channel depth to RGB.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": ("IMAGE",),
            },
            "optional": {
                "normalize": ("BOOLEAN", {"default": True}),
                "colorize": (["grayscale", "viridis", "plasma", "inferno", "magma", "turbo"], {"default": "grayscale"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "format_depth"
    CATEGORY = "CRT/Image"

    def format_depth(self, depth_map, normalize=True, colorize="grayscale"):
        """
        Convert depth map tensor to proper ComfyUI image format

        Args:
            depth_map: Input tensor (various shapes possible)
            normalize: Whether to normalize depth values to 0-1 range
            colorize: Color map to apply to depth

        Returns:
            Properly formatted image tensor (batch, height, width, 3)
        """

        # Convert to torch tensor if needed
        if not isinstance(depth_map, torch.Tensor):
            depth_map = torch.tensor(depth_map)

        # Handle different input shapes
        original_shape = depth_map.shape

        # Case 1: (height, width) - single depth map, no batch, no channel
        if len(original_shape) == 2:
            depth_map = depth_map.unsqueeze(0).unsqueeze(-1)  # -> (1, H, W, 1)

        # Case 2: (frames, height, width) - multiple depth maps or batch
        elif len(original_shape) == 3:
            # Check if last dimension is small (likely channels)
            if original_shape[-1] <= 4:
                # Shape is (batch, H, W, C)
                pass
            else:
                # Shape is (batch/frames, H, W) - add channel dimension
                depth_map = depth_map.unsqueeze(-1)  # -> (B, H, W, 1)

        # Case 3: (batch, height, width, channels) - already correct format
        elif len(original_shape) == 4:
            pass

        else:
            raise ValueError(f"Unexpected tensor shape: {original_shape}")

        # Ensure we have the right shape now: (B, H, W, C)
        if len(depth_map.shape) != 4:
            raise ValueError(f"Failed to reshape tensor. Current shape: {depth_map.shape}")

        # If single channel, we need to convert to 3-channel
        if depth_map.shape[-1] == 1:
            # Normalize if requested
            if normalize:
                depth_min = depth_map.min()
                depth_max = depth_map.max()
                if depth_max > depth_min:
                    depth_map = (depth_map - depth_min) / (depth_max - depth_min)

            # Apply colorization
            if colorize == "grayscale":
                # Simple grayscale - repeat the channel 3 times
                depth_map = depth_map.repeat(1, 1, 1, 3)
            else:
                # Apply matplotlib-style colormaps
                depth_map = self.apply_colormap(depth_map, colorize)

        # Ensure values are in 0-1 range
        depth_map = torch.clamp(depth_map, 0.0, 1.0)

        # Ensure float32
        depth_map = depth_map.float()

        return (depth_map,)

    def apply_colormap(self, depth_map, colormap_name):
        """
        Apply a colormap to single-channel depth map

        Args:
            depth_map: Single channel depth map (B, H, W, 1)
            colormap_name: Name of colormap to apply

        Returns:
            RGB depth map (B, H, W, 3)
        """
        batch, height, width, _ = depth_map.shape

        # Squeeze channel dimension for processing
        depth_values = depth_map.squeeze(-1)  # (B, H, W)

        # Define simple colormaps (approximations of matplotlib colormaps)
        colormaps = {
            "viridis": [
                (0.267004, 0.004874, 0.329415),
                (0.282623, 0.140926, 0.457517),
                (0.253935, 0.265254, 0.529983),
                (0.206756, 0.371758, 0.553117),
                (0.163625, 0.471133, 0.558148),
                (0.127568, 0.566949, 0.550556),
                (0.134692, 0.658636, 0.517649),
                (0.266941, 0.748751, 0.440573),
                (0.477504, 0.821444, 0.318195),
                (0.741388, 0.873449, 0.149561),
                (0.993248, 0.906157, 0.143936),
            ],
            "plasma": [
                (0.050383, 0.029803, 0.527975),
                (0.287675, 0.010384, 0.627419),
                (0.475369, 0.000000, 0.657247),
                (0.627193, 0.004556, 0.634007),
                (0.748703, 0.066235, 0.566995),
                (0.838124, 0.147607, 0.476841),
                (0.901871, 0.240841, 0.373293),
                (0.947055, 0.346559, 0.264610),
                (0.975418, 0.464885, 0.157663),
                (0.987622, 0.593205, 0.063536),
                (0.940015, 0.975158, 0.131326),
            ],
            "inferno": [
                (0.001462, 0.000466, 0.013866),
                (0.087411, 0.044556, 0.224813),
                (0.258234, 0.038571, 0.406485),
                (0.416331, 0.090203, 0.432943),
                (0.578304, 0.148039, 0.404411),
                (0.735683, 0.215906, 0.330245),
                (0.865006, 0.316822, 0.226055),
                (0.955552, 0.451462, 0.119041),
                (0.987622, 0.621069, 0.062409),
                (0.976819, 0.805244, 0.182822),
                (0.988362, 0.998364, 0.644924),
            ],
            "magma": [
                (0.001462, 0.000466, 0.013866),
                (0.087411, 0.044556, 0.224813),
                (0.258234, 0.038571, 0.406485),
                (0.420252, 0.090646, 0.472529),
                (0.585367, 0.180653, 0.498536),
                (0.746971, 0.294279, 0.486621),
                (0.881443, 0.446213, 0.442340),
                (0.967671, 0.630633, 0.415686),
                (0.993248, 0.832305, 0.486307),
                (0.997500, 0.993248, 0.678032),
                (0.987053, 0.991438, 0.749504),
            ],
            "turbo": [
                (0.18995, 0.07176, 0.23217),
                (0.23955, 0.26529, 0.63125),
                (0.12873, 0.56202, 0.76122),
                (0.15104, 0.75898, 0.66022),
                (0.41440, 0.89832, 0.44269),
                (0.69841, 0.93775, 0.22696),
                (0.94162, 0.88648, 0.13188),
                (0.99655, 0.72247, 0.13750),
                (0.95004, 0.51998, 0.07355),
                (0.82952, 0.28581, 0.01273),
                (0.47960, 0.01583, 0.01055),
            ],
        }

        # Get colormap
        if colormap_name not in colormaps:
            colormap_name = "viridis"

        colormap = torch.tensor(colormaps[colormap_name], device=depth_map.device, dtype=depth_map.dtype)

        # Map depth values to colormap indices
        num_colors = len(colormap)
        indices = (depth_values * (num_colors - 1)).long()
        indices = torch.clamp(indices, 0, num_colors - 1)

        # Create RGB output
        rgb_output = torch.zeros(batch, height, width, 3, device=depth_map.device, dtype=depth_map.dtype)

        # Apply colormap
        for b in range(batch):
            for i in range(num_colors):
                mask = indices[b] == i
                rgb_output[b, mask] = colormap[i]

        # Interpolate for smoother gradients
        depth_float = depth_values.unsqueeze(-1)  # (B, H, W, 1)
        idx_float = depth_float * (num_colors - 1)
        idx_low = torch.floor(idx_float).long().clamp(0, num_colors - 2)
        idx_high = (idx_low + 1).clamp(0, num_colors - 1)
        weight = idx_float - idx_low.float()

        # Interpolate colors
        for b in range(batch):
            colors_low = colormap[idx_low[b].squeeze(-1)]
            colors_high = colormap[idx_high[b].squeeze(-1)]
            rgb_output[b] = colors_low * (1 - weight[b]) + colors_high * weight[b]

        return rgb_output


NODE_CLASS_MAPPINGS = {"DepthAnythingTensorrtFormat": DepthAnythingTensorrtFormat}

NODE_DISPLAY_NAME_MAPPINGS = {"DepthAnythingTensorrtFormat": "Depth Anything Tensorrt Format (CRT)"}
