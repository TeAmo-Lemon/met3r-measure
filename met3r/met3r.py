

import sys
import os
import os.path as path

from typing import Literal, Optional, Union

import torch

from torch import Tensor
from torch.nn import Identity, functional as F
from pathlib import Path
from torch.nn import Module
from jaxtyping import Float, Bool
from typing import Union, Tuple
from einops import rearrange, repeat
from torchvision.models.optical_flow import raft_large
from torchmetrics.functional.image import structural_similarity_index_measure


# Load Pytorch3D
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)


from lpips import LPIPS

HERE_PATH = path.normpath(path.dirname(__file__))
MASt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, '../mast3r'))
DUSt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, '../mast3r/dust3r'))
MASt3R_LIB_PATH = path.join(MASt3R_REPO_PATH, 'mast3r')
DUSt3R_LIB_PATH = path.join(DUSt3R_REPO_PATH, 'dust3r')
# check the presence of models directory in repo to be sure its cloned
if path.isdir(MASt3R_LIB_PATH) and path.isdir(DUSt3R_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, MASt3R_REPO_PATH)
    sys.path.insert(0, DUSt3R_REPO_PATH)
else:
    raise ImportError(f"mast3r and dust3r is not initialized, could not find: {MASt3R_LIB_PATH}.\n "
                    "Did you forget to run 'git submodule update --init --recursive' ?")
from dust3r.utils.geometry import xy_grid

def freeze_model(m: Module) -> None:
    for param in m.parameters():
        param.requires_grad = False
    m.eval()

def convert_to_buffer(module: torch.nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


def load_hub_model(repo_or_dir: Union[str, Path], model_name: str, **kwargs):
    repo_or_dir = Path(repo_or_dir).expanduser()
    if repo_or_dir.exists() and repo_or_dir.is_dir():
        return torch.hub.load(str(repo_or_dir), model_name, source="local", **kwargs)
    return torch.hub.load(str(repo_or_dir), model_name, **kwargs)

backbone_to_weights = {
    "mast3r": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    "dust3r": "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
}

class MEt3R(Module):

    def __init__(
        self, 
        img_size: Optional[int] = 256, 
        use_norm: Optional[bool]=True,
        backbone: Literal["mast3r", "dust3r", "raft"] = "mast3r",
        feature_backbone: Optional[Literal["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]] = "dino16",
        feature_backbone_weights: Optional[Union[str, Path]] = "mhamilton723/FeatUp",
        upsampler: Optional[Literal["featup", "nearest", "bilinear", "bicubic"]] = "featup",
        distance: Literal["cosine", "lpips", "rmse", "psnr", "mse", "ssim"] = "cosine",
        freeze: bool=True,
        rasterizer_kwargs: dict = {}
    ) -> None:
        """Initialize MET3R

        Args:
            img_size (int, optional): Image size for rasterization. Set to None to allow for rasterization with the input resolution on the fly. Defaults to 224.
            use_norm (bool, optional): Whether to use norm layers in FeatUp. Refer to https://github.com/mhamilton723/FeatUp?tab=readme-ov-file#using-pretrained-upsamplers. Defaults to True.
            feature_backbone (str, optional): Feature backbone for FeatUp. Select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]. Defaults to "dino16".
            feature_backbone_weights (str | Path, optional): Weight path for FeatUp upsampler. Defaults to "mhamilton723/FeatUp".
            upsampler (str, optional): Set upsampling types. Defaults to "featup".
            distance (str): Select which distance to compute. Default to "cosine" for computing feature dissimilarity.
            freeze (bool, optional): Set whether to freeze the model. Defaults to True.
            rasterizer_kwargs (dict): Additional argument for point cloud render from PyTorch3D. Default to an empty dict. 
        """
        super().__init__()
        self.img_size = img_size
        self.upsampler = upsampler
        self.backbone = backbone
        self.distance = distance
        if upsampler == "featup" and "FeatUp" not in feature_backbone_weights:
            raise ValueError("Need to specify the correct weight path on huggingface for using `upsampler=\"featup\"`. Set `feature_backbone_weights=\"mhamilton723/FeatUp\"`")
            
        if distance == "cosine":
            if "FeatUp" in feature_backbone_weights:
                # Load featup
                from featup.util import norm, unnorm
                self.norm = norm
                if feature_backbone not in ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]:
                    raise ValueError("Provide `feature_backone` is not implemented for `FeatUp`. Please select from [\"dino16\", \"dinov2\", \"maskclip\", \"vit\", \"clip\", \"resnet50\"] in conjunction with `feature_backbone_weights=\"mhamilton723/FeatUp\"`")
                if use_norm is None:
                    raise ValueError("When using `FeatUp`, specify `use_norm` as either `True` or `False`. Currently it is set to `None`")
                
                featup = load_hub_model(feature_backbone_weights, feature_backbone, use_norm=use_norm)
                self.feature_model = featup.model
                if upsampler == "featup":
                    self.upsampler_model = featup.upsampler
                    if freeze:
                        freeze_model(self.upsampler_model)
                        convert_to_buffer(self.upsampler_model, persistent=False)
                
            else:
                self.norm = Identity()
                self.feature_model = load_hub_model(feature_backbone_weights, feature_backbone)
            
            if freeze:
                freeze_model(self.feature_model) 
                convert_to_buffer(self.feature_model, persistent=False)
            

        
        
        if backbone == "mast3r":
            from mast3r.model import AsymmetricMASt3R 
            self.backbone_model = AsymmetricMASt3R.from_pretrained(backbone_to_weights[backbone])
        elif backbone == "dust3r":
            from dust3r.model import AsymmetricCroCo3DStereo 
            self.backbone_model = AsymmetricCroCo3DStereo.from_pretrained(backbone_to_weights[backbone])
        elif backbone == "raft":
            self.backbone_model = raft_large(pretrained=True, progress=False)
        else:
            raise NotImplementedError("Specificed backbone for warping is not available. Please select from ['mast3r', 'dust3r', 'raft']")  

        if freeze:
            freeze_model(self.backbone_model) 
            convert_to_buffer(self.backbone_model, persistent=False)

        if backbone in ["mast3r", "dust3r"]:

            if self.img_size is not None:
                self.set_rasterizer(
                    image_size=img_size, 
                    points_per_pixel=10,
                    bin_size=0,
                    **rasterizer_kwargs
                )
            
            self.compositor = AlphaCompositor()
        
        if distance == "lpips":
            self.lpips = LPIPS(spatial=True)

    def _distance(self, inp1: Tensor, inp2: Tensor, mask: Optional[Tensor]=None, eps: float=1e-5):

        if self.distance == "cosine":
            # Get feature dissimilarity score map
            score_map = 1 - (inp1 * inp2).sum(1) / (torch.linalg.norm(inp1, dim=1) * torch.linalg.norm(inp2, dim=1) + eps) 
            score_map = score_map[:, None]
        elif self.distance == "mse":
            score_map = ((inp1 - inp2)**2).mean(1, keepdim=True)
        elif self.distance == "psnr":
            score_map = 20 * torch.log10(255.0 / (torch.sqrt(((inp1 - inp2)**2)).mean(1, keepdim=True) + eps))
        elif self.distance == "rmse":
            score_map = ((inp1 - inp2)**2).mean(1, keepdim=True)**0.5
        elif self.distance == "lpips":
            score_map = self.lpips(2 * inp1 - 1, 2 * inp2 - 1)
            score_map = score_map[:, None]
        elif self.distance == "ssim":
            _, score_map = structural_similarity_index_measure(inp1, inp2, return_full_image=True)
            print(score_map.shape)
            print(mask.shape)
        result = [score_map[:, 0]]
        if mask is not None: 
            # Weighted averate of score map with computed mask
            weighted = (score_map * mask[:, None]).sum(-1).sum(-1)  / (mask[:, None].sum(-1).sum(-1) + eps)
            result.append(weighted.mean(1))

        return tuple(result)
    
    def _interpolate(self, inp1: Tensor, inp2: Tensor):

        if self.upsampler == "featup":
            feat = self.upsampler_model(inp1, inp2)
            # Important for specific backbone which may not return with correct dimensions
            feat = F.interpolate(feat, (inp2.shape[-2:]), mode="bilinear")
        else:

            feat = F.interpolate(inp1, (inp2.shape[-2:]), mode=self.upsampler)

        return feat
    
    def _get_features(self, images):
        
        return self.feature_model(self.norm(images))

    def set_rasterizer(
        self,
        image_size, 
        points_per_pixel=10,
        bin_size=0,
        **kwargs
    ) -> None:
        raster_settings = PointsRasterizationSettings(
            image_size=image_size, 
            points_per_pixel=points_per_pixel,
            bin_size=bin_size,
            **kwargs
        )

        self.rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings)

    def render(
        self, 
        point_clouds: Pointclouds, 
        **kwargs
    ) -> Tuple[
            Float[Tensor, "b h w c"], 
            Float[Tensor, "b 2 h w n"]
        ]:
        """Adoped from Pytorch3D https://pytorch3d.readthedocs.io/en/latest/modules/renderer/points/renderer.html

        Args:
            point_clouds (pytorch3d.structures.PointCloud): Point cloud object to render 

        Returns:
            images (Float[Tensor, "b h w c"]): Rendered images
            zbuf (Float[Tensor, "b k h w n"]): Z-buffers for points per pixel
        """
        with torch.autocast("cuda", enabled=False):
            fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments.zbuf
    
    def warp_image(self, image: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Warp an input image using an optical flow field and compute a mask for gaps.

        Args:
            image (torch.Tensor): The input image of shape (B, C, H, W), where
                                B is the batch size,
                                C is the number of channels,
                                H is the height,
                                W is the width.
            flow (torch.Tensor): The optical flow of shape (B, 2, H, W), where the 2 channels
                                correspond to the horizontal and vertical flow components.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The warped image of shape (B, C, H, W).
                - A mask of shape (B, 1, H, W) indicating gaps due to warping (1 for valid pixels, 0 for gaps).
        """
        B, C, H, W = image.shape

        # Generate a grid of coordinates for the image
        y, x = torch.meshgrid(
            torch.arange(H, device=image.device, dtype=torch.float32),
            torch.arange(W, device=image.device, dtype=torch.float32),
            indexing="ij"
        )

        # Normalize the grid coordinates to the range [-1, 1]
        x = x / (W - 1) * 2 - 1
        y = y / (H - 1) * 2 - 1

        grid = torch.stack((x, y), dim=2).unsqueeze(0)  # Shape: (1, H, W, 2)
        grid = grid.repeat(B, 1, 1, 1)  # Repeat for batch size

        # Normalize flow from pixel space to normalized coordinates
        flow = flow.clone()
        flow[:, 0, :, :] = flow[:, 0, :, :] / (W - 1) * 2  # Normalize horizontal flow
        flow[:, 1, :, :] = flow[:, 1, :, :] / (H - 1) * 2  # Normalize vertical flow

        # Add the flow to the grid
        flow = flow.permute(0, 2, 3, 1)  # Shape: (B, H, W, 2)
        warped_grid = grid + flow

        # Clip grid values to ensure they are within bounds
        warped_grid[..., 0] = torch.clamp(warped_grid[..., 0], -1, 1)
        warped_grid[..., 1] = torch.clamp(warped_grid[..., 1], -1, 1)

        # Use grid_sample to warp the image
        warped_image = F.grid_sample(image, warped_grid, mode="bilinear", padding_mode="border", align_corners=True)

        # Compute a mask for valid pixels
        mask = F.grid_sample(
            torch.ones((B, 1, H, W), device=image.device, dtype=image.dtype),
            warped_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        mask = (mask > 0.999).float()  # Threshold to create a binary mask

        return warped_image, mask

    def forward(
        self, 
        images: Float[Tensor, "b 2 c h w"], 
        return_overlap_mask: bool=False, 
        return_score_map: bool=False, 
        return_projections: bool=False
    ) -> Tuple[
            float, 
            Bool[Tensor, "b h w"] | None, 
            Float[Tensor, "b h w"] | None, 
            Float[Tensor, "b 2 c h w"] | None
        ]:
        
        """Forward function to compute MET3R
        Args:
            images (Float[Tensor, "b 2 c h w"]): Normalized input image pairs with values ranging in [-1, 1],
            return_overlap_mask (bool, False): Return 2D map overlapping mask
            return_score_map (bool, False): Return 2D map of feature dissimlarity (Unweighted) 
            return_projections (bool, False): Return projected feature maps

        Return:
            score (Float[Tensor, "b"]): MET3R score which consists of weighted mean of feature dissimlarity
            mask (bool[Tensor, "b c h w"], optional): Overlapping mask
            feat_dissim_maps (bool[Tensor, "b h w"], optional): Feature dissimilarity score map
            proj_feats (bool[Tensor, "b h w c"], optional): Projected and rendered features
        """
        
        *_, h, w = images.shape
        
        # Set rasterization settings on the fly based on input resolution
        if self.img_size is None:
            raster_settings = PointsRasterizationSettings(
                    image_size=(h, w), 
                    radius = 0.01,
                    points_per_pixel = 10,
                    bin_size=0
                )
            self.rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings)

        
        b, k, *_ = images.shape
        images = rearrange(images, "b k c h w -> (b k) c h w")
        images = (images + 1) / 2

        if self.distance == "cosine":
            # NOTE: Compute features
            lr_feat = self._get_features(images)
            # NOTE: Transform feature to higher resolution either using `interpolate` or `FeatUp`
            hr_feat = self._interpolate(lr_feat, images)
            # K=2 since we only compare an image pairs
            hr_feat = rearrange(hr_feat, "(b k) ... -> b k ...", k=2)
        images = rearrange(images, "(b k) ... -> b k ...", k=2)
        images = 2 * images - 1

        # NOTE: Apply Backbone MASt3R/DUSt3R/RAFT to warp one view to the other and compute overlap masks
        if self.backbone == "raft":
            flow = self.backbone_model(images[:, 0, ...], images[:, 1, ...])[-1]

            if self.distance == "cosine":
                view1 = hr_feat[:, 0, ...]
                view2 = hr_feat[:, 1, ...]
            else:
                view1 = images[:, 0, ...]
                view2 = images[:, 1, ...]

            warped_view, mask = self.warp_image(view2, flow)
            rendering = torch.stack([view1, warped_view], dim=1)

        else:
            view1 = {"img": images[:, 0, ...], "instance": [""]}
            view2 = {"img": images[:, 1, ...], "instance": [""]}
            pred1, pred2 = self.backbone_model(view1, view2)

            ptmps = torch.stack([pred1["pts3d"], pred2["pts3d_in_other_view"]], dim=1).detach()
            conf = torch.stack([pred1["conf"], pred2["conf"]], dim=1).detach()

            # NOTE: Get canonical point map using the confidences
            confs11 = conf.unsqueeze(-1) - 0.999
            canon = (confs11 * ptmps).sum(1) / confs11.sum(1)
            
            # Define principal point
            pp = torch.tensor([w /2 , h / 2], device=canon.device)
            
            
            # NOTE: Estimating fx and fy for a given canonical point map
            B, H, W, THREE = canon.shape
            assert THREE == 3

            # centered pixel grid
            pixels = xy_grid(W, H, device=canon.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
            canon = canon.flatten(1, 2)  # (B, HW, 3)

            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = canon.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.stack((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-2)[0]
            
            # Normalized focal length
            focal[..., 0] = 1 + focal[..., 0]/w
            focal[..., 1] = 1 + focal[..., 1]/h
            focal = repeat(focal, "b c -> (b k) c", k=2)
            # NOTE: Unproject feature on the point cloud
            ptmps = rearrange(ptmps, "b k h w c -> (b k) (h w) c", b=b, k=2)
            if self.distance == "cosine":
                features = rearrange(hr_feat, "b k c h w -> (b k) (h w) c", k=2)

            else:
                images = (images + 1) / 2
                features = rearrange(images, "b k c h w-> (b k) (h w) c", k=2)
            point_cloud = Pointclouds(points=ptmps, features=features)
            
            # NOTE: Project and Render
            R = torch.eye(3)
            R[0, 0] *= -1
            R[1, 1] *= -1
            R = repeat(R, "... -> (b k) ...", b=b, k=2)
            T = torch.zeros((3, ))
            T = repeat(T, "... -> (b k) ...", b=b, k=2)

            # Define Pytorch3D camera for projection
            cameras = PerspectiveCameras(device=ptmps.device, R=R, T=T, focal_length=focal)
            # Render via point rasterizer to get projected features
            with torch.autocast("cuda", enabled=False):
                rendering, zbuf = self.render(point_cloud, cameras=cameras, background_color=[-10000] * features.shape[-1])
            rendering = rearrange(rendering, "(b k) h w c -> b k c h w",  b=b, k=2)
            
            # Compute overlapping mask
            non_overlap_mask = (rendering == -10000)
            overlap_mask = (1 - non_overlap_mask.float()).prod(2).prod(1)
            
            # Zero out regions which do not overlap
            rendering[non_overlap_mask] = 0.0

            # Mask for weighted sum
            mask = overlap_mask

        # NOTE: Uncomment for incorporating occlusion masks along with overlap mask
        # zbuf = rearrange(zbuf, "(b k) ... -> b k ...",  b=b, k=2)
        # closest_z = zbuf[..., 0]
        # diff = (closest_z[:, 0, ...] - closest_z[:, 1, ...]).abs()
        # mask = (~(diff > 0.5) * (closest_z != -1).prod(1)) * mask
        
        # NOTE: Compute scores as either feature dissimilarity, RMSE, LPIPS, SSIM, MSE, or PSNR 
        score_map, weighted = self._distance(rendering[:, 0, ...], rendering[:, 1, ...], mask=mask)

        outputs = [weighted]
        if return_overlap_mask:
            outputs.append(mask)
            
        if return_score_map:
            outputs.append(score_map)
        
        if return_projections:
            outputs.append(rendering)

        return (*outputs, )

