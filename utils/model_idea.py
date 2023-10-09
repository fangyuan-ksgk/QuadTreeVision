import unicodedata
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import einops
import torch.optim as optim
import torchvision
import time
import argparse
import mediapy as media
import json
import argparse
from quadtree import *

# Get the code point for the smiling face emoji
# code_point = ord(unicodedata.lookup('SMILING FACE WITH OPEN MOUTH'))
# print(f'The code point for the smiling face emoji is {code_point:X}')

# Import the smiling face emoji
smiling_face = chr(0x1F603)
thumbs_up = chr(0x1F44D)
heart_eyes = chr(0x1F60D)


# Define Convolution-Based detail & feature predictor
# Residual Prediction should somehow takes Original Values as input ! Not Guessing on the stuff ....

# PatchConvolution Layer: Default dim / patch size for convmixer CIFAR10 training setup
class PatchConv(nn.Module):
    def __init__(self, dim_patch_feature=256, patch_size=2):
        super(PatchConv, self).__init__()

        # Common initial layers shared by both branches
        self.common_layers = nn.Sequential(
            nn.Conv2d(3, dim_patch_feature, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim_patch_feature)
        )

        # Feature prediction branch
        self.feature_branch = nn.Identity()

        # Detail prediction branch
        # ISSUE: Rescaled to the same size, a BIG Image patch's detail level should be higher than 
        # detail level of an actual small image patch's detail
        self.detail_branch = nn.Sequential(
            nn.Conv2d(dim_patch_feature, 1, kernel_size=1),
            # nn.Sigmoid()  # Scales output to the [0, 1] range
        )

    def forward(self, x):
        # Ensure x has a batch dimension of size 1 if it's a single image
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add a batch dimension of size 1
            
        # Forward pass through common initial layers
        common_output = self.common_layers(x)

        # Feature prediction branch
        feature_output = self.feature_branch(common_output)

        # Detail prediction branch
        detail_output = self.detail_branch(common_output)
        

        return feature_output, detail_output
    


        
        

    
# class PatchConv(nn.Module):
#     def __init__(self, dim_patch_feature=256, patch_size=2):
#         super(PatchConv, self).__init__()

#         # Common initial layers shared by both branches
#         self.common_layers = nn.Sequential(
#             nn.Conv2d(3, dim_patch_feature, kernel_size=patch_size, stride=patch_size),
#             nn.GELU(),
#             nn.BatchNorm2d(dim_patch_feature)
#         )

#         # Feature prediction branch
#         self.feature_branch = nn.Identity()

#         # Detail prediction branch
#         self.detail_branch = nn.Sequential(
#             nn.Conv2d(dim_patch_feature, 1, kernel_size=1),
#             nn.Sigmoid()  # Scales output to the [0, 1] range
#         )
        
#         # Residual regressor head
#         self.residual_regressor = nn.Sequential(
#             nn.Conv2d(dim_patch_feature, dim_patch_feature, kernel_size=1),  # Adjust output channels
#             nn.Sigmoid()  # Adjust activation function as needed
#         )

#     def forward(self, x):
#         # Ensure x has a batch dimension of size 1 if it's a single image
#         if x.dim() == 3:
#             x = x.unsqueeze(0)  # Add a batch dimension of size 1
            
#         # Forward pass through common initial layers
#         common_output = self.common_layers(x)

#         # Feature prediction branch
#         feature_output = self.feature_branch(common_output)

#         # Detail prediction branch
#         detail_output = self.detail_branch(common_output)
        
#         # Residual regressor head
#         residual_output = self.residual_regressor(common_output)
        
#         return feature_output, detail_output, residual_output
#         # return feature_output, detail_output, feature_output
    
    
# PreProcessor 
sudo_mean = (0.3792, 0.3239, 0.2440)
sudo_std = (0.2530, 0.2531, 0.2321)
test_transform = transforms.Compose([
    transforms.Normalize(sudo_mean, sudo_std)
])


# Scale PatchFeature with PatchDetail for end-to-end trainable model design
# Adaptive Exploration under certain threshold values
def scale_feature(feature, patch_detail, target_patch_size):
    reshape_detail = patch_detail.repeat_interleave(target_patch_size[1]//2).reshape(1,1,1,-1)
    assert len(feature.shape)==len(reshape_detail.shape), 'Shape mismatch for scale feature'
    assert feature.shape[0]==reshape_detail.shape[0], 'BatchSize mismatch for scale feature'
    return feature * reshape_detail


# Pytorch Implementation -- Only ZeroMask, otherwise training will be slow without Pytorch-CUDA kernel
def rescale_feature(input_tensor, target_size):
    if len(input_tensor.shape)==3:
        resized_tensor = F.interpolate(input_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    else:
        resized_tensor = F.interpolate(input_tensor, size=target_size, mode='bilinear', align_corners=False)
    return resized_tensor

def average_over_kernels(input_tensor, kernel_size = (25, 40)):
    # Get the dimensions of the input tensor
    batch_size, channels, height, width = input_tensor.size()

    kernel_height, kernel_width = kernel_size

    # Calculate the number of rows and columns for the output tensor
    output_rows = height // kernel_height
    output_columns = width // kernel_width

    # Reshape the input tensor into (batch_size, channels, output_rows, kernel_height, output_columns, kernel_width)
    input_tensor = input_tensor.view(
        batch_size, channels, output_rows, kernel_height, output_columns, kernel_width
    )

    # Compute the average over each (1, 1, 25, 40) kernel
    averaged_tensor = input_tensor.mean(dim=(3, 5))

    return averaged_tensor


from einops import rearrange

def slice_img_with_mask_into_patches(img_data, mask):
    # Get the dimensions of the original image
    C, H, W = img_data.shape

    # Get the dimensions of the mask
    mask = mask.to(torch.int64)
    h, w = mask.shape

    # Calculate the number of sub-images along each dimension
    num_sub_images_h = H // h
    num_sub_images_w = W // w

    # Flatten the image along the height dimension
    # h number of vertical slices, w number of horizonal slices
    slice_patches = rearrange(img_data, 'c (nh h) (nw w) -> c nh nw h w', nh=h, nw=w)
    # Index w Mask
    indices = mask.nonzero()
    kept_patches = slice_patches[...,indices[:,0],indices[0,:],:,:]
    # Kept patches contatenated horizontally
    kept_patches = rearrange(kept_patches, 'c np h w -> c (np h) w')
    return kept_patches

def get_detail_scale_factor(layer, const = 50.):
    scale_factor = const * (1 / 4**layer)
    return scale_factor

# Mask is for the aligned patches (full mask)
def rescale_n_patchify_with_mask(img_data, target_patch_size, layer, mask):
    # layer decides the level of quadtree node
    # try to separate patches from its location
    h_num = (2**layer)
    w_num = (2**layer)
    # print(h_num, w_num)
    h_patch = target_patch_size[0]
    w_patch = target_patch_size[1]
    target_size = (int(h_num * h_patch), int(w_num * w_patch))
    rescale_img = rescale_feature(img_data, target_size)
    patches = rearrange(rescale_img, 'c (nh h) (nw w) -> c (nh nw) h w', h=h_patch, w=w_patch)
    if mask is not None:
        rep_mask = mask.repeat_interleave(4)
        patches = patches[..., rep_mask, :, :]
    align_patches = rearrange(patches, 'c n h w -> c h (n w)')
    # print(f'At layer {layer}: In total we have {patches.shape[-3]} number of un-masked patches out of {int(h_num * w_num)} patches')
    return patches, align_patches  

def update_mask(mask, patch_detail, detail_threshold):
    if len(mask.shape)==1:
        orig_mask = mask
        exp_mask = mask.repeat_interleave(4)
        index_nonzero = exp_mask.nonzero()[:,0]
        exp_mask[index_nonzero] *= (patch_detail>detail_threshold)
        return exp_mask
    else:
        orig_mask = mask
        exp_mask = mask.repeat_interleave(4, dim=-1)
        put_mask = (patch_detail>detail_threshold)
        indices = torch.where(exp_mask)
        b,n = put_mask.shape
        assert len(indices[0]) == int(b*n), 'shape mismatch here!'
        exp_mask[indices] = rearrange(put_mask, 'b n -> (b n)', b=b, n=n)
        return exp_mask

def process_detail_prediction_n_update_mask(detail, layer, target_patch_size, mask,
                                            detail_threshold=0.5,
                                            const=50.):
    # Predicteed Patch Detail are for Resized Image Patch
    realign_detail = rearrange(detail, 'a b h (n w) -> a b n h w', 
                               h = int(target_patch_size[0]/2), 
                               w=int(target_patch_size[1]/2))
    
    # Scale Avg Detail on ImagePatch by its original Area (before resize to image patch)
    detail_scaler = get_detail_scale_factor(layer, const=50.)
    patch_detail = (realign_detail.mean(axis=(-1,-2)) * detail_scaler).squeeze()
    
    # Update Mask with prev mask & current prediction result
    if mask is not None:
        updated_mask = update_mask(mask, patch_detail, detail_threshold)
    else:
        updated_mask = (patch_detail>detail_threshold)
        
    return patch_detail, updated_mask


# Visualize Pathes & Mask
def visualize_patches_with_mask(patches, mask):
    # Assuming patches is your input array
    c, pp, h, w = patches.shape
    p = int(np.sqrt(pp))
    final_image = np.zeros((c, int(p*h), int(p*w)))

    # Loop through the patches
    for i in range(p):
        for j in range(p):
            patch = patches[:, i*p + j]
            # Masked Patch, do not present
            if not mask[i*p + j]:
                continue
            final_image[:, i*h:(i+1)*h, j*w:(j+1)*w] = patch

            # Draw lines to indicate separation
            # Draw Separation Lines
            start_u, start_v = (j*w, i*h)
            end_u, end_v = (j+1)*w, (i+1)*h
            boundary_width = 1
            final_image[:,start_v: start_v + boundary_width, start_u: end_u + 1] = np.array([1.,1.,1.]).reshape(3,1,1)
            final_image[:,end_v-boundary_width: end_v + 1, start_u: end_u + 1] = np.array([1.,1.,1.]).reshape(3,1,1)
            final_image[:,start_v: end_v + 1, start_u: start_u + boundary_width] = np.array([1.,1.,1.]).reshape(3,1,1)
            final_image[:,start_v: end_v + 1, end_u-boundary_width: end_u + 1] = np.array([1.,1.,1.]).reshape(3,1,1)

    # Rearrange the axes to be (width, height, channels) for visualization
    final_image = np.moveaxis(final_image, 0, -1)
    return final_image

# ConvMixer Connection
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
    
# Batch-wise Operation
def index_patches_with_mask(patches, rep_mask):
    # Batch-wise Index with Mask
    b,c,n,h,w = patches.shape
    assert b==rep_mask.shape[0] and n==rep_mask.shape[1], ' mismatch shape '
    keep_patches = rearrange(patches.permute(0,2,1,3,4)[rep_mask], '(b n) c h w -> b c n h w', b=b, c=c, h=h, w=w)
    return keep_patches

# Batch-wise Operation
def patchify_with_mask(img_data, target_patch_size, layer, mask):
    # layer decides the level of quadtree node
    # try to separate patches from its location
    h_num = (2**layer)
    w_num = (2**layer)
    # print(h_num, w_num)
    h_patch = target_patch_size[0]
    w_patch = target_patch_size[1]
    target_size = (int(h_num * h_patch), int(w_num * w_patch))
    rescale_img = rescale_feature(img_data, target_size)
    patches = rearrange(rescale_img, 'b c (nh h) (nw w) -> b c (nh nw) h w', h=h_patch, w=w_patch)
    if mask is not None:
        rep_mask = mask.repeat_interleave(4, axis=-1)
        patches = index_patches_with_mask(patches, rep_mask)
    align_patches = rearrange(patches, 'b c n h w -> b c h (n w)')
    # print(f'At layer {layer}: In total we have {patches.shape[-3]} number of un-masked patches out of {int(h_num * w_num)} patches')
    return patches, align_patches 

def update_mask(mask, patch_detail, detail_threshold):
    if len(mask.shape)==1:
        orig_mask = mask
        exp_mask = mask.repeat_interleave(4)
        index_nonzero = exp_mask.nonzero()[:,0]
        exp_mask[index_nonzero] *= (patch_detail>detail_threshold)
        return exp_mask
    else:
        orig_mask = mask
        exp_mask = mask.repeat_interleave(4, dim=-1)
        put_mask = (patch_detail>detail_threshold)
        indices = torch.where(exp_mask)
        b,n = put_mask.shape
        assert len(indices[0]) == int(b*n), 'shape mismatch here!'
        exp_mask[indices] = rearrange(put_mask, 'b n -> (b n)', b=b, n=n)
        return exp_mask
    
def process_detail_and_update_mask(detail, layer, target_patch_size, mask,
                                   detail_threshold=0.5,
                                   const=50.):
    if mask is None or len(mask.shape)==1:
        return process_detail_prediction_n_update_mask(detail, layer, target_patch_size, mask,
                                                detail_threshold, const)
    # Batch-wise Computation
    # Predicted Patch Detail
    realign_detail = rearrange(detail, 'b c h (n w) -> b c n h w', 
                               h = int(target_patch_size[0]/2), 
                               w=int(target_patch_size[1]/2))
    
    # Scale Avg Detail on ImagePatch by its original Area (before resize to image patch)
    detail_scaler = get_detail_scale_factor(layer, const=50.)
    patch_detail = (realign_detail.mean(axis=(-1,-2)) * detail_scaler).squeeze()
    
    # Update Mask with prev mask & current prediction result
    if mask is not None:
        updated_mask = update_mask(mask, patch_detail, detail_threshold)
    else:
        updated_mask = (patch_detail>detail_threshold)
        
    return patch_detail, updated_mask


def upsample_detail(patch_detail, max_layer):
    input_tensor = patch_detail
    target_size = (int(2**max_layer), int(2**max_layer))
    resized_detail = F.interpolate(input_tensor.unsqueeze(1), size=target_size, mode='nearest')
    return resized_detail.squeeze(1)

def upsample_detail_size(patch_detail, target_size):
    input_tensor = patch_detail
    feature_detail = F.interpolate(input_tensor.unsqueeze(1), size=target_size, mode='nearest')
    return feature_detail

def batch_patch_process_n_pad(bimg, layer, target_patch_size, mask):
    h_num = (2**layer)
    w_num = (2**layer)
    h_patch = target_patch_size[0]
    w_patch = target_patch_size[1]
    target_size = (int(h_num * h_patch), int(w_num * w_patch))
    rescale_img = rescale_feature(bimg, target_size)
    rescale_mask = upsample_detail_size(mask, rescale_img.shape[-2:])
    rescale_img = rescale_img * rescale_mask # ZeroPad with Mask
    return rescale_img

detail_threshold = 0.5
def batch_update_mask_feature_with_detail(feature, detail, layer, target_patch_size, max_layer, mask,
                                         detail_threshold):
    # Update Pixel-wise Detail with Mask, first, as detail will be multiplied with feature
    detail *= upsample_detail_size(mask, detail.shape[-2:])
    realign_detail = rearrange(detail, 'b c (n h) (m w) -> b c n m h w', 
                                   h = int(target_patch_size[0]/2), 
                                   w = int(target_patch_size[1]/2))
    # Scale Avg Detail on ImagePatch by its original Area (before resize to image patch)
    # Naturally, The same PConv as a detail extractor works by the same logic on bigger & smaller patch
    # Bigger Patch alwasys contains more detail, therefore having bigger avg detail values
    
    detail_scaler = get_detail_scale_factor(layer, const=50.)
    patch_detail = (realign_detail.mean(axis=(-1,-2)) * detail_scaler).squeeze() # 面积越大，细节越多
    
    # Small Patch naturally has less detail, therefore we do NOT need to scale it anymore
    # patch_detail = (realign_detail.mean(axis=(-1,-2))).squeeze() # Avg Detail level
    
    resized_detail = upsample_detail(patch_detail, max_layer)
    mask = upsample_detail(mask, max_layer)
    mask *= (resized_detail>detail_threshold)
    
    # To Avoid Exploding values, use another sigmoid function 
    input_tensor = patch_detail / (detail_scaler)
    # input_tensor = torch.sigmoid(patch_detail)
    # input_tensor = torch.nn.Sigmoid(patch_detail)
    target_size = feature.shape[-2:]
    feature_detail = F.interpolate(input_tensor.unsqueeze(1), size=target_size, mode='nearest')
    feature_detail = upsample_detail_size(patch_detail, target_size)
    # Consider doing a Sigmoid to normalized on the feature values here ...
    feature *= feature_detail
    return mask, feature

# Do Not put more emphasis on coarser detail over finer detail here 
def ensemble_feature_upsample(ensemble_feature, residual, layer, pconv_scale):
    target_size = int(pconv_scale[0] * (2**layer)), int(pconv_scale[1] * (2**layer))
    # print('target size: ', target_size)
    ensemble_feature = F.interpolate(ensemble_feature, size=target_size, mode='nearest')
    # feature_scale = 4**layer
    residual_scale = 1.
    return ensemble_feature + residual*residual_scale

# Added Feature Scaler to Scale Feature to be more similar across layers
def upsample_n_ensemble(ensemble_feature, feature, layer, target_patch_size):
    target_size = int(target_patch_size[0]//2*(2**layer)), int(target_patch_size[1]//2*(2**layer))
    ensemble_feature = F.interpolate(ensemble_feature, size=target_size, mode='nearest')
    feature_scale = 2**layer
    return (ensemble_feature + feature*feature_scale)

def Mixer(dim=256, depth=5, kernel_size=5, n_classes=10):
    return nn.Sequential(
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

def propagate_batch(bimg, 
                    pconv,
                    mixer,
                    target_patch_size,
                    detail_threshold=0.5,
                    max_layer=5):
    
    B,C,H,W = bimg.shape
    # Batch-wise Implementation: Initial Mask
    max_layer = 5
    mask = torch.ones((B, int(2**1), int(2**1)))
    ensemble_feature = torch.zeros((B, 256, int(target_patch_size[0]//2*(2**0)), int(target_patch_size[1]//2*(2**0))))
    
    res = {}
    # Rescale Image & Propagation
    for layer in range(1, max_layer):
        # print('layer: ', layer)
        # Rescale
        rescale_img = batch_patch_process_n_pad(bimg, layer, target_patch_size, mask)
        # print('rescale completed')
        # Predict
        feature, detail = pconv(rescale_img)
        # print('prediction completed -- feature shape: ', feature.shape)
        # Update Mask, Feature with Patch Detail in BatchOperation
        # Trainable-Detail, Use the Patch-Detail to Scale on Feature Prediction
        mask, feature = batch_update_mask_feature_with_detail(feature, detail, layer,
                                                              target_patch_size,
                                                              max_layer=(layer+1), 
                                                              mask=mask,
                                                              detail_threshold = detail_threshold)


        # print('mask & feature updated')
        # Ensemble Addition of Feature
        ensemble_feature = upsample_n_ensemble(ensemble_feature, feature, layer, target_patch_size)
        
        res[layer] = {'feature': feature, 'mask': mask}
        
    res['ensemble'] = ensemble_feature
    res['pred'] = mixer(ensemble_feature)
    return res


def get_rescale_shape(layer, rescale_patch_size):
    return rescale_patch_size[0] * int(2**layer), rescale_patch_size[1] * int(2**layer)

    
    
def upscaled_topk_mask_conv(mask, args, patches, pconv, ratio=0.75):
    # Upsample Mask
    upscaled_mask = F.interpolate(mask, (int(2**layer), int(2**layer)), mode='nearest')

    # TopK selection from Mask
    ratio = 0.75
    flat_mask = rearrange(upscaled_mask, 'b 1 h w -> b 1 (h w)')
    flat_topk_index = torch.argsort(flat_mask, axis=-1, descending=True)[...,:int(flat_mask.shape[-1]*ratio)]
    flat_topk_index = torch.sort(flat_topk_index, axis=-1).values
    topk = flat_topk_index.shape[-1]

    # Index from Patches
    patch_of_patch = rearrange(patches, 'b c (n h) (m w) -> b c (n m) h w', h=h_patch, w=w_patch)
    batch_index = torch.arange(B).repeat_interleave(topk)
    sample_index = rearrange(flat_topk_index, 'b 1 topk -> (b topk)')
    realign_patch = rearrange(patch_of_patch[batch_index, :, sample_index, :, :], '(b topk) c h w -> b c h (topk w)', b=B)

    # Prediction
    realign_feature, realign_detail = em.pconv(realign_patch)

    # Put Back to Full Shaped Feature & Detail
    nh = nw = int(2**layer)
    h, w = args.pconv_patch_scale
    dim = args.pconv_feature_dim
    exp_feature = torch.zeros(B, dim, nh*nw, h, w).to(realign_feature.device)
    exp_detail = torch.zeros(B, 1, nh*nw, h, w).to(realign_detail.device)

    put_feature = rearrange(realign_feature, 'b c h (topk w) -> (b topk) c h w', topk=topk)
    exp_feature[batch_index, :, sample_index] = put_feature
    feature = rearrange(exp_feature, 'b c (nh nw) h w -> b c (nh h) (nw w)', nh=nh, nw=nw, h=h, w=w)

    put_detail = rearrange(realign_detail, 'b c h (topk w) -> (b topk) c h w', topk=topk)
    exp_detail[batch_index, :, sample_index] = put_detail
    detail = rearrange(exp_detail, 'b c (nh nw) h w -> b c (nh h) (nw w)', nh=nh, nw=nw, h=h, w=w)
    
    return feature, detail


def decide_mask_ratios(max_layer, layer_ratio):
    mask_ratios = {}
    for l in range(0, max_layer + 1):
        if l<=1:
            mask_ratios[l] = 1.
        else:
            mask_ratios[l] = layer_ratio * mask_ratios[l-1]
    return mask_ratios



            
            
def save_args(args, config_file):
    args_dict = vars(args)
    with open(config_file, 'w') as fp:
        json.dump(args_dict, fp)
    
def load_args(args_dict):
    parser = argparse.ArgumentParser()
    for key, value in args_dict.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    return parser.parse_args([])

def read_args(config_file):
    with open(config_file, 'r') as fp:
        args_dict = json.load(fp)
    return load_args(args_dict)

def read_info(config_file):
    with open(config_file, 'r') as fp:
        args_dict = json_load(fp)
    return args_dict

def process_args(args):
    # Max layers used to correct / refine on rescaled sizes of Image & Patches
    rescale_img_h = args.preprocess_img_size[0]//(2**args.max_layer) * (2**args.max_layer)
    rescale_img_w = args.preprocess_img_size[1]//(2**args.max_layer) * (2**args.max_layer)
    rescale_patch_h = rescale_img_h / (2**args.max_layer)
    rescale_patch_w = rescale_img_w / (2**args.max_layer)

    args.preprocess_img_size = int(rescale_img_h), int(rescale_img_w)
    args.rescale_patch_size = int(rescale_patch_h), int(rescale_patch_w)
    
    # Given Pconv kernel scale, decide on kernel/stide size
    pconv_kernel_h = rescale_patch_h // args.pconv_patch_scale[0]
    pconv_kernel_w = rescale_patch_w // args.pconv_patch_scale[1]
    args.pconv_patch_size = int(pconv_kernel_h), int(pconv_kernel_w)
    

def parse_eyemixer_args(comm = []):
    parser = argparse.ArgumentParser(
                        prog='HumanViewer',
                        description='Perception like Human -- Learnable Image Perception with QuadTreeNodes',
                        epilog='Text at the bottom of help')

    parser.add_argument('--name', type=str, default="EyeMixer")

    parser.add_argument('-max_layer', '--max_layer', default=5)
    parser.add_argument('-imsize', '--preprocess_img_size', default=(192 * 2,256 * 2))
    parser.add_argument('-pconv_dim', '--pconv_feature_dim', default=256)
    parser.add_argument('-pconv_scale', '--pconv_patch_scale', default=(2,2))
    parser.add_argument('-depth', '--mixer_depth', default=4)
    parser.add_argument('-conv_ks', '--mixer_kernel_size', default=5)
    parser.add_argument('-n_classes', '--num_classes', default=10)
    parser.add_argument('-dt', '--detail_threshold', default=0.5)
    parser.add_argument('-m', '--pconv_mode', default=0) # ['topk', 'zeropad'] for Pconv Mode
    parser.add_argument('-r', '--pconv_layer_ratio', default=0.75)
    
    parser.add_argument('--wd', default=0.01, type=float)
    parser.add_argument('--clip-norm', action='store_true')
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--lr-max', default=0.01, type=float)
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--scale', default=0.75, type=float)
    parser.add_argument('--reprob', default=0.25, type=float)
    parser.add_argument('--ra-m', default=8, type=int)
    parser.add_argument('--ra-n', default=1, type=int)
    parser.add_argument('--jitter', default=0.1, type=float)

    args = parser.parse_args(comm)
    process_args(args)
    
    return args


def read_info(config_file):
    with open(config_file, 'r') as fp:
        args_dict = json.load(fp)
    return args_dict


def load_cifar10_test():
    
    # Try to Enable a Training Routine with CIFAR10 dataset 
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])


    testset = torchvision.datasets.CIFAR10(root='/mnt/d/Data', train=False,
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=1)
    return testset, testloader


cifar10_label_to_class = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# Brightness represents the degree to which we emphasize it (no mask)
def divide_image_with_mask(image, mask):
    H,W,C = image.shape
    h,w = mask.shape
    nh,nw = H//h,W//w
    rimg = image
    for i in range(h):
        for j in range(w):
            alpha = mask[i,j]
            rimg[int(nh*i):int((i+1)*nh), int(j*nw):int((j+1)*nw)] *= alpha
            lwidth = 1
            # Top Line
            rimg[int(nh*i):int(nh*i)+lwidth, int(j*nw):int((j+1)*nw)] = 1.
            # Bottom Line
            rimg[int(nh*(i+1))-lwidth:int(nh*(i+1)), int(j*nw):int((j+1)*nw)] = 1.
            # Left Line
            rimg[int(nh*i):int((i+1)*nh), int(j*nw):int(j*nw)+lwidth] = 1.
            # Right Line
            rimg[int(nh*i):int((i+1)*nh), int((j+1)*nw)-lwidth:int((j+1)*nw)] = 1.
    return rimg

def get_norm_feat(feat):
    min_feat = torch.min(feat, dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
    max_feat = torch.max(feat, dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
    norm_feat = ((feat - min_feat) / (max_feat - min_feat))
    return norm_feat[0].permute(1,2,0).detach().numpy()

# if mode=='topk':
#                 # TopK Selective Convolution: Ratio at each layer to uniform masked number of patches 
#                 # ----------------------------------- TopK Selective Convolution
#                 # Upsample Mask
#                 upscaled_mask = F.interpolate(mask, (int(2**layer), int(2**layer)), mode='nearest')

#                 # TopK selection from Mask
#                 ratio *= self.mask_ratios[layer]
#                 flat_mask = rearrange(upscaled_mask, 'b 1 h w -> b 1 (h w)')
#                 flat_topk_index = torch.argsort(flat_mask, axis=-1, descending=True)[...,:int(flat_mask.shape[-1]*ratio)]
#                 flat_topk_index = torch.sort(flat_topk_index, axis=-1).values
#                 topk = flat_topk_index.shape[-1]

#                 # Index from Patches
#                 patch_of_patch = rearrange(patches, 'b c (n h) (m w) -> b c (n m) h w', h=h_patch, w=w_patch)
#                 batch_index = torch.arange(B).repeat_interleave(topk)
#                 sample_index = rearrange(flat_topk_index, 'b 1 topk -> (b topk)')
#                 realign_patch = rearrange(patch_of_patch[batch_index, :, sample_index, :, :], '(b topk) c h w -> b c h (topk w)', b=B)

#                 # Prediction
#                 # realign_feature, realign_detail = self.pconvs[layer](realign_patch)
#                 if layer==0:
#                     feature, detail = self.pconvs[0](patches)
#                 else:
#                     # feature, detail = self.pconvs[1](patches)
#                     feature, detail = self.pconvs[layer](patches)
                                    
#                 # Put Back to Full Shaped Feature & Detail
#                 nh = nw = int(2**layer)
#                 h, w = self.pconv_patch_scale
#                 dim = self.pconv_feature_dim
#                 exp_feature = torch.zeros(B, dim, nh*nw, h, w).to(realign_feature.dtype).to(realign_feature.device)
#                 exp_detail = torch.zeros(B, 1, nh*nw, h, w).to(realign_detail.dtype).to(realign_detail.device)

#                 put_feature = rearrange(realign_feature, 'b c h (topk w) -> (b topk) c h w', topk=topk)
#                 exp_feature[batch_index, :, sample_index] = put_feature
#                 feature = rearrange(exp_feature, 'b c (nh nw) h w -> b c (nh h) (nw w)', nh=nh, nw=nw, h=h, w=w)

#                 put_detail = rearrange(realign_detail, 'b c h (topk w) -> (b topk) c h w', topk=topk)
#                 exp_detail[batch_index, :, sample_index] = put_detail
#                 detail = rearrange(exp_detail, 'b c (nh nw) h w -> b c (nh h) (nw w)', nh=nh, nw=nw, h=h, w=w)
#                 # ----------------------------------- TopK Selective Convolution

class EyeMixer(nn.Module):
    def __init__(self, args):
        super(EyeMixer, self).__init__()
        self.pconv_feature_dim = args.pconv_feature_dim
        self.pconv_patch_scale = args.pconv_patch_scale
        self.subimg_size = args.subimg_size
        self.detail_threshold = args.detail_threshold
        # Maximum layer of explorable QuadTreeNode 
        self.max_layer = args.max_layer
        
        # Image Preprocessor : Augmentation Free Version
        self.preprocess =  transforms.Compose([
                                crop_to_multiple(args.rescale_patch_size),
                                transforms.Resize(args.preprocess_img_size),
                                transforms.ToTensor()
                            ])
        # PatchExploration Network: Residual Inference for FineLayer, Direct Inferece for CoarseLayer
        self.pconvs = nn.ModuleList()  # Create a ModuleList to hold the PatchConv modules
                
        # Strategy 1: Completely Independent Pconv for each layer -- performance ok, but lacks consistency
        for layer in range(0, self.max_layer+1):
            self.pconvs.append(PatchConv(dim_patch_feature=args.pconv_feature_dim, 
                                           patch_size=args.pconv_patch_size))
        
            
        # Mixer: Depthwise & Pointwise Convolution
        self.mixer = Mixer(dim=args.pconv_feature_dim,
                           depth=args.mixer_depth,
                           kernel_size=args.mixer_kernel_size,
                           n_classes=args.num_classes)
        
        # Mask Ratios Scheduler
        self.mask_ratios = decide_mask_ratios(self.max_layer, layer_ratio=args.pconv_layer_ratio)
        modes = ['topk', 'zeropad']
        self.mode = modes[args.pconv_mode]
        
    # mode: ['zeropad', 'skipconv']
    def patch_conv(self, x, mode='zeropad'):
        B,N,H,W = x.shape
        subimg_h, subimg_w = self.subimg_size
        mask = torch.ones((B,1,1,1)).float().to(x.device)
        ensemble_feature = torch.zeros((B, self.pconv_feature_dim, self.pconv_patch_scale[0], self.pconv_patch_scale[1])).to(x.device)
        ratio = 1.
        res = {}
        for layer in range(0, self.max_layer+1):
            
            # (H_layer, W_layer) = (2^layer * SubImg_h, 2^layer * SubImg_w)
            subimgs_size = get_rescale_shape(layer, self.subimg_size)
            
            # SubImgs: (B, C, H_layer, W_layer)
            subimgs = F.interpolate(x, subimgs_size, mode='bilinear', align_corners=True)
            
            if mode=='zeropad':
                # Shape -- Mask: (B, 1, 2^{layer-1}, w^{layer-1})
                
                # Rescaled Mask at Current Layer
                # Shape --- upscaled_mask: (B, 1, H(layer), W(layer))
                upscaled_mask = F.interpolate(mask, subimgs_size, mode='nearest')

                # Zero-Pad on Masked Patches || Discrete Masking Mechanism
                bin_mask = torch.bernoulli(mask)
                # --- 1. Hard Tresholding
                subimgs *= (upscaled_mask > self.detail_threshold)

                # Propagate & Prediction
                # Each SubImg is converted to (PatchScale_h, Patch_Scale_w) features
                # Each Feature is (Dim_feature,) in shape
                
                # feature: (B, Dim_feature, PatchScale_h, PatchScale_w)
                # detail:  (B,           1, PatchScale_h, PatchScale_w)
                feature, detail = self.pconvs[layer](subimgs)

                # Default value for topk selection
                topk = int(4**layer)
                
            
            # Mask Update require Detail Scaler
            # detail_scaler = get_detail_scale_factor(layer, const=50.)
            # patch_detail = detail * detail_scaler
            mask_detail = rearrange(detail, 'c 1 (n h) (m w) -> c 1 n m (h w)', h=self.pconv_patch_scale[0], w=self.pconv_patch_scale[1]).mean(axis=-1)

            pre_mask = F.interpolate(mask, mask_detail.shape[-2:], mode='nearest').clone()
            # Update Mask by Cumulative Product
            mask = F.interpolate(mask, mask_detail.shape[-2:], mode='nearest') * mask_detail

            # Mask Out Feature : With float mask values
            residual = F.interpolate(mask, feature.shape[-2:], mode='nearest') * feature

            # Ensemble Addition of Feature
            ensemble_feature = ensemble_feature_upsample(ensemble_feature, residual, layer, self.pconv_patch_scale)

            res[layer] = {'feature':ensemble_feature, 'mask': mask, 
                          'topk': topk, 'mask_detail': mask_detail, 'prev_mask': pre_mask, 
                          'residual': residual}
        # This is redundant ! 
        res['ensemble_feature'] = ensemble_feature
        return res   
    
    def forward(self, x):
        res = self.patch_conv(x)
        pred = self.mixer(res['ensemble_feature'])
        res['pred'] = pred
        return res
    
    def visualize_res(self, res, sample_idx, channel_idx):
        keep_ratios = np.cumprod([self.mask_ratios[l] for l in range(0, self.max_layer+1)])
        if not isinstance(channel_idx, list):
            fmaps = {f'Layer {layer} Feature': res[layer]['feature'][sample_idx,channel_idx].detach().clip(0.,1.).numpy() for layer in range(1,self.max_layer+1)}
            fmaps['Ensemble Feature'] = res['ensemble_feature'][sample_idx,channel_idx].detach().clip(0.,1.).numpy()
        else:
            fmaps = {f'Layer {layer} Feature': res[layer]['feature'][sample_idx,channel_idx].detach().permute(1,2,0).clip(0.,1.).numpy() for layer in range(1,self.max_layer+1)}
            fmaps['Ensemble Feature'] = res['ensemble_feature'][sample_idx,channel_idx].detach().permute(1,2,0).clip(0.,1.).numpy()

        media.show_images(fmaps, height=300)
        # Mask Maps
        dmaps = {f'Layer {layer} Mask': res[layer]['mask'][sample_idx,0].detach().numpy() for layer in range(0, self.max_layer+1)}
        
        # Predicted Deatail Maps
        pmaps = {f'Layer {layer} Detail': res[layer]['mask_detail'][sample_idx,0].detach().numpy() for layer in range(0, self.max_layer+1)}
        
        vmaps = {f'Layer {layer} Detail': res[layer]['prev_mask'][sample_idx,0].detach().numpy() for layer in range(0, self.max_layer+1)}
        
        # media.show_images(dmaps, height=300)
        for layer in range(0, self.max_layer+1):
            print(f'Layer {layer} Feature Avg Value: ', res[layer]['feature'][sample_idx, channel_idx].mean().item())
            count = keep_ratios[layer-1] * (4**layer)
            topk = res[layer]['topk']
            info2 = f'--- KeepRatios {keep_ratios[layer-1]} KeepPatches {int(count)} Actual KeepCount {int(topk)} outof Total {int(4**layer)} Patches'
            print(info2)

        return fmaps, dmaps, pmaps, vmaps
    
    
#################
# Loss Function #
#################

def decide_sparsity_ratios(max_layer,
                           layer_sparse_ratio=0.8, 
                           initial_sparse_layer=2):
    return [min(1.0, layer_sparse_ratio ** (layer - initial_sparse_layer + 1)) for layer in range(max_layer+1)]

def initialize_weight_dict(max_layers, layer_weight_decay=0.0, reg_weight=1.0, binary_weight=1.0, sparsity_weight=1.0):
    """
    Initialize a weight_dict for different loss components with layer-wise decay.

    Args:
        max_layers (int): The maximum number of layers.
        layer_weight_decay (float): The weight decay ratio for each layer (default is 0.5).
        reg_weight (float): Weight for regression loss (default is 1.0).
        binary_weight (float): Weight for binary loss (default is 1.0).
        sparsity_weight (float): Weight for sparsity loss (default is 1.0).

    Returns:
        dict: The weight_dict with layer-wise decay for different loss components.
    """
    weight_dict = {}
    map_loss = {}
    
    # Assign weights for regression, binary, and sparsity losses
    weight_dict['loss_reg'] = reg_weight
    map_loss['regression'] = []
    map_loss['binary'] = []
    map_loss['sparsity'] = []
    
    for layer in range(0, max_layers + 1):
        layer_name = f'layer{layer}'
        
        map_loss['regression'].append(f'loss_reg_{layer}')
        # Latter Layer have a stronger regression KPIs
        weight_dict[f'loss_reg_{layer}'] = reg_weight * (layer_weight_decay ** (max_layers - layer))
        
        if layer == max_layers:
            continue
        map_loss['binary'].append(f'loss_bin_{layer}')
        map_loss['sparsity'].append(f'loss_sparsity_{layer}')
        
        # Latter Layer have a stronger Sparsity KPIs
        weight_dict[f'loss_bin_{layer}'] = binary_weight * (layer_weight_decay ** (layer - 1))
        weight_dict[f'loss_sparsity_{layer}'] = sparsity_weight * (layer_weight_decay ** (layer - 1))
    
    return weight_dict, map_loss

class SetCriterion(nn.Module):
    """
    Custom Loss Criteria
    """
    def __init__(self, 
                 max_layers,
                 losses=['regression','binary','sparsity'],
                 layer_weight_decay=0., 
                 reg_weight=1.0, 
                 binary_weight=1.0, 
                 sparsity_weight=1.0,
                 layer_sparse_ratio=1.0, 
                 initial_sparse_layer=2):
        """
        weight_dict: dict containing as key the names of the losses and as values their relative weight.
        losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.max_layers = max_layers
        self.layer_weight_decay = layer_weight_decay
        self.reg_weight = reg_weight
        self.binary_weight = binary_weight
        self.sparsity_weight = sparsity_weight
        self.layer_sparse_ratio = layer_sparse_ratio
        self.initial_sparse_layer = initial_sparse_layer
        
        self.weight_dict, self.map_loss = initialize_weight_dict(self.max_layers, self.layer_weight_decay, self.reg_weight, self.binary_weight, self.sparsity_weight)
        self.losses = losses
        self.sparsity_ratios = decide_sparsity_ratios(self.max_layers, self.layer_sparse_ratio, self.initial_sparse_layer)
        
    def update(self, layer_weight_decay=None, 
                 reg_weight=None, 
                 binary_weight=None, 
                 sparsity_weight=None,
                 layer_sparse_ratio=None, 
                 initial_sparse_layer=None):
        if layer_weight_decay is not None:
            self.layer_weight_decay = layer_weight_decay
        if reg_weight is not None:
            self.reg_weight = reg_weight
        if binary_weight is not None:
            self.binary_weight = binary_weight
        if sparsity_weight is not None:
            self.sparsity_weight = sparsity_weight
        if layer_sparse_ratio is not None:
            self.layer_sparse_ratio = layer_sparse_ratio
        if initial_sparse_layer is not None:
            self.initial_sparse_layer = initial_sparse_layer
        
        self.weight_dict, self.map_loss = initialize_weight_dict(self.max_layers, self.layer_weight_decay, self.reg_weight, self.binary_weight, self.sparsity_weight)
        self.sparsity_ratios = decide_sparsity_ratios(self.max_layers, self.layer_sparse_ratio, self.initial_sparse_layer)
        
        
    def loss_regression(self, outputs, targets, log=True):
        """
        Regression loss: L2 Image construction loss
        """
        assert 'ensemble_feature' in outputs        
        losses = {}
        
        for v in outputs:
            if not isinstance(v, int):
                continue
            src_feat = outputs[v]['feature']
            tgt_feat = F.interpolate(targets, size=src_feat.shape[-2:], mode='bilinear', align_corners=False)
            loss_reg = nn.L1Loss()(src_feat, tgt_feat)
            losses[f'loss_reg_{v}'] = loss_reg
            
        if log:
            losses[f'regression_error_at_{v}_layer'] = loss_reg
            
        return losses
    
    
    def loss_binary(self, outputs, log=True):
        """
        Binary Loss: Encourage 0s & 1s in Mask Maps
                     Scaling should happen in feature/residual values
        """
        losses = {}
        for v in outputs:
            if not isinstance(v, int):
                continue
            
            loss_bin = torch.sin(outputs[v]['mask_detail'] * np.pi).mean()
            losses[f'loss_bin_{v}'] = loss_bin
            
            if log:
                losses[f'binary_error_at_{v}_layer'] = loss_bin
                
        return losses
                
    def loss_sparsity(self, outputs, log=True):
        """
        Sparsity Loss: Control the number of zeros / nonzero values within the Mask
        """
        losses = {}
        for v in outputs:
            if not isinstance(v, int):
                continue
            loss_sparse = torch.abs(outputs[v]['mask_detail'].sum()/(4**v) - (1-self.sparsity_ratios[v]))
            losses[f'loss_sparsity_{v}'] = loss_sparse
            
            if log:
                losses[f'sparsity_error_at_{v}_layer'] = loss_sparse
                
        return losses
    
    # Layer-wise incremental Mask value inheritance loss
    # Latter layer Mask should be small in areas where earlier layer predict a small mask
    
    def get_loss(self, loss, outputs, targets, log=False):
        loss_map = {
            'regression': self.loss_regression,
            'binary': self.loss_binary,
            'sparsity': self.loss_sparsity
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        # Supervised Loss
        if loss in ['regression']:
            return loss_map[loss](outputs, targets, log)
        # Self-Supervised Loss
        if loss in ['binary', 'sparsity']:
            return loss_map[loss](outputs, log)
    
    def forward(self, outputs, targets, log=False):
        """ 
        Performs the loss computation
        """
        losses = {}
        for loss_desc in self.losses:
            loss = self.get_loss(loss_desc, outputs, targets, log)            
            # print(f'Loss description: {loss_desc}, Computed Loss Key Values: {loss.keys()}')
            for loss_name in self.map_loss[loss_desc]:
                assert loss_name in loss, f'Loss desc: {loss_desc} do not contain key {loss_name}'
                assert loss_name in self.weight_dict, f'Weight dict do not contain key {loss_name}'
                # Get loss value & weight value
                loss_val, loss_weight = loss.get(loss_name), self.weight_dict.get(loss_name, 1.0)
                weighted_loss_val = loss_val * loss_weight
                losses.update({loss_name:loss_val})
                losses.update({'weighted_'+loss_name:weighted_loss_val})
            
        # Calculate the total loss
        total_loss = sum(value for key, value in losses.items() if key.startswith('weighted'))
        
        # Initialize total_loss as a tensor with value 1.0
        total_prod = torch.tensor(1.0, requires_grad=True, dtype=torch.float32)
        total_prod = total_prod.to(total_loss.dtype).to(total_loss.device)
        # Calculate the total loss as a cumulative product
        for key, value in losses.items():
            if key.startswith('weighted'):
                total_prod *= value
        
        # Include both individual losses and the total loss in the output
        losses['total_loss'] = total_loss
        losses['total_prod'] = total_prod
        return losses
    
    
def get_trainlog(loss_dict, criterion, weighted=True, losses=['regression', 'binary', 'sparsity']):
    tlog = ''
    for k in losses:
        for name in criterion.map_loss[k]:
            if not weighted:
                val = np.round(loss_dict[name].item(),2)
                tlog += f'{name} loss: {val} '
            else:
                val = np.round(loss_dict[f'weighted_{name}'].item(),2)
                tlog += f'Weighted {name} loss: {val} '
    return tlog


# BackUp
# Modified Global Response Normalization Layer
class GRN(nn.Module):
    """ 
    Modified GRN (Global Response Normalization) layer
    For Spatial Feature Diversification instead of Channel Feature Diversification
    """
    def __init__(self, h, w):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, h, w))
        self.beta = nn.Parameter(torch.zeros(1, 1, h, w))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=(-1,-2), keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
# Structure Idea
# 1. Residual Connection (Concatenate on Channel Dimension)
# -- 1.1 Cost efficient approch -- 256->3->256 (256*3*2 << 256*256)
class ResPatchConv(nn.Module):
    def __init__(self, h_feat, w_feat, dim_patch_feature=256, patch_size=2):
        super(ResPatchConv, self).__init__()
        
        # Common initial layers shared by both branches
        self.common_layers = nn.Sequential(
            # nn.Conv2d(3 + dim_patch_feature, dim_patch_feature, kernel_size=patch_size, stride=patch_size)
            nn.Conv2d(3 + dim_patch_feature, 3, 1),
            nn.Conv2d(3, dim_patch_feature, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            GRN(h_feat, w_feat)
            # nn.BatchNorm2d(dim_patch_feature)
        )

        # Feature prediction branch
        self.feature_branch = nn.Identity()

        # Detail prediction branch
        # ISSUE: Rescaled to the same size, a BIG Image patch's detail level should be higher than 
        # detail level of an actual small image patch's detail
        self.detail_branch = nn.Sequential(
            nn.Conv2d(dim_patch_feature, 1, kernel_size=1),
            nn.Sigmoid()  # Scales output to the [0, 1] range
        )

    # Upsample, Concatenate, Propagate
    def forward(self, x, feat):
        # Ensure x has a batch dimension of size 1 if it's a single image
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add a batch dimension of size 1
            
        # Upsample prev_feature to match the spatial dimensions of rescaled_patch
        upsampled_feat = F.interpolate(feat, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        # Concatenate
        x = torch.cat((x, upsampled_feat), dim=1)
        
        # Forward pass through common initial layers
        common_output = self.common_layers(x)

        # Feature prediction branch
        feature_output = self.feature_branch(common_output)

        # Use std of feature prediction as detail threshold
        # Detail prediction branch
        detail_output = self.detail_branch(common_output)

        return feature_output, detail_output
    
    
def normalize_mask_detail(mask_detail, mode, pconv_patch_scale):
    
    if mode == 'softmax':
        shape_before = mask_detail.shape        
        mask_detail_flattened = mask_detail.view(shape_before[0], shape_before[1], -1)
        # Apply softmax along the combined spatial dimensions
        mask_detail = F.softmax(mask_detail_flattened, dim=-1)
        
        # Reshape back to original dimensions
        mask_detail = mask_detail.view(shape_before)

    elif mode == 'minmax':
        # Min-Max normalization
        min_val = torch.min(mask_detail)
        max_val = torch.max(mask_detail)
        mask_detail = (mask_detail - min_val) / (max_val - min_val + 1e-8)
        
    elif mode == 'sigmoid':
        # Sigmoid normalization
        mask_detail = torch.sigmoid(mask_detail)
        
    else:
        raise ValueError("Invalid mode. Choose between 'softmax', 'minmax', and 'sigmoid'.")
    
    # Take the mean across the channel dimension
    mask_detail = mask_detail.mean(dim=1, keepdim=True)
    
    return mask_detail
    

    
    
# Detail is computed by the Mean Absolute Deviation of the feature prediction
# Averaged over each channel, MAD among each patch
# The layer-wise detail mask should have a inheritance relationship
class EyeMixerSim(nn.Module):
    def __init__(self, args):
        super(EyeMixerSim, self).__init__()
        self.pconv_feature_dim = args.pconv_feature_dim
        self.pconv_patch_scale = args.pconv_patch_scale
        self.subimg_size = args.subimg_size
        self.detail_threshold = args.detail_threshold
        # Maximum layer of explorable QuadTreeNode 
        self.max_layer = args.max_layer
        
        # Image Preprocessor : Augmentation Free Version
        self.preprocess =  transforms.Compose([
                                crop_to_multiple(args.subimg_size),
                                transforms.Resize(args.preprocess_img_size),
                                transforms.ToTensor()
                            ])
        # PatchExploration Network: Residual Inference for FineLayer, Direct Inferece for CoarseLayer
        self.pconvs = nn.ModuleList()  # Create a ModuleList to hold the PatchConv modules
                
        # Strategy 1: Completely Independent Pconv for each layer -- performance ok, but lacks consistency
        for layer in range(0, self.max_layer+1):
            if layer == 0:
                self.pconvs.append(PatchConv(dim_patch_feature=args.pconv_feature_dim, 
                                               patch_size=args.pconv_patch_size))
            else:
                self.pconvs.append(ResPatchConv(
                    h_feat=int(args.pconv_patch_scale[0] * (2**layer)),
                    w_feat=int(args.pconv_patch_scale[1] * (2**layer)),
                    dim_patch_feature=args.pconv_feature_dim,
                    patch_size=args.pconv_patch_size))
            
        # Mixer: Depthwise & Pointwise Convolution
        self.mixer = Mixer(dim=args.pconv_feature_dim,
                           depth=args.mixer_depth,
                           kernel_size=args.mixer_kernel_size,
                           n_classes=args.num_classes)
        
        # Mask Ratios Scheduler
        self.mask_ratios = decide_mask_ratios(self.max_layer, layer_ratio=args.pconv_layer_ratio)
        modes = ['topk', 'zeropad']
        self.mode = modes[args.pconv_mode]
        
    # mode: ['zeropad', 'skipconv']
    def patch_conv(self, x, mode='zeropad'):
        B,N,H,W = x.shape
        subimg_h, subimg_w = self.subimg_size
        patchscale_h, patchscale_w = self.pconv_patch_scale
        mask = torch.ones((B,1,1,1)).float().to(x.device)
        ensemble_feature = torch.zeros((B, self.pconv_feature_dim, patchscale_h, patchscale_w)).to(x.device)
        ratio = 1.
        res = {}
        for layer in range(0, self.max_layer+1):
            # (H_layer, W_layer) = (2^layer * SubImg_h, 2^layer * SubImg_w)
            subimgs_size = get_rescale_shape(layer, self.subimg_size)
            
            # Rescaled Img at Current Layer
            # SubImgs: (B, C, H_layer, W_layer)
            subimgs = F.interpolate(x, subimgs_size, mode='bilinear', align_corners=True)
            
            # SkipConv is equivalent to multiplication of mask before and after the Convo
            # Here we only have multiplication before the Convolution
            if mode=='zeropad':
                
                # Zero-Pad before Patch-Convolution
                # --- 1. Hard Thresholding Selection
                subimgs *= (F.interpolate(mask, subimgs_size, mode='nearest') > self.detail_threshold)     
                # --- 2. Bernouli Random Selection
                # bmask = torch.bernoulli(mask)
                # subimgs *= F.interpolate(bmask, subimgs_size, mode='nearest').bool()
                # --- 3. TopK Hard Selection (?)
                
                # Propagate & Prediction
                # Each SubImg is converted to (PatchScale_h, Patch_Scale_w) features
                # Each Feature is (Dim_feature,) in shape
                
                # feature: (B, Dim_feature, PatchScale_h, PatchScale_w)
                # detail:  (B,           1, PatchScale_h, PatchScale_w) 
                # TBD: Getting rid of detail values 
                if layer==0:
                    feature, detail = self.pconvs[0](subimgs)
                else:
                    feature, detail = self.pconvs[layer](subimgs, ensemble_feature)

                # ZeroPad after Patch Convolution
                # --- 1. Hard Threshold Selection
                feature *= (F.interpolate(mask, feature.shape[-2:], mode='nearest') > self.detail_threshold)
                # --- 2. Bernouli Random Selection
                # feature *= (F.interpolate(bmask, feature.shape[-2:], mode='nearest').bool())
                
                # Default value for topk selection
                topk = int(4**layer)
                

            # Mask Detail:
            # Mean-Absolute-Deviation (In-Subimg-Features) | Detail Level for each SubImg | Better Numerical Stability 
            # Each SubImg is converted to PatchScale_h x PatchScale_w number of features through Patch Convolution
            res_feature = rearrange(feature, 'b c (n h) (m w) -> b c n m (h w)', h=patchscale_h, w=patchscale_w)

            mad = torch.abs(res_feature - res_feature.mean(axis=-1, keepdim=True)).mean(axis=-1)
            
            # Mask_Detail: (B, 1, 2^l, 2^l) | Same shape as the current-level Mask
            mask_detail = mad.mean(axis=1, keepdim=True) # Avg Detail Level across all channels
            
            # Then, to reduce the effect of feature scale on the MAD, Normalize it
            # Norm mask mode: ['softmax', 'minmax', 'sigmoid']
            norm_mode = 'sigmoid'
            mask_detail = normalize_mask_detail(mask_detail, norm_mode, self.pconv_patch_scale)
            
            # Useless Part -- Disposable once debug complete
            pre_mask = F.interpolate(mask, mask_detail.shape[-2:], mode='nearest').clone()
            
            # Update Mask by Cumulative Product
            mask = F.interpolate(mask, mask_detail.shape[-2:], mode='nearest') * mask_detail
            # Residual can be scaled by Current Layer Mask instead
            # This way mask value doesn't exponentially drops across layers
            # mask_rand = F.interpolate(bmask, mask_detail.shape[-2:], mode='nearest') * mask_detail
            
            # Feature Scaling w. Detail Mask enhances Performance & Encourages Sparsity
            residual = F.interpolate(mask, feature.shape[-2:], mode='nearest') * feature
            # No exponential cumulative product & Inherit Random Binary Mask
            # residual = F.interpolate(mask_rand, feature.shape[-2:], mode='nearest') * feature
            
            # Ensemble Addition of Feature
            ensemble_feature = ensemble_feature_upsample(ensemble_feature, residual, layer, self.pconv_patch_scale)

            # Current Layer Binary Mask Decides which part of Current Layer Prediction contribute to Ensemble Features
            # binary_mask = mask_rand
            
            # Did not Update Mask here -- my bad
            # mask = mask_rand
            binary_mask = (mask > self.detail_threshold)
            
            
            res[layer] = {'feature':ensemble_feature, 
                          # 'mask': mask_rand,
                          'mask': mask,
                          'topk': topk, 
                          'mask_detail': mask_detail,
                          'prev_mask': pre_mask, 
                          'residual': residual,
                          'binary_mask': binary_mask}
            
        res['ensemble_feature'] = ensemble_feature
        return res   
    
    def forward(self, x):
        res = self.patch_conv(x)
        pred = self.mixer(res['ensemble_feature'])
        res['pred'] = pred
        return res
    
    def visualize_res(self, res, sample_idx, channel_idx):
        keep_ratios = np.cumprod([self.mask_ratios[l] for l in range(0, self.max_layer+1)])
        if not isinstance(channel_idx, list):
            fmaps = {f'Layer {layer} Feature': res[layer]['feature'][sample_idx,channel_idx].detach().clip(0.,1.).numpy() for layer in range(1,self.max_layer+1)}
            fmaps['Ensemble Feature'] = res['ensemble_feature'][sample_idx,channel_idx].detach().clip(0.,1.).numpy()
        else:
            fmaps = {f'Layer {layer} Feature': res[layer]['feature'][sample_idx,channel_idx].detach().permute(1,2,0).clip(0.,1.).numpy() for layer in range(1,self.max_layer+1)}
            fmaps['Ensemble Feature'] = res['ensemble_feature'][sample_idx,channel_idx].detach().permute(1,2,0).clip(0.,1.).numpy()

        media.show_images(fmaps, height=300)
        # Mask Maps
        dmaps = {f'Layer {layer} Mask': res[layer]['mask'][sample_idx,0].detach().numpy() for layer in range(0, self.max_layer+1)}
        
        # Predicted Deatail Maps
        pmaps = {f'Layer {layer} Detail': res[layer]['mask_detail'][sample_idx,0].detach().numpy() for layer in range(0, self.max_layer+1)}
        
        vmaps = {f'PrevLayer {layer} Mask': res[layer]['prev_mask'][sample_idx,0].detach().numpy() for layer in range(0, self.max_layer+1)}
        
        bmaps = {f'Layer {layer} Binary Mask': res[layer]['binary_mask'][sample_idx,0].detach().numpy() for layer in range(0, self.max_layer+1)}
        
        # media.show_images(dmaps, height=300)
        for layer in range(0, self.max_layer+1):
            print(f'Layer {layer} Feature Avg Value: ', res[layer]['feature'][sample_idx, channel_idx].mean().item())
            count = keep_ratios[layer-1] * (4**layer)
            topk = res[layer]['topk']
            info2 = f'--- KeepRatios {keep_ratios[layer-1]} KeepPatches {int(count)} Actual KeepCount {int(topk)} outof Total {int(4**layer)} Patches'
            print(info2)

        return fmaps, dmaps, pmaps, vmaps, bmaps
    
    
