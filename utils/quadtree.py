# Image processing
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm as tqdm
from tqdm import trange
import glob
import torch
from sortedcontainers import SortedListWithKey
# Progress bar
from tqdm import trange
# Binary encoding and compression
from io import BytesIO
import lzma

from sklearn.decomposition import IncrementalPCA
# 


# This Object assumes x-axis pointing to the right, y-axis point to the top, which is not consistent with standard uv image plane setup
# We should respect the fact that We are dealing with Image, not something else here ! 
class QuadNode:
    def __init__(self, position: tuple, size: tuple):
        
        # LeftTop position (u, v) for current QuadNode on Image
        self.position = position
        # Height, Width size of Current QuadNode
        self.size = size
        # print('Quad Initialized Pos: ', position, ' Size: ', size)
        
        self.color = None

        self.is_subdivided = False
        self.bottom_left_node = None
        self.bottom_right_node = None
        self.top_left_node = None
        self.top_right_node = None
    
    def _create_child_node(self, position: tuple, size: tuple):
        return QuadNode(position, size)
    
    # Divide on current QuadNode
    def sub_divide(self):
        if self.is_subdivided:
            return []
        
        height, width = self.size
        u, v = self.position
        
        # It only makes sense to sub-divide when there are more than 2 pixels of length
        if width <= 1 or height <= 1:
            return []
        
        self.is_subdivided = True
        split_width, split_height = width // 2, height // 2
        # Child Node parameters: leftbottom position + (width, height) in size
        child_size = tuple([split_height, split_width])
        topleft = tuple([u, v])
        topright = tuple([u + split_width, v])
        bottomleft = tuple([u, v + split_height])
        bottomright = tuple([u + split_width, v + split_height])
        # Split into subnodes
        self.bottom_left_node = self._create_child_node(bottomleft, child_size)
        self.bottom_right_node = self._create_child_node(bottomright, child_size)
        self.top_left_node = self._create_child_node(topleft, child_size)
        self.top_right_node = self._create_child_node(topright, child_size)
        
        return self.bottom_left_node, self.bottom_right_node, self.top_left_node, self.top_right_node
    
    # Draw & Replace Node color on ImageData
    def draw_self(self, image_data: np.array, pad_color: bool = True):
        start_u, start_v = self.position
        height, width = self.size
        end_u = start_u + width
        end_v = start_v + height
        if self.color is None:
            # print('Use Avg Color to represent QuadNode by Default -- ')
            self.color = image_data[start_v: end_v + 1, start_u: end_u + 1].mean(axis=(0,1))
        # print('Drawing v from ', start_v, ' to ', end_v, ' || u from ', start_u, ' to ', end_u)
        if pad_color:
            image_data[start_v: end_v + 1, start_u: end_u + 1] =  self.color
        # return image_data
        
    def draw(self, image_data: np.array, pad_color: bool = True):
        if self.is_subdivided: # Subdivision exists, traverse down the quadTree
            self.bottom_left_node.draw(image_data, pad_color)
            self.bottom_right_node.draw(image_data, pad_color)
            self.top_left_node.draw(image_data, pad_color)
            self.top_right_node.draw(image_data, pad_color)
        else:
            self.draw_self(image_data, pad_color) # Terminate at leaf operation
            
    def draw_separate_self(self, image_data: np.array, pad_color: bool = True):
        self.draw_self(image_data, pad_color)
        # Draw Separation Lines
        start_u, start_v = self.position
        height, width = self.size
        end_u = start_u + width
        end_v = start_v + height
        # Draw Boundary
        boundary_width = 1
        image_data[start_v: start_v + boundary_width, start_u: end_u + 1] = np.array([255.,255.,255.])
        image_data[end_v-boundary_width: end_v + 1, start_u: end_u + 1] = np.array([255.,255.,255.])
        image_data[start_v: end_v + 1, start_u: start_u + boundary_width] = np.array([255.,255.,255.])
        image_data[start_v: end_v + 1, end_u-boundary_width: end_u + 1] = np.array([255.,255.,255.])
    
    def draw_separate(self, image_data:np.array, pad_color: bool = True):
        if self.is_subdivided:
            self.bottom_left_node.draw_separate(image_data, pad_color)
            self.bottom_right_node.draw_separate(image_data, pad_color)
            self.top_left_node.draw_separate(image_data, pad_color)
            self.top_right_node.draw_separate(image_data, pad_color)
        else:
            self.draw_separate_self(image_data, pad_color)
            
class CompressNode(QuadNode):
    # Each Node represent a compressed Subsection of Image Patch
    def __init__(self, position, img):
        height, width, _ = img.shape 
        super().__init__(position, (height, width))
        
        self.img = img
        # Detail Info: Scaled Standard Deviation of Image Pixel Values
        self.detail = np.sum(np.std(self.img, axis=(0,1))) * self.img.size
        self.color = np.mean(self.img, axis=(0,1)).astype(np.int16)
    # Updates on the Child Node creation Functional, then the subdivide will be updated automatically
    def _create_child_node(self, position, size):
        height, width = size
        u, v = position
        
        start_u = u - self.position[0]
        end_u = start_u + width
        start_v = v - self.position[1]
        end_v = start_v + height
        # We note that the Image here is a sub-Slice of the original image
        # Which only corresponds to the image patch on the current Node
        child_img = self.img[start_v:end_v+1, start_u:end_u+1] 
        return CompressNode(position, child_img)
    # Overrode version w memory clearance
    def sub_divide(self):
        # call on original sub_divide functional with overrode create_child_node functionals
        childs = super().sub_divide()
        self.img = None # Clear up memory
        return childs
    # Overrode draw when we need to use the leaf-node image patch
    def draw_self(self, image_data:np.array, pad_color: bool = True):
        if pad_color == True:
            super().draw_self(image_data, pad_color)
            return
        else:
            # print('Override draw self version')
            start_u, start_v = self.position
            # h1, w1 = self.size
            h2, w2 = self.img.shape[:2]
            h1, w1 = image_data[start_v:start_v+h2, start_u:start_u+w2].shape[:2]
            h = min(h1, h2)
            w = min(w1, w2)
            # print('min val: ', h, w, ' shape 1: ', image_data[start_v:start_v+h, start_u:start_u+w].shape, ' shape 2: ', self.img[:h+1, :w+1].shape)
            image_data[start_v:start_v+h, start_u:start_u+w] = self.img[:h, :w]
                
            
class ImageCompressor:
    def __init__(self, image_data: np.array):
        self.areas = SortedListWithKey(key=lambda node: node.detail)
        self._image_shape = image_data.shape
        self.height, self.width = self._image_shape[:2]
        self.root_node = CompressNode((0,0), image_data)
        self.areas.add(self.root_node)
        
    def add_detail(self, max_iterations: int = 1, detail_error_threshold: float = 1000.):
        n_iter = 0        
        for n_explore in trange(max_iterations, leave=False):
            node_with_most_detail = self.areas.pop()
            # print('Max detail on node: ', node_with_most_detail.detail)
            if node_with_most_detail.detail < detail_error_threshold:
                break
            for child_node in node_with_most_detail.sub_divide():
                self.areas.add(child_node)
        
    def draw(self, pad_color: bool = True):
        empty_img = np.zeros(self._image_shape)
        self.root_node.draw(empty_img, pad_color)
        return empty_img.astype(np.int32)
    
    def draw_separate(self, pad_color: bool = True):
        empty_img = np.zeros(self._image_shape)
        self.root_node.draw_separate(empty_img, pad_color)
        return empty_img.astype(np.int32)
    
    def animate(self, iter_skip: int=50, save_dir='/mnt/d/Implementation/VD/Compression/Orig/quadcomp'):
        # 
        for i in range(200):
            self.add_detail(iter_skip)
            img_ckpt = self.draw()
            n_explore = int(i*iter_skip)
            Image.fromarray(img_ckpt.astype(np.uint8)).save(f"{save_dir}/frame_{n_explore}.jpg")
        

            
def convert_int64_to_uint8(img_o):
    # Clip the values within the valid range (0 to 255)
    clipped_image_data = np.clip(img_o, 0, 255)
    # Convert it to uint8
    uint8_image_data = clipped_image_data.astype(np.uint8)
    return uint8_image_data


def pca_recon(k, img_data):
    ipca = IncrementalPCA(n_components=k)
    img_recon_r = ipca.inverse_transform(ipca.fit_transform(img_data[...,0]))
    img_recon_g = ipca.inverse_transform(ipca.fit_transform(img_data[...,1]))
    img_recon_b = ipca.inverse_transform(ipca.fit_transform(img_data[...,2]))
    img_recon = np.stack([img_recon_r, img_recon_g, img_recon_b], axis=-1).astype(int)
    return img_recon


# PreProcess for Patch-Based Hierachy propagation
import torchvision.transforms as transforms
import torch.nn.functional as F


def crop_to_multiple(patch_size):
    def transform(image):
        # Get the dimensions of the image
        image_height, image_width = image.height, image.width
        patch_height, patch_width = patch_size
        # Calculate the new dimensions that are multiples of the patch size
        uni_scale = min(image_width//patch_width, image_height//patch_height)
        new_width = patch_width * uni_scale
        new_height = patch_height * uni_scale

        # Calculate the crop region coordinates
        left_crop = (image_width - new_width) // 2
        top_crop = (image_height - new_height) // 2

        # Crop the image
        cropped_image = image.crop((left_crop, top_crop, left_crop + new_width, top_crop + new_height))

        return cropped_image

    return transform


# 
def resize_image_tensor(image_tensor, target_size):
    """
    Resize a tensor image to a target size using torch.nn.functional.interpolate.

    Args:
        image_tensor (torch.Tensor): Input image tensor (C x H x W).
        target_size (tuple or int): Target size as a tuple (H, W) or a single integer (H, where H = W).

    Returns:
        torch.Tensor: Resized image tensor.
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    # Reshape image tensor to (1, C, H, W) if it's not already batched
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    resized_image = F.interpolate(image_tensor, size=target_size, mode='bilinear', align_corners=False)

    # Remove the batch dimension if the input tensor was not batched
    if resized_image.shape[0] == 1:
        resized_image = resized_image.squeeze(0)

    return resized_image



class QuadNode_t:
    def __init__(self, position: tuple, size: tuple):
        self.position = position
        self.size = size
        self.color = None
        self.is_subdivided = False
        self.bottom_left_node = None
        self.bottom_right_node = None
        self.top_left_node = None
        self.top_right_node = None

    def _create_child_node(self, position: tuple, size: tuple):
        return QuadNode_t(position, size)

    def sub_divide(self):
        if self.is_subdivided:
            return []

        height, width = self.size
        u, v = self.position

        if width <= 1 or height <= 1:
            return []

        self.is_subdivided = True
        split_width, split_height = width // 2, height // 2
        child_size = (split_height, split_width)
        topleft = (u, v)
        topright = (u + split_width, v)
        bottomleft = (u, v + split_height)
        bottomright = (u + split_width, v + split_height)
        self.bottom_left_node = self._create_child_node(bottomleft, child_size)
        self.bottom_right_node = self._create_child_node(bottomright, child_size)
        self.top_left_node = self._create_child_node(topleft, child_size)
        self.top_right_node = self._create_child_node(topright, child_size)

        return self.bottom_left_node, self.bottom_right_node, self.top_left_node, self.top_right_node

    def draw_self(self, image_data: torch.Tensor, pad_color: bool = True):
        start_u, start_v = self.position
        height, width = self.size
        end_u = start_u + width
        end_v = start_v + height

        if self.color is None:
            self.color = image_data[:, start_v:end_v + 1, start_u:end_u + 1].mean(dim=(0, 1))

        if pad_color:
            image_data[:, start_v:end_v + 1, start_u:end_u + 1] = self.color

    def draw(self, image_data: torch.Tensor, pad_color: bool = True):
        if self.is_subdivided:
            self.bottom_left_node.draw(image_data, pad_color)
            self.bottom_right_node.draw(image_data, pad_color)
            self.top_left_node.draw(image_data, pad_color)
            self.top_right_node.draw(image_data, pad_color)
        else:
            self.draw_self(image_data, pad_color)

    def draw_separate_self(self, image_data: torch.Tensor, pad_color: bool = True):
        self.draw_self(image_data, pad_color)
        start_u, start_v = self.position
        height, width = self.size
        end_u = start_u + width
        end_v = start_v + height

        boundary_width = 1
        image_data[:, start_v:start_v + boundary_width, start_u:end_u + 1] = torch.tensor([1., 1., 1.]).reshape(3,1,1)
        image_data[:, end_v - boundary_width:end_v + 1, start_u:end_u + 1] = torch.tensor([1., 1., 1.]).reshape(3,1,1)
        image_data[:, start_v:end_v + 1, start_u:start_u + boundary_width] = torch.tensor([1., 1., 1.]).reshape(3,1,1)
        image_data[:, start_v:end_v + 1, end_u - boundary_width:end_u + 1] = torch.tensor([1., 1., 1.]).reshape(3,1,1)

    def draw_separate(self, image_data: torch.Tensor, pad_color: bool = True):
        if self.is_subdivided:
            self.bottom_left_node.draw_separate(image_data, pad_color)
            self.bottom_right_node.draw_separate(image_data, pad_color)
            self.top_left_node.draw_separate(image_data, pad_color)
            self.top_right_node.draw_separate(image_data, pad_color)
        else:
            self.draw_separate_self(image_data, pad_color)
            
def count_nodes_at_each_layer(tree_structure):
    layer_count = {}

    def traverse_and_count_layers(tree_node):
        layer = tree_node['layer']
        if layer in layer_count:
            layer_count[layer] += 1
        else:
            layer_count[layer] = 1

        if 'children' in tree_node:
            for child in tree_node['children']:
                traverse_and_count_layers(child)

    traverse_and_count_layers(tree_structure)
    return layer_count

class ConvCompressNode(QuadNode_t):
    def __init__(self, position, img: torch.Tensor, patch_size: torch.Tensor, depth: int = 0):
        # Img size (C, H, W)
        _, height, width = img.shape
        super().__init__(position, (height, width))
        # Img 2 Patch
        self.patch_size = patch_size
        # print('Img size: ', img.shape, ' Patch Size: ', self.patch_size)
        self.patch = resize_image_tensor(img, self.patch_size)
        self.img = img
        self.depth = depth
        self.target_depth = 5
        self.color = img.mean(dim=(1, 2)).int()
    
        self.single_feature = None
        self.feature = None
        self.detail = None
        
    
    def _create_child_node(self, position, size):
        height, width = size
        u, v = position
        start_u = u - self.position[0]
        end_u = start_u + width
        start_v = v - self.position[1]
        end_v = start_v + height
        child_img = self.img[:, start_v:end_v + 1, start_u:end_u + 1]
        
        patch_size = self.patch_size
        depth = self.depth + 1
        return ConvCompressNode(position, child_img, patch_size, depth)
    
    def sub_divide(self):
        if self.is_subdivided:
            return []
        # Constrained to PatchSize exploration here -- or not 
        height, width = self.size
        u, v = self.position
        patch_height, patch_width = self.patch_size
        if width < 2*patch_width or height < 2*patch_height:
            return []
        childs = super().sub_divide()
        self.img = None # Clear up memory, while keeping features and details level
        return childs
    
    # QuadTree Traverse and Explore with PatchConv network prediction values
    def explore_with_nn(self, preprocessor, patch_model):
        # Preprocessing on Image Patch
        prep = preprocessor(self.patch)
        # Prediction with Patch-wise model
        patch_feature, patch_detail = patch_model(prep)
        
        # Feature should be assembled from all the child node's inference result
        self.single_feature = patch_feature
        self.detail = patch_detail
        
        # Treat the detail value as a Binary prediction
        if (patch_detail >= 0.5) and (self.depth <= self.target_depth):
            # Subdivide & Explore child nodes
            for child_node in self.sub_divide():
                child_node.explore_with_nn(preprocessor, patch_model)
                
    # Fixed Number of Layer, then upsample & addition for root node feature values
    def ensemble_feature(self):
        # raise NotImplementedError
        assert self.single_feature is not None, '-- Run explore with nn first !'
        if self.feature is not None:
            return
        if not self.is_subdivided:
            self.feature = self.single_feature
        else:
            # When current node is explored, it will have all 4 childs for sure
            # Dynamic collection of feature
            self.bottom_left_node.ensemble_feature()
            self.bottom_right_node.ensemble_feature()
            self.top_left_node.ensemble_feature()
            self.top_right_node.ensemble_feature()
            
            # Upsample and Add
            child_scale_factor = 1
            topleft_feat = rescale_feature(self.top_left_node.feature, child_scale_factor)
            topright_feat = rescale_feature(self.top_right_node.feature, child_scale_factor)
            bottomleft_feat = rescale_feature(self.bottom_left_node.feature, child_scale_factor)
            bottomright_feat = rescale_feature(self.bottom_right_node.feature, child_scale_factor)
            
            
            target_depth = self.depth + 1
            scale_factor = target_depth - self.depth
            
            # Each Time we combine the child feat with the parent feat
            # Strategy : upsample on parent feat, add child feat onto specific location
            parent_feat = rescale_feature(self.single_feature, scale_factor)
            
            self.feature = ensemble_add(parent_feat = parent_feat,
                         topleft_feat = self.top_left_node.feature,
                         topright_feat = self.top_right_node.feature,
                         bottomleft_feat = self.bottom_left_node.feature,
                         bottomright_feat = self.bottom_right_node.feature)
            # At the end of this function, features for different leaf node is of different size
            
            
              
    # Traverse for information collection: from current node, count info for sub-tree structure
    def traverse_and_count(self, depth=0):
        tree = self.to_tree_structure(node_name='root', layer=0)
        node_count_per_layer = count_nodes_at_each_layer(tree)
        num_layer = len(node_count_per_layer)
        num_node = sum(node_count_per_layer.values())
        
        repr_tree = f'In total, QuadTree has {num_layer} layers and {num_node} nodes.'
        print(repr_tree)
        for layer in node_count_per_layer:
            print(f'- On layer {layer}, {node_count_per_layer[layer]} nodes are explored')
        return node_count_per_layer
    
    def draw_self(self, image_data: torch.Tensor, pad_color: bool = True):
        if pad_color:
            super().draw_self(image_data, pad_color)
        else:
            start_u, start_v = self.position
            _, h2, w2 = self.img.shape
            _, h1, w1 = image_data[:, start_v:start_v + h2, start_u:start_u + w2].shape
            h = min(h1, h2)
            w = min(w1, w2)
            image_data[:, start_v:start_v + h, start_u:start_u + w] = self.img[:, :h, :w]            
            
    # Might be a crucial Step towards parallel implementation
    def to_tree_structure(self, node_name, layer=0):
        tree = {'name': node_name, 'color': 'blue', 'layer': layer}

        if self.is_subdivided:
            children = [self.bottom_left_node, self.bottom_right_node, self.top_left_node, self.top_right_node]
            child_names = [f"{node_name}_{i}" for i in range(4)]
            tree['children'] = [child.to_tree_structure(child_name, layer + 1) for child, child_name in zip(children, child_names) if child is not None]

        return tree
    
    # Get layer-wise node feature & position 
    def traverse_for_node_feature(self, node_feature = {}):
        if self.depth not in node_feature:
            node_feature[self.depth] = []
        # Add node information into the dictionary
        node_feature[self.depth].append({'single_feature': self.single_feature, 
                                      'feature': self.feature,
                                      'depth': self.depth,
                                      'position': self.position,
                                      'size': self.size,
                                      'node': self})

        if self.is_subdivided:
            children = [self.bottom_left_node, self.bottom_right_node, self.top_left_node, self.top_right_node]
            for child in children:
                child.traverse_for_node_feature(node_feature)            
