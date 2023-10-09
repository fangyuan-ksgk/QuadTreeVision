from model_idea import *

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
            # nn.Sigmoid()  # Scales output to the [0, 1] range
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
    # print('Input MaskDetail ToBeNormalized: ', mask_detail)
    if mode == 'softmax':
        shape_before = mask_detail.shape        
        mask_detail_flattened = mask_detail.view(shape_before[0], shape_before[1], -1)
        # Apply softmax along the combined spatial dimensions
        mask_detail = F.softmax(mask_detail_flattened, dim=-1)
        
        # Reshape back to original dimensions
        mask_detail = mask_detail.view(shape_before)

    elif mode == 'minmax':
        B,_,_,_ = mask_detail.shape
        # Min-Max normalization per image
        min_val = torch.min(torch.min(torch.min(mask_detail, dim=3, keepdim=True).values, dim=2, keepdim=True).values, dim=1, keepdim=True).values
        max_val = torch.max(torch.max(torch.max(mask_detail, dim=3, keepdim=True).values, dim=2, keepdim=True).values, dim=1, keepdim=True).values
        if not torch.eq(min_val, max_val):
            min_val = min_val.view(B,1,1,1)
            max_val = max_val.view(B,1,1,1)        
            mask_detail = (mask_detail - min_val / 5) / (max_val - min_val / 5 + 1e-8)
            # print('Normalized Mask (MinMax) shape: ', mask_detail.shape)
            # print('Normalized Mask (MinMax) value: ', mask_detail)
            

    elif mode == 'sigmoid':
        # Sigmoid normalization
        mask_detail = torch.sigmoid(mask_detail)
        # print('Normalized Mask (Sigmoid) shape: ', mask_detail.shape)
        # print('Normalized Mask (Sigmoid) value: ', mask_detail)
    else:
        raise ValueError("Invalid mode. Choose between 'softmax', 'minmax', and 'sigmoid'.")
    
    # Take the maximum across the channel dimension
    # mask_detail, _ = mask_detail.max(dim=1, keepdim=True)
    mask_detail = mask_detail.mean(dim=1, keepdim=True)
    
    return mask_detail



# (B, 1, H, W) mask_detail shape
def regulate_mask(mask_detail, ratio):
    # print('RegulateMask Information: ')
    # print('Mask detail ', mask_detail)
    h,w = mask_detail.shape[-2:]
    require = int(h*w*ratio)
    regular_mask = (mask_detail * (require / mask_detail.sum(axis=(-2,-1), keepdim=True))).clip(0.000001,1-0.000001)
    # print('regular mask ', regular_mask)
    return regular_mask
    

    
    
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
    # Reverie: Walk inside the Maze of Selective Masking
    # -- Sometimes unveil the mask to clear up thougts
    def patch_conv(self, x, mode='zeropad', rand_mask=True):
        B,N,H,W = x.shape
        subimg_h, subimg_w = self.subimg_size
        patchscale_h, patchscale_w = self.pconv_patch_scale
        mask = torch.ones((B,1,1,1)).float().to(x.device)
        regular_mask = torch.ones((B,1,1,1)).float().to(x.device)
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
                # subimgs *= (F.interpolate(mask, subimgs_size, mode='nearest') > self.detail_threshold)     
                # --- 2. Bernouli Random Selection
                # Regulation is only for RandomTraining Stability here 
                # Random Bernouli required regular-valued mask : regular mask
                # print('Layer ', layer)
                # print(f'Regular Mask Info: -- max {regular_mask.max()} -- min {regular_mask.min()} | --value {regular_mask}')
                
                if rand_mask:
                    bmask = torch.bernoulli(mask)
                    subimgs *= F.interpolate(bmask, subimgs_size, mode='nearest').bool()
                    
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
                # feature *= (F.interpolate(mask, feature.shape[-2:], mode='nearest') > self.detail_threshold)
                # --- 2. Bernouli Random Selection
                if rand_mask:
                    feature *= (F.interpolate(bmask, feature.shape[-2:], mode='nearest').bool())
                
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
            norm_mode = 'minmax'
            # norm_mode = 'sigmoid' # Sigmoid Mask centralize all the values around 0.5
            mask_detail = normalize_mask_detail(mask_detail, norm_mode, self.pconv_patch_scale)
            
            
            # Regulate current mask_detail values -- Regularize
            mask_detail = regulate_mask(mask_detail, ratio=self.mask_ratios[layer])
            
            # Residual can be scaled by Current Layer Mask instead
            if rand_mask:
                # Bmask have already scaled on the feature after Conv
                residual = feature
                # Include Current Layer Details into Mask
                mask = F.interpolate(bmask, mask_detail.shape[-2:], mode='nearest') * mask_detail
                
            else:
                # Mimic Bmask Scaling is to scale with Mask_rand
                # As Bmaks is not used to scale feature, here we scale it with Mask_rand
                residual = F.interpolate(mask, feature.shape[-2:], mode='nearest') * feature
                # Update on MaskRand to include current layer details
                mask_rand = F.interpolate(mask, mask_detail.shape[-2:], mode='nearest') * mask_detail
            
            
            # Ensemble Addition of Feature
            ensemble_feature = ensemble_feature_upsample(ensemble_feature, residual, layer, self.pconv_patch_scale)

            if not rand_mask:
                bmask = torch.ones_like(mask).to(mask.dtype).to(mask.device)
                        
            
            res[layer] = {'feature':ensemble_feature, 
                          'mask': mask,
                          'mask_detail': mask_detail,
                          'mad':mad.mean(axis=1, keepdim=True),
                          'residual': residual,
                          'binary_mask': bmask}
            
        res['ensemble_feature'] = ensemble_feature
        return res   
    
    def forward(self, x, rand_mask=True):
        res = self.patch_conv(x, rand_mask=rand_mask)
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
        
        # MAD avg over channels
        mads = {f'Layer {layer} MAD': res[layer]['mad'][sample_idx,0].detach().numpy() for layer in range(0, self.max_layer+1)}
        
        bmaps = {f'Layer {layer} Binary Mask': res[layer]['binary_mask'][sample_idx,0].detach().numpy() for layer in range(0, self.max_layer+1)}
        
        # media.show_images(dmaps, height=300)
        for layer in range(0, self.max_layer+1):
            print(f'Layer {layer} Feature Avg Value: ', res[layer]['feature'][sample_idx, channel_idx].mean().item())
            count = keep_ratios[layer-1] * (4**layer)
            # topk = res[layer]['topk']
            # info2 = f'--- KeepRatios {keep_ratios[layer-1]} KeepPatches {int(count)} Actual KeepCount {int(topk)} outof Total {int(4**layer)} Patches'
            # print(info2)

        return fmaps, dmaps, pmaps, bmaps, mads
    
    
# Bernouli Random Selection -- EyeMixer
# Detail direct learned from PatchConvolution Layer -- in logits
class EyeSelect(nn.Module):
    def __init__(self, args):
        super(EyeSelect, self).__init__()
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
        modes = ['gumbel', 'ste']
        self.mode = modes[args.pconv_mode]
        
    # mode: ['gumbel', 'ste']
    # Reverie: Walk inside the Maze of Selective Masking
    # -- Sometimes unveil the mask to clear up thougts
    # Things to try here are
    # --- From the Gumbel Trick, we need logits, separate detail prediction worth trying
    # --- 
    def patch_conv(self, x, mode='ste'): 
        B,N,H,W = x.shape
        subimg_h, subimg_w = self.subimg_size
        patchscale_h, patchscale_w = self.pconv_patch_scale
        mask = torch.ones((B,1,1,1)).float().to(x.device)
        regular_mask = torch.ones((B,1,1,1)).float().to(x.device)
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
            epsilon = 1e-10
            if mode == 'ste':
                bmask = torch.bernoulli(mask) + mask - mask.detach()   
            elif mode == 'gumbel':
                logits_cat = torch.stack([(mask + epsilon).log(), (1 - mask + epsilon).log()], dim=-1)
                gumbel_out = F.gumbel_softmax(logits_cat, tau=1.0, hard=True)
                bmask = gumbel_out[:, :, :, :, 0]
            else:
                raise ValueError(f"Invalid mode: {mode}. Choose between 'ste' and 'gumbel'.")
                
            subimgs *= F.interpolate(bmask, subimgs_size, mode='nearest').bool()
                    
            # feature: (B, Dim_feature, PatchScale_h, PatchScale_w)
            # detail:  (B,           1, PatchScale_h, PatchScale_w) 
            if layer==0:
                feature, detail = self.pconvs[0](subimgs)
            else:
                feature, detail = self.pconvs[layer](subimgs, ensemble_feature)

            feature *= (F.interpolate(bmask, feature.shape[-2:], mode='nearest').bool())
            
            # Termination Logits
            logits = rearrange(detail, 'b 1 (n h) (m w) -> b 1 n m (h w)', h=patchscale_h, w=patchscale_w).mean(axis=-1)
            terminate_conf = torch.sigmoid(logits)
            mask = (1 - terminate_conf) * F.interpolate(bmask, terminate_conf.shape[-2:], mode='nearest')
            
            # Ensemble Addition of Feature
            ensemble_feature = ensemble_feature_upsample(ensemble_feature, feature, layer, self.pconv_patch_scale)         
            
            res[layer] = {'feature':ensemble_feature, 
                          'mask': mask,
                          'terminate_confidence': terminate_conf,
                          'residual': feature,
                          'binary_mask': bmask}
            
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
        pmaps = {f'Layer {layer} Termination Confidence': res[layer]['terminate_confidence'][sample_idx,0].detach().numpy() for layer in range(0, self.max_layer+1)}
        
        
        # # MAD avg over channels
        # mads = {f'Layer {layer} MAD': res[layer]['mad'][sample_idx,0].detach().numpy() for layer in range(0, self.max_layer+1)}
        
        bmaps = {f'Layer {layer} Binary Mask': res[layer]['binary_mask'][sample_idx,0].detach().numpy() for layer in range(0, self.max_layer+1)}
        
        # media.show_images(dmaps, height=300)
        for layer in range(0, self.max_layer+1):
            print(f'Layer {layer} Feature Avg Value: ', res[layer]['feature'][sample_idx, channel_idx].mean().item())
            count = keep_ratios[layer-1] * (4**layer)
            # topk = res[layer]['topk']
            # info2 = f'--- KeepRatios {keep_ratios[layer-1]} KeepPatches {int(count)} Actual KeepCount {int(topk)} outof Total {int(4**layer)} Patches'
            # print(info2)

        return fmaps, dmaps, pmaps, bmaps
    
    
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
        Regression loss: L1 Image construction loss
        """
        assert 'ensemble_feature' in outputs        
        losses = {}
        
        for v in outputs:
            if not isinstance(v, int):
                continue
            if 'feature' in outputs[v]:
                src_feat = outputs[v]['feature']
            else:
                src_feat = outputs[v]['ensemble_feature']
            tgt_feat = F.interpolate(targets, size=src_feat.shape[-2:], mode='bilinear', align_corners=False)
            loss_reg = nn.L1Loss()(src_feat, tgt_feat)
            losses[f'loss_reg_{v}'] = loss_reg
            
        if log:
            losses[f'regression_error_at_{v}_layer'] = loss_reg
            
        return losses
    
    
    def loss_binary(self, outputs, log=True):
        """
        Binary Loss: Encourage 0s & 1s in Termination Confidence Scores
        """
        losses = {}
        for v in outputs:
            if not isinstance(v, int):
                continue
            if v == self.max_layers:
                continue
            p = outputs[v]['terminate_confidence']
            loss_bin = (p * (1 - p)).sum(axis=(-1,-2)).mean()
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
            if v == self.max_layers:
                continue
                
            terminate_conf = outputs[v]['terminate_confidence']
            loss_sparsity = (1-terminate_conf).sum(axis=(-1,-2)).mean()
#             tgt = self.sparsity_ratios[v]*terminate_conf.shape[-1]*terminate_conf.shape[-2]
            
#             pred_sparsity = (1-terminate_conf).sum(axis=(-1,-2)).mean()
#             loss_sparse = torch.abs(terminate_conf - tgt)            
            # loss_sparse = (1 - terminate_conf).sum(axis=(-1,-2)).mean()
                     
            losses[f'loss_sparsity_{v}'] = loss_sparsity

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
    
    
def process_args(args):
    # Convert tuple strings to actual tuples for preprocess_img_size and pconv_patch_scale
    args.preprocess_img_size = tuple(map(int, args.preprocess_img_size.split(',')))
    args.pconv_patch_scale = tuple(map(int, args.pconv_patch_scale.split(',')))
    
    # Max layers used to correct / refine on rescaled sizes of Image & Patches
    rescale_img_h = args.preprocess_img_size[0]//(2**args.max_layer) * (2**args.max_layer)
    rescale_img_w = args.preprocess_img_size[1]//(2**args.max_layer) * (2**args.max_layer)
    subimg_h = rescale_img_h / (2**args.max_layer)
    subimg_w = rescale_img_w / (2**args.max_layer)

    args.preprocess_img_size = int(rescale_img_h), int(rescale_img_w)
    args.subimg_size = int(subimg_h), int(subimg_w)
    
    # Given Pconv kernel scale, decide on kernel/stide size
    pconv_kernel_h = subimg_h // args.pconv_patch_scale[0]
    pconv_kernel_w = subimg_w // args.pconv_patch_scale[1]
    args.pconv_patch_size = int(pconv_kernel_h), int(pconv_kernel_w)


def parse_eyemixer_args(comm = []):
    parser = argparse.ArgumentParser(
                        prog='HumanViewer',
                        description='Perception like Human -- Learnable Image Perception with QuadTreeNodes',
                        epilog='Text at the bottom of help')
    
    parser.add_argument('--name', type=str, default="EyeMixer")
        
    parser.add_argument('-max_layer', '--max_layer', default=1, type=int)
    parser.add_argument('-imsize', '--preprocess_img_size', default='384,768', type=str)  # Accept as a comma-separated string
    parser.add_argument('-pconv_dim', '--pconv_feature_dim', default=3, type=int)
    parser.add_argument('-pconv_scale', '--pconv_patch_scale', default='6,12', type=str)  # Accept as a comma-separated string
    parser.add_argument('-depth', '--mixer_depth', default=4, type=int)
    parser.add_argument('-conv_ks', '--mixer_kernel_size', default=5, type=int)
    parser.add_argument('-n_classes', '--num_classes', default=10, type=int)
    parser.add_argument('-dt', '--detail_threshold', default=0.10, type=float)
    parser.add_argument('-m', '--pconv_mode', default=1, type=int) # ['ste', 'gumbel'] for Pconv Mode
    parser.add_argument('-r', '--pconv_layer_ratio', default=0.6, type=float)
    
    parser.add_argument('--wd', default=0.01, type=float)
    parser.add_argument('--clip-norm', action='store_true')
    parser.add_argument('--batch-size', default=128, type=int)
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
