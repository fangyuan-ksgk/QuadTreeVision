# Hierachical Selection Experiment
from quadtree import *
from model_idea import *
from experiment_model import *


# Recurrent Gating Function -- Recurrency in Vision is cricial
# ======================
# Recurrent Gate  Design
# ======================

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class RNNGate(nn.Module):
    """given the fixed input size, return a single layer lstm """
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm'):
        super(RNNGate, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim)
        else:
            self.rnn = None
        self.hidden = None

        # reduce dim
        self.proj = nn.Conv2d(in_channels=hidden_dim, out_channels=1,
                              kernel_size=1, stride=1)
        self.prob = nn.Sigmoid()

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()))

    def repackage_hidden(self):
        self.hidden = repackage_hidden(self.hidden)

    def forward(self, x):
        batch_size = x.size(0)
        self.rnn.flatten_parameters()

        out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)

        out = out.squeeze()
        proj = self.proj(out.view(out.size(0), out.size(1), 1, 1,)).squeeze()
        prob = self.prob(proj)

        disc_prob = (prob > 0.5).float().detach() - prob.detach() + prob
        disc_prob = disc_prob.view(batch_size, 1, 1, 1)
        return disc_prob, prob



# mode
# def ensemble_update(ensemble, feat, mode):
#     if ensemble is not None:
#         target_size = feat.shape[-2:]
#         upsample_ensemble = F.interpolate(ensemble, size=target_size, mode='nearest')
#     else: # Initialize
#         if feat is None:
#             return None
#         if mode =='add':
#             upsample_ensemble = torch.zeros_like(feat).to(feat.dtype).to(feat.device)
#         if mode == 'prob':
#             upsample_ensemble = torch.ones_like(feat).to(feat.dtype).to(feat.device)
#     if mode == 'add':
#         return upsample_ensemble + feat
#     if mode == 'prod':
#         return upsample_ensemble * feat
    
def ensemble_update(ensemble, feat, mode):
    if feat is None:
        return ensemble
    if ensemble is None:
        return feat
    target_size = feat.shape[-2:]
    upsample_ensemble = F.interpolate(ensemble, size=target_size, mode='nearest')
    if mode == 'add':
        return upsample_ensemble + feat
    if mode == 'prod':
        return upsample_ensemble * feat

def ensemble_mask_addition(ensemble, feat, mask):
    
    if ensemble is not None:
        target_size = feat.shape[-2:]
        upsample_ensemble = F.interpolate(ensemble, size=target_size, mode='nearest')
    else: # Initialize
        upsample_ensemble = torch.zeros_like(feat).to(feat.dtype).to(feat.device)
    
    if mask is not None:
        target_size = feat.shape[-2:]
        upsample_mask = F.interpolate(mask, size=target_size, mode='nearest')
    else:
        upsample_mask = torch.zeros_like(feat).to(feat.dtype).to(feat.device)
    
    return upsample_ensemble + feat * upsample_mask

# The crux of SkipNet's Structural Design is the separation of MaskGate
# Such MaskGate has recurrent structure and works better (!)
# To achieve that, we need to deal with the fact that each layer has different
# Mask shape here. 

# PatchConv Output
# Mask, Prob, 
class UnifiedPatchConv(nn.Module):
    def __init__(self, feat_h, feat_w, dim_patch_feature=256, patch_size=2, residual=False, patchscale_h=1, patchscale_w=1, mode='gumbel'):
        super(UnifiedPatchConv, self).__init__()
        self.residual = residual
        self.mode = mode

        # Define the non-residual and residual layers separately for clarity
        non_residual_layers = [
            nn.Conv2d(3, 1, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            GRN(feat_h, feat_w),
            nn.AvgPool2d((patchscale_h, patchscale_w))
        ]

        residual_layers = [
            nn.Conv2d(3 + dim_patch_feature, 3, 1),
            nn.Conv2d(3, 1, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            GRN(feat_h, feat_w),
            nn.AvgPool2d((patchscale_h, patchscale_w))
        ]
        
        if self.residual:
            layers = residual_layers
        else:
            layers = non_residual_layers
        
        # Here is the Gating Structure
        # Current Design has already Recurrency Built-in with residual connection
        # The 'residual' can be viewed as the internal state here
        # We are basically constructing a very rough RNN structure for mask prediction
        # Formal setup requires prediction & gating together -- related
        self.proj = nn.Sequential(*layers)
        self.prob = nn.Sigmoid()
        self.temperature = 1.
        
        
    def forward(self, x, feat=None, mode=None):
        if mode is not None:
            self.mode = mode
            
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add a batch dimension of size 1

        if self.residual:
            assert feat is not None, "Previous feature 'feat' is required for residual operation"
            upsampled_feat = F.interpolate(feat, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            # In reality we estimate on the residual term, then add with feat.
            residual = x - upsampled_feat
            x = torch.cat((x, upsampled_feat), dim=1)
            
            logit = self.proj(x)
            prob = self.prob(logit)
            if self.mode == 'gumbel':
                # Gumbel
                logits_cat = torch.stack([logit, -logit], dim=-1)
                gumbel_out = F.gumbel_softmax(logits_cat, tau=self.temperature, hard=True)
                disc_prob = gumbel_out[..., 0]
            if self.mode == 'threshold':
                # Threshold
                disc_prob = (prob > 0.5).float().detach() - prob.detach() + prob
                        
            return residual, disc_prob, prob, logit
        
        else:
            logit = self.proj(x)
            prob = self.prob(logit)
            if self.mode == 'gumbel':
                # Gumbel
                logits_cat = torch.stack([logit, -logit], dim=-1)
                gumbel_out = F.gumbel_softmax(logits_cat, tau=self.temperature, hard=True)
                disc_prob = gumbel_out[..., 0]
            if self.mode == 'threshold':
                # Threshold
                disc_prob = (prob > 0.5).float().detach() - prob.detach() + prob
            return x, disc_prob, prob, logit   
            
    
    
class EyeHS(nn.Module):
    def __init__(self, args):
        super(EyeHS, self).__init__()
        # sudo
        self.mode = 'threshold'
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
                self.pconvs.append(UnifiedPatchConv(
                    feat_h=int(args.pconv_patch_scale[0] * (2**layer)),
                    feat_w=int(args.pconv_patch_scale[1] * (2**layer)),
                    dim_patch_feature=args.pconv_feature_dim, 
                    patch_size=args.pconv_patch_size,
                    patchscale_h = args.pconv_patch_scale[0],
                    patchscale_w = args.pconv_patch_scale[1],
                    residual = False,
                    mode = self.mode
                ))
            else:
                self.pconvs.append(UnifiedPatchConv(
                    feat_h=int(args.pconv_patch_scale[0] * (2**layer)),
                    feat_w=int(args.pconv_patch_scale[1] * (2**layer)),
                    dim_patch_feature=args.pconv_feature_dim, 
                    patch_size=args.pconv_patch_size,
                    patchscale_h = args.pconv_patch_scale[0],
                    patchscale_w = args.pconv_patch_scale[1],
                    residual = True,
                    mode = self.mode
                ))
            
        # Mixer: Depthwise & Pointwise Convolution
        self.mixer = Mixer(dim=args.pconv_feature_dim,
                           depth=args.mixer_depth,
                           kernel_size=args.mixer_kernel_size,
                           n_classes=args.num_classes)
        
        # Mask Ratios Scheduler
        self.mask_ratios = decide_mask_ratios(self.max_layer, layer_ratio=args.pconv_layer_ratio)
        modes = ['gumbel', 'threshold']
        self.mode = modes[args.pconv_mode]
        
    # Patch Convolution on SubImages
    # Ensemble_Mask    accumulate Discrete Mask by Multiplicative
    # Ensemble_Feature accumulate Feature       by Addition
    def patch_conv(self, x, mode='threshold'): 
        if mode is not None:
            self.mode = mode
        B,N,H,W = x.shape
        ensemble_mask = torch.ones((B,1,1,1)).float().to(x.device)   
        ensemble_feature, ensemble_logits, ensemble_prob = None,None,None
        
        ratio = 1.
        res = {}
        for layer in range(0, self.max_layer+1):
            subimgs_size = get_rescale_shape(layer, self.subimg_size)
            subimgs = F.interpolate(x, subimgs_size, mode='bilinear', align_corners=True)
            
            # SkipConv is equivalent to mask before & after Conv
            subimgs *= F.interpolate(ensemble_mask, subimgs_size, mode='nearest').bool()
                    
            if layer==0:
                residual, mask, prob, logits = self.pconvs[0](subimgs, mode=self.mode)
            else:
                residual, mask, prob, logits = self.pconvs[layer](subimgs, ensemble_feature, mode = self.mode)
            
            residual *= (F.interpolate(ensemble_mask, residual.shape[-2:], mode='nearest').bool())
              
            
            # Ensemble
            ensemble_feature = ensemble_update(ensemble_feature, 
                                               residual, 
                                               mode='add')
            ensemble_mask = ensemble_update(ensemble_mask, 
                                            mask, 
                                            mode='prod')
            ensemble_logits = ensemble_update(ensemble_logits,
                                              logits,
                                              mode='add')
            ensemble_prob = ensemble_update(ensemble_prob,
                                            prob,
                                            mode='prod')
            
            
            res[layer] = {'ensemble_feature':ensemble_feature, 
                          'ensemble_mask': ensemble_mask,
                          'prob': prob,
                          'residual': residual,
                          'binary_mask': mask,
                          'logit': logits}
            
        res['ensemble_feature'] = ensemble_feature
        return res   
    
    def forward(self, x, mode=None):
        res = self.patch_conv(x, mode=mode)
        pred = self.mixer(res['ensemble_feature'])
        res['pred'] = pred
        return res
    
    def visualize_res(self, res, sample_idx, channel_idx):
        keep_ratios = np.cumprod([self.mask_ratios[l] for l in range(0, self.max_layer+1)])
        if not isinstance(channel_idx, list):
            fmaps = {f'Layer {layer} Ensemble Feature': res[layer]['ensemble_feature'][sample_idx,channel_idx].detach().clip(0.,1.).numpy() for layer in range(1,self.max_layer+1)}
        else:
            fmaps = {f'Layer {layer} Ensemble Feature': res[layer]['ensemble_feature'][sample_idx,channel_idx].detach().permute(1,2,0).clip(0.,1.).numpy() for layer in range(1,self.max_layer+1)}

        media.show_images(fmaps, height=300)
        # Mask Maps
        dmaps = {f'Layer {layer} Ensemble Mask': res[layer]['ensemble_mask'][sample_idx,0].detach().numpy() for layer in range(0, self.max_layer+1)}
        
        # Predicted Deatail Maps
        pmaps = {f'Layer {layer} ExploreProb': res[layer]['prob'][sample_idx,0].detach().numpy() for layer in range(0, self.max_layer+1)}
        
        
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
            p = outputs[v]['ensemble_mask']
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
                
            mask = outputs[v]['ensemble_mask']
            loss_sparsity = mask.sum(axis=(-1,-2)).mean()
                     
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
    
    
normalize = lambda x: (x - x.min())/(x.max() - x.min())



class ResNetRecurrentGateSP(nn.Module):
    """SkipNet with Recurrent Gate Model"""
    def __init__(self, block, layers, num_classes=10, embed_dim=10,
                 hidden_dim=10, gate_type='rnn'):
        self.inplanes = 16
        super(ResNetRecurrentGateSP, self).__init__()

        self.num_layers = layers
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self._make_group(block, 16, layers[0], group_id=1, pool_size=32)
        self._make_group(block, 32, layers[1], group_id=2, pool_size=16)
        self._make_group(block, 64, layers[2], group_id=3, pool_size=8)

        # define recurrent gating module
        if gate_type == 'rnn':
            self.control = RNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        elif gate_type == 'soft':
            self.control = SoftRNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        else:
            print('gate type {} not implemented'.format(gate_type))
            self.control = None

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_group(self, block, planes, layers, group_id=1, pool_size=16):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            meta = self._make_layer_v2(block, planes, stride=stride,
                                       pool_size=pool_size)

            setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), meta[1])
            setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])

    def _make_layer_v2(self, block, planes, stride=1, pool_size=16):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),

            )
        layer = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion

        gate_layer = nn.Sequential(
            nn.AvgPool2d(pool_size),
            nn.Conv2d(in_channels=planes * block.expansion,
                      out_channels=self.embed_dim,
                      kernel_size=1,
                      stride=1))
        if downsample:
            return downsample, layer, gate_layer
        else:
            return None, layer, gate_layer

    def forward(self, x):

        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # reinitialize hidden units
        self.control.hidden = self.control.init_hidden(batch_size)

        masks = []
        gprobs = []
        
        # Ok so basically Prediction and Gate is independent with each other
        # No wonder it would need a recurrent gate -- just mimic resnet feature
        # Otherwise Previous Mask has no way to get through the next layer's Gate
        # The internel state is another efficient (?) way of doing residual connection
        x = getattr(self, 'group1_layer0')(x)
        gate_feature = getattr(self, 'group1_gate0')(x)
        mask, gprob = self.control(gate_feature)
        gprogs.append(gprob)
        masks.append(mask.squeeze())
        prev = x
        
        # must pass through the first layer in first group
        x = getattr(self, 'group1_layer0')(x)
        # gate takes the output of the current layer

        gate_feature = getattr(self, 'group1_gate0')(x)
        mask, gprob = self.control(gate_feature)
        gprobs.append(gprob)
        masks.append(mask.squeeze())
        prev = x  # input of next layer

        for g in range(3):
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                # new mask is taking the current output
                prev = x = mask.expand_as(x) * x \
                           + (1 - mask).expand_as(prev) * prev
                gate_feature = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                mask, gprob = self.control(gate_feature)
                gprobs.append(gprob)
                masks.append(mask.squeeze())

        # last block doesn't have gate module
        del masks[-1]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, masks, gprobs
