# Hierachical Selection Experiment
from quadtree import *
from model_idea import *
from experiment_model import *
from mid_select_test import *
from torch.autograd import Variable
import torch.nn.functional as F
import time

# Arguments
def process_args(args):
    # Convert tuple strings to actual tuples for preprocess_img_size and pconv_patch_scale
    args.preprocess_img_size = tuple(map(int, args.preprocess_img_size.split(',')))
    args.pconv_patch_scale = tuple(map(int, args.pconv_patch_scale.split(',')))
    
    # Max layers used to correct / refine on rescaled sizes of Image & Patches
    subimg_h = args.preprocess_img_size[0]//(2**(args.num_layer-1))
    subimg_w = args.preprocess_img_size[1]//(2**(args.num_layer-1))
    rescale_img_h = subimg_h * (2**(args.num_layer-1))
    rescale_img_w = subimg_w * (2**(args.num_layer-1))

    args.preprocess_img_size = int(rescale_img_h), int(rescale_img_w)
    args.subimg_size = int(subimg_h), int(subimg_w)
    
    # Given Pconv kernel scale, decide on kernel/stide size
    pconv_kernel_h = subimg_h // args.pconv_patch_scale[0]
    pconv_kernel_w = subimg_w // args.pconv_patch_scale[1]
    args.pconv_patch_size = int(pconv_kernel_h), int(pconv_kernel_w)
    




# Simultaneous Residual Prediction & Gate Prediction
class ConvGRUCell(nn.Module):
    """
    Intuition:
    -- Introduce extra recurrency in mask generation
    -- Variable Spatial Dimension Adaptive power is why we use not LSTM/GRU
    -- As cheap as possible
    -- Mind the upsample requirement !
    """
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size     = kernel_size
        self.ConvGates       = nn.Conv2d(self.input_channels + self.hidden_channels, 
                                         2*self.hidden_channels,
                                         self.kernel_size,
                                         padding=self.kernel_size//2)
        self.Conv_ct         = nn.Conv2d(self.input_channels + self.hidden_channels, 
                                         self.hidden_channels,
                                         self.kernel_size,
                                         padding=self.kernel_size//2)
        dtype                = torch.FloatTensor
        
    def forward(self, x, h):
        if h is None:
            size_h = [x.data.size()[0], self.hidden_channels] + list(x.data.size()[2:])
            hidden = Variable(torch.zeros(size_h)).to(x.device)
        else:
            # Upsample the hidden state to match the spatial dimensions of x
            hidden = nn.functional.interpolate(h, size=(x.size(2), x.size(3)), mode='nearest')
        c1           = self.ConvGates(torch.cat((x,hidden),1))
        (rt,ut)      = c1.chunk(2, 1)
        reset_gate   = torch.sigmoid(rt)
        update_gate  = torch.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate,hidden)
        p1           = self.Conv_ct(torch.cat((x,gated_hidden),1))
        ct           = torch.tanh(p1)
        next_h       = torch.mul(update_gate,hidden) + (1-update_gate)*ct
        return next_h
    
    
class PatchConv(nn.Module):
    def __init__(self, feat_h, feat_w, dim_patch_feature=256, patch_size=2, residual=False, patchscale_h=1, patchscale_w=1, mode=1):
        super(PatchConv, self).__init__()
        self.residual = residual
        
        # Residual Connection & Feature Prediction
        if mode == 1:
            self.layer = self._make_residual_layer(residual, 
                                              dim_patch_feature,
                                              patch_size, 
                                              feat_h, feat_w)
        if mode == 2:
            self.layer = self._make_residual_layer2(residual,
                                                    dim_patch_feature,
                                                    patch_size)
    
    def _make_residual_layer(self, residual, dim_patch_feature, patch_size, feat_h, feat_w):
        if residual:
            layers = [
                nn.Conv2d(3 + dim_patch_feature, 3, 1),
                nn.Conv2d(3, dim_patch_feature, kernel_size=patch_size, stride=patch_size),
                nn.GELU(),
                GRN(feat_h, feat_w)
            ]

        else:
            layers = [
                nn.Conv2d(3, dim_patch_feature, kernel_size=patch_size, stride=patch_size),
                nn.GELU(),
                GRN(feat_h, feat_w)
            ]
        return nn.Sequential(*layers)
    
    # when residual=False, this is equivalent with ConvNet patch convolution sec
    def _make_residual_layer2(self, residual, dim_patch_feature, patch_size):
        hidden_dim = 3
        if residual:
            layers = [
                nn.Conv2d(3 + dim_patch_feature, hidden_dim, kernel_size=patch_size, stride=patch_size),
                nn.Conv2d(hidden_dim, dim_patch_feature, kernel_size=patch_size, stride=patch_size),
                nn.GELU(),
                nn.BatchNorm2d(dim_patch_feature)
            ]
        else:
            layers = [
                nn.Conv2d(3, dim_patch_feature, kernel_size=patch_size, stride=patch_size),
                nn.GELU(),
                nn.BatchNorm2d(dim_patch_feature)
            ]
        return nn.Sequential(*layers)

    def forward(self, x, feat=None):
        assert (feat is None) != self.residual
        
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add a batch dimension of size 1
        
        if self.residual:    
            upsampled_feat = F.interpolate(feat, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            x = torch.cat((x, upsampled_feat), dim=1)
        
        residual = self.layer(x)
                
        return residual
    
#       
        
# Recurrency in Mask Prediction
class ConvRNNGate(nn.Module):
    def __init__(self, patchscale_h, patchscale_w, dim_patch_feature, dim_hidden, mode='gumbel'):
        super(ConvRNNGate, self).__init__()
        self.patchscale_h = patchscale_h
        self.patchscale_w = patchscale_w
        self.dim_patch_feature = dim_patch_feature
        self.dim_hidden = dim_hidden
        
        self.downsample = nn.AvgPool2d((patchscale_h, patchscale_w))
        self.rnn = ConvGRUCell(dim_patch_feature, dim_hidden, kernel_size=3)
        self.proj = nn.Conv2d(dim_hidden, 1, 1, 1)
        self.prob = nn.Sigmoid()
        self.mode = mode
        self.temperature = 1.0
        
    def forward(self, x, hidden):
        x_downsample = self.downsample(x)
        
        if hidden is None:
            size_h = [x_downsample.shape[0], self.dim_hidden] + list(x.shape[2:])
            hidden = torch.zeros(size_h, device=x.device)
        
        hidden_next = self.rnn(x, hidden)
        
        logit = self.proj(hidden_next)
        prob = self.prob(logit)
        
        if self.mode == 'gumbel':
            logits_cat = torch.stack([logit, -logit], dim=-1)
            gumbel_out = F.gumbel_softmax(logits_cat, tau=self.temperature, hard=True)
            disc_prob = gumbel_out[..., 0]
        elif self.mode == 'threshold':
            disc_prob = (prob > 0.5).float().detach() - prob.detach() + prob
        else:
            raise ValueError("Invalid mode")
            
        return disc_prob, prob, logit, hidden_next
    
    
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
    if ensemble is not None and mask is not None:
        target_size = feat.shape[-2:]
        upsample_ensemble = F.interpolate(ensemble, size=target_size, mode='nearest')
        upsample_mask = F.interpolate(mask, size=target_size, mode='nearest')
        return upsample_ensemble + feat * upsample_mask
    else:
        return feat   
    
class ConvGate(nn.Module):
    def __init__(self, patchscale_h, patchscale_w, dim_patch_feature, mode='gumbel'):
        super(ConvGate, self).__init__()
        self.patchscale_h = patchscale_h
        self.patchscale_w = patchscale_w
        self.dim_patch_feature = dim_patch_feature
        
        self.downsample = nn.AvgPool2d((patchscale_h, patchscale_w))
        self.proj = nn.Conv2d(dim_patch_feature, 1, 1, 1)
        self.prob = nn.Sigmoid()
        self.mode = mode
        self.temperature = 1.0
    
    def forward(self, x):
        x_downsample = self.downsample(x)
        logit = self.proj(x_downsample)
        prob = self.prob(logit)
        
        if self.mode == 'gumbel':
            logits_cat = torch.stack([logit, -logit], dim=-1)
            gumbel_out = F.gumbel_softmax(logits_cat, tau=self.temperature, hard=True)
            disc_prob = gumbel_out[..., 0]
        elif self.mode == 'threshold':
            disc_prob = (prob > 0.5).float().detach() - prob.detach() + prob
        else:
            raise ValueError("Invalid mode")
            
        return disc_prob, prob, logit
    
class EyeNet_raw(nn.Module):
    def __init__(self, inp_size, mode='gumbel', num_layer=2):
        super(EyeNet_raw, self).__init__()
        
        # Calculate sizes
        self.mode = mode
        self.num_layer = num_layer
        self.subimg_size = inp_size[0]//int(2**(num_layer-1)), inp_size[1]//int(2**(num_layer-1))        
        self.patch_size = (3, 3)
        self.patchscale_h = self.subimg_size[0]//self.patch_size[0]
        self.patchscale_w = self.subimg_size[1]//self.patch_size[1]
        self.dim_patch_feature = 3
        self.dim_hidden = 1
        self.mode = 'gumbel'
        
        # Patch Convolution for Feature Prediction
        self._make_layers()        
        
    def _make_layers(self):
        self.subimgs_size = []
        for layer in range(self.num_layer):
            subimgs_size = self.subimg_size[0]*2**layer, self.subimg_size[1]*2**layer
            self.subimgs_size.append(subimgs_size)
            feature_layer = PatchConv(feat_h = subimgs_size[0]//self.patch_size[0],
                                  feat_w = subimgs_size[1]//self.patch_size[1],
                                  dim_patch_feature = self.dim_patch_feature,
                                  residual = (layer!=0),
                                  patch_size = self.patch_size,
                                  patchscale_h = self.patchscale_h,
                                  patchscale_w = self.patchscale_w)
            setattr(self, f'feature_layer_{layer}', feature_layer)
            
            mask_layer = ConvGate(self.patchscale_h, self.patchscale_w,
                                  self.dim_patch_feature, self.mode)
            setattr(self, f'mask_layer_{layer}', mask_layer)
            
        self.output_size = subimgs_size[0]//self.patch_size[0], subimgs_size[1]//self.patch_size[1]
        
    def forward(self, x, temperature = 1.0):
        total_forward_time = 0
        stat = {}
        f,m,p,next_m,next_p,res = None,None,None,None,None,{}
        for layer in range(self.num_layer):
            
            t0 = time.time()
            m = ensemble_update(m, next_m, mode='prod')
            p = ensemble_update(p, next_p, mode='prod')
            subimgs = F.interpolate(x, (self.subimg_size[0]*2**layer, self.subimg_size[1]*2**layer), mode='bilinear', align_corners=True)
            
            t1 = time.time()
            stat[f'Layer {layer} Upsample Time'] = t1-t0
            
            
            
            # print(f'Layer {layer} | SubImg Size: {subimgs.shape}')
            r = getattr(self, f'feature_layer_{layer}')(subimgs, f)
            
            t2 = time.time()
            stat[f'Layer {layer} PatchConvolution Time'] = t2-t1
            
            
            # print(f'-- Residual shape: {r.shape}')
            getattr(self, f'mask_layer_{layer}').temperature = temperature
            next_m, next_p, next_l = getattr(self, f'mask_layer_{layer}')(r)
            
            # next_m, next_p, next_l, next_h = self.control(r, h)
            t3 = time.time()
            stat[f'Layer {layer} Gating Time'] = t3-t2
            
            f = ensemble_mask_addition(f, r, m)
            t4 = time.time()
            stat[f'Layer {layer} Ensemble Mask Addition'] = t4-t3
            
            res[f'feature_{layer}'] = f
            res[f'mask_{layer}'] = m
            res[f'prob_{layer}'] = p
            res[f'residual_{layer}'] = r
            
        res['feature'] = f
        res['mask'] = m
        res['prob'] = p

        return res, stat
    
    
    
# Feature Extraction from QuadVision
class EyeFeat(nn.Module):
    def __init__(self, args):
        super(EyeFeat, self).__init__()
        self.num_layer = args.num_layer
        self.subimg_size = args.subimg_size
        self.patch_size = args.pconv_patch_size
        self.patchscale_h = args.pconv_patch_scale[0]
        self.patchscale_w = args.pconv_patch_scale[1]
        self.dim_patch_feature = args.pconv_feature_dim
        self.mode = 'gumbel'
        
        self._make_layers(mode=1)
              
        
    def _make_layers(self, mode):
        self.subimgs_size = []
        for layer in range(self.num_layer):
            subimgs_size = self.subimg_size[0]*2**layer, self.subimg_size[1]*2**layer
            self.subimgs_size.append(subimgs_size)
            feature_layer = PatchConv(feat_h = subimgs_size[0]//self.patch_size[0],
                                  feat_w = subimgs_size[1]//self.patch_size[1],
                                  dim_patch_feature = self.dim_patch_feature,
                                  residual = (layer!=0),
                                  patch_size = self.patch_size,
                                  patchscale_h = self.patchscale_h,
                                  patchscale_w = self.patchscale_w,
                                  mode = mode)
            setattr(self, f'feature_layer_{layer}', feature_layer)
            
            mask_layer = ConvGate(self.patchscale_h, self.patchscale_w,
                                  self.dim_patch_feature, self.mode)
            setattr(self, f'mask_layer_{layer}', mask_layer)
            
        self.output_size = subimgs_size[0]//self.patch_size[0], subimgs_size[1]//self.patch_size[1]
            
    def forward(self, x, temperature = 1.0):
        total_forward_time = 0
        stat = {}
        f,m,p,next_m,next_p,res = None,None,None,None,None,{}
        for layer in range(self.num_layer):
            
            t0 = time.time()
            m = ensemble_update(m, next_m, mode='prod')
            p = ensemble_update(p, next_p, mode='prod')
            subimgs = F.interpolate(x, (self.subimg_size[0]*2**layer, self.subimg_size[1]*2**layer), mode='bilinear', align_corners=True)
            
            t1 = time.time()
            stat[f'Layer {layer} Upsample Time'] = t1-t0
            
            
            # print(f'Layer {layer} | SubImg Size: {subimgs.shape}')
            r = getattr(self, f'feature_layer_{layer}')(subimgs, f)
            
            t2 = time.time()
            stat[f'Layer {layer} PatchConvolution Time'] = t2-t1
            
            
            # print(f'-- Residual shape: {r.shape}')
            getattr(self, f'mask_layer_{layer}').temperature = temperature
            next_m, next_p, next_l = getattr(self, f'mask_layer_{layer}')(r)
            
            # next_m, next_p, next_l, next_h = self.control(r, h)
            t3 = time.time()
            stat[f'Layer {layer} Gating Time'] = t3-t2
            
            f = ensemble_mask_addition(f, r, m)
            t4 = time.time()
            stat[f'Layer {layer} Ensemble Mask Addition'] = t4-t3
            
            res[f'feature_{layer}'] = f
            res[f'mask_{layer}'] = m # Current Layer's Adpopted Mask get recorded
            res[f'prob_{layer}'] = p
            res[f'residual_{layer}'] = r
            
        res['feature'] = f
        res['mask'] = m
        res['prob'] = p

        return res, stat
    
    
class TestFeat1layer(nn.Module):
    def __init__(self, args, mode=2):
        super(TestFeat1layer, self).__init__()
        self.num_layer = args.num_layer
        self.subimg_size = args.subimg_size
        self.patch_size = args.pconv_patch_size
        self.patchscale_h = args.pconv_patch_scale[0]
        self.patchscale_w = args.pconv_patch_scale[1]
        self.dim_patch_feature = args.pconv_feature_dim
        self.mode = 'gumbel'
        self._make_layers(mode=mode)
        
        # Additional initialization code for the child class (if any)

    def _make_layers(self, mode):
        subimgs_size = self.subimg_size[0], self.subimg_size[1]
        feature_layer = PatchConv(feat_h = subimgs_size[0]//self.patch_size[0],
                                  feat_w = subimgs_size[1]//self.patch_size[1],
                                  dim_patch_feature = self.dim_patch_feature,
                                  residual = False,
                                  patch_size = self.patch_size,
                                  patchscale_h = self.patchscale_h,
                                  patchscale_w = self.patchscale_w,
                                  mode = mode)
        setattr(self, 'feature_layer_0', feature_layer)
        
        
        self.output_size = subimgs_size[0]//self.patch_size[0], subimgs_size[1]//self.patch_size[1]
        
    def forward(self, x, temperature=1.0):
        subimgs = F.interpolate(x, (self.subimg_size[0], self.subimg_size[1]), mode='bilinear', align_corners=True)
        feat = getattr(self, 'feature_layer_0')(subimgs)
        res = {}
        res[f'feature_0'] = feat
        res[f'mask_0'] = None
        res[f'prob_0'] = None
        res[f'residual_0'] = feat
        res['feature'] = feat
        res['mask'] = None
        res['prob'] = None
        return res, None



class SetCriterion(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        num_layers = args.num_layer
        weights = {'acc': args.acc_scale, 'cost':args.cost_scale}
        weight_decay = args.wd
        min_cost = max(args.sparsity_ratio * (2**(args.num_layer-1)) / 4, args.min_cost)
        max_acc=1e-08
        window_size=50 # Bigger Window size stabilize the reweighting
        
        losses = []
        for key in weights:
            if weights[key]!=0:
                losses.append(key)
     
        self.update_param(num_layers, losses, weight_decay, min_cost, max_acc, weights, window_size)
        self.initialize_running_average(losses)
        self.adaptive_weighting()
        
    def update_param(self, num_layers=None, losses=None, weight_decay=None, min_cost=None,
                     max_acc=None, weights=None, window_size=None):
        self.num_layers = num_layers if num_layers is not None else self.num_layers
        self.losses = losses if losses is not None else self.losses
        self.weight_decay = weight_decay if weight_decay is not None else self.weight_decay
        self.min_cost = min_cost if min_cost is not None else self.min_cost
        self.max_acc = max_acc if max_acc is not None else self.max_acc
        self.weights = weights if weights is not None else self.weights
        self.window_size = window_size if window_size is not None else self.window_size
        
    def initialize_running_average(self, losses):
        self.runavg = {}
        if 'acc' in losses:
            self.runavg.update({f'loss_acc_{l}':None for l in range(self.num_layers-1, self.num_layers)})
        if 'cost' in losses:
            self.runavg.update({f'loss_cost_{l}':None for l in range(1, self.num_layers) if l>0})
            
        self.buffers = {}  # Dictionary to store the last N data points for each loss
        self.window_size = 50
        
    def adaptive_weighting(self):
        self.weight_dict = {}
        
        for name, ravg in self.runavg.items():
            layer = int(name.split('_')[-1])
            if ravg == None:
                self.weight_dict[name] = 1.
            else: # Balance
                self.weight_dict[name] = 1/(ravg+1e-06)

            # Layer-wise Weight Decay
            scale = self.weights[name.split('_')[1]]
            self.weight_dict[name] *= (scale * self.weight_decay**(self.num_layers-layer))
        
            # No penalize for cost less than MinCost
            if name.startswith('loss_cost') and isinstance(ravg,torch.Tensor) and ravg<=self.min_cost:
                self.weight_dict[name] *= 0.
                
            # No more reward for accuracy more than MacAcc
            if name.startswith('loss_acc') and isinstance(ravg,torch.Tensor) and ravg<=self.max_acc:
                self.weight_dict[name] *= 0
                
    def update_running_average(self, losses):
        for name, previous_avg in self.runavg.items():
            if name not in losses:
                continue
                
            # print('Name ', name, ' Losses keys: ', losses.keys())
            current_loss = losses.get(name).detach()
            
            # Initialize the buffer for this loss if it doesn't exist
            if name not in self.buffers:
                self.buffers[name] = []
            
            # Add the current loss to the buffer
            self.buffers[name].append(current_loss)
            
            # Remove the oldest loss if the buffer exceeds the window size
            if len(self.buffers[name]) > self.window_size:
                self.buffers[name].pop(0)
                
                
            if previous_avg is None:
                updated_avg = current_loss
            else:
                # EMA
                # updated_avg = self.alpha * previous_avg + (1 - self.alpha) * current_loss
                # CMA
                # updated_avg = self.runavg[name] + (current_loss - self.runavg[name]) / self.counters[name]
                # SMA
                updated_avg = sum(self.buffers[name]) / len(self.buffers[name])
                
            self.runavg[name] = updated_avg

    def loss_accuracy(self, outputs, targets):
        losses = {}
        if 'pred' in outputs:
            pred_feat = outputs['pred']
            tgt_feat = targets
            loss_acc = nn.CrossEntropyLoss()(pred_feat, tgt_feat)
        else:
            pred_feat = outputs[f'feature_{self.num_layers-1}']
            tgt_feat = F.interpolate(targets, size=pred_feat.shape[-2:], mode='bilinear', align_corners=False)
            loss_acc = F.mse_loss(pred_feat, tgt_feat)
        losses[f'loss_acc_{self.num_layers-1}'] = loss_acc
        return losses
        
    def loss_cost(self, outputs):
        losses = {}
        for l in range(1, self.num_layers):
            mask = outputs[f'mask_{l}']
            loss_cost = mask.sum(axis=(1,2,3))
            losses[f'loss_cost_{l}'] = loss_cost.mean()
        return losses
    
    # Initialization Loss -- to improve quality of weights & features
    def loss_init(self, outputs):
        losses = {}
        for l in range(1, self.num_layers):
            # orthogonality
            res = outputs[f'residual_{l}']
            prev_feat = outputs[f'feature_{l}']
            F.interpolate(targets, size=pred_feat.shape[-2:], mode='bilinear', align_corners=False)
            
    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'acc': self.loss_accuracy,
            'cost': self.loss_cost
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        if loss in ['acc']:
            return loss_map[loss](outputs, targets)
        if loss in ['cost']:
            return loss_map[loss](outputs)
        
    def forward(self, output, target):
        losses = {}
        for desc in self.losses:
            loss = self.get_loss(desc, output, target)
            for name in loss:
                loss_val, loss_weight = loss.get(name),self.weight_dict.get(name)
                weighted_loss_val = loss_val * loss_weight
                losses.update({name:loss_val})
                losses.update({f'weighted_{name}':weighted_loss_val})
        # Update Running Average
        self.update_running_average(losses)
        # Calculate Total loss
        total_loss = sum(value for key, value in losses.items() if key.startswith('weighted'))
        losses['total'] = total_loss
        return losses
    


        
cifar10_label_to_name = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}


class EyeMixer(nn.Module):
    def __init__(self, args):
        super(EyeMixer, self).__init__()
        
        self.name = 'eyemix'
        self.feature_extractor = EyeFeat(args)
        
        self.mixer = Mixer(dim=args.pconv_feature_dim,
                           depth=args.mixer_depth,
                           kernel_size=args.mixer_kernel_size,
                           n_classes=args.num_classes)
        
    def get_feature(self, inp, temperature=1.0):
        res, stat = self.feature_extractor(inp, temperature=temperature)
        return res, stat
    
    def forward(self, inp, temperature=1.0):
        res, stat = self.get_feature(inp, temperature=temperature)
        res['pred'] = self.mixer(res['feature'])
        return res
    
def ConvMixer(dim, depth, kernel_size=5, patch_size=2, n_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
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


def prepare_cifar10(args):
    # Cifar10 trainig 
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
        transforms.ColorJitter(args.jitter, args.jitter, args.jitter),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
        transforms.RandomErasing(p=args.reprob)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    trainset = torchvision.datasets.CIFAR10(root='/mnt/d/Data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(root='/mnt/d/Data', train=False,
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers)
    return trainset, testset, trainloader, testloader


import torch.nn.functional as f
print('MeaningFul Residual Is Indepedent from Previous Ensemble Feature')
def residual_loss(residual):
    window_var = f.avg_pool2d(residual**2, kernel_size=2, stride=2) - (f.avg_pool2d(residual, kernel_size=2, stride=2))**2
    return window_var.sum(axis=(-1,-2)).mean()

