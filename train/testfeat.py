# Build EyeNet on CIFAR10
# Torch Tensor Versioplotf CompressNode
import sys
conv3d_path = '/mnt/d/Implementation/VD/3DConv/utils'
sys.path.append(conv3d_path)
# QuadTree Img Compression Package
from quadtree import *
from model_idea import *
from experiment_model import *
from mid_select_test import *
from high_selection_experiment import *


# Read From Pre-stored Args for CIFAR10
config_path = '/mnt/d/Implementation/VD/3DConv/config'
# testmix -- 1layer Pconv with BatchNorm instead of GRN (this is the only difference as far as I can see ??)
config_names = ['r1_acc_loss_compare_1layer_eyemix']


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--cfg', type=int, default=0)
p = parser.parse_args()

config_name = config_names[p.cfg]
print('Preparing with Config: ', config_name)
config_file = f'{config_path}/{config_name}.json'
args = read_args(config_file)

config_name = 'r1_acc_loss_compare_1layer_testmix'
# Read CIFAR10 dataset
trainset, testset, trainloader, testloader = prepare_cifar10(args)
    
    
if not args.eyemix:
    model = ConvMixer(dim=args.pconv_feature_dim,
                    depth=args.mixer_depth,
                    kernel_size=args.mixer_kernel_size,
                    patch_size=args.pconv_patch_size,
                    n_classes=args.num_classes)
else:
    model = EyeMixer(args)
    model.feature_extractor = TestFeat1layer(args) # Replace w another mode
    # There seems to be a bug that I have yet to spot out
    
    
model = nn.DataParallel(model).cuda()

lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], 
                                  [0, args.lr_max, args.lr_max/20.0, 0])[0]
opt = optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.wd)
criterion = SetCriterion(args)

if args.penalize_cost:
    losses_use = ['acc', 'cost']
else:
    losses_use = ['acc']

criterion.update_param(losses=losses_use) 

scaler = torch.cuda.amp.GradScaler()

if args.use_temperature_schedule:
    tem_schedule = lambda t: max(5. - 8. * (t/args.epochs), 1.0)
else:
    tem_schedule = lambda t: 1.
    
training_info = {'epochs': [], 'train_acc': [], 'test_acc': []}

for epoch in range(args.epochs):
    start = time.time()
    train_loss, train_acc, n = 0, 0, 0
    for i, (X, y) in enumerate(trainloader):
        model.train()
        X, y = X.cuda(), y.cuda()

        lr = lr_schedule(epoch + (i + 1)/len(trainloader))
        opt.param_groups[0].update(lr=lr)

        opt.zero_grad()
        with torch.cuda.amp.autocast():
            if hasattr(model.module, 'feature_extractor'):
                output = model(X, temperature=tem_schedule(epoch))
            else:
                output = {'pred': model(X)}
                
            loss = criterion.loss_accuracy(output, y)[f'loss_acc_{args.num_layer-1}']
            # loss = criterion(output, y)['total']

        scaler.scale(loss).backward()
        if args.clip_norm:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        
        train_loss += loss.item() * y.size(0)
        train_acc += (output['pred'].max(1)[1] == y).sum().item()
        n += y.size(0)
        
    model.eval()
    test_acc, m = 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):
            X, y = X.cuda(), y.cuda()
            with torch.cuda.amp.autocast():
                output = model(X)
            if not isinstance(output, dict):
                output = {'pred':output}
            test_acc += (output['pred'].max(1)[1] == y).sum().item()
            m += y.size(0)

    print(f'[{args.name}] Epoch: {epoch} | Train Acc: {train_acc/n:.4f}, Test Acc: {test_acc/m:.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}')
    print(f'-- train loss: {train_loss}')
    
    training_info['epochs'].append(epoch)
    training_info['train_acc'].append(train_acc/n)
    training_info['test_acc'].append(test_acc/m)
    info_file = f'{config_path}/{config_name}_train_info.json'
    with open(info_file, 'w') as fp:
        json.dump(training_info, fp)
    
# Save checkpoint
checkpoint_path = f'{config_path}/{config_name}_{epoch}.pth'
torch.save({
    'epoch': epoch,
    'model_state_dict': model.module.state_dict()
}, checkpoint_path)
print('CheckPoint File Saved')
    