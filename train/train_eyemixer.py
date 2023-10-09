import sys
conv3d_path = '/mnt/d/Implementation/VD/3DConv/utils'
sys.path.append(conv3d_path)

# QuadTree Img Compression Package
from quadtree import *
from model_idea import *

# Read From Pre-stored Args for CIFAR10
config_path = '/mnt/d/Implementation/VD/3DConv/config'
# Works -- BatchSize 128 -- 24 epochs -- 0.9 TestAcc
# config_file = f'{config_path}/cifar10_psize2_maxlayer0.json'
# Tries -- Information
# sepconv: Uses separate Pconv paramter to inference on ImagePatches at different layer -- Residual & Direct prediction should be different task, therefore using the same Pconv parameter does not make sens
# nodt: detail threshold set to zero -- no discrete skip-conv applied

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=int, default=0)
p = parser.parse_args()

config_names = ['cifar10_psize2_maxlayer1', 'cifar10_psize2_maxlayer0', 'cifar10_psize2_maxlayer1_sepconv', 'cifar10_psize2_maxlayer1_sepconv_nodt']
if p.cfg !=0:
    config_names = config_names[p.cfg-1:p.cfg]
    
for config_name in config_names:
    print('Preparing with Config: ', config_name)
    config_file = f'{config_path}/{config_name}.json'
    args = read_args(config_file)
    args.batch_size = 128

    # CIFAR10 Data Preparation
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

    model = EyeMixer(args)
    model = nn.DataParallel(model).cuda()

    lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], 
                                      [0, args.lr_max, args.lr_max/20.0, 0])[0]

    opt = optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

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
                output = model(X)['pred']
                loss = criterion(output, y)

            scaler.scale(loss).backward()
            if args.clip_norm:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

        model.eval()
        test_acc, m = 0, 0
        with torch.no_grad():
            for i, (X, y) in enumerate(testloader):
                X, y = X.cuda(), y.cuda()
                with torch.cuda.amp.autocast():
                    output = model(X)['pred']
                test_acc += (output.max(1)[1] == y).sum().item()
                m += y.size(0)

        print(f'[{args.name}] Epoch: {epoch} | Train Acc: {train_acc/n:.4f}, Test Acc: {test_acc/m:.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}')

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
        'model_state_dict': model.state_dict()
    }, checkpoint_path)
    print('CheckPoint File Saved')
