import os.path as osp
from argparse import ArgumentParser
import pathlib
from datetime import datetime
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms

from name import get_model

#region PARSE ARGUMENTS
parser = ArgumentParser(description='Implementation of (SE-)VGG16 and (SE-)ResNe(X)t')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU index or -1 (CPU)')
parser.add_argument('--tensorboard-logdir', '-tl', type=str, default='runs/')
parser.add_argument('--logdir', '-l', type=str, default='log/')
parser.add_argument('--log_id', '-li', type=str, default=None, help='default: yyyyMMdd-HHmmss')
parser.add_argument('--log_id_postfix', '-lip', type=str, default=None)
parser.add_argument('--model-save-step', '-s', type=int, default=5)
parser.add_argument('--dataroot', '-d', type=str, default='datasets/')
parser.add_argument('--dataset', '-D', type=str, default='cifar10', help='cifar10 or cifar100 or ilsvrc2012')
parser.add_argument('--batch-size', '-b', type=int, default=128)
parser.add_argument('--model', '-m', type=str, default='resnet50', help='(se_)vgg[n](_bn) or (se_)resnet[n] or (se_)resnext[m_n]d')
parser.add_argument('--optimizer', '-o', type=str, default='sgd', help='sgd or adam')
parser.add_argument('--momentum', '-M', type=float, default=0.9, help='Only for SGD, momentum in [0, 1]')
parser.add_argument('--learning-rate', '-lr', type=float, default=7.5E-2)
parser.add_argument('--epochs', '-e', type=int, default=100)

args = parser.parse_args()
GPU = args.gpu
TENSORBOARD_LOGDIR = args.tensorboard_logdir
LOGDIR = args.logdir
LOG_ID = args.log_id if args.log_id else datetime.now().strftime('%Y%m%d_%H%M%S')
if args.log_id_postfix:
    LOG_ID += '-' + args.log_id_postfix
TENSORBOARD_LOGDIR = osp.join(TENSORBOARD_LOGDIR, LOG_ID)
LOGDIR = osp.join(LOGDIR, LOG_ID)
MODEL_LOGDIR = osp.join(LOGDIR, 'models')
MODEL_SAVE_STEP = args.model_save_step
MODEL = args.model.lower()
DATAROOT = args.dataroot
DATASET = args.dataset.lower()
BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer.lower()
MOMENTUM = args.momentum
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
#endregion

#region DEVICE SELECTION
if GPU < 0:
    DEVICE = 'cpu'
elif GPU <= torch.cuda.device_count():
    DEVICE = f'cuda:{GPU}'
else:
    raise ValueError('Cannot find the selected CUDA device.')
#endregion

#region CREATEING LOG DIRECTORY IF NOT EXISTS AND SETUP LOGGER
path = pathlib.Path(osp.join(LOGDIR, 'models'))
path.mkdir(parents=True, exist_ok=True)
logfile = open(osp.join(LOGDIR, 'training.log'), 'a')
writer = SummaryWriter(TENSORBOARD_LOGDIR)

def _print(*values, sep=' ', end='\n', flush=True):
    print(*values, sep=sep, end=end)
    print(*values, sep=sep, end=end, file=logfile)
    if flush:
        logfile.flush()
#endregion

#region SELECT DATASET, CREATE DATALOADER
if 'cifar' in DATASET:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size=(224, 224)),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(224, 224)),
    ])

    if DATASET == 'cifar10':
        train_dataset = CIFAR10(DATAROOT, train=True, transform=train_transform, download=True)
        eval_dataset = CIFAR10(DATAROOT, train=False, transform=eval_transform, download=True)
        NUM_CLASSES = 10
    elif DATASET == 'cifar100':
        train_dataset = CIFAR100(DATAROOT, train=True, transform=train_transform, download=True)
        eval_dataset = CIFAR100(DATAROOT, train=False, transform=eval_transform, download=True)
        NUM_CLASSES = 100
    else:
        raise ValueError('Dataset not supported.')
elif DATASET == 'ilsvrc2012':
    NUM_CLASSES = 1000
    raise NotImplementedError('ILSVRC2012 is not implemented yet.')
else:
    raise ValueError('Dataset not supported.')

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
eval_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
#endregion

#region SELECT MODEL AND OPTIMIZER
model = get_model(MODEL, num_classes=NUM_CLASSES).to(DEVICE)
if OPTIMIZER == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
elif OPTIMIZER == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
else:
    raise NotImplementedError(f'{OPTIMIZER} optimizer is not supported.')
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#endregion
    
#region TRAIN-EVAL LOOP
for epoch in tqdm(range(1, EPOCHS + 1), desc='EPOCH', position=1, leave=False):
    _print('EPOCH %03d' % epoch)

    model.train()
    loss = 0
    for img, label in tqdm(train_dataloader, desc='TRAIN', position=2, leave=True):
        img = img.to(DEVICE)
        onehot = F.one_hot(label, num_classes=NUM_CLASSES).float().to(DEVICE)
        output = model(img)

        batch_loss = F.cross_entropy(output, onehot)
        batch_loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        loss += batch_loss.item()

    mean_loss = loss / len(train_dataloader)
    writer.add_scalar('Train/Loss', mean_loss, epoch)  
    _print('train_loss=%.4f' % mean_loss)
    if epoch % MODEL_SAVE_STEP == 0:
        torch.save(model.cpu(), osp.join(MODEL_LOGDIR, '%03d.pkl' % epoch))

    model.eval()
    with torch.no_grad():
        loss, num_correct = 0, 0
        num_class_correct = torch.zeros(NUM_CLASSES)
        num_class_total = torch.zeros_like(num_class_correct)
        for img, label in tqdm(eval_dataloader, desc=' EVAL', position=2, leave=True):
            img = img.to(DEVICE)
            onehot = F.one_hot(label, num_classes=NUM_CLASSES).float().to(DEVICE)
            output = model(img)
            output_label = output.argmax(dim=1)

            # eval loss
            batch_loss = F.cross_entropy(output, onehot)
            loss += batch_loss.item()

            # class-wise accuracy
            num_class_correct += torch.sum(onehot.cpu() * (label == output_label.cpu()).unsqueeze(-1), dim=0)
            num_class_total += torch.sum(onehot.cpu(), dim=0)

            # accuracy
            num_correct += torch.sum(label == output_label.cpu()).item()

        mean_loss = loss / len(eval_dataloader)
        mean_class_accuracy = torch.mean(num_class_correct / num_class_total)
        accuracy = num_correct / len(eval_dataset)
        writer.add_scalar('Eval/Loss', mean_loss, epoch)
        writer.add_scalar('Eval/Class Mean Accuracy', mean_class_accuracy, epoch)
        writer.add_scalar('Eval/Accuracy', accuracy, epoch)
        _print('eval_loss=%.4f' % mean_loss)
        _print('eval_mean_class_acc=%.4f' % mean_class_accuracy)
        _print('eval_accuracy=%.4f' % accuracy)
        _print()

        logfile.flush()
        writer.flush()
#endregion
    
#region FLUSH, CLOSE LOGGERS
logfile.close()
writer.close()
#endregion
