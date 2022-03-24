from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import torch.nn.functional as F
from networks.resnet_scatter import Scattering2dResNet
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model, seed_everything
from networks.resnet_big import SupConResNet, LinearClassifier2, LinearClassifier
from losses import SupConLoss


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--train_data_folder', type=str, default=None, help='path to custom train dataset')
    parser.add_argument('--val_data_folder', type=str, default=None, help='path to custom val dataset')
    parser.add_argument('--size', type=int, default=128, help='parameter for ResizedCrop')
    
    #Augments 
    parser.add_argument('--augment', type=str, default='randcrop', help='augmentations to apply')
    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # Wavelet Scatter Setting 
    parser.add_argument('--K', type=int, default=81*3)
    parser.add_argument('--width', type=int, default=2)
    parser.add_argument('--latent', type=int, default=128)

    # other setting
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--n_cls', type=int, default=2)
    parser.add_argument('--model_number', type=str, default="1",
                        help='Ensemble number')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()
    seed_everything(opt.seed)
    # check if dataset is path that passed required arguments
    
    
    opt.model_path = './save/SupConScatter/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupConScatter/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupConScatter_{}_{}_modelnumber_{}_aug_{}'.\
        format(opt.dataset, opt.model, opt.model_number,opt.augment)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        #mean = eval(opt.mean)
        #std = eval(opt.std)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    transforms_array = []

    transform_text = opt.augment.split(",")

    for tf in transform_text: 

        if tf == "color": 
            print("Color Augmentations Selected")
            transforms_array.append(transforms.Resize((opt.size,opt.size)))
            transforms_array.append(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
            transforms_array.append(transforms.RandomGrayscale(p=0.5))
            #transforms_array.append(transforms.RandomApply([transforms.RandomCrop(200)], p=0.2))
            transforms_array.append(transforms.Resize((opt.size,opt.size)))

        elif tf == "rotate":
            print("Affine Augmentations Selected")
            transforms_array.append(transforms.Resize((opt.size,opt.size)))
            #transforms_array.append(transforms.RandomApply([transforms.RandomCrop(200)], p=0.2))
            transforms_array.append(transforms.RandomApply([transforms.RandomRotation(45)], p=0.5))
            transforms_array.append(transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5))
            transforms_array.append(transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.5))
            transforms_array.append(transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.3), scale=(0.8, 1.1))], p=0.2))
            transforms_array.append(transforms.Resize((opt.size,opt.size)))


        elif tf == "all":
            print("All Augmentations Selected")
            transforms_array.append(transforms.Resize((opt.size,opt.size)))
            #transforms_array.append(transforms.RandomApply([transforms.RandomCrop(200)], p=0.2))
            transforms_array.append(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
            transforms_array.append(transforms.RandomGrayscale(p=0.5))
            transforms_array.append(transforms.RandomApply([transforms.RandomRotation(45)], p=0.5))
            transforms_array.append(transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5))
            transforms_array.append(transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.5))
            transforms_array.append(transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.3), scale=(0.8, 1.1))], p=0.2))
            transforms_array.append(transforms.Resize((opt.size,opt.size)))

    transforms_array.append(transforms.ToTensor()) 
    transforms_array.append(normalize) 
    train_transform = transforms.Compose(transforms_array)
        
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    """

    val_transform = transforms.Compose([
        transforms.Resize((opt.size,opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.train_data_folder,
                                            transform=TwoCropTransform(train_transform))

        val_dataset = datasets.ImageFolder(root=opt.val_data_folder,
                                            transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader,val_loader


def set_model(opt):
    model_scatter = Scattering2dResNet(opt.K, opt.width, num_classes =opt.latent)
    model = SupConResNet(name=opt.model)
    criterion_supcon = SupConLoss(temperature=opt.temp)
    criterion_ce = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls,extra_dim=opt.latent)
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        model_scatter = model_scatter.cuda()
        classifier = classifier.cuda()
        criterion_supcon = criterion_supcon.cuda()
        criterion_ce     = criterion_ce.cuda()
        cudnn.benchmark = True

    return model,model_scatter, classifier, criterion_supcon, criterion_ce


def train(train_loader, model, model_scatter,classifier, criterion_supcon, criterion_ce, optimizer_model,optimizer_classifier, epoch, opt):
    """one epoch training"""

    model.train()
    model_scatter.train()
    classifier.train()


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_supcon = AverageMeter()
    losses_ce = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        labels_ce = torch.cat([labels, labels], dim=0).cuda()
        
        if torch.cuda.is_available():

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        

        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer_model)
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer_classifier)

        #Scattering Features 
        features_scat,f_scat_encoder = model_scatter(images)
        #print(f_scat_encoder.shape)
        # compute loss Cross Entropy
        #print(features_scat.shape)
        with torch.no_grad():
            features_ce = model.encoder(images)
            features_ce_scat = torch.cat((features_ce,features_scat),1)
        output  = classifier(features_ce_scat.detach())
        loss_ce = criterion_ce(output, labels_ce)

        # update metric
        losses_ce.update(loss_ce.item(), bsz)

        # SGD 
        optimizer_classifier.zero_grad()
        loss_ce.backward()
        optimizer_classifier.step()


        #Compute Loss SupCon
        features = model(images)
        
        features_all = torch.cat((features,features_scat),1)
        features_all = F.normalize(features_all,dim=1)
        #print(features_all.shape)

        #print(features.shape)
        f1, f2 = torch.split(features_all, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss_cont = criterion_supcon(features, labels)
        elif opt.method == 'SimCLR':
            loss_cont = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses_supcon.update(loss_cont.item(), bsz)

        # SGD
        optimizer_model.zero_grad()
        loss_cont.backward()
        optimizer_model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss_ce {loss_ce.val:.3f} ({loss_ce.avg:.3f}) \t'
                  'loss_supcon {loss_supcon.val:.3f} ({loss_supcon.avg:.3f}) \t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_ce=losses_ce,loss_supcon=losses_supcon))
            sys.stdout.flush()

    return losses_supcon.avg,losses_ce.avg



def validate(val_loader, model,model_scatter,classifier, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            #Scattering Features 
            features_scat,f_scat_encoder = model_scatter(images)
            # forward
            output = model.encoder(images)
            features_ce_scat = torch.cat((output,features_scat),1)
            output = classifier(features_ce_scat)
            

            output_2 = F.softmax(output,dim=1)
            #print(labels,output_2)

            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels)
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      #'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses)) # top1=top1

    #print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print("VAL ACCURACY:",top1.avg)
    return losses.avg, top1.avg


def main():

    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader,val_loader = set_loader(opt)

    # build model and criterion
    model, model_scatter, classifier, criterion_supcon,criterion_ce = set_model(opt)

    # build optimizer
    optimizer_model      = set_optimizer(opt, model,model_scatter)
    optimizer_classifier = set_optimizer(opt, classifier)


    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(opt, optimizer_model, epoch)
        adjust_learning_rate(opt, optimizer_classifier, epoch)

        # train for one epoch
        time1 = time.time()
        loss_supcon,loss_ce  = train(train_loader, model,model_scatter,classifier, criterion_supcon, criterion_ce, optimizer_model,optimizer_classifier, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss_supcon', loss_supcon, epoch)
        logger.log_value('loss_ce', loss_ce, epoch)
        logger.log_value('learning_rate', optimizer_model.param_groups[0]['lr'], epoch)

        # evaluation
        loss, val_acc = validate(val_loader, model, model_scatter, classifier, criterion_ce, opt)
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)

        if val_acc >= best_acc:

            best_acc = val_acc
            save_file = os.path.join(
                opt.save_folder, 'best_model.pth'.format(epoch=epoch))
            save_model(model, optimizer_model, opt, epoch, save_file)

            save_file = os.path.join(
                opt.save_folder, 'best_scatter.pth'.format(epoch=epoch))
            save_model(model_scatter, optimizer_model, opt, epoch, save_file)

            save_file = os.path.join(
                opt.save_folder, 'best_classifier.pth'.format(epoch=epoch))
            save_model(classifier, optimizer_classifier, opt, epoch, save_file)



if __name__ == '__main__':
    main()
