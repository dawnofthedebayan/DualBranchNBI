from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
import tensorboard_logger as tb_logger
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model, seed_everything
from networks.resnet_big import SupCEResNet
from sklearn.metrics import confusion_matrix,f1_score,average_precision_score

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():

    parser = argparse.ArgumentParser('argument for testing')
    parser.add_argument('--model_folder', type=str, default="/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/NBI/code/SupContrast-master/save/CE/path_models/",
                        help='Ensemble model folder')
    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--method', type=str, choices=["ce","supcon"], default='ce')

    parser.add_argument('--data_folder', type=str, default=None, help='path to test custom dataset')
    parser.add_argument('--size', type=int, default=128, help='parameter for ResizedCrop')
    parser.add_argument('--n_cls', type=int, default=2, help='Total classes')
    parser.add_argument('--expt_name', type=str, default="dummy", help='Name of experiment')
    
    # other setting

    opt = parser.parse_args()
    # set the path according to the environment
    #opt.data_folder = './datasets/'
    return opt
    



def set_loader(opt):
   
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)


    normalize = transforms.Normalize(mean=mean, std=std)

    test_transform = transforms.Compose([
        transforms.Resize((opt.size,opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(root=opt.data_folder,
                                        transform=test_transform)


    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)

    return test_loader


def set_model(opt,ckpt_path):

    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model"])

    criterion = torch.nn.CrossEntropyLoss()


    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def test(loader, model):
    """validation"""
    model.eval()

    with torch.no_grad():

        for idx, (images, labels) in enumerate(loader):

            images = images.float().cuda()
            labels = labels.cuda()
            
            # forward
            output = model(images)
            output = F.softmax(output,dim=1)
            
            #print(output.shape)
            
    #print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    
    return output,labels

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    test_loader = set_loader(opt)

    #subdirs = get_immediate_subdirectories(opt.model_folder)
    logits_ensemble = None 

    #for direc in subdirs: 

    ckpt_path =  opt.model_folder  + "/best_model.pth" #+ "/" + direc + "/best_model.pth"
    # build model and criterion
    model, criterion = set_model(opt,ckpt_path)
    logits, labels = test(test_loader, model)

    if logits_ensemble == None: 
        
        logits_ensemble = logits.unsqueeze(0) 
        
    else: 

        logits_ensemble = torch.cat((logits_ensemble,logits.unsqueeze(0)),0)
        print('best accuracy: {:.2f}'.format(best_acc))
    
    logits_ensemble = torch.mean(logits_ensemble,dim=0)
    y_pred = 1 - torch.argmax(logits_ensemble,dim=1)
    y_pred = y_pred.cpu().numpy()
    labels = 1 - labels 
    labels = labels.cpu().numpy()
    
    y_conf = logits_ensemble[:,0]
    y_conf = y_conf.cpu().numpy()
    print(y_conf)
    print(labels)
    print(y_pred)
    tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()

    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    precision   = tp / (tp+fp)
    f1          = f1_score(labels, y_pred, average='weighted')
    auprc       = average_precision_score(labels, y_conf)


    with open("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/NBI/code/SupContrast-master/results/metrics_latent_final.txt","a+") as f: 

        f.write(f"{opt.expt_name} {specificity} {sensitivity} {precision} {f1} {auprc}\n")


if __name__ == '__main__':
    main()
