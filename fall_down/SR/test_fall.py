#!/usr/bin/env python
# this is used to deal low-res inputs rather input high-res and dowmsample them to get low-res, and don't eval result

import argparse
import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models import Generator, Discriminator, FeatureExtractor
import warnings

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='folder', help='cifar10 | cifar100 | folder')
    parser.add_argument('--dataroot', type=str, default='./data', help='path to dataset')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--upSampling', type=int, default=4, help='low to high resolution scaling factor')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--nGPU', type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--generatorWeights', type=str, default='checkpoints/generator_final.pth', help="path to generator weights (to continue training)")
    parser.add_argument('--discriminatorWeights', type=str, default='checkpoints/discriminator_final.pth', help="path to discriminator weights (to continue training)")

    opt = parser.parse_args()
    print(opt)
	


    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    transform = transforms.Compose([transforms.ToTensor()])

    normalize = transforms.Compose([transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                    std = [0.229, 0.224, 0.225])
                                ])

    # Equivalent to un-normalizing ImageNet (for correct visualization)
    unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

    if opt.dataset == 'folder':
        # folder dataset
        dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)
    elif opt.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root=opt.dataroot, download=True, train=False, transform=transform)
    elif opt.dataset == 'cifar100':
        dataset = datasets.CIFAR100(root=opt.dataroot, download=True, train=False, transform=transform)
    assert dataset
    
    #print(dataset)
    image_name = dataset.imgs  # image path
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers))

    generator = Generator(16, opt.upSampling)
    if opt.generatorWeights != '':
        generator.load_state_dict(torch.load(opt.generatorWeights))
    print(generator)

    discriminator = Discriminator()
    if opt.discriminatorWeights != '':
        discriminator.load_state_dict(torch.load(opt.discriminatorWeights))
    print(discriminator)

    # For the content loss
    feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
    print(feature_extractor)

    # if gpu is to be used
    if opt.cuda:
        #generator.cuda()
        #discriminator.cuda()
        #feature_extractor.cuda()
        gpu_ids = [0,2] 
        torch.cuda.set_device(gpu_ids[0])
        generator = torch.nn.DataParallel(generator, device_ids=gpu_ids).cuda()
        discriminator = torch.nn.DataParallel(discriminator, device_ids=gpu_ids).cuda()
        feature_extractor = torch.nn.DataParallel(feature_extractor, device_ids=gpu_ids).cuda()

    print('Test started...')

    # Set evaluation mode (not training)
    generator.eval()
    discriminator.eval()
    #print(len(dataloader))
    for i, data in enumerate(dataloader):
        # Generate data
        low_res, _ = data
        #print(low_res.shape)
        #print(image_name[i])
        # eg: image_type_path, image_detail_name = bounding_box_test , -1_c1s3_065901_04.jpg
        image_type_path, image_detail_name = [],[]
        for j in range(len(low_res)):  # not opt.batchSize means never skip final batch
            # 'replace' is for window path issue
            image_type_path.append(image_name[i*opt.batchSize+j][0].replace('\\','/').split('/')[-2])
            image_detail_name.append(image_name[i*opt.batchSize+j][0].replace('\\','/').split('/')[-1])
        #print(len(image_type_path), len(image_detail_name))
        for j in range(len(low_res)):  # never skip final batch
            low_res[j] = normalize(low_res[j])

        # Generate real and fake inputs
        if opt.cuda:
            high_res_fake = generator(Variable(low_res).cuda())
        else:
            high_res_fake = generator(Variable(low_res))
        
        # high_res_fake = high_res_fake.to(torch.device('cuda:0'))
        for j in range(len(low_res)):  # not opt.batchSize means never skip final batch
            print(image_type_path[j],image_detail_name[j])
            if not os.path.exists('output/high_res_fake/A/{}'.format(image_type_path[j])):
                os.makedirs('output/high_res_fake/A/{}'.format(image_type_path[j]))
            #print(high_res_fake[j])
            #print(low_res[j])
            # if use unnormalize would lead error, why? sys say `high_res_real[j]` is not cuda but i print it show is cuda
            # must be cpu???

            # comment 1,uncommnet 2 when get full market high_res dataset; uncomment 1,comment 2 when only test some images
            # 1. when only test some images
            ##################################################################################################
            #save_image(unnormalize(high_res_fake[j].cpu()), 'output/high_res_fake/' + str(i * opt.batchSize + j) + '.png')  
            #save_image(unnormalize(low_res[j]), 'output/low_res/' + str(i*opt.batchSize + j) + '.png')  # save raw low_res images
            ##################################################################################################

            # 2. when get full dataset
            ##################################################################################################
            save_image(unnormalize(high_res_fake[j].cpu()), 'output/high_res_fake/A/{}/{}'.format(image_type_path[j], image_detail_name[j]))
            ##################################################################################################




if __name__ == '__main__':
    main()
