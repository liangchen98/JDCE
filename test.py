import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils import cluster_evaluation, load_imagenet10, load_cifar10, load_cifar100, load_imagedog, load_stl10, \
    JDCEDatasets, cifar100_to_cifar20
from modules import resnet, network


def test(net, dataloader):
    net.eval()
    net.cuda()
    out_list = []
    true_list = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            img, label = data
            img = data[0].cuda()
            out, _ = net.forward_cluster(img)
            out = out.cpu()
            out = out.data.numpy()
            for j in range(len(out)):
                out_list.append(out[j])
                true_list.append(label[j])
    out_npy = np.array(out_list)
    true_label = np.array(true_list)
    if args.dataset == 'cifar-100':
        true_label = [cifar100_to_cifar20(i) for i in true_label]
        true_label = np.array(true_label)

    cluster_model1 = KMeans(n_clusters=classes_num, n_init=50, init='k-means++')
    cluster_model1.fit(out_npy)
    label_kmeans = cluster_model1.labels_
    nmi, acc,ari = cluster_evaluation(true_label, label_kmeans)

    return nmi, acc, ari



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='my project')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--dataset', type=str, default='imagenet-dog', help='Datasets')
    # cifar-10, cifar-100, stl-10, imagenet-10, imagenet-dog

    args = parser.parse_args()
    feature_dim, batch_size = args.feature_dim, args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == 'cifar-10':
        data_loader, data_num, classes_num = load_cifar10(batch_size)
        save_path = './save_model/CIFAR10'
    elif args.dataset == 'cifar-100':
        data_loader, data_num, classes_num = load_cifar100(batch_size)
        save_path = './save_model/CIFAR100'
    elif args.dataset == 'stl-10':
        data_loader, data_num, classes_num = load_stl10(batch_size)
        save_path = './save_model/STL10'
    elif args.dataset == 'imagenet-10':
        data_loader, data_num, classes_num = load_imagenet10(batch_size)
        save_path = './save_model/ImageNet10'
    elif args.dataset == 'imagenet-dog':
        data_loader, data_num, classes_num = load_imagedog(batch_size)
        save_path = './save_model/ImageDog'
    else:
        raise TypeError
    print('# Classes: {}'.format(classes_num))

    res = resnet.get_resnet("ResNet34")
    model = network.Network(res, args.feature_dim, classes_num)
    model_fp = os.path.join(save_path, "checkpoint_{}.tar".format(5))  # load trained model here
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model = model.to(device)

    NMI, ACC, ARI = test(model, data_loader)
    print(f'NMI={NMI}, ACC={ACC}, ARI={ARI}')
