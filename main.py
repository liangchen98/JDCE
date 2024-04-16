import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from utils import cluster_evaluation, load_imagenet10, load_cifar10, load_cifar100, load_imagedog, load_stl10, \
    JDCEDatasets, cifar100_to_cifar20
from modules import resnet, network


def extract(net, Dataloader):
    feature = []
    image = []
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(Dataloader, 0):
            img, _ = data
            img = img.cuda(non_blocking=True)
            out, _ = net.forward_cluster(img)
            feature.append(out)
            image.append(img)
    feature = torch.cat(feature, dim=0)  # n * c
    image = torch.cat(image, dim=0)  # n * d
    return feature, image


def get_base(net, dataloader, base_num):
    base_tensor = torch.zeros(size=(2 * base_num, data_num, classes_num))
    base_final = np.zeros(shape=(base_num, data_num, classes_num))
    base_ = np.zeros(shape=(2 * base_num, data_num, classes_num))
    predict = np.zeros(shape=(2 * base_num, data_num))
    nmi_matrix = np.zeros(shape=(2 * base_num, 2 * base_num))
    for i in range(base_num * 2):
        base_tensor[i], _ = extract(net, dataloader)
        base_[i] = base_tensor[i].cpu().numpy()
    cluster_model_ = KMeans(n_clusters=10, n_init=50)
    for i in range(base_num * 2):
        cluster_model_.fit(base_[i])
        predict[i] = cluster_model_.labels_
    for i in range(base_num * 2):
        for j in range(base_num * 2):
            nmi_matrix[i, j] = normalized_mutual_info_score(predict[i], predict[j])
    nmi_sum = np.sum(nmi_matrix, axis=1)
    nmi_order = np.argsort(nmi_sum)
    for i in range(base_num):
        index = nmi_order[i]
        base_final[i] = base_[index]
    print('# Get base clusters')
    return base_final


def ensemble(base_cluster, rotation_matrix):
    # base_cluster: m * n * c, ndarray
    # rotation_matrix: m * c * c, ndarray
    alpha = np.full(base_num, 1.0 / base_num)
    A = np.zeros(shape=(base_cluster.shape[1], base_cluster.shape[2]))
    h = np.zeros_like(A)
    for epoch in range(ensemble_epoch):
        # update h
        for i in range(base_num):
            A += np.matmul(base_cluster[i], rotation_matrix[i]) * alpha[i] * alpha[i]
        U1, sigma1, ST1 = np.linalg.svd(A, full_matrices=False)
        h = np.matmul(U1, ST1)

        # update rotation_matrix
        for i in range(base_num):
            B = np.matmul(base_cluster[i].T, h) * alpha[i] * alpha[i]
            U2, sigma2, ST2 = np.linalg.svd(B, full_matrices=False)
            rotation_matrix[i] = np.matmul(U2, ST2)

        # update alpha
        d = np.zeros(shape=base_num)
        for i in range(base_num):
            d[i] = np.square(np.linalg.norm(h - np.matmul(base_cluster[i], rotation_matrix[i])))
            if d[i] == 0:
                raise ValueError
            else:
                d[i] = 1.0 / d[i]
        alpha = d / np.sum(d)

    # h: ensemble cluster result, n * c, ndarray
    # rotation_matrix: m * c * c, ndarray
    return h, rotation_matrix


def save_model(args, model, optimizer):
    out = os.path.join(save_path, "checkpoint_{}.tar".format(iteration))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, out)


def train(net, train_optimizer, dataloader):
    loss_history = []
    count = []
    net.train()
    for epoch in range(1, epochs + 1):
        train_bar = tqdm(dataloader)
        loss_sum = 0.0
        for data, y in train_bar:
            # data : batch_size * d
            # y : batch_size * c, pseudo label
            train_bar.set_description('Train Epoch: [{}/{}]'.format(epoch, epochs))
            data, y = data.cuda(), y.cuda()
            x, _ = net.forward_cluster(data)
            train_optimizer.zero_grad()
            loss = F.cross_entropy(x, y.long())
            loss_sum += loss.item()
            loss.backward()
            train_optimizer.step()
            train_bar.set_postfix(loss='{:.5f}'.format(loss.item()))

        loss_history.append(loss_sum)
        count.append(epoch + 1)

    total_loss = sum(loss_history)
    return total_loss, net


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

    cluster_model1 = KMeans(n_clusters=classes_num, n_init=50)
    cluster_model1.fit(out_npy)
    label_kmeans = cluster_model1.labels_
    nmi, acc, ari = cluster_evaluation(true_label, label_kmeans)

    return nmi, acc, ari


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='my project')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=20, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--iterations', default=5, type=int, help='Number of sweeps over the model to train')
    parser.add_argument('--base_num', default=10, type=int, help='Number of base cluster labels')
    parser.add_argument('--lr', default=0.0001, type=float, help='Optimizer learning rate')
    parser.add_argument('--ensemble_epoch', default=10, type=int, help='Number of ensemble epoches')
    parser.add_argument('--dataset', type=str, default='imagenet-10', help='Datasets')
    # cifar-10, cifar-100, stl-10, imagenet-10, imagenet-dog

    args = parser.parse_args()
    feature_dim, batch_size, epochs = args.feature_dim, args.batch_size, args.epochs
    base_num = args.base_num
    lr = args.lr
    iterations = args.iterations
    ensemble_epoch = args.ensemble_epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'cifar-10':
        data_loader, data_num, classes_num = load_cifar10(batch_size)
        pre_model_path = './pre_model/CIFAR10'
        save_path = './save_model/CIFAR10'
    elif args.dataset == 'cifar-100':
        data_loader, data_num, classes_num = load_cifar100(batch_size)
        pre_model_path = './pre_model/CIFAR100'
        save_path = './save_model/CIFAR100'
    elif args.dataset == 'stl-10':
        data_loader, data_num, classes_num = load_stl10(batch_size)
        pre_model_path = './pre_model/STL10'
        save_path = './save_model/STL10'
    elif args.dataset == 'imagenet-10':
        data_loader, data_num, classes_num = load_imagenet10(batch_size)
        pre_model_path = './pre_model/ImageNet10'
        save_path = './save_model/ImageNet10'
    elif args.dataset == 'imagenet-dog':
        data_loader, data_num, classes_num = load_imagedog(batch_size)
        pre_model_path = './pre_model/ImageDog'
        save_path = './save_model/ImageDog'
    else:
        raise TypeError
    print('# Classes: {}'.format(classes_num))

    res = resnet.get_resnet("ResNet34")
    model = network.Network(res, feature_dim, classes_num)
    model_fp = os.path.join(pre_model_path, "checkpoint_{}.tar".format(10))  # load pre_trained model here
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    # Initialize
    Rotation_matrix = np.zeros(shape=(base_num, classes_num, classes_num))
    for i in range(base_num):
        Rotation_matrix[i] = np.eye(classes_num)

    for iteration in range(1, iterations + 1):
        _, image = extract(model, data_loader)  # n * c
        print('# Begin to get base clusters')
        Y_base = get_base(model, data_loader, base_num)  # m * n * c
        H, Rotation_matrix = ensemble(Y_base, Rotation_matrix)  # n * c , m * c * c
        cluster_model = KMeans(n_clusters=10, n_init=50)
        cluster_model.fit(H)
        label_pseudo = cluster_model.labels_
        label_tensor = torch.from_numpy(label_pseudo)
        train_datasets = JDCEDatasets(image, label_tensor)
        train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
        train_loss, model = train(model, optimizer, train_loader)
        NMI, ACC, ARI = test(model, data_loader)
        save_model(args, model, optimizer)
        print(
            f'Iter{iteration} Kmeans：NMI = {NMI}, ACC = {ACC}, ARI = {ARI}')
