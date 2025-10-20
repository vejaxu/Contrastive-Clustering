import os
import argparse
import torch
import torchvision
import numpy as np
from utils import yaml_config_hook
from modules import resnet, network, transform
from evaluation import evaluation
from torch.utils import data
import copy


import scipy


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
from matplotlib.colors import ListedColormap


def get_distinct_colors(n):
    """生成 n 个视觉上可区分的颜色"""
    if n <= 10:
        # 使用 Tab10 调色板（专为分类设计）
        cmap = plt.cm.get_cmap('tab10')
        return [cmap(i) for i in range(n)]
    else:
        # 使用 hsv 或其他循环调色板
        cmap = plt.cm.get_cmap('hsv')
        return [cmap(i / n) for i in range(n)]

def visualize_ac_clustering(X_raw, y_true, y_pred, save_dir="fig/AC"):
    """
    X_raw: 原始未归一化的特征，shape [N, D]
    y_true: 真实标签，shape [N]
    y_pred: 预测聚类标签，shape [N]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 降维到 2D
    print(">>> Running t-SNE on raw features...")
    if X_raw.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_raw) - 1))
        X_embedded = tsne.fit_transform(X_raw)
    else:
        X_embedded = X_raw

    # --- 真实标签图 ---
    unique_true = np.unique(y_true)
    n_true = len(unique_true)
    true_colors = get_distinct_colors(n_true)
    true_cmap = ListedColormap(true_colors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_true, cmap='viridis', alpha=0.7, s=15) # true_cmap
    plt.title("Ground Truth Labels - AC")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.colorbar(scatter, label='Class Labels', ticks=unique_true)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/AC_dataset.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 预测聚类图 ---
    unique_pred = np.unique(y_pred)
    n_pred = len(unique_pred)
    pred_colors = get_distinct_colors(n_pred)
    pred_cmap = ListedColormap(pred_colors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_pred, cmap='viridis', alpha=0.7, s=15)
    plt.title("Predicted Clusters - AC")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.colorbar(scatter, label='Cluster Labels', ticks=unique_pred)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/AC_clusters.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    print(f">>> Visualizations saved to {save_dir}/")


class MatDataset(Dataset):
    def __init__(self, mat_file, dataset_name=None, transform=None):
        data = scipy.io.loadmat(mat_file)
        
        if dataset_name == "AC":
            X_raw = data['data'].astype(np.float32)
            y = data['class']
        else:
            # ... 其他数据集逻辑（略）...
            raise NotImplementedError

        # 保存原始数据（用于可视化）
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(-1, 1)
        self.raw_features = torch.from_numpy(X_raw.copy())  # ← 未归一化！

        # 归一化后的数据（用于训练）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_raw)
        self.features = torch.from_numpy(X_norm)

        # 处理标签（同前）
        if y is not None:
            y = np.asarray(y).flatten().astype(int)
            unique_labels = np.unique(y)
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in y])
            self.labels = torch.from_numpy(y).long()
        else:
            self.labels = torch.full((len(self.features),), -1)
        
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        x = self.transform(x) if self.transform else x
        return x, self.labels[idx].item()


def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config_mat.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=True,
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=False,
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif args.dataset == "STL-10":
        train_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-10',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-dogs',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/tiny-imagenet-200/train',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        class_num = 200
    elif args.dataset == "AC":
        dataset = MatDataset(
            mat_file="E:/Code/data/AC.mat", 
            dataset_name="AC",
            transform=transform.IdentityTransform()
        )
        class_num = 2
        input_dim = dataset.features.shape[1]
    else:
        raise NotImplementedError
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    if args.dataset == "AC":
        from modules.mlp import MLP
        res = MLP(input_dim=input_dim, hidden_dim=512, output_dim=512)
    else:
        from modules import resnet
        res = resnet.get_resnet(args.resnet)

    # res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model_fp = os.path.join(args.model_path, f"checkpoint_{args.epochs}.tar")
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)

    print("### Creating features from model ###")
    X, Y = inference(data_loader, model, device)
    if args.dataset == "CIFAR-100":  # super-class
        super_label = [
            [72, 4, 95, 30, 55],
            [73, 32, 67, 91, 1],
            [92, 70, 82, 54, 62],
            [16, 61, 9, 10, 28],
            [51, 0, 53, 57, 83],
            [40, 39, 22, 87, 86],
            [20, 25, 94, 84, 5],
            [14, 24, 6, 7, 18],
            [43, 97, 42, 3, 88],
            [37, 17, 76, 12, 68],
            [49, 33, 71, 23, 60],
            [15, 21, 19, 31, 38],
            [75, 63, 66, 64, 34],
            [77, 26, 45, 99, 79],
            [11, 2, 35, 46, 98],
            [29, 93, 27, 78, 44],
            [65, 50, 74, 36, 80],
            [56, 52, 47, 59, 96],
            [8, 58, 90, 13, 48],
            [81, 69, 41, 89, 85],
        ]
        Y_copy = copy.copy(Y)
        for i in range(20):
            for j in super_label[i]:
                Y[Y_copy == j] = i
    nmi, ari, f, acc = evaluation.evaluate(Y, X)


    if args.dataset == "AC":
        # 使用未归一化的原始数据
        original_data = dataset.raw_features.numpy()  # 确保 MatDataset 有 raw_features
        visualize_ac_clustering(original_data, Y, X, save_dir="fig/AC")
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
