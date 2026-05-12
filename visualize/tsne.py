import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    perplexity: int = 30,
    save_path: str = "tsne.png",
    title: str = "t-SNE of GNN Features",
    cmap: str = "viridis",
    pca_dim: int = 50
):
    """
    features: [N, D]  你提取的 h_pooled
    labels:   [N]     对应的 property 值（连续值）
    """
    # 1. 可选 PCA 预处理（强烈推荐当 D > 50 时）
    if features.shape[1] > pca_dim:
        features = PCA(n_components=pca_dim, random_state=42).fit_transform(features)

    # 2. t-SNE 降维
    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        max_iter=1000,
        init="pca",
        random_state=42,
        n_jobs=-1
    )
    xy = tsne.fit_transform(features)

    # 3. 绘图
    fig, ax = plt.subplots(dpi=200, figsize=(8, 6.5))
    scatter = ax.scatter(
        xy[:, 0], xy[:, 1],
        c=labels,
        cmap=cmap,
        s=30,
        alpha=0.8,
        edgecolors="none"
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    # cbar.set_label("Property Value", fontsize=12)

    ax.set_title(title, fontsize=14, pad=15)
    # ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    # ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    # print(f"Saved to {save_path}")