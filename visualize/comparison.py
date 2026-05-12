import numpy as np
from sklearn import metrics
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

def _error_caculation(target, pred, weight=None):
    mae = metrics.mean_absolute_error(target, pred, sample_weight=weight)
    # mse = metrics.mean_squared_error(target, pred)
    pcc = pearsonr(target, pred).correlation.tolist()[0]
    r2 = metrics.r2_score(target,pred)
    return mae, pcc, r2

# 旧版density, 图像更美观
def _density_v1(x,y,xmin=None,xmax=None):
    if xmin is not None:
        mask = x>=xmin
        x,y = x[mask],y[mask]
    if xmax is not None:
        mask = x<=xmax
        x,y = x[mask],y[mask]
    
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    return x, y, z

def _density_v2(x, y, xmin=None, xmax=None, bins=100):
    if xmin is not None:
        mask = x >= xmin
        x, y = x[mask], y[mask]
    if xmax is not None:
        mask = x <= xmax
        x, y = x[mask], y[mask]
    # 计算二维直方图
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    # 转换为点坐标
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, heatmap.shape[0]-1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, heatmap.shape[1]-1)
    z = heatmap[xidx, yidx]
    return x, y, z
    
def plot_comparison(target, pred, plot_range=None, colorbar_range=None, atom_num=None):
    if plot_range is None:
        plot_range = (np.minimum(np.min(target), np.min(pred)), np.maximum(np.max(target), np.max(pred)))
    x_min, x_max = plot_range
        
    x,y,z = _density_v2(target, pred, x_min, x_max)
    
    if colorbar_range is not None:
        min_mask = z < colorbar_range[0]
        max_mask = z > colorbar_range[1]
        
        z[min_mask] = colorbar_range[0]
        z[max_mask] = colorbar_range[1]
    
    mae, pcc, r2 = _error_caculation(target, pred)
    if atom_num is not None:
        atom_num = atom_num.reshape(-1)
        mae_peratom = metrics.mean_absolute_error(target, pred, sample_weight=1/atom_num) / np.mean(atom_num)
        text = f'Total: PCC={pcc:.3f}\nMAE={mae:.3f}\nR2={r2:.3f}\n\nper atom: MAE={mae_peratom * 1e+3:.3f}meV\n'
    else:
        text = f'Total: PCC={pcc:.3f}\nMAE={mae:.3f}\nR2={r2:.3f}\n'
    plt.text(0.95, 0.05, text, transform=plt.gca().transAxes,ha='right', va='bottom')
    
    plt.scatter(x,y,edgecolors='none',c=z,s=6,marker='o')
    plt.plot([x_min, x_max], [x_min, x_max], color='r', linestyle='-',linewidth=0.5)
    plt.colorbar()
    plt.tight_layout()
    pass