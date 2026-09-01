import os
import numpy as np
import matplotlib.pyplot as plt
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量
INPUT_DIR = "processed_data"
OUTPUT_DIR = "visualizations"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 物理常数
C = 299792458  # 光速 (m/s)

# 加载处理参数
def load_processing_params():
    """
    加载处理参数
    """
    params_path = os.path.join(INPUT_DIR, 'processing_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        return params
    else:
        logging.warning("处理参数文件不存在，使用默认参数")
        return {
            'wavelength_range': [200e-9, 2500e-9],
            'target_points': 500,
            'input_features': ['energy', 'tau', 'chirp', 'lambda0', 'pressure', 'length', 'diameter',
                             'beta2', 'gamma', 'N', 'L0', 'gamma_K', 'P_ratio', 'Aeff', 'neff']
        }

# 生成波长数组
def generate_wavelengths(params):
    """
    生成波长数组
    """
    wavelength_range = params['wavelength_range']
    target_points = params['target_points']
    # 确保波长范围是正确的
    if len(wavelength_range) != 2:
        wavelength_range = [200e-9, 2500e-9]
    # 生成正确的波长数组
    return np.linspace(wavelength_range[0], wavelength_range[1], target_points) * 1e9  # 转换为nm

# 可视化光谱数据
def visualize_spectra(X, y, wavelengths, title="Spectra"):
    """
    可视化光谱数据
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制前10个样本的光谱
    for i in range(min(10, len(y))):
        # 确保数据是一维的
        spectrum = y[i]
        if len(spectrum.shape) > 1:
            spectrum = spectrum.flatten()
        # 确保光谱长度与波长数组长度匹配
        if len(spectrum) != len(wavelengths):
            logging.warning(f"光谱长度 {len(spectrum)} 与波长数组长度 {len(wavelengths)} 不匹配，跳过此样本")
            continue
        ax.plot(wavelengths, spectrum, label=f'Sample {i+1}')
    
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized Log Power')
    ax.set_title(title)
    ax.set_xlim(200, 2500)
    ax.set_ylim(0, 1)  # 确保y轴范围正确
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    save_path = os.path.join(OUTPUT_DIR, f'{title.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"保存光谱可视化到: {save_path}")
    plt.close()

# 可视化特征分布
def visualize_features(X, feature_names, title="Feature Distributions"):
    """
    可视化特征分布
    """
    n_features = X.shape[1]
    n_rows = (n_features + 2) // 3  # 每行3个图
    n_cols = min(3, n_features)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        ax.hist(X[:, i], bins=50, alpha=0.7)
        ax.set_title(feature_names[i])
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{title.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"保存特征分布可视化到: {save_path}")
    plt.close()

# 检查数据质量
def check_data_quality(X, y):
    """
    检查数据质量
    """
    logging.info("检查数据质量...")
    
    # 检查NaN值
    has_nan_X = np.isnan(X).any()
    has_nan_y = np.isnan(y).any()
    logging.info(f"X包含NaN值: {has_nan_X}")
    logging.info(f"y包含NaN值: {has_nan_y}")
    
    # 检查无穷值
    has_inf_X = np.isinf(X).any()
    has_inf_y = np.isinf(y).any()
    logging.info(f"X包含无穷值: {has_inf_X}")
    logging.info(f"y包含无穷值: {has_inf_y}")
    
    # 检查值范围
    logging.info(f"X值范围: [{X.min():.4f}, {X.max():.4f}]")
    logging.info(f"y值范围: [{y.min():.4f}, {y.max():.4f}]")
    
    # 检查数据形状
    logging.info(f"X形状: {X.shape}")
    logging.info(f"y形状: {y.shape}")

# 主函数
def main():
    logging.info("开始可视化预处理数据...")
    
    # 加载处理参数
    params = load_processing_params()
    
    # 生成波长数组
    wavelengths = generate_wavelengths(params)
    
    # 加载数据
    try:
        X_train = np.load(os.path.join(INPUT_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(INPUT_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(INPUT_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(INPUT_DIR, 'y_val.npy'))
        X_test = np.load(os.path.join(INPUT_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(INPUT_DIR, 'y_test.npy'))
    except Exception as e:
        logging.error(f"加载数据时出错: {e}")
        return
    
    # 检查数据质量
    check_data_quality(X_train, y_train)
    
    # 可视化光谱数据
    visualize_spectra(X_train, y_train, wavelengths, "Training Spectra")
    visualize_spectra(X_val, y_val, wavelengths, "Validation Spectra")
    visualize_spectra(X_test, y_test, wavelengths, "Test Spectra")
    
    # 可视化特征分布
    feature_names = params['input_features']
    visualize_features(X_train, feature_names, "Training Feature Distributions")
    
    # 可视化单个光谱的详细视图
    if len(y_train) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        # 确保数据是一维的
        spectrum = y_train[0]
        if len(spectrum.shape) > 1:
            spectrum = spectrum.flatten()
        # 确保光谱长度与波长数组长度匹配
        if len(spectrum) != len(wavelengths):
            logging.warning(f"光谱长度 {len(spectrum)} 与波长数组长度 {len(wavelengths)} 不匹配，无法绘制详细视图")
        else:
            ax.plot(wavelengths, spectrum, label='Sample 1')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Normalized Log Power')
            ax.set_title('Detailed View of a Single Spectrum')
            ax.set_xlim(200, 2500)
            ax.set_ylim(0, 1)  # 确保y轴范围正确
            ax.grid(True, alpha=0.3)
            ax.legend()
        save_path = os.path.join(OUTPUT_DIR, 'single_spectrum_detail.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"保存单个光谱详细视图到: {save_path}")
        plt.close()
    
    # 可视化特征相关性
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    im = ax.imshow(corr_matrix, cmap='coolwarm')
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names)
    plt.colorbar(im, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    save_path = os.path.join(OUTPUT_DIR, 'feature_correlation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"保存特征相关性矩阵到: {save_path}")
    plt.close()
    
    logging.info("数据可视化完成！")

if __name__ == "__main__":
    main()
