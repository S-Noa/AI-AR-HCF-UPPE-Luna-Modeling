import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import logging
import time
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 标准MLP模型定义（向后兼容）
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.01):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, features=None):
        spectral_loss = self.mse(pred, target)
        
        energy_loss = 0
        if features is not None:
            pred_energy = torch.sum(pred, dim=1)
            input_energy = features[:, 0]
            energy_loss = torch.mean(torch.relu(pred_energy - input_energy) ** 2)
        
        rdw_loss = 0
        if features is not None:
            N = features[:, 7]
            mask = N > 1
            if mask.sum() > 0:
                pass
        
        total_loss = spectral_loss + self.alpha * energy_loss + self.beta * rdw_loss
        
        return total_loss, spectral_loss, energy_loss, rdw_loss

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=300, patience=20):
    best_val_loss = float('inf')
    early_stop_count = 0
    
    # 记录训练历史
    training_history = {
        'train_loss': [],
        'train_spectral_loss': [],
        'val_loss': [],
        'val_spectral_loss': [],
        'val_r2': [],
        'learning_rate': []
    }
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_spectral_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss, spectral_loss, _, _ = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_spectral_loss += spectral_loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_spectral_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss, spectral_loss, _, _ = criterion(output, target)
                val_loss += loss.item()
                val_spectral_loss += spectral_loss.item()
                val_predictions.extend(output.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        # 计算R²分数
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        r2 = r2_score(val_targets.flatten(), val_predictions.flatten())
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录训练历史
        training_history['train_loss'].append(train_loss/len(train_loader))
        training_history['train_spectral_loss'].append(train_spectral_loss/len(train_loader))
        training_history['val_loss'].append(val_loss/len(val_loader))
        training_history['val_spectral_loss'].append(val_spectral_loss/len(val_loader))
        training_history['val_r2'].append(r2)
        training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 打印日志
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Time: {epoch_time:.2f}s")
        logging.info(f"Train Loss: {train_loss/len(train_loader):.6f}, Spectral Loss: {train_spectral_loss/len(train_loader):.6f}")
        logging.info(f"Val Loss: {val_loss/len(val_loader):.6f}, Spectral Loss: {val_spectral_loss/len(val_loader):.6f}")
        logging.info(f"Val R²: {r2:.4f}")
        logging.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            logging.info("保存最佳模型")
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                logging.info(f"早停触发，停止训练")
                break
    
    # 保存训练历史
    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=4)
    logging.info("保存训练历史")
    
    return model

# 计算评估指标
def calculate_metrics(y_true, y_pred):
    """
    计算回归模型的评估指标
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# 测试函数
def test_model(model, test_loader):
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_predictions.extend(output.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    
    # 计算所有评估指标
    metrics = calculate_metrics(test_targets, test_predictions)
    
    # 打印评估指标
    logging.info("测试集评估指标:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # 保存评估指标
    with open(os.path.join(OUTPUT_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

# 主函数
def main():
    logging.info("开始训练MLP模型...")
    
    # 加载数据
    X_train = np.load(os.path.join(INPUT_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(INPUT_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(INPUT_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(INPUT_DIR, 'y_val.npy'))
    X_test = np.load(os.path.join(INPUT_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(INPUT_DIR, 'y_test.npy'))
    
    logging.info(f"训练集: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"验证集: X={X_val.shape}, y={y_val.shape}")
    logging.info(f"测试集: X={X_test.shape}, y={y_test.shape}")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 初始化模型
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = MLP(input_dim, output_dim)
    
    # 初始化损失函数和优化器
    criterion = CustomLoss(alpha=0.1, beta=0.01)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True)
    
    # 训练模型
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=500, patience=20)
    
    # 测试模型
    test_metrics = test_model(model, test_loader)
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model.pth'))
    
    # 保存模型配置
    model_config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'batch_size': batch_size,
        'test_metrics': test_metrics
    }
    
    with open(os.path.join(OUTPUT_DIR, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)
    
    logging.info("模型训练完成！")
    for metric, value in test_metrics.items():
        logging.info(f"测试集{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
