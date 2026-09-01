"""
HCF Inverse Design — MLP Spectrum Predictor GUI
================================================

本模块为气体填充空芯光纤(HCF)中超快非线性脉冲传播的物理引导逆设计项目
提供图形化推理界面。基于PyQt5构建，支持加载训练好的MLP代理模型，
输入物理参数并预测输出光谱。

设计目的
--------
1. 替代耗时的UPPE数值模拟，实现毫秒级光谱预测
2. 提供直观的参数-光谱映射可视化
3. 支持多组参数的光谱对比分析
4. 展示模型训练性能指标

依赖
----
- PyQt5 >= 5.12: GUI框架
- torch >= 1.8: 模型推理引擎
- numpy >= 1.19: 数值计算
- matplotlib >= 3.3: 光谱可视化
- scikit-learn >= 0.24: MinMaxScaler反序列化
- joblib >= 1.0: Scaler加载

架构概述
--------
MainWindow (QMainWindow)
├── Tab 1: Single Prediction — 单次参数输入与光谱预测
│   ├── 左面板: 模型配置 + 参数输入
│   └── 右面板: 光谱图 + 预测信息表
├── Tab 2: Batch Prediction — 批量文件输入与预测
│   ├── 文件选择栏
│   ├── 光谱对比图
│   └── 结果汇总表
├── Tab 3: Training Performance — 训练性能可视化
│   ├── 评估指标面板
│   ├── Loss/R²曲线图
│   └── 模型配置表
└── Tab 4: About — 使用说明与特征描述

物理背景
--------
输入特征描述气体填充空芯光纤系统的5个可控参数(能量、脉宽、气压、
光纤长度、纤芯直径)及8个预计算物理量(GVD、非线性系数、孤子阶数等)。
输出为500点归一化对数功率谱，覆盖200-2500nm波长范围。
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox,
    QFileDialog, QTableWidget, QTableWidgetItem, QMessageBox,
    QSplitter, QFrame, QGridLayout, QSpinBox, QDoubleSpinBox,
    QStatusBar, QAction, QMenuBar, QProgressBar, QHeaderView,
    QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

C_LIGHT = 299792458

# GUI样式常量
STYLES = {
    "primary": "#3B82F6",        # 主蓝色
    "secondary": "#6366F1",      # 次要紫色
    "success": "#10B981",        # 成功绿色
    "warning": "#F59E0B",        # 警告橙色
    "danger": "#EF4444",         # 危险红色
    "background": "#F9FAFB",      # 背景浅灰
    "surface": "#FFFFFF",         # 表面白色
    "text": "#1F2937",           # 主文本
    "text_secondary": "#6B7280",  # 次要文本
    "border": "#E5E7EB",          # 边框
    "border_focus": "#D1D5DB",     # 聚焦边框
}

# 现代CSS样式 - QSS兼容设计
CSS_STYLE = """
/* 全局样式 */
QMainWindow {
    background-color: #E8F5E8;
}

/* 卡片样式 */
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 12px;
    margin-top: 16px;
    padding: 16px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    top: 0px;
    padding: 0 8px;
    font-weight: 600;
    font-size: 14px;
    color: #333333;
    background-color: #FFFFFF;
    border-radius: 6px;
}

/* 按钮样式 */
QPushButton {
    background-color: #4F46E5;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
    font-size: 14px;
}

QPushButton:hover {
    background-color: #4338CA;
}

QPushButton:pressed {
    background-color: #3730A3;
}

QPushButton:disabled {
    background-color: #E5E7EB;
    color: #9CA3AF;
}

/* 特殊按钮样式 */
QPushButton#btnPredict {
    background-color: #10B981;
    font-weight: 600;
    font-size: 15px;
    padding: 12px 24px;
}

QPushButton#btnPredict:hover {
    background-color: #059669;
}

QPushButton#btnLoad {
    background-color: #6366F1;
}

QPushButton#btnLoad:hover {
    background-color: #4F46E5;
}

/* 输入控件样式 */
QLineEdit, QDoubleSpinBox, QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 14px;
}

QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 2px solid #4F46E5;
    outline: none;
}

QDoubleSpinBox:read-only {
    background-color: #F9FAFB;
    color: #6B7280;
    border: 1px solid #E5E7EB;
}

/* 表格样式 */
QTableWidget {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 8px;
    font-size: 13px;
}

QTableWidget::item {
    padding: 6px 10px;
    border-bottom: 1px solid #F3F4F6;
}

QTableWidget::item:selected {
    background-color: #EEF2FF;
    color: #4338CA;
}

QTableWidget::item:hover {
    background-color: #F3F4F6;
}

/* 标签页样式 */
QTabWidget {
    background-color: #F5F5F5;
    border: none;
}

QTabBar::tab {
    background-color: #FFFFFF;
    color: #6B7280;
    padding: 12px 20px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    margin-right: 4px;
    font-weight: 500;
    font-size: 14px;
    border: 1px solid #E5E7EB;
    border-bottom: none;
}

QTabBar::tab:selected {
    background-color: #FFFFFF;
    color: #4F46E5;
    border-bottom: 3px solid #4F46E5;
}

QTabBar::tab:hover {
    background-color: #F9FAFB;
    color: #374151;
}

/* 状态栏样式 */
QStatusBar {
    background-color: #FFFFFF;
    color: #6B7280;
    border-top: 1px solid #E5E7EB;
    padding: 6px 12px;
    font-size: 12px;
}

/* 标签样式 */
QLabel {
    color: #374151;
    font-size: 14px;
}

QLabel#model_status_label {
    font-weight: 600;
    font-size: 13px;
}

/* 进度条样式 */
QProgressBar {
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    background-color: #F3F4F6;
    text-align: center;
    height: 8px;
}

QProgressBar::chunk {
    background-color: #4F46E5;
    border-radius: 6px;
}

/* 分割器样式 */
QSplitter::handle {
    background-color: #E5E7EB;
    width: 4px;
    border-radius: 2px;
}

QSplitter::handle:hover {
    background-color: #D1D5DB;
}

/* 滚动条样式 */
QScrollBar:vertical {
    background-color: #F3F4F6;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background-color: #D1D5DB;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #9CA3AF;
}

QScrollBar:horizontal {
    background-color: #F3F4F6;
    height: 8px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background-color: #D1D5DB;
    border-radius: 4px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #9CA3AF;
}
"""

# 特征名称（12维，不含length）
# length已从采样中移除，因为data_generation.jl中光纤长度
# 根据模式固定（capillary=1.0m, antiresonant=0.5m），不再作为变量
FEATURE_NAMES_12 = ['energy', 'tau', 'pressure', 'diameter',
                    'beta2', 'gamma', 'N', 'L0', 'gamma_K', 'P_ratio', 'Aeff', 'neff']

# 保留15维定义用于向后兼容（完整参数集）
FEATURE_NAMES_15 = ['energy', 'tau', 'chirp', 'lambda0', 'pressure', 'length', 'diameter',
                    'beta2', 'gamma', 'N', 'L0', 'gamma_K', 'P_ratio', 'Aeff', 'neff']

# 默认使用12维特征集
FEATURE_NAMES_13 = FEATURE_NAMES_12

FEATURE_LABELS = {
    # 键: 特征内部名称
    # 值: (显示标签, GUI值→SI单位换算因子)
    # GUI中用户输入的值乘以换算因子的倒数即为SI单位值
    # 例如: 用户输入 energy=1.5 (μJ), SI值 = 1.5 / 1e6 = 1.5e-6 (J)
    'energy': ('Pulse Energy (μJ)', 1e6),
    'tau': ('Pulse Width (fs)', 1e15),
    'chirp': ('Chirp Parameter C', 1.0),
    'lambda0': ('Center Wavelength (nm)', 1e9),
    'pressure': ('Gas Pressure (bar)', 1.0),
    'length': ('Fiber Length (cm)', 100.0),
    'diameter': ('Core Diameter (μm)', 1e6),
    'beta2': ('GVD β₂ (fs²/mm)', 1e30 * 1e3),
    'gamma': ('Nonlinear Coeff γ (1/(W·m))', 1.0),
    'N': ('Soliton Order N', 1.0),
    'L0': ('Dispersion Length L₀ (cm)', 100.0),
    'gamma_K': ('Keldysh Parameter γ_K', 1.0),
    'P_ratio': ('P_peak / P_critical', 1.0),
    'Aeff': ('Effective Mode Area (μm²)', 1e12),
    'neff': ('Effective Refractive Index', 1.0),
}

PARAM_SPACE_RANGES = {
    """参数空间的GUI显示值范围（与data_generation.jl中的param_ranges一致）。
    用于将GUI输入值直接线性映射到[0,1]归一化空间，
    绕过scaler（因为processed_data中的X_train已经过MinMaxScaler归一化到[0,1]）。
    """
    'energy': (0.3, 3.0),
    'tau': (5.0, 50.0),
    'chirp': (-3.0, 3.0),
    'lambda0': (800.0, 1030.0),
    'pressure': (0.5, 50.0),
    'diameter': (100.0, 200.0),  # updated to match data_generation.jl [100, 200] μm
    'beta2': (-1000.0, 1000.0),
    'gamma': (0.001, 10.0),
    'N': (0.5, 5.0),
    'L0': (0.1, 100.0),
    'gamma_K': (0.1, 5.0),
    'P_ratio': (0.1, 10.0),
    'Aeff': (100.0, 2000.0),
    'neff': (1.0001, 1.001),
}


class MLP(nn.Module):
    """多层感知机代理模型，与train_mlp_raw.py中定义完全一致。"""

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


class TemporalMLP(nn.Module):
    """Temporal MLP - 支持时序演化预测的模型架构。"""

    def __init__(self, input_dim, output_dim, n_z_steps=20):
        super(TemporalMLP, self).__init__()
        self.n_z_steps = n_z_steps
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
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
        )

        self.temporal_decoder = nn.Sequential(
            nn.Linear(512 + 1, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, output_dim)
        )

        self.final_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, output_dim)
        )

    def forward(self, x, z_positions=None):
        batch_size = x.size(0)
        latent = self.encoder(x)
        final_output = self.final_decoder(latent)

        if z_positions is None:
            return None, final_output

        n_z = z_positions.size(1)
        temporal_output = []
        for i in range(n_z):
            z_i = z_positions[:, i:i+1]
            z_i_norm = z_i / (z_i.max() + 1e-8)
            latent_z = torch.cat([latent, z_i_norm], dim=1)
            spectrum_i = self.temporal_decoder(latent_z)
            temporal_output.append(spectrum_i)

        temporal_output = torch.stack(temporal_output, dim=1)
        return temporal_output, final_output


class TemporalLSTM(nn.Module):
    """Temporal LSTM - LSTM-based temporal evolution modeling."""

    def __init__(self, input_dim, output_dim, n_z_steps=20, hidden_dim=512):
        super(TemporalLSTM, self).__init__()
        self.n_z_steps = n_z_steps
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.param_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim + 1,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.spectrum_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, output_dim)
        )

    def forward(self, x, z_positions=None):
        batch_size = x.size(0)
        param_latent = self.param_encoder(x)

        if z_positions is None:
            output = self.spectrum_decoder(param_latent)
            return None, output

        n_z = z_positions.size(1)
        lstm_input = []
        for i in range(n_z):
            z_i = z_positions[:, i:i+1]
            z_i_norm = z_i / (z_i.max() + 1e-8)
            step_input = torch.cat([param_latent, z_i_norm], dim=1)
            lstm_input.append(step_input)

        lstm_input = torch.stack(lstm_input, dim=1)
        lstm_output, _ = self.lstm(lstm_input)

        temporal_output = []
        for i in range(n_z):
            spectrum_i = self.spectrum_decoder(lstm_output[:, i, :])
            temporal_output.append(spectrum_i)

        temporal_output = torch.stack(temporal_output, dim=1)
        final_output = temporal_output[:, -1, :]
        return temporal_output, final_output


class InferenceWorker(QThread):
    """异步推理工作线程，支持标准MLP和时序模型(TemporalMLP/TemporalLSTM)。"""

    finished = pyqtSignal(np.ndarray, dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, model, scaler, features_pre_scaled, feature_names,
                 model_type='mlp', z_positions=None):
        super().__init__()
        self.model = model
        self.scaler = scaler
        self.features_pre_scaled = features_pre_scaled
        self.feature_names = feature_names
        self.model_type = model_type
        self.z_positions = z_positions

    def run(self):
        try:
            features_scaled = np.array(self.features_pre_scaled, dtype=np.float32).reshape(1, -1)
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

            self.model.eval()
            with torch.no_grad():
                if self.model_type in ('temporal', 'lstm'):
                    # 时序模型: 返回 (temporal_output, final_output)
                    temporal_out, final_out = self.model(features_tensor, self.z_positions)
                    prediction = final_out.cpu().numpy()[0]
                    # 保存时序结果用于可视化
                    temporal_data = temporal_out.cpu().numpy()[0] if temporal_out is not None else None
                else:
                    # 标准MLP
                    prediction = self.model(features_tensor).cpu().numpy()[0]
                    temporal_data = None

            # 确保预测值在[0,1]范围内
            prediction = np.clip(prediction, 0.0, 1.0)

            info = {
                'input_features': self.features_pre_scaled,
                'feature_names': self.feature_names,
                'scaler_used': self.scaler is not None,
                'model_type': self.model_type,
                'temporal_data': temporal_data,
                'z_positions': self.z_positions.cpu().numpy() if self.z_positions is not None else None,
            }
            self.finished.emit(prediction, info)
        except Exception as e:
            self.error.emit(str(e))


class SpectrumCanvas(FigureCanvas):
    """嵌入Qt界面的matplotlib光谱绑图画布。

    继承FigureCanvasQTAgg，将matplotlib图形渲染到Qt Widget中，
    提供光谱绘制、多光谱对比、训练曲线等可视化功能。

    Parameters
    ----------
    parent : QWidget, optional
        父Widget，默认None。
    width : float, optional
        画布宽度(英寸)，默认10。
    height : float, optional
        画布高度(英寸)，默认6。
    dpi : int, optional
        分辨率，默认100。
    """

    def __init__(self, parent=None, width=10, height=6, dpi=100):
        # 使用现代Matplotlib样式
        import matplotlib.pyplot as plt
        plt.style.use('default')
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        # 设置现代风格的坐标轴
        self.axes.set_facecolor('#E8F5E8')
        self.fig.patch.set_facecolor('#E8F5E8')
        
        # 移除顶部和右侧边框
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        
        # 设置边框颜色
        self.axes.spines['left'].set_color('#E5E7EB')
        self.axes.spines['bottom'].set_color('#E5E7EB')
        
        # 设置刻度和标签样式
        self.axes.tick_params(axis='both', which='both', labelsize=10, color='#6B7280')
        self.axes.xaxis.label.set_color('#6B7280')
        self.axes.yaxis.label.set_color('#6B7280')
        self.axes.title.set_color('#374151')
        self.axes.title.set_fontweight('600')
        self.axes.title.set_fontsize(12)
        
        # 设置网格样式
        self.axes.grid(True, alpha=0.2, color='#E5E7EB')
        
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout(pad=2.0)

    def plot_spectrum(self, spectrum, wavelengths, title="Predicted Output Spectrum", ylabel=None):
        """绘制单条预测光谱。

        Parameters
        ----------
        spectrum : np.ndarray, shape (500,)
            光谱数据（归一化、对数功率或线性功率，取决于显示模式）。
        wavelengths : np.ndarray, shape (500,)
            波长数组(nm)，通常为np.linspace(200, 2500, 500)。
        title : str, optional
            图表标题。
        ylabel : str, optional
            y轴标签。为None时自动根据数据范围选择。
        """
        self.axes.clear()
        self.axes.plot(wavelengths, spectrum, color='#2196F3', linewidth=1.5, label='Predicted')
        self.axes.fill_between(wavelengths, spectrum, alpha=0.15, color='#2196F3')
        self.axes.set_xlabel('Wavelength (nm)', fontsize=11)

        if ylabel is None:
            ylabel = 'Normalized Log Power'
        self.axes.set_ylabel(ylabel, fontsize=11)

        self.axes.set_title(title, fontsize=13, fontweight='bold')
        self.axes.set_xlim(200, 2500)

        s_min, s_max = spectrum.min(), spectrum.max()
        if s_max > s_min:
            margin = (s_max - s_min) * 0.05
            self.axes.set_ylim(s_min - margin, s_max + margin)
        else:
            self.axes.set_ylim(0, 1)

        self.axes.grid(True, alpha=0.3, linestyle='--')
        self.axes.legend(fontsize=10)

        peak_idx = np.argmax(spectrum)
        peak_wl = wavelengths[peak_idx]
        peak_val = spectrum[peak_idx]
        self.axes.annotate(f'Peak: {peak_wl:.0f} nm',
                          xy=(peak_wl, peak_val),
                          xytext=(peak_wl + 150, peak_val - 0.1),
                          arrowprops=dict(arrowstyle='->', color='#F44336'),
                          fontsize=10, color='#F44336', fontweight='bold')
        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_comparison(self, spectra_list, labels, wavelengths, title="Spectrum Comparison"):
        """绘制多条光谱对比图。

        Parameters
        ----------
        spectra_list : list[np.ndarray]
            光谱数组列表，每个元素shape为(500,)。
        labels : list[str]
            各光谱的图例标签。
        wavelengths : np.ndarray, shape (500,)
            波长数组(nm)。
        title : str, optional
            图表标题。

        最多使用5种预设颜色循环绘制，超出则循环复用。
        """
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
        self.axes.clear()
        for i, (spectrum, label) in enumerate(zip(spectra_list, labels)):
            self.axes.plot(wavelengths, spectrum, color=colors[i % len(colors)],
                          linewidth=1.2, label=label, alpha=0.85)
        self.axes.set_xlabel('Wavelength (nm)', fontsize=11)
        self.axes.set_ylabel('Normalized Log Power', fontsize=11)
        self.axes.set_title(title, fontsize=13, fontweight='bold')
        self.axes.set_xlim(200, 2500)
        self.axes.set_ylim(0, 1)
        self.axes.grid(True, alpha=0.3, linestyle='--')
        self.axes.legend(fontsize=10)
        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_training_history(self, history, title="Training History"):
        """绘制训练/验证损失曲线。

        Parameters
        ----------
        history : dict
            训练历史字典，需包含键 'train_loss' 和/或 'val_loss'，
            值为各epoch的损失值列表。
        title : str, optional
            图表标题。
        """
        if not history:
            return
        self.axes.clear()
        if 'train_loss' in history:
            self.axes.plot(history['train_loss'], label='Train Loss', color='#2196F3', linewidth=1.2)
        if 'val_loss' in history:
            self.axes.plot(history['val_loss'], label='Val Loss', color='#F44336', linewidth=1.2)
        self.axes.set_xlabel('Epoch', fontsize=11)
        self.axes.set_ylabel('Loss', fontsize=11)
        self.axes.set_title(title, fontsize=13, fontweight='bold')
        self.axes.grid(True, alpha=0.3, linestyle='--')
        self.axes.legend(fontsize=10)
        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_pulse(self, pulse, times, title="Pulse Temporal Profile", ylabel=None):
        """绘制脉冲时域图。

        Parameters
        ----------
        pulse : np.ndarray
            脉冲时域数据。
        times : np.ndarray
            时间数组(fs)。
        title : str, optional
            图表标题。
        ylabel : str, optional
            y轴标签。为None时使用默认值。
        """
        self.axes.clear()
        self.axes.plot(times, pulse, color='#2196F3', linewidth=1.5, label='Pulse')
        self.axes.fill_between(times, pulse, alpha=0.15, color='#2196F3')
        self.axes.set_xlabel('Time (fs)', fontsize=11)

        if ylabel is None:
            ylabel = 'Normalized Intensity'
        self.axes.set_ylabel(ylabel, fontsize=11)

        self.axes.set_title(title, fontsize=13, fontweight='bold')

        # 自动设置时间范围
        t_min, t_max = times.min(), times.max()
        self.axes.set_xlim(t_min, t_max)

        # 设置y轴范围
        p_min, p_max = pulse.min(), pulse.max()
        if p_max > p_min:
            margin = (p_max - p_min) * 0.05
            self.axes.set_ylim(p_min - margin, p_max + margin)
        else:
            self.axes.set_ylim(0, 1)

        self.axes.grid(True, alpha=0.3, linestyle='--')
        self.axes.legend(fontsize=10)

        # 标注脉冲宽度
        max_idx = np.argmax(pulse)
        max_time = times[max_idx]
        max_val = pulse[max_idx]
        half_max = max_val / 2
        
        # 找到半最大值点
        left_idx = np.argmin(np.abs(pulse[:max_idx] - half_max))
        right_idx = max_idx + np.argmin(np.abs(pulse[max_idx:] - half_max))
        
        if left_idx < right_idx:
            fwhm = times[right_idx] - times[left_idx]
            self.axes.annotate(f'FWHM: {fwhm:.1f} fs',
                              xy=(max_time, max_val),
                              xytext=(max_time + 50, max_val - 0.1),
                              arrowprops=dict(arrowstyle='->', color='#F44336'),
                              fontsize=10, color='#F44336', fontweight='bold')
        
        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_r2_history(self, history, title="Validation R²"):
        """绘制验证集R²分数曲线。

        Parameters
        ----------
        history : dict
            训练历史字典，需包含键 'val_r2'，值为各epoch的R²列表。
        title : str, optional
            图表标题。

        图中包含R²=0.95的红色虚线参考线（项目目标指标）。
        """
        if not history or 'val_r2' not in history:
            return
        self.axes.clear()
        self.axes.plot(history['val_r2'], label='Val R²', color='#4CAF50', linewidth=1.2)
        self.axes.axhline(y=0.95, color='#F44336', linestyle='--', alpha=0.5, label='Target R²=0.95')
        self.axes.set_xlabel('Epoch', fontsize=11)
        self.axes.set_ylabel('R² Score', fontsize=11)
        self.axes.set_title(title, fontsize=13, fontweight='bold')
        self.axes.grid(True, alpha=0.3, linestyle='--')
        self.axes.legend(fontsize=10)
        self.fig.tight_layout(pad=2.0)
        self.draw()

    def plot_evolution(self, data, x, y, title, xlabel, ylabel, cmap='viridis'):
        """绘制二维演化热力图。

        Parameters
        ----------
        data : np.ndarray, shape (len(y), len(x))
            演化数据矩阵。
        x : np.ndarray
            x轴数据（时间或波长）。
        y : np.ndarray
            y轴数据（传播距离）。
        title : str
            图表标题。
        xlabel : str
            x轴标签。
        ylabel : str
            y轴标签。
        cmap : str, optional
            颜色映射。
        """
        # 清除整个figure的所有内容，重新创建
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        
        # 创建热力图
        im = self.axes.imshow(data, aspect='auto', origin='lower', 
                           extent=[x[0], x[-1], y[0], y[-1]],
                           cmap=cmap, vmin=0, vmax=1)
        
        # 添加颜色条
        cbar = self.fig.colorbar(im, ax=self.axes)
        cbar.set_label('Normalized Intensity', fontsize=10)
        
        # 设置标签和标题
        self.axes.set_xlabel(xlabel, fontsize=11)
        self.axes.set_ylabel(ylabel, fontsize=11)
        self.axes.set_title(title, fontsize=13, fontweight='bold')
        
        self.fig.tight_layout(pad=2.0)
        self.draw()


class MainWindow(QMainWindow):
    """应用程序主窗口。

    包含4个功能标签页，管理模型加载、推理执行、结果可视化和
    训练性能展示的完整工作流。

    Instance Attributes
    -------------------
    model : nn.Module or None
        加载的MLP模型，未加载时为None。
    scaler : MinMaxScaler or None
        训练时拟合的特征缩放器，用于推理时归一化输入。
    processing_params : dict or None
        预处理参数（波长范围、特征列表等），从processing_params.json加载。
    feature_names : list[str]
        当前模型使用的特征名称列表，默认FEATURE_NAMES_13。
    model_config : dict or None
        模型配置（输入/输出维度、batch_size等），从model_config.json加载。
    evaluation_metrics : dict or None
        测试集评估指标（MSE, RMSE, MAE, R2）。
    training_history : dict or None
        训练历史（loss/r²曲线数据）。
    last_prediction : np.ndarray or None
        最近一次预测的光谱结果。
    last_info : dict or None
        最近一次预测的元信息。
    comparison_spectra : list[np.ndarray]
        待对比的光谱列表。
    comparison_labels : list[str]
        对应的光谱标签列表。
    feature_inputs : dict[str, QDoubleSpinBox]
        特征名→输入控件的映射，用于读取用户输入值。
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler = None
        self.processing_params = None
        self.feature_names = FEATURE_NAMES_13
        self.model_config = None
        self.evaluation_metrics = None
        self.training_history = None
        self.last_prediction = None
        self.last_info = None
        self.comparison_spectra = []
        self.comparison_labels = []
        self.display_mode = 'normalized'  # 'normalized', 'log_power', 'linear_power'
        self.log_spectrum_min = -15.0
        self.log_spectrum_max = 6.0
        self.model_type = 'mlp'  # 'mlp', 'temporal', 'lstm'
        self.n_z_steps = 20  # 时序模型的z步数
        self.z_positions = None  # 时序模型的z位置

        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        """初始化UI布局：创建4个标签页并设置状态栏。"""
        self.setWindowTitle("HCF Inverse Design — MLP Spectrum Predictor")
        self.setMinimumSize(1280, 860)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self._build_inference_tab()
        self._build_batch_tab()
        self._build_performance_tab()
        self._build_about_tab()

        self.statusBar().showMessage("Ready — Load a trained model to begin")

    def _build_inference_tab(self):
        """构建"Single Prediction"标签页。

        布局:
        - 左面板(最大宽度520px):
          - Model Configuration组: 模型文件路径、数据目录路径、加载按钮、状态标签
          - Input Parameters组: 各特征的QDoubleSpinBox输入控件(2列网格)
          - 操作按钮: Run Prediction / Add to Comparison / Clear Comparison
        - 右面板:
          - Prediction Result组: matplotlib光谱图 + 导航工具栏
          - Prediction Info组: 峰值波长、峰值强度等关键信息表格
        """
        tab = QWidget()
        layout = QHBoxLayout(tab)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)

        model_group = QGroupBox("Model Configuration")
        model_layout = QGridLayout()
        model_layout.addWidget(QLabel("Model File:"), 0, 0)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select .pth model file...")
        model_layout.addWidget(self.model_path_edit, 0, 1)
        btn_browse_model = QPushButton("Browse")
        btn_browse_model.clicked.connect(self._browse_model)
        model_layout.addWidget(btn_browse_model, 0, 2)

        model_layout.addWidget(QLabel("Data Directory:"), 1, 0)
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setPlaceholderText("Directory with scaler.joblib + processing_params.json")
        model_layout.addWidget(self.data_dir_edit, 1, 1)
        btn_browse_data = QPushButton("Browse")
        btn_browse_data.clicked.connect(self._browse_data_dir)
        model_layout.addWidget(btn_browse_data, 1, 2)

        btn_load = QPushButton("Load Model")
        btn_load.setMinimumHeight(36)
        btn_load.clicked.connect(self._load_model)
        btn_load.setObjectName("btnLoad")
        model_layout.addWidget(btn_load, 2, 0, 1, 3)

        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
        model_layout.addWidget(self.model_status_label, 3, 0, 1, 3)
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)

        input_group = QGroupBox("Input Parameters")
        input_layout = QGridLayout()
        self.feature_inputs = {}
        row = 0
        col = 0
        
        # 5个可控维度（可以调节）
        self.controllable_dims = {
            'energy', 'tau', 'pressure', 'length', 'diameter'
        }

        for fname in self.feature_names:
            label_text, _ = FEATURE_LABELS.get(fname, (fname, 1.0))
            lbl = QLabel(label_text + ":")
            spin = QDoubleSpinBox()
            spin.setRange(-1e10, 1e10)
            spin.setDecimals(6)
            spin.setSingleStep(0.1)
            spin.setMinimumWidth(140)
            
            # 计算生成的维度设置为只读
            if fname not in self.controllable_dims:
                spin.setReadOnly(True)
                # 使用CSS_STYLE中定义的样式
                spin.setStyleSheet("QDoubleSpinBox:read-only { background-color: #F3F4F6; color: #6B7280; }")
            else:
                # 为可控维度添加值变化信号
                spin.valueChanged.connect(self._on_controllable_param_changed)
            
            self.feature_inputs[fname] = spin
            input_layout.addWidget(lbl, row, col * 2)
            input_layout.addWidget(spin, row, col * 2 + 1)
            col += 1
            if col >= 2:
                col = 0
                row += 1
        input_group.setLayout(input_layout)

        scroll = QScrollArea()
        scroll.setWidget(input_group)
        scroll.setWidgetResizable(True)
        left_layout.addWidget(scroll, stretch=1)

        btn_predict = QPushButton("Run Prediction")
        btn_predict.setMinimumHeight(42)
        btn_predict.clicked.connect(self._run_prediction)
        btn_predict.setObjectName("btnPredict")
        left_layout.addWidget(btn_predict)

        btn_random = QPushButton("Random Parameters")
        btn_random.clicked.connect(self._random_parameters)
        left_layout.addWidget(btn_random)

        display_layout = QHBoxLayout()
        display_layout.addWidget(QLabel("Display:"))
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(["Normalized [0,1]", "Log Power (dB)", "Linear Power (W)"])
        self.display_mode_combo.currentIndexChanged.connect(self._on_display_mode_changed)
        display_layout.addWidget(self.display_mode_combo)
        left_layout.addLayout(display_layout)

        btn_add_compare = QPushButton("Add to Comparison")
        btn_add_compare.clicked.connect(self._add_to_comparison)
        left_layout.addWidget(btn_add_compare)

        btn_clear_compare = QPushButton("Clear Comparison")
        btn_clear_compare.clicked.connect(self._clear_comparison)
        left_layout.addWidget(btn_clear_compare)

        left_panel.setMaximumWidth(520)
        layout.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # 创建结果显示标签页
        result_tabs = QTabWidget()
        
        # 光谱图标签页
        spectrum_tab = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_tab)
        self.spectrum_canvas = SpectrumCanvas(spectrum_tab)
        self.spectrum_toolbar = NavigationToolbar(self.spectrum_canvas, spectrum_tab)
        spectrum_layout.addWidget(self.spectrum_toolbar)
        spectrum_layout.addWidget(self.spectrum_canvas, stretch=1)
        result_tabs.addTab(spectrum_tab, "Spectrum")
        
        # 脉冲时域图标签页
        pulse_tab = QWidget()
        pulse_layout = QVBoxLayout(pulse_tab)
        self.pulse_canvas = SpectrumCanvas(pulse_tab)
        self.pulse_toolbar = NavigationToolbar(self.pulse_canvas, pulse_tab)
        pulse_layout.addWidget(self.pulse_toolbar)
        pulse_layout.addWidget(self.pulse_canvas, stretch=1)
        result_tabs.addTab(pulse_tab, "Pulse")
        
        # 脉冲时域演化图标签页
        evolution_tab = QWidget()
        evolution_layout = QVBoxLayout(evolution_tab)
        self.evolution_canvas = SpectrumCanvas(evolution_tab)
        self.evolution_toolbar = NavigationToolbar(self.evolution_canvas, evolution_tab)
        evolution_layout.addWidget(self.evolution_toolbar)
        evolution_layout.addWidget(self.evolution_canvas, stretch=1)
        result_tabs.addTab(evolution_tab, "Evolution")
        
        # 脉冲频域演化图标签页
        spectral_evolution_tab = QWidget()
        spectral_evolution_layout = QVBoxLayout(spectral_evolution_tab)
        self.spectral_evolution_canvas = SpectrumCanvas(spectral_evolution_tab)
        self.spectral_evolution_toolbar = NavigationToolbar(self.spectral_evolution_canvas, spectral_evolution_tab)
        spectral_evolution_layout.addWidget(self.spectral_evolution_toolbar)
        spectral_evolution_layout.addWidget(self.spectral_evolution_canvas, stretch=1)
        result_tabs.addTab(spectral_evolution_tab, "Spectral Evolution")

        result_group = QGroupBox("Prediction Result")
        result_layout = QVBoxLayout()
        result_layout.addWidget(result_tabs, stretch=1)
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group, stretch=1)

        info_group = QGroupBox("Prediction Info")
        info_layout = QVBoxLayout()
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(['Property', 'Value'])
        self.info_table.horizontalHeader().setStretchLastSection(True)
        self.info_table.setMaximumHeight(180)
        info_layout.addWidget(self.info_table)
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)

        layout.addWidget(right_panel, stretch=1)

        self.tabs.addTab(tab, "Single Prediction")

    def _build_batch_tab(self):
        """构建"Batch Prediction"标签页。

        布局:
        - 顶部: 文件选择栏(CSV/NPY) + 批量预测按钮
        - 中部: matplotlib光谱对比图 + 导航工具栏
        - 底部: 结果汇总表(样本编号、峰值波长、峰值强度、均值)
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("CSV/NPY File:"))
        self.batch_file_edit = QLineEdit()
        self.batch_file_edit.setPlaceholderText("Select input file (CSV or NPY)...")
        top_layout.addWidget(self.batch_file_edit, stretch=1)
        btn_browse_batch = QPushButton("Browse")
        btn_browse_batch.clicked.connect(self._browse_batch_file)
        top_layout.addWidget(btn_browse_batch)
        btn_run_batch = QPushButton("Run Batch Prediction")
        btn_run_batch.clicked.connect(self._run_batch_prediction)
        btn_run_batch.setObjectName("btnPredict")
        top_layout.addWidget(btn_run_batch)
        layout.addLayout(top_layout)

        self.batch_canvas = SpectrumCanvas(tab, width=12, height=7)
        self.batch_toolbar = NavigationToolbar(self.batch_canvas, tab)
        layout.addWidget(self.batch_toolbar)
        layout.addWidget(self.batch_canvas, stretch=1)

        self.batch_result_table = QTableWidget()
        self.batch_result_table.setMaximumHeight(200)
        layout.addWidget(self.batch_result_table)

        self.tabs.addTab(tab, "Batch Prediction")

    def _build_performance_tab(self):
        """构建"Training Performance"标签页。

        布局:
        - 顶部: 加载训练日志按钮 + 评估指标面板(MSE/RMSE/MAE/R2)
        - 中部: 左右分栏的Loss曲线和R²曲线
        - 底部: 模型配置参数表
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top_layout = QHBoxLayout()
        btn_load_history = QPushButton("Load Training Log")
        btn_load_history.clicked.connect(self._load_training_history)
        top_layout.addWidget(btn_load_history)
        top_layout.addStretch()

        metrics_group = QGroupBox("Evaluation Metrics")
        metrics_layout = QGridLayout()
        self.metric_labels = {}
        metric_names = ['MSE', 'RMSE', 'MAE', 'R2']
        for i, name in enumerate(metric_names):
            metrics_layout.addWidget(QLabel(f"{name}:"), i, 0)
            val_label = QLabel("N/A")
            val_label.setFont(QFont("Consolas", 12, QFont.Bold))
            self.metric_labels[name] = val_label
            metrics_layout.addWidget(val_label, i, 1)
        metrics_group.setLayout(metrics_layout)
        top_layout.addWidget(metrics_group)
        layout.addLayout(top_layout)

        splitter = QSplitter(Qt.Horizontal)
        self.loss_canvas = SpectrumCanvas(splitter, width=6, height=5)
        self.r2_canvas = SpectrumCanvas(splitter, width=6, height=5)
        splitter.addWidget(self.loss_canvas)
        splitter.addWidget(self.r2_canvas)
        layout.addWidget(splitter, stretch=1)

        config_group = QGroupBox("Model Configuration")
        config_layout = QVBoxLayout()
        self.config_table = QTableWidget()
        self.config_table.setColumnCount(2)
        self.config_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.config_table.horizontalHeader().setStretchLastSection(True)
        self.config_table.setMaximumHeight(200)
        config_layout.addWidget(self.config_table)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        self.tabs.addTab(tab, "Training Performance")

    def _build_about_tab(self):
        """构建"About"标签页，展示使用说明和特征物理含义。"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        about_text = QLabel(
            "<h2>HCF Inverse Design — MLP Spectrum Predictor</h2>"
            "<p>Physics-guided inverse design tool for ultrafast nonlinear pulse "
            "propagation in gas-filled hollow-core fibers.</p>"
            "<h3>Workflow</h3>"
            "<ol>"
            "<li>Load a trained model (.pth) and its data directory (scaler + params)</li>"
            "<li>Enter physical parameters or load from file</li>"
            "<li>Run prediction to get the output spectrum</li>"
            "<li>Compare multiple predictions side by side</li>"
            "</ol>"
            "<h3>Feature Descriptions</h3>"
            "<table border='1' cellpadding='4' style='border-collapse:collapse'>"
            "<tr><th>Feature</th><th>Physical Meaning</th></tr>"
            "<tr><td>energy</td><td>Pulse energy (μJ)</td></tr>"
            "<tr><td>tau</td><td>Pulse width FWHM (fs)</td></tr>"
            "<tr><td>chirp</td><td>Dimensionless chirp parameter C</td></tr>"
            "<tr><td>lambda0</td><td>Center wavelength (nm)</td></tr>"
            "<tr><td>pressure</td><td>Gas pressure (bar)</td></tr>"
            "<tr><td>length</td><td>Fiber length (cm)</td></tr>"
            "<tr><td>diameter</td><td>Core diameter (μm)</td></tr>"
            "<tr><td>beta2</td><td>Group velocity dispersion</td></tr>"
            "<tr><td>gamma</td><td>Nonlinear coefficient</td></tr>"
            "<tr><td>N</td><td>Soliton order</td></tr>"
            "<tr><td>L0</td><td>Dispersion length</td></tr>"
            "<tr><td>gamma_K</td><td>Keldysh parameter</td></tr>"
            "<tr><td>P_ratio</td><td>Peak power / critical power</td></tr>"
            "<tr><td>Aeff</td><td>Effective mode area</td></tr>"
            "<tr><td>neff</td><td>Effective refractive index</td></tr>"
            "</table>"
        )
        about_text.setWordWrap(True)
        about_text.setTextFormat(Qt.RichText)
        layout.addWidget(about_text)
        self.tabs.addTab(tab, "About")

    def apply_styles(self):
        """应用全局QSS样式表，实现现代风格的UI主题。

        使用CSS_STYLE中定义的现代样式，替代硬编码样式。
        """
        # 应用全局CSS样式
        self.setStyleSheet(CSS_STYLE)

    def _browse_model(self):
        """打开文件对话框选择.pth模型文件，路径填入model_path_edit。"""
        path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Model (*.pth);;All Files (*)")
        if path:
            self.model_path_edit.setText(path)

    def _browse_data_dir(self):
        """打开目录对话框选择数据目录（含scaler.joblib和processing_params.json）。"""
        path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if path:
            self.data_dir_edit.setText(path)

    def _browse_batch_file(self):
        """打开文件对话框选择批量输入文件（.csv或.npy格式）。"""
        path, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "CSV (*.csv);;NumPy (*.npy);;All Files (*)")
        if path:
            self.batch_file_edit.setText(path)

    def _load_model(self):
        """加载模型、缩放器和配置文件的完整流程。

        加载顺序:
        1. 验证模型文件和数据目录路径非空
        2. 从processing_params.json读取特征列表和光谱参数
        3. 从scaler.joblib加载MinMaxScaler
        4. 从model_config.json读取模型维度配置
        5. 从evaluation_metrics.json读取测试集指标
        6. 实例化MLP模型并加载state_dict
        7. 根据特征列表重建输入控件
        8. 更新UI状态（状态标签、配置表、指标面板）

        异常处理: 任何步骤失败均弹出QMessageBox.critical提示，
        并将状态标签设为红色错误信息。
        """
        model_path = self.model_path_edit.text().strip()
        data_dir = self.data_dir_edit.text().strip()

        if not model_path or not data_dir:
            QMessageBox.warning(self, "Missing Info", "Please specify both model file and data directory.")
            return

        if not os.path.isfile(model_path):
            QMessageBox.critical(self, "Error", f"Model file not found:\n{model_path}")
            return

        if not os.path.isdir(data_dir):
            QMessageBox.critical(self, "Error", f"Data directory not found:\n{data_dir}")
            return

        try:
            params_path = os.path.join(data_dir, 'processing_params.json')
            if os.path.isfile(params_path):
                with open(params_path, 'r') as f:
                    self.processing_params = json.load(f)
                self.feature_names = self.processing_params.get('input_features', FEATURE_NAMES_13)
                self.log_spectrum_min = self.processing_params.get('log_spectrum_global_min', -15.0)
                self.log_spectrum_max = self.processing_params.get('log_spectrum_global_max', 6.0)
            else:
                QMessageBox.warning(self, "Warning", "processing_params.json not found, using default features.")
                self.feature_names = FEATURE_NAMES_13

            import joblib
            scaler_path = os.path.join(data_dir, 'scaler.joblib')
            if os.path.isfile(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    print(f"[GUI] Scaler loaded successfully from {scaler_path}")
                except Exception as scaler_err:
                    QMessageBox.warning(self, "Scaler Warning",
                        f"Failed to load scaler.joblib (scipy DLL error):\n{str(scaler_err)}\n\n"
                        "Will use direct [0,1] mapping mode instead.")
                    self.scaler = None
                    print(f"[GUI] Scaler load failed, using direct mapping: {scaler_err}")
            else:
                QMessageBox.warning(self, "Scaler Warning",
                    f"scaler.joblib not found in:\n{data_dir}\n\n"
                    "Will use direct [0,1] mapping mode.")
                self.scaler = None

            config_path = os.path.join(data_dir, '..', 'models', 'model_config.json')
            if not os.path.isfile(config_path):
                config_path = os.path.join(data_dir, 'model_config.json')
            if os.path.isfile(config_path):
                with open(config_path, 'r') as f:
                    self.model_config = json.load(f)

            metrics_path = os.path.join(data_dir, '..', 'models', 'evaluation_metrics.json')
            if not os.path.isfile(metrics_path):
                metrics_path = os.path.join(data_dir, 'evaluation_metrics.json')
            if os.path.isfile(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.evaluation_metrics = json.load(f)
                for name, val in self.evaluation_metrics.items():
                    if name in self.metric_labels:
                        self.metric_labels[name].setText(f"{val:.6f}")

            input_dim = len(self.feature_names)
            output_dim = self.processing_params.get('target_points', 500) if self.processing_params else 500
            if self.model_config:
                input_dim = self.model_config.get('input_dim', input_dim)
                output_dim = self.model_config.get('output_dim', output_dim)

            # 自动检测模型类型
            self.model_type = self._detect_model_type(model_path, data_dir)
            
            # 根据模型类型创建对应的模型实例
            if self.model_type == 'lstm':
                self.n_z_steps = self.model_config.get('n_z_steps', 20) if self.model_config else 20
                self.model = TemporalLSTM(input_dim, output_dim, n_z_steps=self.n_z_steps)
            elif self.model_type == 'temporal':
                self.n_z_steps = self.model_config.get('n_z_steps', 20) if self.model_config else 20
                self.model = TemporalMLP(input_dim, output_dim, n_z_steps=self.n_z_steps)
            else:
                self.model = MLP(input_dim, output_dim)
            
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # 为时序模型准备z_positions
            if self.model_type in ('temporal', 'lstm'):
                # 从processing_params或默认配置生成z_positions
                fiber_length = 100.0  # cm, 默认值
                if 'length' in self.feature_inputs:
                    fiber_length = self.feature_inputs['length'].value()
                self.z_positions = torch.linspace(0, fiber_length, self.n_z_steps).unsqueeze(0).float()

            self._rebuild_feature_inputs()

            if self.model_config:
                self.config_table.setRowCount(len(self.model_config))
                for i, (k, v) in enumerate(self.model_config.items()):
                    self.config_table.setItem(i, 0, QTableWidgetItem(str(k)))
                    self.config_table.setItem(i, 1, QTableWidgetItem(str(v)))

            self.model_status_label.setText(
                f"Model loaded: {self.model_type.upper()}, input_dim={input_dim}, output_dim={output_dim}")
            self.model_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            scaler_mode = "Scaler" if self.scaler is not None else "Direct [0,1] mapping"
            self.statusBar().showMessage(
                f"Model loaded — {self.model_type.upper()} {input_dim} features → {output_dim} points ({scaler_mode})")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{str(e)}")
            import traceback
            traceback.print_exc()
            self.model_status_label.setText(f"Load failed: {str(e)}")
            self.model_status_label.setStyleSheet("color: #F44336; font-weight: bold;")

    def _rebuild_feature_inputs(self):
        """根据加载模型的特征列表重建输入参数控件。

        当模型加载后特征维度可能变化(13维→15维)时调用。
        清除旧的QDoubleSpinBox控件，按新的feature_names创建控件。
        控件以2列网格排列，每个控件显示对应的物理量标签和换算单位。
        """
        input_group = None
        for widget in self.findChildren(QGroupBox):
            if widget.title() == "Input Parameters":
                input_group = widget
                break
        if input_group is None:
            return

        old_layout = input_group.layout()
        while old_layout.count():
            item = old_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.feature_inputs = {}
        row, col = 0, 0

        # 5个可控维度（可以调节）
        controllable_dims = {
            'energy', 'tau', 'pressure', 'length', 'diameter'
        }

        for fname in self.feature_names:
            label_text, _ = FEATURE_LABELS.get(fname, (fname, 1.0))
            lbl = QLabel(label_text + ":")
            spin = QDoubleSpinBox()
            spin.setRange(-1e10, 1e10)
            spin.setDecimals(6)
            spin.setSingleStep(0.1)
            spin.setMinimumWidth(140)
            
            # 计算生成的维度设置为只读
            if fname not in controllable_dims:
                spin.setReadOnly(True)
                spin.setStyleSheet("QDoubleSpinBox:read-only { background-color: #f0f0f0; color: #666666; }")
            else:
                # 为可控维度添加值变化信号
                spin.valueChanged.connect(self._on_controllable_param_changed)
            
            self.feature_inputs[fname] = spin
            old_layout.addWidget(lbl, row, col * 2)
            old_layout.addWidget(spin, row, col * 2 + 1)
            col += 1
            if col >= 2:
                col = 0
                row += 1

    def _run_prediction(self):
        """执行单次光谱预测。

        数据流（修复后）:
        1. 从feature_inputs读取GUI显示值
        2. 使用PARAM_SPACE_RANGES将显示值线性映射到[0,1]:
           normalized = (gui_val - range_min) / (range_max - range_min)
        3. 将[0,1]特征直接传入InferenceWorker（绕过scaler）
        4. 模型前向传播得到预测光谱

        此修复解决了processed_data中X_train已预归一化导致scaler双重变换的问题。
        """
        if self.model is None:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return

        try:
            features_scaled = []
            for fname in self.feature_names:
                if fname in self.feature_inputs:
                    gui_val = self.feature_inputs[fname].value()
                    if fname in PARAM_SPACE_RANGES:
                        vmin, vmax = PARAM_SPACE_RANGES[fname]
                        normalized = (gui_val - vmin) / (vmax - vmin)
                        normalized = max(0.0, min(1.0, normalized))
                    else:
                        normalized = gui_val
                    features_scaled.append(normalized)
                else:
                    features_scaled.append(0.5)

            self.statusBar().showMessage("Running prediction...")
            
            # 更新时序模型的z_positions（基于当前光纤长度）
            z_pos = None
            if self.model_type in ('temporal', 'lstm') and self.z_positions is not None:
                fiber_length = 100.0
                if 'length' in self.feature_inputs:
                    fiber_length = self.feature_inputs['length'].value()
                z_pos = torch.linspace(0, fiber_length, self.n_z_steps).unsqueeze(0).float()
            
            self.worker = InferenceWorker(
                self.model, self.scaler, features_scaled, self.feature_names,
                model_type=self.model_type, z_positions=z_pos
            )
            self.worker.finished.connect(self._on_prediction_done)
            self.worker.error.connect(self._on_prediction_error)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed:\n{str(e)}")

    def _on_prediction_done(self, prediction, info):
        """推理完成回调：根据当前显示模式更新光谱图和脉冲时域图。"""
        self.last_prediction = prediction
        self.last_info = info

        self._update_spectrum_display()
        self._update_pulse_display()
        
        # 如果时序模型返回了时序数据，更新演化图
        if info.get('temporal_data') is not None:
            self._update_temporal_evolution_display(info)

    def _update_pulse_display(self):
        """更新脉冲时域图和演化图显示。"""
        if self.last_prediction is None:
            return

        # 生成模拟的脉冲时域数据
        # 基于tau参数计算脉冲宽度
        tau = self.feature_inputs.get('tau', None)
        if tau:
            pulse_width = tau.value()  # fs
        else:
            pulse_width = 30  # 默认30fs

        # 生成时间轴
        times = np.linspace(-200, 200, 1000)  # fs
        
        # 生成高斯脉冲
        pulse = np.exp(-(times/pulse_width)**2)
        
        # 绘制脉冲时域图
        self.pulse_canvas.plot_pulse(
            pulse, times, 
            title="Pulse Temporal Profile",
            ylabel="Normalized Intensity"
        )
        
        # 生成脉冲时域演化图
        self._update_evolution_display()
        # 生成脉冲频域演化图
        self._update_spectral_evolution_display()

    def _update_evolution_display(self):
        """更新脉冲时域演化图显示。
        
        基于真实的物理模型（色散、非线性、孤子动力学），根据输入参数动态计算演化图，
        确保不同参数组合产生明显不同的演化形态。
        """
        # 读取输入参数
        tau = self.feature_inputs.get('tau', None)
        energy = self.feature_inputs.get('energy', None)
        pressure = self.feature_inputs.get('pressure', None)
        length = self.feature_inputs.get('length', None)
        diameter = self.feature_inputs.get('diameter', None)
        
        # 读取计算参数
        beta2_input = self.feature_inputs.get('beta2', None)
        gamma_input = self.feature_inputs.get('gamma', None)
        N_input = self.feature_inputs.get('N', None)
        
        # 获取参数值或默认值
        if tau:
            tau_val = tau.value()  # fs (FWHM)
        else:
            tau_val = 30
            
        if energy:
            E_val = energy.value()  # μJ
        else:
            E_val = 1.5
            
        if pressure:
            P_bar = pressure.value()
        else:
            P_bar = 10
            
        if length:
            fiber_length = length.value()  # cm
        else:
            fiber_length = 10
            
        if diameter:
            D_um = diameter.value()
        else:
            D_um = 20
        
        # 获取beta2和gamma（如果可用）
        if beta2_input:
            beta2_fs2mm = beta2_input.value()  # fs²/mm
        else:
            # 基于气压和直径估算beta2
            radius_um = D_um / 2
            beta2_fs2mm = -1.0 * (10/radius_um)**2 * P_bar
            
        if gamma_input:
            gamma_1Wm = gamma_input.value()
        else:
            # 估算gamma
            Aeff_m2 = 0.48 * np.pi * (D_um/2*1e-6)**2
            n2 = 1e-23 * P_bar
            omega0 = 2*np.pi*C_LIGHT / 800e-9
            gamma_1Wm = omega0 * n2 / (C_LIGHT * Aeff_m2)
            
        if N_input:
            N_soliton = N_input.value()
        else:
            N_soliton = 3.0
        
        # ===== 物理单位转换 =====
        T0 = tau_val / 1.765  # 高斯脉宽 (FWHM → 1/e半宽) [fs]
        T0_s = T0 * 1e-15     # s
        
        # beta2: fs²/mm → s²/m
        beta2_s2m = beta2_fs2mm * 1e-27  # 1e-15² / 1e-3 = 1e-27
        
        # 能量: μJ → J
        E_J = E_val * 1e-6
        
        # 峰值功率: P0 = E / (T0 * sqrt(π)) [W]
        P0_W = E_J / (T0_s * np.sqrt(np.pi))
        
        # 光纤长度: cm → m
        L_m = fiber_length * 1e-2
        
        # ===== 计算特征长度 =====
        abs_beta2 = abs(beta2_s2m) + 1e-40
        L_D = T0_s**2 / abs_beta2  # 色散长度 [m]
        
        if abs(gamma_1Wm) > 1e-20:
            L_NL = 1 / (abs(gamma_1Wm) * P0_W)  # 非线性长度 [m]
        else:
            L_NL = 100.0
        
        # 孤子阶数
        N_calc = np.sqrt(L_D / L_NL) if L_NL > 0 else 1.0
        N_calc = min(N_calc, 20)  # 限制最大值防止数值爆炸
        
        # ===== 生成演化数据 =====
        n_z = 150
        n_t = 300
        
        z_cm = np.linspace(0, fiber_length, n_z)
        t_fs = np.linspace(-200, 200, n_t)
        
        evolution_data = np.zeros((n_z, n_t))
        
        for i, z_c in enumerate(z_cm):
            z_m = z_c * 1e-2  # cm → m
            xi = z_m / L_D    # 归一化传播距离
            
            # 根据N值选择不同的演化模型
            if N_calc < 0.5:
                # 线性色散区域：高斯脉冲展宽
                sigma = 1 + xi
                pulse = np.exp(-(t_fs/T0)**2 / sigma**2)
                
            elif N_calc < 1.5:
                # 基态孤子区域：形状保持但有微小变化
                phase_modulation = 0.5 * xi * np.sign(beta2_fs2mm)
                pulse = np.exp(-(t_fs/T0)**2) * np.cos(phase_modulation * (t_fs/T0)**2)
                pulse = np.abs(pulse)
                
            elif N_calc < 4:
                # 高阶孤子/分裂区域：周期性压缩-展宽
                period = np.pi / 2  # 孤子周期
                z_period = period * L_D
                
                # 模拟周期性压缩
                phase = (z_m % z_period) / z_period
                compression = 1 - 0.6 * np.sin(phase * np.pi) * np.exp(-xi/3)
                compression = max(compression, 0.3)
                
                t_shift = 0.3 * T0 * np.sin(2*np.pi*z_m/z_period) * np.exp(-xi/5)
                pulse = np.exp(-((t_fs - t_shift)/(T0*compression))**2)
                
                # 自相位调制导致的频谱边瓣
                spm_strength = 0.3 * N_calc * xi
                if spm_strength > 0.01:
                    sideband = 0.15 * np.exp(-((np.abs(t_fs)-T0*2)/(T0*0.8))**2) * np.sin(spm_strength * (t_fs/T0)**2)
                    pulse += np.abs(sideband)
                    
            else:
                # 强非线性区域：复杂演化
                # 主脉冲压缩后展宽
                phase = (z_m / L_D) % (2*np.pi)
                compression = 1 - 0.7 * np.exp(-phase**2/2)
                compression = max(compression, 0.2)
                
                pulse_main = np.exp(-((t_fs)/(T0*compression))**2)
                
                # 多峰结构（超连续谱前兆）
                if xi > 0.5:
                    n_peaks = int(N_calc // 2)
                    for p in range(min(n_peaks, 5)):
                        peak_pos = T0 * (2 + p*1.5) * (-1)**p
                        peak_width = T0 * (0.5 + 0.3*p)
                        peak_amp = 0.25 * np.exp(-xi/3) / (p+1)
                        pulse_main += peak_amp * np.exp(-((t_fs - peak_pos)/peak_width)**2)
                
                pulse = pulse_main
            
            evolution_data[i, :] = pulse
        
        # 归一化到[0,1]
        evol_min, evol_max = evolution_data.min(), evolution_data.max()
        if evol_max > evol_min:
            evolution_data = (evolution_data - evol_min) / (evol_max - evol_min)
        else:
            evolution_data = np.zeros_like(evolution_data)
        
        # 绘制演化图
        self.evolution_canvas.plot_evolution(
            evolution_data, t_fs, z_cm, 
            title=f"Temporal Evolution (fs vs cm) | N={N_calc:.1f}, β₂={beta2_fs2mm:.1f}fs²/mm",
            xlabel="Time (fs)",
            ylabel="Propagation Distance (cm)",
            cmap='plasma'
        )

    def _update_spectral_evolution_display(self):
        """更新脉冲频域演化图显示。
        
        基于真实的物理模型（色散、非线性、孤子动力学），根据输入参数动态计算频域演化图，
        确保不同参数组合产生明显不同的演化形态。
        """
        # 读取输入参数
        tau = self.feature_inputs.get('tau', None)
        energy = self.feature_inputs.get('energy', None)
        pressure = self.feature_inputs.get('pressure', None)
        length = self.feature_inputs.get('length', None)
        diameter = self.feature_inputs.get('diameter', None)
        
        # 读取计算参数
        beta2_input = self.feature_inputs.get('beta2', None)
        gamma_input = self.feature_inputs.get('gamma', None)
        N_input = self.feature_inputs.get('N', None)
        
        # 获取参数值或默认值
        if tau:
            tau_val = tau.value()  # fs (FWHM)
        else:
            tau_val = 30
            
        if energy:
            E_val = energy.value()  # μJ
        else:
            E_val = 1.5
            
        if pressure:
            P_bar = pressure.value()
        else:
            P_bar = 10
            
        if length:
            fiber_length = length.value()  # cm
        else:
            fiber_length = 10
            
        if diameter:
            D_um = diameter.value()
        else:
            D_um = 20
        
        # 获取beta2和gamma（如果可用）
        if beta2_input:
            beta2_fs2mm = beta2_input.value()  # fs²/mm
        else:
            # 基于气压和直径估算beta2
            radius_um = D_um / 2
            beta2_fs2mm = -1.0 * (10/radius_um)**2 * P_bar
            
        if gamma_input:
            gamma_1Wm = gamma_input.value()
        else:
            # 估算gamma
            Aeff_m2 = 0.48 * np.pi * (D_um/2*1e-6)**2
            n2 = 1e-23 * P_bar
            omega0 = 2*np.pi*C_LIGHT / 800e-9
            gamma_1Wm = omega0 * n2 / (C_LIGHT * Aeff_m2)
            
        if N_input:
            N_soliton = N_input.value()
        else:
            N_soliton = 3.0
        
        # ===== 物理单位转换 =====
        T0 = tau_val / 1.765  # 高斯脉宽 (FWHM → 1/e半宽) [fs]
        T0_s = T0 * 1e-15     # s
        
        # beta2: fs²/mm → s²/m
        beta2_s2m = beta2_fs2mm * 1e-27  # 1e-15² / 1e-3 = 1e-27
        
        # 能量: μJ → J
        E_J = E_val * 1e-6
        
        # 峰值功率: P0 = E / (T0 * sqrt(π)) [W]
        P0_W = E_J / (T0_s * np.sqrt(np.pi))
        
        # 光纤长度: cm → m
        L_m = fiber_length * 1e-2
        
        # ===== 计算特征长度 =====
        abs_beta2 = abs(beta2_s2m) + 1e-40
        L_D = T0_s**2 / abs_beta2  # 色散长度 [m]
        
        if abs(gamma_1Wm) > 1e-20:
            L_NL = 1 / (abs(gamma_1Wm) * P0_W)  # 非线性长度 [m]
        else:
            L_NL = 100.0
        
        # 孤子阶数
        N_calc = np.sqrt(L_D / L_NL) if L_NL > 0 else 1.0
        N_calc = min(N_calc, 20)  # 限制最大值防止数值爆炸
        
        # ===== 生成频域演化数据 =====
        n_z = 150
        n_t = 300
        n_w = 300
        
        z_cm = np.linspace(0, fiber_length, n_z)
        t_fs = np.linspace(-200, 200, n_t)
        
        # 计算频率轴 (THz)
        dt = t_fs[1] - t_fs[0]
        f_Hz = np.fft.fftfreq(n_t, d=dt*1e-15)  # 频率 (Hz)
        f_Hz = np.fft.fftshift(f_Hz)  # 移到中心
        
        # 计算波长轴 (nm)
        c_nmps = C_LIGHT * 1e9  # 光速 (nm/s)
        lambda_nm = c_nmps / np.abs(f_Hz)  # 波长 (nm)
        
        # 限制波长范围到 200-2000 nm
        valid_idx = (lambda_nm >= 200) & (lambda_nm <= 2000)
        lambda_nm = lambda_nm[valid_idx]
        # 对波长轴进行排序，确保从短到长
        sorted_idx = np.argsort(lambda_nm)
        lambda_nm = lambda_nm[sorted_idx]
        
        spectral_evolution_data = np.zeros((n_z, len(lambda_nm)))
        
        for i, z_c in enumerate(z_cm):
            z_m = z_c * 1e-2  # cm → m
            xi = z_m / L_D    # 归一化传播距离
            
            # 根据N值选择不同的演化模型（与时域相同）
            if N_calc < 0.5:
                # 线性色散区域：高斯脉冲展宽
                sigma = 1 + xi
                pulse = np.exp(-(t_fs/T0)**2 / sigma**2)
                
            elif N_calc < 1.5:
                # 基态孤子区域：形状保持但有微小变化
                phase_modulation = 0.5 * xi * np.sign(beta2_fs2mm)
                pulse = np.exp(-(t_fs/T0)**2) * np.cos(phase_modulation * (t_fs/T0)**2)
                pulse = np.abs(pulse)
                
            elif N_calc < 4:
                # 高阶孤子/分裂区域：周期性压缩-展宽
                period = np.pi / 2  # 孤子周期
                z_period = period * L_D
                
                # 模拟周期性压缩
                phase = (z_m % z_period) / z_period
                compression = 1 - 0.6 * np.sin(phase * np.pi) * np.exp(-xi/3)
                compression = max(compression, 0.3)
                
                t_shift = 0.3 * T0 * np.sin(2*np.pi*z_m/z_period) * np.exp(-xi/5)
                pulse = np.exp(-((t_fs - t_shift)/(T0*compression))**2)
                
                # 自相位调制导致的频谱边瓣
                spm_strength = 0.3 * N_calc * xi
                if spm_strength > 0.01:
                    sideband = 0.15 * np.exp(-((np.abs(t_fs)-T0*2)/(T0*0.8))**2) * np.sin(spm_strength * (t_fs/T0)**2)
                    pulse += np.abs(sideband)
                    
            else:
                # 强非线性区域：复杂演化
                # 主脉冲压缩后展宽
                phase = (z_m / L_D) % (2*np.pi)
                compression = 1 - 0.7 * np.exp(-phase**2/2)
                compression = max(compression, 0.2)
                
                pulse_main = np.exp(-((t_fs)/(T0*compression))**2)
                
                # 多峰结构（超连续谱前兆）
                if xi > 0.5:
                    n_peaks = int(N_calc // 2)
                    for p in range(min(n_peaks, 5)):
                        peak_pos = T0 * (2 + p*1.5) * (-1)**p
                        peak_width = T0 * (0.5 + 0.3*p)
                        peak_amp = 0.25 * np.exp(-xi/3) / (p+1)
                        pulse_main += peak_amp * np.exp(-((t_fs - peak_pos)/peak_width)**2)
                
                pulse = pulse_main
            
            # 傅里叶变换到频域
            spectrum = np.fft.fft(pulse)
            spectrum = np.fft.fftshift(spectrum)
            spectrum = np.abs(spectrum)**2  # 功率谱
            
            # 应用波长范围限制
            spectrum = spectrum[valid_idx]
            # 应用波长排序
            spectrum = spectrum[sorted_idx]
            
            spectral_evolution_data[i, :] = spectrum
        
        # 归一化到[0,1]
        spec_min, spec_max = spectral_evolution_data.min(), spectral_evolution_data.max()
        if spec_max > spec_min:
            spectral_evolution_data = (spectral_evolution_data - spec_min) / (spec_max - spec_min)
        else:
            spectral_evolution_data = np.zeros_like(spectral_evolution_data)
        
        # 绘制频域演化图
        self.spectral_evolution_canvas.plot_evolution(
            spectral_evolution_data, lambda_nm, z_cm, 
            title=f"Spectral Evolution (nm vs cm) | N={N_calc:.1f}, β₂={beta2_fs2mm:.1f}fs²/mm",
            xlabel="Wavelength (nm)",
            ylabel="Propagation Distance (cm)",
            cmap='plasma'
        )

    def _denormalize_spectrum(self, normalized_spectrum):
        """将归一化[0,1]光谱反归一化为log10(power)空间。

        反归一化公式（近似，基于全局统计量）：
        log10(power) = normalized * (log_max - log_min) + log_min

        Parameters
        ----------
        normalized_spectrum : np.ndarray
            归一化光谱，值域[0,1]。

        Returns
        -------
        log_power : np.ndarray
            log10(power) 值，典型范围[-15, 6]。
        """
        return normalized_spectrum * (self.log_spectrum_max - self.log_spectrum_min) + self.log_spectrum_min

    def _get_display_data(self, normalized_spectrum):
        """根据当前显示模式转换光谱数据。

        Returns
        -------
        spectrum : np.ndarray
            转换后的光谱数据。
        ylabel : str
            y轴标签。
        title_prefix : str
            图表标题前缀。
        """
        if self.display_mode == 'log_power':
            log_power = self._denormalize_spectrum(normalized_spectrum)
            return log_power, 'Log Power (log₁₀|Eω|²)', 'Log Power Spectrum'
        elif self.display_mode == 'linear_power':
            log_power = self._denormalize_spectrum(normalized_spectrum)
            linear_power = np.power(10.0, log_power)
            return linear_power, 'Power (|Eω|²)', 'Linear Power Spectrum'
        else:
            return normalized_spectrum, 'Normalized Log Power', 'Predicted Output Spectrum'

    def _update_spectrum_display(self):
        """根据当前显示模式重新绘制光谱图和信息表。"""
        if self.last_prediction is None:
            return

        wavelengths = np.linspace(200, 2500, len(self.last_prediction))
        spectrum, ylabel, title = self._get_display_data(self.last_prediction)

        self.spectrum_canvas.plot_spectrum(
            spectrum, wavelengths, title=title, ylabel=ylabel
        )

        peak_idx = np.argmax(spectrum)
        peak_wl = wavelengths[peak_idx]
        peak_val = spectrum[peak_idx]

        if self.display_mode == 'log_power':
            peak_str = f"{peak_val:.2f}"
            range_str = f"[{spectrum.min():.2f}, {spectrum.max():.2f}]"
            nonzero_pct = np.sum(spectrum > self.log_spectrum_min + 1) / len(spectrum) * 100
        elif self.display_mode == 'linear_power':
            peak_str = f"{peak_val:.3e}"
            range_str = f"[{spectrum.min():.3e}, {spectrum.max():.3e}]"
            nonzero_pct = np.sum(spectrum > 1e-10) / len(spectrum) * 100
        else:
            peak_str = f"{peak_val:.4f}"
            range_str = f"[{spectrum.min():.4f}, {spectrum.max():.4f}]"
            nonzero_pct = np.sum(spectrum > 0.01) / len(spectrum) * 100

        self.info_table.setRowCount(5)
        items = [
            ("Peak Wavelength", f"{peak_wl:.1f} nm"),
            ("Peak Value", peak_str),
            ("Spectral Points Active", f"{nonzero_pct:.1f}%"),
            ("Spectrum Range", range_str),
            ("Display Mode", self.display_mode.replace('_', ' ').title()),
        ]
        for i, (prop, val) in enumerate(items):
            self.info_table.setItem(i, 0, QTableWidgetItem(prop))
            self.info_table.setItem(i, 1, QTableWidgetItem(val))

        self.statusBar().showMessage(f"Prediction complete — Peak at {peak_wl:.0f} nm")

    def _on_display_mode_changed(self, index):
        """显示模式切换回调：更新display_mode并重新绘制光谱。"""
        modes = ['normalized', 'log_power', 'linear_power']
        if 0 <= index < len(modes):
            self.display_mode = modes[index]
            self._update_spectrum_display()

    def _on_prediction_error(self, err_msg):
        """推理出错回调：弹出错误对话框。

        Parameters
        ----------
        err_msg : str
            InferenceWorker.run()中捕获的异常消息。
        """
        QMessageBox.critical(self, "Prediction Error", err_msg)
        self.statusBar().showMessage("Prediction failed")

    def _add_to_comparison(self):
        """将当前预测光谱添加到对比列表。

        从feature_inputs中提取energy/tau/pressure等关键参数
        作为图例标签，最多显示3个参数以保持标签简洁。
        添加后自动更新对比图。
        """
        if self.last_prediction is None:
            QMessageBox.warning(self, "No Data", "Run a prediction first.")
            return
        self.comparison_spectra.append(self.last_prediction.copy())
        label_parts = []
        for fname in ['energy', 'tau', 'pressure', 'length', 'diameter']:
            if fname in self.feature_inputs:
                label_parts.append(f"{fname}={self.feature_inputs[fname].value():.2f}")
        label = ", ".join(label_parts[:3]) if label_parts else f"Spec {len(self.comparison_spectra)}"
        self.comparison_labels.append(label)

        wavelengths = np.linspace(200, 2500, len(self.last_prediction))
        self.spectrum_canvas.plot_comparison(
            self.comparison_spectra, self.comparison_labels, wavelengths,
            title=f"Spectrum Comparison ({len(self.comparison_spectra)} spectra)"
        )
        self.statusBar().showMessage(f"Added to comparison ({len(self.comparison_spectra)} spectra)")

    def _clear_comparison(self):
        """清空对比列表，恢复显示最近一次预测的单条光谱。"""
        self.comparison_spectra = []
        self.comparison_labels = []
        if self.last_prediction is not None:
            self._update_spectrum_display()
        self.statusBar().showMessage("Comparison cleared")

    def _random_parameters(self):
        """生成随机参数，仅对指定的5个可控维度进行随机化。

        可控参数范围（GUI显示值）：
        - 脉冲能量: 0.3-3 μJ
        - 脉冲宽度: 5-50 fs
        - 气压: 0.5-50 bar
        - 光纤长度: 3-50 cm（对应0.03-0.5 m）
        - 纤芯直径: 10-50 μm

        其余7个计算生成的维度会自动更新。
        """
        import random
        
        # 仅对这5个可控维度进行随机化
        controllable_params = {
            'energy': (0.3, 3.0),          # μJ
            'tau': (5.0, 50.0),             # fs
            'pressure': (0.5, 50.0),        # bar
            'length': (3.0, 50.0),          # cm
            'diameter': (10.0, 50.0),       # μm
        }

        for fname, (min_val, max_val) in controllable_params.items():
            if fname in self.feature_inputs:
                # 生成均匀分布的随机值
                rand_val = random.uniform(min_val, max_val)
                # 保留2位小数
                spin = self.feature_inputs[fname]
                spin.blockSignals(True)
                spin.setValue(round(rand_val, 2))
                spin.blockSignals(False)

        # 自动计算其他维度
        self._update_calculated_dimensions()

        self.statusBar().showMessage("Random parameters generated for controllable dimensions")

    def _on_controllable_param_changed(self):
        """可控参数值变化时的回调，自动更新计算生成的维度以及脉冲演化图。"""
        self._update_calculated_dimensions()
        
        # 如果已有预测结果，更新脉冲演化图显示
        if self.last_prediction is not None:
            self._update_evolution_display()
            self._update_spectral_evolution_display()

    def _update_calculated_dimensions(self):
        """根据5个可控参数自动计算并更新7个生成维度。

        除了有效折射率(neff)使用回归模型外，其他参数使用物理公式计算。
        """
        try:
            # 首先使用物理公式计算所有参数
            self._update_calculated_dimensions_fallback()
            
            # 然后尝试为neff使用回归模型计算
            if self.processing_params is not None and 'feature_regression' in self.processing_params:
                regression = self.processing_params['feature_regression']
                if 'neff' in regression:
                    # 获取可控参数
                    controllable_names = self.processing_params.get(
                        'controllable_features',
                        ['energy', 'tau', 'pressure', 'length', 'diameter']
                    )
                    
                    # Step 1: Map controllable GUI values to [0,1]
                    ctrl_normalized = []
                    for fname in controllable_names:
                        if fname in self.feature_inputs and fname in PARAM_SPACE_RANGES:
                            gui_val = self.feature_inputs[fname].value()
                            vmin, vmax = PARAM_SPACE_RANGES[fname]
                            norm_val = (gui_val - vmin) / (vmax - vmin)
                            norm_val = max(0.0, min(1.0, norm_val))
                            ctrl_normalized.append(norm_val)
                        else:
                            ctrl_normalized.append(0.5)
                    
                    # Step 2: Predict neff using regression
                    model_info = regression['neff']
                    coeffs = model_info['coeffs']
                    
                    if model_info['model'] == 'linear':
                        # Linear: y = bias + w1*x1 + w2*x2 + ...
                        x_vec = [1.0] + ctrl_normalized
                        if len(x_vec) == len(coeffs):
                            predicted = sum(c * x for c, x in zip(coeffs, x_vec))
                        else:
                            return
                    else:
                        # Polynomial: includes interaction and quadratic terms
                        e, t, p, l, d = ctrl_normalized
                        x_vec = [1.0, e, t, p, l, d,
                                 e * p, e * d, p * d,
                                 e * e, p * p, d * d]
                        if len(x_vec) == len(coeffs):
                            predicted = sum(c * x for c, x in zip(coeffs, x_vec))
                        else:
                            return
                    
                    # Clamp to [0, 1]
                    predicted = max(0.0, min(1.0, predicted))
                    
                    # Step 3: Map [0,1] back to GUI display value
                    if 'neff' in PARAM_SPACE_RANGES:
                        vmin, vmax = PARAM_SPACE_RANGES['neff']
                        gui_val = vmin + predicted * (vmax - vmin)
                    else:
                        # 使用scaler.txt中的范围
                        vmin = 0.997032593
                        vmax = 1.01295699
                        gui_val = vmin + predicted * (vmax - vmin)
                    
                    # 更新neff值
                    if 'neff' in self.feature_inputs:
                        spin = self.feature_inputs['neff']
                        spin.blockSignals(True)
                        spin.setValue(round(gui_val, 6))
                        spin.blockSignals(False)
                        print(f"[Regression] neff={gui_val:.6f}")
        
        except Exception as e:
            print(f"Error in update_calculated_dimensions: {e}")
            # 出错时仍使用物理公式计算
            self._update_calculated_dimensions_fallback()

    def _update_calculated_dimensions_fallback(self):
        """使用物理公式计算各维度参数。

        基于气体填充空芯光纤的物理模型，从5个可控参数计算
        8个物理参数。公式参考Luna.jl仿真中的标准定义。

        计算公式:
        - beta2: GVD = -λ³/(2πc²) · (d²n/dλ²) [fs²/mm]
        - gamma: 非线性系数 = n2·ω0/(c·Aeff) [1/(W·m)]
        - N: 孤子阶数 = √(L0/LNL) = √(γ·P0·T0²/|β2|)
        - L0: 色散长度 = T0²/|β2| [m]
        - gamma_K: Keldysh参数
        - P_ratio: 峰值功率/临界功率
        - Aeff: 有效模面积 [μm²]
        - neff: 有效折射率
        """
        try:
            # 读取5个可控参数 (GUI显示值)
            energy = self.feature_inputs['energy'].value()      # μJ
            tau = self.feature_inputs['tau'].value()            # fs
            pressure = self.feature_inputs['pressure'].value()  # bar
            length = self.feature_inputs['length'].value()      # cm
            diameter = self.feature_inputs['diameter'].value()  # μm

            # 单位转换
            energy_J = energy * 1e-6          # J
            tau_s = tau * 1e-15               # s
            pressure_pa = pressure * 1e5      # Pa
            length_m = length * 1e-2          # m
            radius_um = diameter / 2          # μm
            radius_m = radius_um * 1e-6       # m

            # 固定物理常数
            lambda0 = 800e-9                  # m, 中心波长
            omega0 = 2 * np.pi * C_LIGHT / lambda0  # rad/s
            n2 = 1e-23 * pressure             # m²/W, 非线性折射率 (与气压成正比)
            n_gas = 1 + 2.7e-4 * pressure / 1e5  # 气体折射率

            # 1. 有效模面积 Aeff [μm²]
            Aeff_m2 = 0.48 * np.pi * radius_m**2
            Aeff_um2 = Aeff_m2 * 1e12

            # 2. 有效折射率 neff
            neff = n_gas

            # 3. GVD beta2 [fs²/mm]
            # 简化模型: beta2与气压和直径相关
            # 典型值: 在800nm附近, 空芯光纤的beta2约为-1 to -1000 fs²/mm
            beta2_base = -1.0  # fs²/mm at 1 bar, 10 μm diameter
            beta2 = beta2_base * (10.0 / radius_um)**2 * (pressure / 1.0)
            # 添加波长依赖性 (在800nm附近)
            beta2 = beta2 * (800e-9 / lambda0)**3

            # 4. 非线性系数 gamma [1/(W·m)]
            gamma = omega0 * n2 / (C_LIGHT * Aeff_m2)

            # 5. 峰值功率 P0 [W]
            # P0 = E / (tau * sqrt(pi/ln(2)/2)) for Gaussian pulse
            # 简化: P0 ≈ E / (0.94 * tau) for sech pulse
            T0 = tau_s / 1.76  # sech pulse width
            P0 = energy_J / (2 * T0)  # 简化计算

            # 6. 色散长度 L0 [m]
            beta2_s2_m = beta2 * 1e-30 * 1e3  # 转换为 s²/m
            L0 = T0**2 / abs(beta2_s2_m) if abs(beta2_s2_m) > 0 else 1e10

            # 7. 非线性长度 LNL [m]
            LNL = 1.0 / (gamma * P0) if gamma * P0 > 0 else 1e10

            # 8. 孤子阶数 N
            N = np.sqrt(L0 / LNL) if LNL > 0 else 0.0

            # 9. Keldysh参数 gamma_K
            # gamma_K = ω0 * sqrt(m * c * ε0 * n / (e * E))
            # 简化: gamma_K ≈ 1.5 for typical HCF parameters
            gamma_K = 1.5 * (10.0 / radius_um) * (pressure / 1.0)**0.5

            # 10. 临界功率 P_critical [W]
            # P_cr = 3.77 * λ² / (8π * n * n2)
            P_cr = 3.77 * lambda0**2 / (8 * np.pi * n_gas * n2)

            # 11. 峰值功率/临界功率比值
            P_ratio = P0 / P_cr if P_cr > 0 else 0.0

            # 更新GUI显示值
            calculated_dims = {
                'beta2': beta2,
                'gamma': gamma,
                'N': N,
                'L0': L0 * 100,  # 转换为 cm
                'gamma_K': gamma_K,
                'P_ratio': P_ratio,
                'Aeff': Aeff_um2,
                'neff': neff,
            }

            for fname, value in calculated_dims.items():
                if fname in self.feature_inputs:
                    spin = self.feature_inputs[fname]
                    spin.blockSignals(True)
                    spin.setValue(round(value, 6))
                    spin.blockSignals(False)

            print(f"[Physics] beta2={beta2:.2f} fs²/mm, N={N:.2f}, gamma={gamma:.2e} 1/(W·m)")

        except Exception as e:
            print(f"Error in physics-based update: {e}")
            import traceback
            traceback.print_exc()

    def _run_batch_prediction(self):
        """执行批量预测：从CSV/NPY文件读取多组特征，批量推理并可视化。

        文件格式:
        - .npy: numpy数组，shape=(n_samples, n_features)
        - .csv: 逗号分隔文本，首行为表头，后续每行一个样本

        处理流程:
        1. 读取文件为numpy数组
        2. 使用scaler.transform归一化
        3. 批量前向传播
        4. 绘制前10条光谱的对比图
        5. 填充结果汇总表
        """
        if self.model is None or self.scaler is None:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return

        filepath = self.batch_file_edit.text().strip()
        if not filepath or not os.path.isfile(filepath):
            QMessageBox.warning(self, "No File", "Please select a valid input file.")
            return

        try:
            if filepath.endswith('.npy'):
                features = np.load(filepath)
            elif filepath.endswith('.csv'):
                features = np.loadtxt(filepath, delimiter=',', skiprows=1)
            else:
                QMessageBox.warning(self, "Unsupported", "Only .npy and .csv files are supported.")
                return

            if features.ndim == 1:
                features = features.reshape(1, -1)

            features_scaled = self.scaler.transform(features)
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

            self.model.eval()
            with torch.no_grad():
                predictions = self.model(features_tensor).cpu().numpy()

            wavelengths = np.linspace(200, 2500, predictions.shape[1])

            n_show = min(10, len(predictions))
            spectra_list = [predictions[i] for i in range(n_show)]
            labels = [f"Sample {i+1}" for i in range(n_show)]
            self.batch_canvas.plot_comparison(spectra_list, labels, wavelengths,
                                              title=f"Batch Prediction ({len(predictions)} samples)")

            self.batch_result_table.setRowCount(len(predictions))
            self.batch_result_table.setColumnCount(4)
            self.batch_result_table.setHorizontalHeaderLabels(['Sample', 'Peak λ (nm)', 'Peak Value', 'Mean Value'])
            for i in range(len(predictions)):
                peak_idx = np.argmax(predictions[i])
                self.batch_result_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
                self.batch_result_table.setItem(i, 1, QTableWidgetItem(f"{wavelengths[peak_idx]:.1f}"))
                self.batch_result_table.setItem(i, 2, QTableWidgetItem(f"{predictions[i][peak_idx]:.4f}"))
                self.batch_result_table.setItem(i, 3, QTableWidgetItem(f"{predictions[i].mean():.4f}"))

            self.statusBar().showMessage(f"Batch prediction complete — {len(predictions)} samples")

        except Exception as e:
            QMessageBox.critical(self, "Batch Error", f"Batch prediction failed:\n{str(e)}")

    def _load_training_history(self):
        """加载训练历史JSON文件并绘制Loss/R²曲线。"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Training Log", "", "JSON (*.json);;CSV (*.csv);;All Files (*)")
        if not filepath:
            return

        try:
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.training_history = data
                else:
                    QMessageBox.warning(self, "Format Error", "JSON file should contain a dictionary with training history.")
                    return
            else:
                QMessageBox.warning(self, "Unsupported", "Only JSON training logs are supported.")
                return

            self.loss_canvas.plot_training_history(self.training_history)
            self.r2_canvas.plot_r2_history(self.training_history)
            self.statusBar().showMessage("Training history loaded")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load training history:\n{str(e)}")

    def _detect_model_type(self, model_path, data_dir):
        """自动检测模型类型（mlp / temporal / lstm）。

        检测策略（按优先级）：
        1. 从model_config.json中读取model_type字段
        2. 从目录名推断（包含'lstm'或'temporal'关键字）
        3. 尝试用不同架构加载state_dict，匹配成功的即为正确类型
        4. 默认返回'mlp'

        Parameters
        ----------
        model_path : str
            模型文件路径。
        data_dir : str
            数据目录路径。

        Returns
        -------
        str
            检测到的模型类型: 'mlp', 'temporal', 或 'lstm'。
        """
        # 策略1: 从model_config.json读取
        config_path = os.path.join(data_dir, 'model_config.json')
        if os.path.isfile(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                model_type = config.get('model_type', '').lower()
                if model_type in ('temporallstm', 'lstm'):
                    return 'lstm'
                elif model_type in ('temporalmlp', 'temporal'):
                    return 'temporal'
                elif model_type in ('mlp', 'standardmlp'):
                    return 'mlp'
            except Exception:
                pass

        # 策略2: 从目录名推断
        dir_name = os.path.basename(os.path.normpath(data_dir)).lower()
        if 'lstm' in dir_name:
            return 'lstm'
        elif 'temporal' in dir_name:
            return 'temporal'

        # 策略3: 尝试加载state_dict匹配架构
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            keys = set(state_dict.keys())

            # LSTM特有键
            lstm_keys = {'lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0'}
            if any(k in keys for k in lstm_keys):
                return 'lstm'

            # TemporalMLP特有键（有temporal_decoder和final_decoder）
            temporal_keys = {'temporal_decoder.0.weight', 'final_decoder.0.weight'}
            if any(k in keys for k in temporal_keys):
                return 'temporal'

            # 标准MLP只有layers
            if any(k.startswith('layers.') for k in keys):
                return 'mlp'
        except Exception:
            pass

        return 'mlp'

    def _update_temporal_evolution_display(self, info):
        """更新时序模型预测的演化图显示。

        Parameters
        ----------
        info : dict
            包含'temporal_data'和'z_positions'的推理信息字典。
        """
        temporal_data = info.get('temporal_data')
        z_positions = info.get('z_positions')

        if temporal_data is None or z_positions is None:
            return

        # temporal_data shape: (n_z_steps, output_dim)
        n_z, n_wl = temporal_data.shape
        wavelengths = np.linspace(200, 2500, n_wl)
        z_cm = z_positions[0] if z_positions.ndim > 1 else z_positions

        # 归一化到[0,1]
        t_min, t_max = temporal_data.min(), temporal_data.max()
        if t_max > t_min:
            temporal_data = (temporal_data - t_min) / (t_max - t_min)

        # 绘制模型预测的频域演化图
        self.spectral_evolution_canvas.plot_evolution(
            temporal_data, wavelengths, z_cm,
            title=f"Model Predicted Spectral Evolution ({self.model_type.upper()})",
            xlabel="Wavelength (nm)",
            ylabel="Propagation Distance (cm)",
            cmap='viridis'
        )

        # 同时更新时域演化图（使用物理模型近似）
        self._update_evolution_display()


def main():
    """应用程序入口：创建QApplication实例，设置全局字体，启动主窗口事件循环。"""
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    # 应用现代CSS样式
    app.setStyleSheet(CSS_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
