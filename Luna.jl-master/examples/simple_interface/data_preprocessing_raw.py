import os
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DIR = "/private/Luna.jl-master/training_data"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
WAVELENGTH_RANGE = (200e-9, 2500e-9)
TARGET_POINTS = 500
EPSILON = 1e-15
C = 299792458

os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_file_complete(filepath, min_size=10000):
    try:
        if not os.path.exists(filepath):
            return False, "文件不存在"
        file_size = os.path.getsize(filepath)
        if file_size < min_size:
            return False, f"文件大小过小 ({file_size} 字节)"
        try:
            with h5py.File(filepath, 'r') as f:
                required_datasets = ['Eω', 'z', 'stats']
                missing_datasets = [d for d in required_datasets if d not in f]
                if missing_datasets:
                    return False, f"缺少数据集: {', '.join(missing_datasets)}"
                if 'Eω' in f:
                    eω_shape = f['Eω'].shape
                    if len(eω_shape) < 2 or eω_shape[0] == 0 or eω_shape[1] == 0:
                        return False, f"Eω形状无效: {eω_shape}"
                if 'physics_features' not in f:
                    return False, "缺少physics_features组"
            return True, "文件完整"
        except Exception as e:
            return False, f"无法打开: {str(e)}"
    except Exception as e:
        return False, f"检查出错: {str(e)}"

def determine_eω_layout(Eω, z):
    """
    自动判断Eω的维度布局
    
    根据Luna源码，Eω的标准布局是 (频率点, z保存点)
    但某些版本/配置可能产生 (z保存点, 频率点) 的布局
    
    判断方法：z数组长度应匹配Eω的某个维度
    """
    z_len = len(z)
    if z_len == Eω.shape[1]:
        return 'freq_z'
    elif z_len == Eω.shape[0]:
        return 'z_freq'
    else:
        logging.warning(f"无法确定Eω布局: z长度={z_len}, Eω形状={Eω.shape}")
        return 'freq_z'

def get_output_spectrum(Eω, z):
    """
    从Eω中提取输出光谱（最后一个z位置的光谱）
    
    自动适配不同的维度布局
    """
    layout = determine_eω_layout(Eω, z)
    if layout == 'freq_z':
        return Eω[:, -1]
    else:
        return Eω[-1, :]

def get_all_spectra(Eω, z):
    """
    从Eω中提取所有z位置的光谱（全过程时序数据）
    
    返回: (n_z, n_freq) 的功率谱数组
    """
    layout = determine_eω_layout(Eω, z)
    if layout == 'freq_z':
        # Eω[频率, z] → 转置为 [z, 频率]
        spectra = np.abs(Eω)**2
        return spectra.T  # (n_z, n_freq)
    else:
        # Eω[z, 频率]
        return np.abs(Eω)**2

def get_freq_count(Eω, z):
    """
    获取频率点数量（自动适配维度布局）
    """
    layout = determine_eω_layout(Eω, z)
    if layout == 'freq_z':
        return Eω.shape[0]
    else:
        return Eω.shape[1]

def parse_byte_str(val):
    """将HDF5中的字节串或普通值转换为Python值
    
    处理Luna/Julia存储的参数格式:
    - 数值: b'3.14e-7' -> float
    - 空数组: b'Float64[]' -> None (表示未设置)
    - 非空数组: b'[0.0, 0.0, 1.5e-28]' -> list of floats
    - 特殊值: b'nothing', b'NaN', b'Inf' -> None
    - 布尔值: b'true', b'false' -> bool
    """
    if isinstance(val, bytes):
        s = val.decode('utf-8').strip()
        if s in ('nothing', 'NaN', 'Inf', '-Inf'):
            return None
        if s in ('true', 'True'):
            return True
        if s in ('false', 'False'):
            return False
        if s.endswith('[]') and '[' not in s[:-2]:
            return None
        if s.startswith('[') and s.endswith(']'):
            inner = s[1:-1].strip()
            if not inner:
                return None
            try:
                return [float(x.strip()) for x in inner.split(',')]
            except ValueError:
                return s
        try:
            return float(s)
        except ValueError:
            return s
    return val

def read_hdf5_file(filepath):
    """
    读取HDF5文件并提取相关数据
    返回: (Eω, z, stats, physics_features, meta, ω_effective, sim_params)
    sim_params: 从prop_capillary_args提取的原始模拟参数
    """
    try:
        with h5py.File(filepath, 'r') as f:
            Eω = f['Eω'][:]
            z = f['z'][:]
            
            stats = {}
            if 'stats' in f:
                for key in f['stats']:
                    stats[key] = f['stats'][key][:]
            
            physics_features = {}
            if 'physics_features' in f:
                for key in f['physics_features']:
                    val = f['physics_features'][key][()]
                    physics_features[key] = val
            
            meta = {}
            if 'meta' in f:
                for key in f['meta']:
                    try:
                        meta[key] = f['meta'][key][()]
                    except:
                        pass
            
            # 从prop_capillary_args提取原始模拟参数
            sim_params = {}
            if 'prop_capillary_args' in f:
                pca = f['prop_capillary_args']
                param_map = {
                    'energy': 'energy',
                    'τfwhm': 'tau',
                    'ϕ': 'phi',
                    'λ0': 'lambda0',
                    'pressure': 'pressure',
                    'flength': 'length',
                    'radius': 'radius',
                }
                for hdf5_key, param_name in param_map.items():
                    if hdf5_key in pca:
                        val = parse_byte_str(pca[hdf5_key][()])
                        if val is not None:
                            sim_params[param_name] = val
            
            # 确定Eω布局和频率点数
            layout = determine_eω_layout(Eω, z)
            freq_count = get_freq_count(Eω, z)
            logging.debug(f"Eω布局: {layout}, 形状={Eω.shape}, 频率点数={freq_count}")
            
            # 智能获取频率网格
            ω_effective = None
            
            if 'grid' in f:
                grid = f['grid']
                
                # 策略1: 直接查找与频率点数匹配的数组字段
                for candidate in ['ω', 'omega', 'w']:
                    if candidate in grid:
                        ds = grid[candidate]
                        if ds.shape == ():
                            continue
                        candidate_data = ds[:]
                        if len(candidate_data) == freq_count:
                            ω_effective = candidate_data
                            logging.debug(f"✅ 找到匹配频率网格: grid/{candidate} ({len(candidate_data)}点)")
                            break
                
                # 策略2: 用sidx裁剪ωo
                if ω_effective is None and 'sidx' in grid and 'ωo' in grid:
                    sidx_ds = grid['sidx']
                    ωo_ds = grid['ωo']
                    if sidx_ds.shape != () and ωo_ds.shape != ():
                        sidx = sidx_ds[:]
                        ωo = ωo_ds[:]
                        ω_cropped = ωo[sidx]
                        if len(ω_cropped) == freq_count:
                            ω_effective = ω_cropped
                            logging.debug(f"✅ 通过sidx裁剪ωo得到匹配网格 ({len(ω_cropped)}点)")
                
                # 策略3: 如果ω是标量数组，尝试从参数重建
                if ω_effective is None:
                    try:
                        grid_params = {}
                        for key in grid.keys():
                            ds = grid[key]
                            val = ds[()] if ds.shape == () else ds[:]
                            grid_params[key] = val
                        
                        if 'ωo' in grid_params and isinstance(grid_params['ωo'], np.ndarray):
                            ωo = grid_params['ωo']
                            if 'sidx' in grid_params and isinstance(grid_params['sidx'], np.ndarray):
                                sidx = grid_params['sidx']
                                ω_cropped = ωo[sidx]
                                if len(ω_cropped) == freq_count:
                                    ω_effective = ω_cropped
                                    logging.debug(f"✅ 从标量参数重建频率网格 ({len(ω_cropped)}点)")
                    except Exception as e:
                        logging.debug(f"从标量参数重建失败: {e}")
                
                # 策略4: 如果以上都失败，记录信息
                if ω_effective is None:
                    logging.warning(f"⚠️ 无法找到与频率点数({freq_count})匹配的网格")
                    grid_keys = list(grid.keys())
                    logging.info(f"   grid可用字段: {grid_keys}")
                    for key in grid_keys:
                        ds = grid[key]
                        try:
                            if ds.shape == ():
                                val = ds[()]
                                if isinstance(val, np.ndarray):
                                    logging.info(f"     grid/{key}: array, shape={val.shape}")
                                else:
                                    logging.info(f"     grid/{key}: scalar, value={val}")
                            else:
                                data = ds[:]
                                logging.info(f"     grid/{key}: shape={data.shape}")
                        except:
                            logging.info(f"     grid/{key}: 无法读取")
            else:
                logging.warning("⚠️ 文件中没有grid组")
            
            return Eω, z, stats, physics_features, meta, ω_effective, sim_params
    
    except Exception as e:
        logging.error(f"读取文件出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None, None, None, None, None, None

def process_spectrum(Eω_output, omega=None):
    """
    处理单时刻光谱数据
    
    参数:
        Eω_output: 输出光谱（一维复数数组）
        omega: 角频率数组（rad/s）
    """
    try:
        n_freq = len(Eω_output)
        
        if omega is not None and len(omega) == n_freq:
            valid_mask = np.abs(omega) > 1e10
            wavelength = np.full(n_freq, np.nan)
            wavelength[valid_mask] = 2 * np.pi * C / omega[valid_mask]
            
            nan_mask = np.isnan(wavelength)
            if np.any(nan_mask):
                valid_idx = np.where(~nan_mask)[0]
                nan_idx = np.where(nan_mask)[0]
                if len(valid_idx) >= 2:
                    wavelength[nan_idx] = np.interp(nan_idx, valid_idx, wavelength[valid_idx])
                else:
                    wavelength = estimate_wavelength(n_freq)
        else:
            wavelength = estimate_wavelength(n_freq)
            logging.info(f"使用估算波长: [{wavelength.min()*1e9:.1f}, {wavelength.max()*1e9:.1f}] nm")
        
        power_spectrum = np.abs(Eω_output)**2
        
        mask = (wavelength >= WAVELENGTH_RANGE[0]) & (wavelength <= WAVELENGTH_RANGE[1])
        valid_points = np.sum(mask)
        
        if valid_points < 10:
            wl_min, wl_max = np.nanmin(wavelength), np.nanmax(wavelength)
            mask = (wavelength >= max(WAVELENGTH_RANGE[0], wl_min)) & (wavelength <= min(WAVELENGTH_RANGE[1], wl_max))
            valid_points = np.sum(mask)
        
        spectrum_cropped = power_spectrum[mask]
        wavelength_cropped = wavelength[mask]
        
        sort_idx = np.argsort(wavelength_cropped)
        wavelength_cropped = wavelength_cropped[sort_idx]
        spectrum_cropped = spectrum_cropped[sort_idx]
        
        if len(spectrum_cropped) < 10:
            logging.warning("有效数据点不足，返回零光谱")
            return np.zeros(TARGET_POINTS)
        
        target_wavelength = np.linspace(WAVELENGTH_RANGE[0], WAVELENGTH_RANGE[1], TARGET_POINTS)
        spectrum_downsampled = np.interp(target_wavelength, wavelength_cropped, spectrum_cropped)
        
        log_spectrum = np.log10(spectrum_downsampled + EPSILON)
        
        min_val, max_val = log_spectrum.min(), log_spectrum.max()
        if max_val > min_val:
            normalized_spectrum = (log_spectrum - min_val) / (max_val - min_val)
        else:
            normalized_spectrum = np.zeros_like(log_spectrum)
        
        return normalized_spectrum
    
    except Exception as e:
        logging.error(f"处理光谱出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return np.zeros(TARGET_POINTS)

def process_temporal_spectra(Eω, z, omega=None):
    """
    处理全过程时序光谱数据
    
    参数:
        Eω: 复数光谱数组 (频率, z) 或 (z, 频率)
        z: 传播距离数组
        omega: 角频率数组（rad/s）
    
    返回:
        temporal_spectra: (n_z, TARGET_POINTS) 的归一化对数功率谱时序数据
    """
    try:
        layout = determine_eω_layout(Eω, z)
        if layout == 'freq_z':
            n_z = Eω.shape[1]
        else:
            n_z = Eω.shape[0]
        
        logging.info(f"处理时序光谱: {n_z}个z位置, 布局={layout}")
        
        # 处理每个z位置的光谱
        temporal_spectra = []
        for i in range(n_z):
            if layout == 'freq_z':
                spectrum_i = Eω[:, i]
            else:
                spectrum_i = Eω[i, :]
            
            processed = process_spectrum(spectrum_i, omega=omega)
            temporal_spectra.append(processed)
        
        temporal_spectra = np.array(temporal_spectra)  # (n_z, TARGET_POINTS)
        
        # 时序标准化: 对每个z位置独立归一化，确保时间一致性
        # 方案: 全局Min-Max标准化（基于所有时间步的最大/最小值）
        global_min = temporal_spectra.min()
        global_max = temporal_spectra.max()
        
        if global_max > global_min:
            temporal_spectra = (temporal_spectra - global_min) / (global_max - global_min)
        
        return temporal_spectra
    
    except Exception as e:
        logging.error(f"处理时序光谱出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def estimate_wavelength(n_points):
    """估算波长数组（fallback）"""
    λ_min, λ_max = WAVELENGTH_RANGE
    freq = np.linspace(C / λ_max, C / λ_min, n_points)
    return C / freq

def extract_features(physics_features, sim_params=None):
    """
    提取输入特征，确保所有值都是float64实数
    
    特征顺序（13维，含wallthickness，不含length）:
    [energy, tau, pressure, diameter, wallthickness,
     beta2, gamma, N, L0, gamma_K, P_ratio, Aeff, neff]
    
    注意: 
    1. chirp和lambda0已移除，因为当前数据集中这两个参数为常数
       (chirp=0, lambda0=1030nm)，不提供判别信息
    2. length已从特征中移除，因为data_generation.jl中光纤长度
       根据模式固定（capillary=1.0m, antiresonant=0.5m），不再作为变量
    """
    features = []
    
    def safe_float(value, default=0.0):
        try:
            if value is None:
                return default
            if isinstance(value, complex):
                return float(value.real)
            if isinstance(value, (list, tuple)):
                arr = np.asarray(value).flatten()
                if len(arr) > 0:
                    v = arr[0]
                    return float(v.real) if isinstance(v, complex) else float(v)
                return default
            if hasattr(value, '__len__') and not isinstance(value, str):
                arr = np.asarray(value).flatten()
                if len(arr) > 0:
                    v = arr[0]
                    return float(v.real) if isinstance(v, complex) else float(v)
                return default
            return float(value)
        except:
            return default
    
    if sim_params:
        features.append(safe_float(sim_params.get('energy', 0)))
        features.append(safe_float(sim_params.get('tau', 0)))
        features.append(safe_float(sim_params.get('pressure', 0)))
        # length已从采样中移除，不再作为特征
        radius = safe_float(sim_params.get('radius', 0))
        features.append(radius * 2 * 1e6 if radius > 0 else 0.0)
        # wallthickness (μm) - 从physics_features或默认值
        wallthickness = safe_float(physics_features.get('wallthickness', 0.5e-6))
        features.append(wallthickness * 1e6 if wallthickness > 0 else 0.5)
    else:
        features.extend([0.0]*5)
    
    feature_keys = ['beta2', 'gamma', 'N', 'L0', 'gamma_K', 'P_ratio', 'Aeff', 'neff']
    non_zero_count = sum(1 for key in feature_keys if safe_float(physics_features.get(key, 0.0)) != 0.0)
    
    for key in feature_keys:
        value = physics_features.get(key, 0.0)
        features.append(safe_float(value))
    
    if non_zero_count == 0:
        logging.warning("⚠️ 所有物理特征均为0！")
    
    result = np.array(features, dtype=np.float64)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

def diagnose_sample(Eω, z, omega, features, sim_params, sample_idx):
    """对样本进行详细诊断"""
    layout = determine_eω_layout(Eω, z)
    output_spectrum = get_output_spectrum(Eω, z)
    freq_count = get_freq_count(Eω, z)
    
    logging.info(f"\n{'='*50}")
    logging.info(f"📊 样本 {sample_idx} 诊断")
    logging.info(f"{'='*50}")
    logging.info(f"  Eω形状: {Eω.shape}, 布局: {layout}")
    logging.info(f"  z长度: {len(z)}, 频率点数: {freq_count}")
    logging.info(f"  输出光谱形状: {output_spectrum.shape}")
    logging.info(f"  |输出光谱|范围: [{np.abs(output_spectrum).min():.6e}, {np.abs(output_spectrum).max():.6e}]")
    
    if omega is not None:
        logging.info(f"  ω: {len(omega)}点, 零值数: {np.sum(np.abs(omega)<1e10)}")
        wl = 2*np.pi*C/omega
        wl_valid = wl[np.isfinite(wl)]
        if len(wl_valid) > 0:
            logging.info(f"  波长范围: [{wl_valid.min()*1e9:.1f}, {wl_valid.max()*1e9:.1f}] nm")
    else:
        logging.info(f"  ω: 未获取")
    
    logging.info(f"  特征非零数: {np.count_nonzero(features)}/{len(features)}")
    logging.info(f"  特征值: {features}")
    
    if sim_params:
        logging.info(f"  原始参数: energy={sim_params.get('energy',0):.3e}J, "
                     f"τfwhm={sim_params.get('tau',0):.3e}s, "
                     f"pressure={sim_params.get('pressure',0):.1f}bar, "
                     f"flength={sim_params.get('length',0):.3f}m, "
                     f"radius={sim_params.get('radius',0):.3e}m")

def main():
    logging.info("="*60)
    logging.info("数据预处理 v3.0 - 时序数据+反谐振光纤支持")
    logging.info("="*60)
    
    hdf5_files = []
    skipped_files = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.h5'):
            filepath = os.path.join(INPUT_DIR, filename)
            is_complete, reason = is_file_complete(filepath)
            if is_complete:
                hdf5_files.append(filepath)
            else:
                skipped_files.append((filename, reason))
                logging.info(f"跳过: {filename} - {reason}")
    
    if skipped_files:
        logging.info(f"共跳过 {len(skipped_files)} 个文件")
    
    hdf5_files = sorted(hdf5_files)
    logging.info(f"找到 {len(hdf5_files)} 个有效文件")
    
    BATCH_SIZE = 100
    all_features = []
    all_temporal_spectra = []  # 时序光谱数据
    all_final_spectra = []     # 最终时刻光谱（用于兼容）
    all_z_positions = []       # z位置信息
    
    num_batches = (len(hdf5_files) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(hdf5_files))
        batch_files = hdf5_files[start_idx:end_idx]
        
        logging.info(f"\n批次 {batch_idx+1}/{num_batches}: 文件 {start_idx+1}-{end_idx}")
        
        batch_features = []
        batch_temporal_spectra = []
        batch_final_spectra = []
        batch_z_positions = []
        
        for i, filepath in enumerate(batch_files):
            filename = os.path.basename(filepath)
            global_idx = start_idx + i + 1
            
            if i % 20 == 0:
                logging.info(f"  处理 {global_idx}/{len(hdf5_files)}: {filename}")
            
            Eω, z, stats, physics_features, meta, omega, sim_params = read_hdf5_file(filepath)
            
            if Eω is None:
                continue
            
            # 提取输入特征
            features = extract_features(physics_features, sim_params=sim_params)
            
            # 处理时序光谱数据
            temporal_spectra = process_temporal_spectra(Eω, z, omega=omega)
            if temporal_spectra is None:
                logging.warning(f"  {filename}: 时序光谱处理失败，跳过")
                continue
            
            # 提取最终时刻光谱（兼容原有输出格式）
            final_spectrum = temporal_spectra[-1, :]
            
            # 记录z位置
            z_positions = z.copy()
            
            # 前3个样本详细诊断
            total_processed = len(all_features) + len(batch_features)
            if total_processed < 3:
                diagnose_sample(Eω, z, omega, features, sim_params, global_idx)
                logging.info(f"  时序光谱形状: {temporal_spectra.shape}")
                logging.info(f"  z位置范围: [{z_positions.min():.4f}, {z_positions.max():.4f}] m")
            
            batch_features.append(features)
            batch_temporal_spectra.append(temporal_spectra)
            batch_final_spectra.append(final_spectrum)
            batch_z_positions.append(z_positions)
        
        all_features.extend(batch_features)
        all_temporal_spectra.extend(batch_temporal_spectra)
        all_final_spectra.extend(batch_final_spectra)
        all_z_positions.extend(batch_z_positions)
        
        if all_features:
            logging.info(f"  累计处理: {len(all_features)} 个样本")
    
    if all_features:
        X = np.array(all_features)
        y_temporal = np.array(all_temporal_spectra)  # (N, n_z, 500)
        y_final = np.array(all_final_spectra)        # (N, 500)
        z_positions = np.array(all_z_positions, dtype=object)  # 变长数组
        
        logging.info(f"\n原始数据:")
        logging.info(f"  X={X.shape} (输入特征)")
        logging.info(f"  y_temporal={y_temporal.shape} (时序光谱)")
        logging.info(f"  y_final={y_final.shape} (最终光谱)")
        logging.info(f"  z_positions={len(z_positions)}个样本")
        
        # 数据清洗
        valid_mask = np.isfinite(X).all(axis=1)
        for i in range(len(y_temporal)):
            if not np.isfinite(y_temporal[i]).all():
                valid_mask[i] = False
        
        X = X[valid_mask]
        y_temporal = y_temporal[valid_mask]
        y_final = y_final[valid_mask]
        z_positions = z_positions[valid_mask]
        
        logging.info(f"清洗后: X={X.shape}, y_temporal={y_temporal.shape}, y_final={y_final.shape}")
        
        # 输入特征标准化
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # 时序输出标准化（每个样本独立标准化）
        # 方案: 对每个样本的时序数据进行全局Min-Max标准化
        y_temporal_scaled = np.zeros_like(y_temporal)
        for i in range(len(y_temporal)):
            sample_min = y_temporal[i].min()
            sample_max = y_temporal[i].max()
            if sample_max > sample_min:
                y_temporal_scaled[i] = (y_temporal[i] - sample_min) / (sample_max - sample_min)
        
        # 保存标准化器
        import joblib
        joblib.dump(scaler_X, os.path.join(OUTPUT_DIR, 'scaler_X.joblib'))
        
        # 数据划分
        X_train, X_temp, y_temporal_train, y_temporal_temp, y_final_train, y_final_temp, z_train, z_temp = \
            train_test_split(X_scaled, y_temporal_scaled, y_final, z_positions, test_size=0.2, random_state=42)
        X_val, X_test, y_temporal_val, y_temporal_test, y_final_val, y_final_test, z_val, z_test = \
            train_test_split(X_temp, y_temporal_temp, y_final_temp, z_temp, test_size=0.5, random_state=42)
        
        # 保存数据
        np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
        np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_val)
        np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
        
        # 时序数据
        np.save(os.path.join(OUTPUT_DIR, 'y_temporal_train.npy'), y_temporal_train)
        np.save(os.path.join(OUTPUT_DIR, 'y_temporal_val.npy'), y_temporal_val)
        np.save(os.path.join(OUTPUT_DIR, 'y_temporal_test.npy'), y_temporal_test)
        
        # 最终光谱数据（兼容）
        np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_final_train)
        np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_final_val)
        np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_final_test)
        
        # z位置信息
        np.save(os.path.join(OUTPUT_DIR, 'z_train.npy'), z_train, allow_pickle=True)
        np.save(os.path.join(OUTPUT_DIR, 'z_val.npy'), z_val, allow_pickle=True)
        np.save(os.path.join(OUTPUT_DIR, 'z_test.npy'), z_test, allow_pickle=True)
        
        # 保存处理参数
        processing_params = {
            'wavelength_range': [WAVELENGTH_RANGE[0], WAVELENGTH_RANGE[1]],
            'target_points': TARGET_POINTS,
            'epsilon': EPSILON,
            'input_features': ['energy', 'tau', 'pressure', 'diameter', 'wallthickness',
                             'beta2', 'gamma', 'N', 'L0', 'gamma_K', 'P_ratio', 'Aeff', 'neff'],
            'fix_version': 'v3.1-fixed-length',
            'data_format': {
                'X': '(N, 13) - 输入特征（length已移除）',
                'y_temporal': '(N, n_z, 500) - 时序光谱数据',
                'y_final': '(N, 500) - 最终时刻光谱',
                'z': '(N,) - z位置数组（变长）'
            },
            'fibre_model': 'antiresonant',
            'diameter_range': [100, 200],
            'wallthickness_range': [0.3, 1.0],
            'fixed_length': 0.5
        }
        
        with open(os.path.join(OUTPUT_DIR, 'processing_params.json'), 'w') as f:
            json.dump(processing_params, f, indent=4)
        
        logging.info("\n" + "="*60)
        logging.info("✅ 数据预处理完成！(v3.0)")
        logging.info("="*60)
        logging.info(f"训练集: X={X_train.shape}, y_temporal={y_temporal_train.shape}")
        logging.info(f"验证集: X={X_val.shape}, y_temporal={y_temporal_val.shape}")
        logging.info(f"测试集: X={X_test.shape}, y_temporal={y_temporal_test.shape}")
    else:
        logging.warning("没有有效的HDF5文件")

if __name__ == "__main__":
    main()
