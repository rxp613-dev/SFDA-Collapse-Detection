import scipy.io as sio
import numpy as np
from scipy.signal import butter, filtfilt
import torch
import os

class BearingDataPreprocessor:
    """
    轴承数据预处理器
    
    功能:
    1. 抗混叠降采样（64kHz→12kHz）
    2. 滑动窗口截取（L=1024, overlap=0.5）
    3. Z-score归一化（每个样本独立）
    
    支持数据集:
    - CWRU: .mat文件，键名格式 X{编号}_DE_time，采样率12kHz
    - PU: .mat嵌套结构，振动数据在Y[6].Data，采样率64kHz
    - JNU: .csv文件，采样率50kHz
    """
    
    def __init__(self, window_length=1024, overlap_ratio=0.5, target_fs=12000):
        self.window_length = window_length
        self.overlap_ratio = overlap_ratio
        self.target_fs = target_fs
        self.step = int(window_length * (1 - overlap_ratio))
        
    def anti_aliasing_downsample(self, signal, orig_fs=64000):
        """
        抗混叠降采样
        
        Args:
            signal: 原始信号 (N,)
            orig_fs: 原始采样率
        
        Returns:
            降采样后的信号
        """
        if orig_fs == self.target_fs:
            return signal
        
        nyquist = self.target_fs / 2
        cutoff = nyquist / (orig_fs / 2)
        
        b, a = butter(4, cutoff, btype='low')
        filtered_signal = filtfilt(b, a, signal)
        
        downsample_factor = int(orig_fs / self.target_fs)
        downsampled = filtered_signal[::downsample_factor]
        
        print(f"降采样: {orig_fs}Hz → {self.target_fs}Hz, 数据点: {len(signal)} → {len(downsampled)}")
        return downsampled
    
    def sliding_window(self, signal):
        """
        滑动窗口截取
        
        Args:
            signal: 输入信号
        
        Returns:
            windows: 窗口化后的样本 (N_samples, window_length)
        """
        if len(signal) < self.window_length:
            print(f"警告: 信号长度 {len(signal)} 小于窗口长度 {self.window_length}，跳过")
            return np.array([])
        
        num_windows = (len(signal) - self.window_length) // self.step + 1
        
        windows = []
        for i in range(num_windows):
            start = i * self.step
            end = start + self.window_length
            window = signal[start:end]
            windows.append(window)
        
        windows = np.array(windows)
        print(f"窗口化: {len(signal)}个点 → {num_windows}个样本 (每个1024点)")
        return windows
    
    def z_score_normalize(self, windows):
        """
        Z-score归一化（每个样本独立）
        
        Args:
            windows: 窗口样本 (N, 1024)
        
        Returns:
            normalized: 归一化后的样本
        """
        normalized = []
        for window in windows:
            mean = np.mean(window)
            std = np.std(window)
            if std == 0:
                std = 1e-10
            norm_window = (window - mean) / std
            normalized.append(norm_window)
        
        normalized = np.array(normalized)
        print(f"归一化完成: 每个样本独立标准化")
        return normalized
    
    def process_cwru_mat(self, mat_path, label, load_condition='0HP'):
        """
        处理CWRU单个.mat文件
        
        Args:
            mat_path: .mat文件路径
            label: 故障类别标签 (0:正常, 1:内圈, 2:滚动体, 3:外圈)
            load_condition: 负载条件
        
        Returns:
            samples: 处理后的样本 (N, 1, 1024)
            labels: 标签数组 (N,)
        """
        mat_data = sio.loadmat(mat_path)
        
        keys = [k for k in mat_data.keys() if 'DE_time' in k]
        if len(keys) == 0:
            print(f"警告: {mat_path} 未找到DE_time键")
            return np.array([]), np.array([])
        
        key = keys[0]
        signal = mat_data[key].flatten()
        
        windows = self.sliding_window(signal)
        if len(windows) == 0:
            return np.array([]), np.array([])
        
        normalized = self.z_score_normalize(windows)
        samples = normalized[:, np.newaxis, :]
        labels = np.full(len(samples), label)
        
        print(f"处理完成: {mat_path} → {len(samples)}个样本")
        return samples, labels
    
    def process_pu_mat(self, mat_path, label):
        """
        处理PU单个.mat文件
        
        Args:
            mat_path: .mat文件路径
            label: 故障类别标签
        
        Returns:
            samples: 处理后的样本 (N, 1, 1024)
            labels: 标签数组 (N,)
        """
        mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if len(keys) == 0:
            print(f"警告: {mat_path} 未找到数据键")
            return np.array([]), np.array([])
        
        key = keys[0]
        data_struct = mat_data[key]
        
        if not hasattr(data_struct, 'Y'):
            print(f"警告: {mat_path} 未找到Y字段")
            return np.array([]), np.array([])
        
        y_channels = data_struct.Y
        
        vibration_channel = None
        for i, channel in enumerate(y_channels):
            if hasattr(channel, 'Name') and 'vibration' in channel.Name.lower():
                vibration_channel = channel
                break
        
        if vibration_channel is None:
            vibration_channel = y_channels[6]
        
        if not hasattr(vibration_channel, 'Data'):
            print(f"警告: {mat_path} 振动通道未找到Data字段")
            return np.array([]), np.array([])
        
        signal = vibration_channel.Data.flatten()
        
        signal = self.anti_aliasing_downsample(signal, orig_fs=64000)
        
        windows = self.sliding_window(signal)
        if len(windows) == 0:
            return np.array([]), np.array([])
        
        normalized = self.z_score_normalize(windows)
        samples = normalized[:, np.newaxis, :]
        labels = np.full(len(samples), label)
        
        print(f"处理完成: {mat_path} → {len(samples)}个样本")
        return samples, labels
    
    def process_jnu_csv(self, csv_path, label):
        """
        处理JNU单个.csv文件
        
        Args:
            csv_path: .csv文件路径
            label: 故障类别标签
        
        Returns:
            samples: 处理后的样本 (N, 1, 1024)
            labels: 标签数组 (N,)
        """
        signal = np.loadtxt(csv_path)
        
        signal = self.anti_aliasing_downsample(signal, orig_fs=50000)
        
        windows = self.sliding_window(signal)
        if len(windows) == 0:
            return np.array([]), np.array([])
        
        normalized = self.z_score_normalize(windows)
        samples = normalized[:, np.newaxis, :]
        labels = np.full(len(samples), label)
        
        print(f"处理完成: {csv_path} → {len(samples)}个样本")
        return samples, labels
    
    def process_cwru_dataset(self, data_dir, save_path, load_conditions=['0HP']):
        """
        处理完整CWRU数据集
        
        Args:
            data_dir: CWRU数据目录
            save_path: 保存路径(.pt文件)
            load_conditions: 负载条件列表
        """
        all_samples = []
        all_labels = []
        
        file_mapping = {
            'normal': {'pattern': 'normal', 'label': 0},
            'inner': {'pattern': 'IR', 'label': 1},
            'ball': {'pattern': 'B', 'label': 2},
            'outer': {'pattern': 'OR', 'label': 3}
        }
        
        for load in load_conditions:
            load_dir = os.path.join(data_dir, load)
            if not os.path.exists(load_dir):
                print(f"警告: 目录 {load_dir} 不存在")
                continue
            
            files = os.listdir(load_dir)
            
            for fault_type, mapping in file_mapping.items():
                pattern = mapping['pattern']
                label = mapping['label']
                
                matching_files = [f for f in files if pattern in f and f.endswith('.mat')]
                
                for mat_file in matching_files[:1]:
                    mat_path = os.path.join(load_dir, mat_file)
                    samples, labels = self.process_cwru_mat(mat_path, label, load)
                    
                    if len(samples) > 0:
                        all_samples.append(samples)
                        all_labels.append(labels)
        
        if not all_samples:
            raise ValueError("没有成功处理任何CWRU文件！")
        
        all_samples = np.concatenate(all_samples, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        torch.save({
            'samples': torch.from_numpy(all_samples).float(),
            'labels': torch.from_numpy(all_labels).long(),
        }, save_path)
        
        print(f"CWRU数据集保存: {save_path}")
        print(f"总样本数: {len(all_samples)}, 类别分布: {np.bincount(all_labels)}")
    
    def process_pu_dataset(self, data_dir, save_path):
        """
        处理完整PU数据集
        
        Args:
            data_dir: PU数据目录 (包含K001, K002, K003, K004子目录)
            save_path: 保存路径(.pt文件)
        """
        all_samples = []
        all_labels = []
        
        k_mapping = {
            'K001': 0,
            'K002': 1,
            'K003': 2,
            'K004': 3
        }
        
        for k_name, label in k_mapping.items():
            k_dir = os.path.join(data_dir, k_name)
            if not os.path.exists(k_dir):
                print(f"警告: 目录 {k_dir} 不存在")
                continue
            
            mat_files = [f for f in os.listdir(k_dir) if f.endswith('.mat')]
            
            for mat_file in mat_files[:5]:
                mat_path = os.path.join(k_dir, mat_file)
                samples, labels = self.process_pu_mat(mat_path, label)
                
                if len(samples) > 0:
                    all_samples.append(samples)
                    all_labels.append(labels)
        
        if not all_samples:
            raise ValueError("没有成功处理任何PU文件！")
        
        all_samples = np.concatenate(all_samples, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        torch.save({
            'samples': torch.from_numpy(all_samples).float(),
            'labels': torch.from_numpy(all_labels).long(),
        }, save_path)
        
        print(f"PU数据集保存: {save_path}")
        print(f"总样本数: {len(all_samples)}, 类别分布: {np.bincount(all_labels)}")
    
    def process_jnu_dataset(self, data_dir, save_path):
        """
        处理完整JNU数据集
        
        Args:
            data_dir: JNU数据目录
            save_path: 保存路径(.pt文件)
        """
        all_samples = []
        all_labels = []
        
        file_mapping = {
            'n': {'pattern': 'n', 'label': 0},
            'ib': {'pattern': 'ib', 'label': 1},
            'tb': {'pattern': 'tb', 'label': 2},
            'ob': {'pattern': 'ob', 'label': 3}
        }
        
        files = os.listdir(data_dir)
        
        for fault_type, mapping in file_mapping.items():
            pattern = mapping['pattern']
            label = mapping['label']
            
            matching_files = [f for f in files if pattern in f and f.endswith('.csv')]
            
            for csv_file in matching_files[:3]:
                csv_path = os.path.join(data_dir, csv_file)
                samples, labels = self.process_jnu_csv(csv_path, label)
                
                if len(samples) > 0:
                    all_samples.append(samples)
                    all_labels.append(labels)
        
        if not all_samples:
            raise ValueError("没有成功处理任何JNU文件！")
        
        all_samples = np.concatenate(all_samples, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        torch.save({
            'samples': torch.from_numpy(all_samples).float(),
            'labels': torch.from_numpy(all_labels).long(),
        }, save_path)
        
        print(f"JNU数据集保存: {save_path}")
        print(f"总样本数: {len(all_samples)}, 类别分布: {np.bincount(all_labels)}")

if __name__ == '__main__':
    preprocessor = BearingDataPreprocessor()
    
    print("数据预处理模块已创建")
    print("使用方法:")
    print("  preprocessor.process_cwru_dataset('data/raw/CWRU/cwru_data', 'data/processed/cwru_0hp.pt', ['0HP'])")
    print("  preprocessor.process_pu_dataset('data/raw/PU', 'data/processed/pu_k001_k004.pt')")
    print("  preprocessor.process_jnu_dataset('data/raw/JNU/JNU-Bearing-Dataset-main', 'data/processed/jnu_variable.pt')")