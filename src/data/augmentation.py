"""
任务3: 数据增强
添加噪声、时间扭曲等增强方法，预期准确率60-65%

数据增强策略:
1. 噪声注入: 添加高斯噪声模拟传感器噪声
2. 时间扭曲: 改变信号时间尺度
3. 幅值扰动: 随机调整信号幅值
4. 切片翻转: 随机翻转信号片段
"""
import torch
import numpy as np

class DataAugmenter:
    """轴承信号数据增强器"""
    
    def __init__(self, 
                 noise_std=0.01,
                 time_warp_range=0.1,
                 amplitude_range=0.2,
                 flip_prob=0.3):
        """
        Args:
            noise_std: 噪声标准差
            time_warp_range: 时间扭曲范围（±%）
            amplitude_range: 幅值扰动范围（±%）
            flip_prob: 翻转概率
        """
        self.noise_std = noise_std
        self.time_warp_range = time_warp_range
        self.amplitude_range = amplitude_range
        self.flip_prob = flip_prob
    
    def add_gaussian_noise(self, signal):
        """添加高斯噪声"""
        noise = torch.randn_like(signal) * self.noise_std
        return signal + noise
    
    def time_warp(self, signal):
        """时间扭曲（速度变化）"""
        warp_factor = 1.0 + np.random.uniform(-self.time_warp_range, self.time_warp_range)
        
        original_length = signal.shape[-1]
        warped_length = max(1, int(original_length * warp_factor))
        
        original_dim = signal.dim()
        
        if original_dim == 1:
            signal_3d = signal.unsqueeze(0).unsqueeze(0)
        else:
            signal_3d = signal.unsqueeze(1)
        
        warped_3d = torch.nn.functional.interpolate(
            signal_3d,
            size=warped_length,
            mode='linear',
            align_corners=True if warped_length > 1 else False
        )
        
        if warped_length > original_length:
            start = (warped_length - original_length) // 2
            warped_3d = warped_3d[:, :, start:start + original_length]
        elif warped_length < original_length:
            pad_left = (original_length - warped_length) // 2
            pad_right = original_length - warped_length - pad_left
            warped_3d = torch.nn.functional.pad(warped_3d, (pad_left, pad_right))
        
        if original_dim == 1:
            warped_signal = warped_3d.squeeze(0).squeeze(0)
        else:
            warped_signal = warped_3d.squeeze(1)
        
        return warped_signal
    
    def amplitude_perturbation(self, signal):
        """幅值扰动"""
        factor = 1.0 + np.random.uniform(-self.amplitude_range, self.amplitude_range)
        return signal * factor
    
    def random_flip(self, signal):
        """随机翻转信号片段"""
        if np.random.random() < self.flip_prob:
            length = signal.shape[-1]
            flip_start = np.random.randint(0, length // 2)
            flip_end = flip_start + np.random.randint(length // 4, length // 2)
            
            if signal.dim() == 1:
                signal[flip_start:flip_end] = torch.flip(signal[flip_start:flip_end], dims=[0])
            else:
                signal[:, flip_start:flip_end] = torch.flip(signal[:, flip_start:flip_end], dims=[1])
        
        return signal
    
    def augment(self, signal):
        """综合数据增强"""
        # 随机选择增强方法组合
        augmented = signal.clone()
        
        # 1. 噪声注入（总是应用）
        augmented = self.add_gaussian_noise(augmented)
        
        # 2. 时间扭曲（50%概率）
        if np.random.random() < 0.5:
            augmented = self.time_warp(augmented)
        
        # 3. 幅值扰动（50%概率）
        if np.random.random() < 0.5:
            augmented = self.amplitude_perturbation(augmented)
        
        # 4. 随机翻转（30%概率）
        augmented = self.random_flip(augmented)
        
        return augmented

def test_augmentation():
    """测试数据增强效果"""
    import matplotlib.pyplot as plt
    
    # 创建测试信号
    signal = torch.randn(1, 1024)
    
    augmenter = DataAugmenter(
        noise_std=0.02,
        time_warp_range=0.1,
        amplitude_range=0.15,
        flip_prob=0.3
    )
    
    # 生成多个增强样本
    augmented_samples = [augmenter.augment(signal) for _ in range(5)]
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.plot(signal.numpy().flatten())
    plt.title('Original')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    for i, aug in enumerate(augmented_samples[:5]):
        plt.subplot(2, 3, i + 2)
        plt.plot(aug.numpy().flatten())
        plt.title(f'Augmented {i + 1}')
        plt.xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('experiments/results/data_augmentation/augmentation_examples.png')
    print("可视化保存: experiments/results/data_augmentation/augmentation_examples.png")

if __name__ == '__main__':
    print("数据增强模块测试")
    print("="*60)
    
    augmenter = DataAugmenter()
    
    signal = torch.randn(10, 1024)
    augmented = augmenter.augment(signal[0])
    
    print(f"原始信号: shape={signal[0].shape}, mean={signal[0].mean():.4f}, std={signal[0].std():.4f}")
    print(f"增强信号: shape={augmented.shape}, mean={augmented.mean():.4f}, std={augmented.std():.4f}")
    
    test_augmentation()