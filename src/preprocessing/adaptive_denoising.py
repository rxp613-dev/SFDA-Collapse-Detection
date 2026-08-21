import torch
import torch.nn as nn

class AdaptiveDenoising(nn.Module):
    """自适应降噪模块（针对PU高噪声水平）"""
    
    def __init__(self, noise_threshold=0.2, preserve_freq_range=(50, 1000)):
        """
        Args:
            noise_threshold: 噪声阈值
            preserve_freq_range: 保持的故障频率范围（Hz）
        """
        super().__init__()
        self.noise_threshold = noise_threshold
        self.preserve_freq = preserve_freq_range
    
    def denoise(self, noisy_signal):
        """
        自适应降噪
        
        Args:
            noisy_signal: [batch, 1, length] 噪声信号
        
        Returns:
            denoised_signal: 降噪后的信号
        """
        spectrum = torch.fft.rfft(noisy_signal.squeeze())
        
        noise_level = torch.std(spectrum).item()
        
        adaptive_threshold = self.noise_threshold * noise_level
        
        magnitude = torch.abs(spectrum)
        mask = (magnitude > adaptive_threshold).float()
        
        denoised_spectrum = spectrum * mask
        
        denoised_signal = torch.fft.irfft(denoised_spectrum, n=noisy_signal.shape[-1])
        denoised_signal = denoised_signal.unsqueeze(0).unsqueeze(0)
        
        return denoised_signal
    
    def estimate_noise_level(self, signal):
        """估算信号噪声水平"""
        spectrum = torch.fft.rfft(signal.squeeze())
        
        high_freq_spectrum = spectrum[len(spectrum) // 2:]
        
        noise_level = torch.std(high_freq_spectrum).item()
        
        return noise_level