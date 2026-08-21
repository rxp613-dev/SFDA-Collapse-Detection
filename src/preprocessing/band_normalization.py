import torch
import numpy as np

class BandNormalization:
    """频带归一化：减少不同采样率数据的domain gap"""
    
    def __init__(self, 
                 cwru_sampling_rate=12e3,
                 pu_sampling_rate=64e3,
                 target_band=(0, 6e3)):
        """
        Args:
            cwru_sampling_rate: CWRU采样率
            pu_sampling_rate: PU采样率
            target_band: 目标统一频带范围（Hz）
        """
        self.cwru_rate = cwru_sampling_rate
        self.pu_rate = pu_sampling_rate
        self.target_band = target_band
    
    def normalize(self, signal, domain='cwru'):
        """
        归一化信号到统一频带
        
        Args:
            signal: [batch, 1, length]
            domain: 'cwru' 或 'pu'
        
        Returns:
            normalized_signal: 归一化后的信号
        """
        if domain == 'cwru':
            source_rate = self.cwru_rate
        elif domain == 'pu':
            source_rate = self.pu_rate
        else:
            raise ValueError(f"Unknown domain: {domain}")
        
        spectrum = torch.fft.rfft(signal.squeeze())
        freqs = torch.fft.rfftfreq(signal.shape[-1], 1.0 / source_rate)
        
        band_mask = (freqs >= self.target_band[0]) & (freqs <= self.target_band[1])
        
        filtered_spectrum = spectrum * band_mask.to(spectrum.device)
        
        normalized_signal = torch.fft.irfft(filtered_spectrum, n=signal.shape[-1])
        normalized_signal = normalized_signal.unsqueeze(0).unsqueeze(0)
        
        return normalized_signal
    
    def get_frequency_response(self, domain='cwru'):
        """获取频率响应曲线（用于分析）"""
        if domain == 'cwru':
            source_rate = self.cwru_rate
        else:
            source_rate = self.pu_rate
        
        freqs = np.linspace(0, source_rate / 2, 1000)
        
        response = np.zeros_like(freqs)
        band_mask = (freqs >= self.target_band[0]) & (freqs <= self.target_band[1])
        response[band_mask] = 1.0
        
        return freqs, response