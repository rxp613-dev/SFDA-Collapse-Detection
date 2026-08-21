import torch
import numpy as np

class OrderTracking:
    """阶次跟踪预处理模块"""
    
    def __init__(self, samples_per_revolution=256):
        """
        Args:
            samples_per_revolution: 每转采样点数（角度分辨率）
        """
        self.samples_per_rev = samples_per_revolution
    
    def transform_with_speed(self, vibration_signal, rpm, sampling_rate=None):
        """
        方案A: 有转速信号的阶次跟踪
        
        Args:
            vibration_signal: [batch, 1, length] 振动信号
            rpm: 转速（RPM）
            sampling_rate: 采样率（Hz），默认从信号长度推断
        
        Returns:
            angle_signal: [batch, 1, length] 角度域信号
        """
        if sampling_rate is None:
            sampling_rate = vibration_signal.shape[-1]
        
        angle_signal = self._resample_to_angle_domain(
            vibration_signal, 
            sampling_rate,
            rpm
        )
        
        return angle_signal
    
    def transform_without_speed(self, vibration_signal, sampling_rate=None):
        """
        方案B: 无转速信号的tachometer-less方法
        
        Args:
            vibration_signal: [batch, 1, length]
            sampling_rate: 采样率
        
        Returns:
            angle_signal: 角度域信号
            estimated_rpm: 估算转速
        """
        if sampling_rate is None:
            sampling_rate = vibration_signal.shape[-1]
        
        estimated_rpm = self._estimate_rpm_from_signal(
            vibration_signal,
            sampling_rate
        )
        
        angle_signal = self.transform_with_speed(
            vibration_signal,
            estimated_rpm,
            sampling_rate
        )
        
        return angle_signal, estimated_rpm
    
    def _estimate_rpm_from_signal(self, signal, sampling_rate):
        """从信号估算转速"""
        signal_np = signal.squeeze().numpy()
        
        fft_result = np.abs(np.fft.rfft(signal_np))
        freqs = np.fft.rfftfreq(len(signal_np), 1.0 / sampling_rate)
        
        mask = freqs > 10
        if not np.any(mask):
            return 1800.0  # 默认转速
        
        main_freq_idx = np.argmax(fft_result[mask])
        main_freq = freqs[mask][main_freq_idx]
        
        estimated_rpm = main_freq * 60.0
        
        return estimated_rpm
    
    def _resample_to_angle_domain(self, signal, sampling_rate, rpm):
        """重采样到角度域"""
        rev_time = 60.0 / rpm
        samples_per_rev_at_sampling = int(sampling_rate * rev_time)
        
        scale_factor = self.samples_per_rev / samples_per_rev_at_sampling
        
        angle_signal = torch.nn.functional.interpolate(
            signal,
            scale_factor=scale_factor,
            mode='linear',
            align_corners=False if scale_factor != 1.0 else True
        )
        
        return angle_signal
    
    def adapt_sampling_rate(self, signal, original_rate, target_rate=32e3):
        """
        采样率自适应（统一不同采样率到相同角度分辨率）
        
        Args:
            signal: [batch, 1, length] 信号
            original_rate: 原始采样率
            target_rate: 目标统一采样率
        
        Returns:
            adapted_signal: 适配后的信号
        """
        scale_factor = target_rate / original_rate
        
        adapted_signal = torch.nn.functional.interpolate(
            signal,
            scale_factor=scale_factor,
            mode='linear',
            align_corners=False
        )
        
        return adapted_signal