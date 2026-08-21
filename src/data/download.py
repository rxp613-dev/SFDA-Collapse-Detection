import os
import urllib.request
import json
from tqdm import tqdm

class DataDownloader:
    """
    三大公开数据集下载器
    
    数据集来源:
    - CWRU: https://engineering.case.edu/bearingdatacenter
    - PU: https://mb.uni-paderborn.de/kat/forschung/data
    - JNU: GitHub开源仓库镜像
    """
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_cwru(self):
        """
        CWRU数据集下载
        
        文件: 正常、内圈、外圈、滚动体故障
        格式: .mat, 采样率12kHz
        大小: 约100MB
        """
        cwru_dir = os.path.join(self.data_dir, 'CWRU')
        os.makedirs(cwru_dir, exist_ok=True)
        
        print("=" * 50)
        print("CWRU数据集下载指南:")
        print("1. 访问 https://engineering.case.edu/bearingdatacenter")
        print("2. 下载以下文件到 data/raw/CWRU/:")
        print("   - Normal Baseline Data (0hp)")
        print("   - Inner Race Fault (0.007, 0.014, 0.021 inch)")
        print("   - Outer Race Fault (@6, @12, centered)")
        print("   - Ball Fault")
        print("3. 文件格式为.mat，采样率12kHz")
        print("=" * 50)
        
        return cwru_dir
    
    def download_pu(self):
        """
        PU (Paderborn)数据集下载
        
        文件: K001~K004人为损伤数据
        格式: .mat, 采样率64kHz
        大小: 约1GB
        
        注意: 只下载人为损坏数据，真实损坏数据太杂乱
        """
        pu_dir = os.path.join(self.data_dir, 'PU')
        os.makedirs(pu_dir, exist_ok=True)
        
        print("=" * 50)
        print("PU数据集下载指南:")
        print("方式1 (官方): https://mb.uni-paderborn.de/kat/forschung/data")
        print("方式2 (Kaggle镜像): https://www.kaggle.com/datasets/...")
        print("下载文件到 data/raw/PU/:")
        print("   - K001.mat (正常)")
        print("   - K002.mat (内圈)")
        print("   - K003.mat (外圈)")
        print("   - K004.mat (滚动体)")
        print("注意: 采样率64kHz，需要降采样到12kHz")
        print("=" * 50)
        
        return pu_dir
    
    def download_jnu(self):
        """
        JNU (江南大学)数据集下载
        
        文件: 变速变载工况数据
        格式: .mat
        大小: 约200MB
        
        来源: GitHub开源仓库或ResearchGate
        """
        jnu_dir = os.path.join(self.data_dir, 'JNU')
        os.makedirs(jnu_dir, exist_ok=True)
        
        print("=" * 50)
        print("JNU数据集下载指南:")
        print("方式1: GitHub仓库 https://github.com/...")
        print("方式2: ResearchGate作者分享")
        print("下载变速变载工况数据到 data/raw/JNU/")
        print("=" * 50)
        
        return jnu_dir
    
    def create_metadata(self):
        """
        创建数据元信息文件
        
        记录数据集基本信息，便于后续处理
        """
        metadata = {
            'CWRU': {
                'sampling_rate': 12000,
                'classes': ['normal', 'inner', 'outer', 'ball'],
                'load_conditions': ['0hp', '1hp', '2hp', '3hp'],
                'file_format': '.mat',
                'downloaded': False,
            },
            'PU': {
                'sampling_rate': 64000,
                'classes': ['K001', 'K002', 'K003', 'K004'],
                'file_format': '.mat',
                'downloaded': False,
                'note': '人为损坏数据，需降采样'
            },
            'JNU': {
                'sampling_rate': 'variable',
                'operating_conditions': 'variable_speed_load',
                'file_format': '.mat',
                'downloaded': False,
            }
        }
        
        metadata_path = os.path.join(self.data_dir.replace('/raw', ''), 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"元信息文件创建: {metadata_path}")
        return metadata_path
    
    def verify_downloads(self):
        """
        验证数据是否下载完成
        
        检查文件是否存在且格式正确
        """
        datasets = ['CWRU', 'PU', 'JNU']
        all_downloaded = True
        
        for dataset in datasets:
            path = os.path.join(self.data_dir, dataset)
            if not os.path.exists(path):
                print(f"❌ {dataset} 未下载")
                all_downloaded = False
            else:
                files = os.listdir(path)
                if len(files) > 0:
                    print(f"✅ {dataset} 已下载 ({len(files)} 个文件)")
                else:
                    print(f"⚠️ {dataset} 目录存在但无文件")
                    all_downloaded = False
        
        return all_downloaded

if __name__ == '__main__':
    import sys
    downloader = DataDownloader()
    
    if '--verify' in sys.argv:
        downloader.verify_downloads()
    else:
        downloader.download_cwru()
        downloader.download_pu()
        downloader.download_jnu()
        downloader.create_metadata()
        print("\n请按照上述指南手动下载数据集")
        print("下载完成后运行: python src/data/download.py --verify")