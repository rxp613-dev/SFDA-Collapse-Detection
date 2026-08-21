import torch
from models.classifier import CompleteModel

class TeacherModelsManager:
    """多源Teacher模型管理"""
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.teachers = {}
    
    def train_teacher(self, domain_name, train_data_path, epochs=50, feature_dim=256):
        from torch.utils.data import DataLoader
        from training.source_pretrain import SourcePreTrainer
        
        config = {
            'source_data_path': train_data_path,
            'feature_dim': feature_dim,
            'epochs': epochs,
            'lr': 1e-3,
            'batch_size': 32
        }
        
        trainer = SourcePreTrainer(config)
        trained_model = trainer.run()
        
        self.teachers[domain_name] = trained_model
        return trained_model
    
    def load_teacher(self, domain_name, model_path):
        model = CompleteModel(num_classes=self.num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.teachers[domain_name] = model
        return model
    
    def get_teacher_outputs(self, domain_name, data_batch):
        teacher = self.teachers[domain_name]
        with torch.no_grad():
            features = teacher.get_features(data_batch)
            logits, probs = teacher.classifier(features)
        return logits, features
    
    def save_all_teachers(self, save_dir='models/teachers/'):
        for name, model in self.teachers.items():
            torch.save(model.state_dict(), f"{save_dir}/{name}.pt")