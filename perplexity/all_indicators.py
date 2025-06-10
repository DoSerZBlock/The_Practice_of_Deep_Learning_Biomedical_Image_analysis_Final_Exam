import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')  # 不彈出圖表
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# 1. 所有路徑根據 root 設定
ROOT = './perplexity'
DATA_DIR = os.path.join(ROOT, 'dataset')
RESULTS_DIR = os.path.join(ROOT, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
CSV_DIR = os.path.join(RESULTS_DIR, 'csv')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

CONFIG = {
    'data_dir': DATA_DIR,
    'batch_size': 16,
    'num_epochs': 200,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'patience': 10
}

class MultiModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.metrics_history = {
            'EfficientNet': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
            'ResNet50': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
            'DenseNet121': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        }
        
    def create_models(self):
        self.models['EfficientNet'] = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.models['EfficientNet'].classifier[1] = nn.Linear(
            self.models['EfficientNet'].classifier[1].in_features, 3
        )
        self.models['ResNet50'] = models.resnet50(weights='IMAGENET1K_V1')
        self.models['ResNet50'].fc = nn.Linear(
            self.models['ResNet50'].fc.in_features, 3
        )
        self.models['DenseNet121'] = models.densenet121(weights='IMAGENET1K_V1')
        self.models['DenseNet121'].classifier = nn.Linear(
            self.models['DenseNet121'].classifier.in_features, 3
        )
        for name, model in self.models.items():
            self.models[name] = model.to(self.device)
            
    def setup_training(self):
        for name, model in self.models.items():
            self.optimizers[name] = optim.Adam(
                model.parameters(), 
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
            self.schedulers[name] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers[name], 'min', patience=5, factor=0.5
            )
    
    def load_data(self):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        full_dataset = datasets.ImageFolder(self.config['data_dir'], transform=train_transform)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_set, val_set, test_set = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        val_set.dataset.transform = val_transform
        test_set.dataset.transform = val_transform
        self.train_loader = DataLoader(
            train_set, batch_size=self.config['batch_size'], 
            shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            val_set, batch_size=self.config['batch_size'],
            shuffle=False, num_workers=4
        )
        self.test_loader = DataLoader(
            test_set, batch_size=self.config['batch_size'],
            shuffle=False, num_workers=4
        )
        self.class_names = full_dataset.classes

    def train_epoch(self, model_name, epoch):
        model = self.models[model_name]
        optimizer = self.optimizers[model_name]
        criterion = nn.CrossEntropyLoss()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.train_loader, desc=f'{model_name} Epoch {epoch+1}')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model_name):
        model = self.models[model_name]
        criterion = nn.CrossEntropyLoss()
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def test_model(self, model_name):
        model = self.models[model_name]
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds, 
            target_names=self.class_names, 
            output_dict=True
        )
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_preds,
            'true_labels': all_labels
        }
    
    def save_training_curves(self):
        for model_name in self.models.keys():
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            epochs = range(1, len(self.metrics_history[model_name]['train_loss']) + 1)
            ax[0].plot(epochs, self.metrics_history[model_name]['train_loss'], 'b-', label='Train Loss')
            ax[0].plot(epochs, self.metrics_history[model_name]['val_loss'], 'r-', label='Val Loss')
            ax[0].set_title(f'{model_name} - Loss')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Loss')
            ax[0].legend()
            ax[1].plot(epochs, self.metrics_history[model_name]['train_acc'], 'b-', label='Train Acc')
            ax[1].plot(epochs, self.metrics_history[model_name]['val_acc'], 'r-', label='Val Acc')
            ax[1].set_title(f'{model_name} - Accuracy')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].legend()
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'{model_name}_training_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_confusion_matrices(self, test_results):
        for model_name, results in test_results.items():
            plt.figure(figsize=(8, 6))
            cm = results['confusion_matrix']
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'{model_name} - Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(self.class_names))
            plt.xticks(tick_marks, self.class_names, rotation=45)
            plt.yticks(tick_marks, self.class_names)
            thresh = cm_normalized.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                        horizontalalignment="center",
                        color="white" if cm_normalized[i, j] > thresh else "black")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'{model_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_metrics_to_csv(self, test_results):
        for model_name in self.models.keys():
            history_df = pd.DataFrame(self.metrics_history[model_name])
            history_df['epoch'] = range(1, len(history_df) + 1)
            history_df = history_df[['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']]
            history_df.to_csv(os.path.join(CSV_DIR, f'{model_name}_training_history.csv'), index=False)
        summary_data = []
        for model_name, results in test_results.items():
            summary_data.append({
                'Model': model_name,
                'Test_Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Score': results['f1_score']
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(CSV_DIR, 'model_comparison_summary.csv'), index=False)
        for model_name, results in test_results.items():
            report_df = pd.DataFrame(results['classification_report']).transpose()
            report_df.to_csv(os.path.join(CSV_DIR, f'{model_name}_classification_report.csv'))
        for model_name, results in test_results.items():
            cm_df = pd.DataFrame(
                results['confusion_matrix'], 
                index=self.class_names, 
                columns=self.class_names
            )
            cm_df.to_csv(os.path.join(CSV_DIR, f'{model_name}_confusion_matrix.csv'))
        for model_name, results in test_results.items():
            predictions_df = pd.DataFrame({
                'True_Label': results['true_labels'],
                'Predicted_Label': results['predictions'],
                'True_Class': [self.class_names[i] for i in results['true_labels']],
                'Predicted_Class': [self.class_names[i] for i in results['predictions']]
            })
            predictions_df.to_csv(os.path.join(CSV_DIR, f'{model_name}_predictions.csv'), index=False)
    
    def train_all_models(self):
        best_val_acc = {name: 0.0 for name in self.models.keys()}
        patience_counters = {name: 0 for name in self.models.keys()}
        for epoch in range(self.config['num_epochs']):
            print(f"\n=== Epoch {epoch+1}/{self.config['num_epochs']} ===")
            for model_name in self.models.keys():
                train_loss, train_acc = self.train_epoch(model_name, epoch)
                val_loss, val_acc = self.validate_epoch(model_name)
                self.metrics_history[model_name]['train_loss'].append(train_loss)
                self.metrics_history[model_name]['val_loss'].append(val_loss)
                self.metrics_history[model_name]['train_acc'].append(train_acc)
                self.metrics_history[model_name]['val_acc'].append(val_acc)
                self.schedulers[model_name].step(val_loss)
                if val_acc > best_val_acc[model_name]:
                    best_val_acc[model_name] = val_acc
                    torch.save(self.models[model_name].state_dict(), 
                             os.path.join(MODELS_DIR, f'best_{model_name}.pth'))
                    patience_counters[model_name] = 0
                else:
                    patience_counters[model_name] += 1
                print(f"{model_name}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            if all(counter >= self.config['patience'] for counter in patience_counters.values()):
                print(f"所有模型觸發早停機制，在第{epoch+1}輪停止訓練")
                break
    
    def run_complete_training(self):
        self.create_models()
        self.setup_training()
        self.load_data()
        print("開始訓練...")
        start_time = time.time()
        self.train_all_models()
        training_time = time.time() - start_time
        print("\n開始測試和評估...")
        test_results = {}
        for model_name in self.models.keys():
            self.models[model_name].load_state_dict(
                torch.load(os.path.join(MODELS_DIR, f'best_{model_name}.pth'))
            )
            test_results[model_name] = self.test_model(model_name)
            print(f"\n{model_name} 測試結果:")
            print(f"準確率: {test_results[model_name]['accuracy']:.4f}")
            print(f"精確率: {test_results[model_name]['precision']:.4f}")
            print(f"召回率: {test_results[model_name]['recall']:.4f}")
            print(f"F1分數: {test_results[model_name]['f1_score']:.4f}")
        print("\n保存結果...")
        self.save_training_curves()
        self.save_confusion_matrices(test_results)
        self.save_metrics_to_csv(test_results)
        print(f"\n✓ 訓練完成！總耗時: {training_time/60:.2f} 分鐘")
        print(f"✓ 所有結果已保存至 {RESULTS_DIR}/")
        return test_results

if __name__ == '__main__':
    trainer = MultiModelTrainer(CONFIG)
    trainer.run_complete_training()
