# ── 0. 套件匯入 ─────────────────────────────────────────────
import os, time, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
import matplotlib
matplotlib.use('Agg')          # 伺服器端繪圖，避免彈窗
import matplotlib.pyplot as plt
from tqdm import tqdm

# ⬇⬇⬇  Optuna（Hyperband）⬇⬇⬇
import optuna
from optuna.pruners import HyperbandPruner

# ── 1. 全域路徑與固定參數 ──────────────────────────────────
ROOT         = './perplexity'
DATA_DIR     = os.path.join(ROOT, 'dataset')
RESULTS_DIR  = os.path.join(ROOT, 'results')
MODELS_DIR   = os.path.join(RESULTS_DIR, 'models')
FIGURES_DIR  = os.path.join(RESULTS_DIR, 'figures')
CSV_DIR      = os.path.join(RESULTS_DIR, 'csv')

for p in (MODELS_DIR, FIGURES_DIR, CSV_DIR):
    os.makedirs(p, exist_ok=True)

# 固定設定（搜尋時會覆寫其中部分欄位）
BASE_CONFIG = {
    'data_dir'     : DATA_DIR,
    'batch_size'   : 16,
    'num_epochs'   : 200,      # 完整訓練用
    'learning_rate': 1e-3,
    'weight_decay' : 1e-4,
    'patience'     : 10
}

# ── 2. 多模型訓練類別 ───────────────────────────────────────
class MultiModelTrainer:
    """
    封裝：資料載入、單模型訓練 / 驗證 / 測試、曲線與 CSV 輸出 ……
    可一次處理多個 CNN backbone。
    """
    def __init__(self, config: dict, model_names: list[str] | None = None):
        """
        :param config:  各種超參數設定
        :param model_names:  要訓練的模型清單；若為 None 則全訓練
        """
        self.cfg         = config
        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_names = model_names or ['EfficientNet', 'ResNet50', 'DenseNet121']
        self.models      = {}   # str → nn.Module
        self.optimizers  = {}   # str → Optimizer
        self.schedulers  = {}   # str → LR‐Scheduler
        self.metrics_history = {n: {'train_loss':[], 'val_loss':[],
                                'train_acc':[],  'val_acc':[]} for n in self.model_names}

    # ---------- 建立模型 ----------
    def create_models(self) -> None:
        """根據 self.model_names 產生對應 CNN 並調整最後線性層。"""
        if 'EfficientNet' in self.model_names:
            m = models.efficientnet_b0(weights='IMAGENET1K_V1')
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, 3)
            self.models['EfficientNet'] = m.to(self.device)

        if 'ResNet50' in self.model_names:
            m = models.resnet50(weights='IMAGENET1K_V1')
            m.fc = nn.Linear(m.fc.in_features, 3)
            self.models['ResNet50'] = m.to(self.device)

        if 'DenseNet121' in self.model_names:
            m = models.densenet121(weights='IMAGENET1K_V1')
            m.classifier = nn.Linear(m.classifier.in_features, 3)
            self.models['DenseNet121'] = m.to(self.device)

    # ---------- 建立 Optimizer + Scheduler ----------
    def setup_training(self) -> None:
        """為每個模型設定 Adam / ReduceLROnPlateau。"""
        for name, model in self.models.items():
            opt = optim.Adam(model.parameters(),
                             lr=self.cfg['learning_rate'],
                             weight_decay=self.cfg['weight_decay'])
            sch = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)
            self.optimizers[name] = opt
            self.schedulers[name] = sch

    # ---------- 載入資料 ----------
    def load_data(self) -> None:
        """建立 train/val/test Dataloader；同時保存 class_names。"""
        train_tf = transforms.Compose([
            transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), transforms.ColorJitter(0.2,0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        val_tf = transforms.Compose([
            transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        full = datasets.ImageFolder(self.cfg['data_dir'], transform=train_tf)
        n_total = len(full)
        n_train = int(0.7*n_total); n_val = int(0.15*n_total)
        n_test  = n_total - n_train - n_val
        train_set, val_set, test_set = random_split(
            full, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
        val_set.dataset.transform  = val_tf
        test_set.dataset.transform = val_tf

        self.train_loader = DataLoader(train_set, batch_size=self.cfg['batch_size'],
                                       shuffle=True,  num_workers=4)
        self.val_loader   = DataLoader(val_set,   batch_size=self.cfg['batch_size'],
                                       shuffle=False, num_workers=4)
        self.test_loader  = DataLoader(test_set,  batch_size=self.cfg['batch_size'],
                                       shuffle=False, num_workers=4)
        self.class_names  = full.classes

    # ---------- 單 epoch 訓練 ----------
    def train_epoch(self, model_name:str, epoch:int):
        model, opt = self.models[model_name], self.optimizers[model_name]
        crit = nn.CrossEntropyLoss()
        model.train(); running_loss=0; correct=0; total=0
        pbar = tqdm(self.train_loader, desc=f'{model_name} Epoch {epoch+1}')
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(self.device), lbls.to(self.device)
            opt.zero_grad()
            out = model(imgs); loss = crit(out, lbls)
            loss.backward(); opt.step()
            running_loss += loss.item()*imgs.size(0)
            _, pred = torch.max(out, 1)
            total += lbls.size(0); correct += (pred==lbls).sum().item()
            pbar.set_postfix({'loss':f'{loss.item():.3f}',
                              'acc':f'{100.*correct/total:.1f}%'})
        return running_loss/len(self.train_loader.dataset), correct/total

    # ---------- 單 epoch 驗證 ----------
    def validate_epoch(self, model_name:str):
        model = self.models[model_name]
        crit  = nn.CrossEntropyLoss()
        model.eval(); running_loss=0; correct=0; total=0
        with torch.no_grad():
            for imgs, lbls in self.val_loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                out = model(imgs); loss = crit(out, lbls)
                running_loss += loss.item()*imgs.size(0)
                _, pred = torch.max(out, 1)
                total += lbls.size(0); correct += (pred==lbls).sum().item()
        return running_loss/len(self.val_loader.dataset), correct/total

    # ---------- 測試集評估 ----------
    def test_model(self, model_name:str):
        model = self.models[model_name]
        model.eval(); preds, labels = [], []
        with torch.no_grad():
            for imgs, lbls in self.test_loader:
                imgs = imgs.to(self.device)
                out, _ = model(imgs).max(1)
                preds.extend(_.cpu().numpy()); labels.extend(lbls.numpy())
        acc = accuracy_score(labels, preds)
        prec= precision_score(labels, preds, average='weighted')
        rec = recall_score(labels, preds, average='weighted')
        f1  = f1_score(labels, preds, average='weighted')
        cm  = confusion_matrix(labels, preds)
        rpt = classification_report(labels, preds, target_names=self.class_names,
                                    output_dict=True)
        return {'accuracy':acc, 'precision':prec, 'recall':rec,
                'f1_score':f1, 'confusion_matrix':cm,
                'classification_report':rpt,
                'predictions':preds, 'true_labels':labels}

    # ---------- ↓↓↓ 以下：曲線 / CSV 輸出（與原版相同，可保留） ----------
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
    

    # ---------- 部分訓練 (for Optuna) ----------
    def train_for_epochs(self, model_name:str, n_epochs:int, trial=None):
        """
        只訓練指定 epochs；若傳入 trial 則回報並檢查剪枝。
        傳回最佳驗證準確率。
        """
        best_val = 0.0
        for ep in range(n_epochs):
            t_loss, t_acc = self.train_epoch(model_name, ep)
            v_loss, v_acc = self.validate_epoch(model_name)
            # 回報給 Optuna
            if trial is not None:
                trial.report(v_acc, step=ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            best_val = max(best_val, v_acc)
        return best_val

# ── 3. Optuna objective ─────────────────────────────────────
def objective(trial):
    """
    由 Optuna 呼叫：隨機抽樣超參數 → 建立 Trainer → 只訓練 20 epochs
    回傳最佳驗證準確率；如表現差則被 HyperbandPruner 剪枝。
    """
    # 3.1 抽樣超參數
    lr  = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    wd  = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    mdl = trial.suggest_categorical('model_name',
                                    ['EfficientNet', 'ResNet50', 'DenseNet121'])

    # 3.2 組合 config & 建立 Trainer（只訓練抽到的單一模型）
    cfg = BASE_CONFIG.copy()
    cfg['learning_rate'] = lr
    cfg['weight_decay']  = wd
    trainer = MultiModelTrainer(cfg, model_names=[mdl])
    trainer.create_models(); trainer.setup_training(); trainer.load_data()

    # 3.3 部分訓練 (20 epochs) + 剪枝機制
    best_val = trainer.train_for_epochs(model_name=mdl, n_epochs=20, trial=trial)
    return best_val

# ── 4. 進行超參數搜尋 ───────────────────────────────────────
if __name__ == '__main__':
    # 4.1 Hyperband Pruner（= Hyperband 早停策略）
    pruner = HyperbandPruner(min_resource=1, max_resource=20, reduction_factor=3)
    study  = optuna.create_study(direction='maximize', pruner=pruner)
    print("⏳  開始 Hyperband 超參數搜尋 …")
    study.optimize(objective, n_trials=30)   # 最多 30 trial 或 1 小時

    # 4.2 搜尋結束，輸出最佳組合
    print("✅  搜尋完成！最佳 trial：")
    best = study.best_trial
    print(f"  ValAcc = {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # 4.3 以最佳組合做完整訓練 + 測試 + 輸出
    final_cfg = BASE_CONFIG.copy()
    final_cfg['learning_rate'] = best.params['lr']
    final_cfg['weight_decay']  = best.params['weight_decay']
    best_model = best.params['model_name']

    trainer = MultiModelTrainer(final_cfg, model_names=[best_model])
    trainer.create_models(); trainer.setup_training(); trainer.load_data()
    print("\n🚀  以最佳參數做完整訓練 …")
    start = time.time()
    trainer.train_for_epochs(best_model, n_epochs=final_cfg['num_epochs'])
    # 儲存最佳權重（此處僅單一模型，可視需要增補早停與存檔）
    torch.save(trainer.models[best_model].state_dict(),
               os.path.join(MODELS_DIR, f'best_{best_model}.pth'))

    # 4.4 測試評估 & 結果輸出
    results = {best_model: trainer.test_model(best_model)}
    trainer.save_training_curves()
    trainer.save_confusion_matrices(results)
    trainer.save_metrics_to_csv(results)
    mins = (time.time() - start) / 60
    print(f"\n🎉  完整流程結束！耗時 {mins:.1f} 分鐘，所有結果已輸出至 {RESULTS_DIR}/")
