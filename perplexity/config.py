
# 實驗配置檔案
import torch

class Config:
    # 資料設定
    DATA_DIR = 'data/BUSI_dataset'  # BUSI資料集路徑
    TRAIN_SPLIT = 0.7  # 訓練集比例
    VAL_SPLIT = 0.15   # 驗證集比例
    TEST_SPLIT = 0.15  # 測試集比例

    # 影像設定
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    NUM_WORKERS = 4

    # 類別設定
    CLASS_NAMES = ['benign', 'malignant', 'normal']
    NUM_CLASSES = len(CLASS_NAMES)

    # 訓練設定
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10

    # 模型設定
    MODELS_TO_COMPARE = ['resnet50', 'efficientnet_b0', 'densenet121']
    PRETRAINED = True

    # 硬體設定
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 資料增強設定
    AUGMENTATION = {
        'horizontal_flip': 0.5,
        'rotation': 15,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        }
    }

    # 優化器設定
    OPTIMIZER_PARAMS = {
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY
    }

    # 學習率調度器設定
    SCHEDULER_PARAMS = {
        'step_size': 10,
        'gamma': 0.1
    }
