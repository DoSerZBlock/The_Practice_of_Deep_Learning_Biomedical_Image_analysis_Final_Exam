# main.py  ── 一支腳本支援 EfficientNet-B0 / ResNet50 / DenseNet121
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from multiprocessing import freeze_support
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# ── 全域設定 ─────────────────────────────────────────────
DATA_DIR     = './perplexity/dataset'
BATCH_SIZE   = 32
NUM_EPOCHS   = 200
LR           = 1e-5
IMG_SIZE     = 224
SEED         = 42
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── 資料增強 / 預處理 ───────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ─── 辅助函式 ───────────────────────────────────────────
def replace_head(net: nn.Module, n_cls: int) -> nn.Module:
    """依不同 backbone 替換分類層"""
    if isinstance(net, models.EfficientNet):
        in_f = net.classifier[-1].in_features          # Sequential 的最後一層
        net.classifier[-1] = nn.Linear(in_f, n_cls)
    elif isinstance(net, models.DenseNet):
        in_f = net.classifier.in_features              # 直接是 Linear
        net.classifier = nn.Linear(in_f, n_cls)
    elif isinstance(net, models.ResNet):
        in_f = net.fc.in_features                      # ResNet 用 fc
        net.fc = nn.Linear(in_f, n_cls)
    else:
        raise ValueError(f"未知 backbone：{net.__class__.__name__}")
    return net

def train_one_epoch(net, loader, optim_, loss_fn):
    net.train()
    tot_loss = tot_ok = tot = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim_.zero_grad()
        out  = net(x)
        loss = loss_fn(out, y)
        loss.backward(); optim_.step()
        tot_loss += loss.item() * x.size(0)
        tot_ok   += (out.argmax(1) == y).sum().item()
        tot      += y.size(0)
    return tot_loss / tot, tot_ok / tot

@torch.no_grad()
def evaluate(net, loader, loss_fn):
    net.eval()
    tot_loss = tot_ok = tot = 0
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out  = net(x)
        tot_loss += loss_fn(out, y).item() * x.size(0)
        pred = out.argmax(1)
        tot_ok += (pred == y).sum().item()
        tot    += y.size(0)
        ys.extend(y.cpu()); ps.extend(pred.cpu())
    return tot_loss / tot, tot_ok / tot, np.array(ys), np.array(ps)

# ─── 主流程 ──────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)

    # 1. 讀資料並切分
    ds_full = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    num_classes = len(ds_full.classes)
    n = len(ds_full)
    n_train, n_val = int(0.7*n), int(0.15*n)
    n_test = n - n_train - n_val
    ds_train, ds_val, ds_test = random_split(
        ds_full, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    ds_val.dataset.transform  = test_tf
    ds_test.dataset.transform = test_tf

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True , num_workers=2)
    dl_val   = DataLoader(ds_val  , batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    dl_test  = DataLoader(ds_test , batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. 指定三種 backbone
    MODELS = [
        models.efficientnet_b0(weights='IMAGENET1K_V1'),
        models.resnet50(weights='IMAGENET1K_V1'),
        models.densenet121(weights='IMAGENET1K_V1')
    ]

    # 3. 逐模型訓練
    for net in MODELS:
        net = replace_head(net, num_classes).to(DEVICE)

        optim_ = optim.Adam(net.parameters(), lr=LR)
        sched  = optim.lr_scheduler.ReduceLROnPlateau(optim_, 'min', patience=5, factor=0.5)
        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0; patience = 0
        tr_loss_hist, val_loss_hist = [], []

        for ep in range(NUM_EPOCHS):
            tl, ta = train_one_epoch(net, dl_train, optim_, loss_fn)
            vl, va, _, _ = evaluate(net, dl_val, loss_fn)
            tr_loss_hist.append(tl); val_loss_hist.append(vl)
            sched.step(vl)

            print(f"[{net.__class__.__name__}] Epoch {ep+1:02d} "
                  f"TrainAcc={ta:.3f} ValAcc={va:.3f}")

            if va > best_acc:
                best_acc = va; patience = 0
                torch.save(net.state_dict(), f'best_{net.__class__.__name__}.pth')
            else:
                patience += 1
            if patience >= 10:
                print("Early stopping."); break

        # 4. 測試
        net.load_state_dict(torch.load(f'best_{net.__class__.__name__}.pth'))
        tl, ta, y_true, y_pred = evaluate(net, dl_test, loss_fn)
        print(f"\n◆ {net.__class__.__name__} TestAcc = {ta:.4f}")

        # 指定 labels 以確保完整形狀
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
        print("Confusion Matrix:\n", cm)
        print(classification_report(
            y_true, y_pred, target_names=ds_full.classes,
            digits=4, zero_division=0
        ))

        # 5. 繪圖
        plt.plot(tr_loss_hist, label='Train')
        plt.plot(val_loss_hist, label='Val')
        plt.title(net.__class__.__name__)
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        plt.savefig(f'./perplexity/loss_{net.__class__.__name__}.png')
        #plt.show()
        

# ─── 程式入口 ────────────────────────────────────────────
if __name__ == "__main__":
    freeze_support()      # Windows spawn 必要
    main()
