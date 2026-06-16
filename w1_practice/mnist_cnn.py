import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    """簡單的卷積神經網路 (CNN)，28x28 灰階手寫數字辨識。

    架構：
        Conv2d(1→32, 3x3) → ReLU
        Conv2d(32→64, 3x3) → ReLU → MaxPool(2x2)
        Dropout(0.25) → Flatten
        Linear(9216→128) → ReLU → Dropout(0.5)
        Linear(128→10) → LogSoftmax

    輸入 shape：[batch, 1, 28, 28]
    輸出 shape：[batch, 10]，為每個類別的 log 機率。
    """

    def __init__(self):
        super(Net, self).__init__()
        # 卷積層1：輸入 1 通道，輸出 32 通道（32個不同filter），kernel 大小 3x3，每次滑動 stride 1
        self.conv1 = nn.Conv2d(1, 32, 3, 1) 
        # 卷積層2：輸入 32 通道，輸出 64 通道，kernel 大小 3x3，每次滑動 stride 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1) 
        # Dropout：訓練時隨機把神經元歸零，降低過擬合
        self.dropout1 = nn.Dropout(0.25) #25%機率會被歸零
        self.dropout2 = nn.Dropout(0.5)
        # 全連接層：flatten 後的 9216 維特徵 → 128 維 → 10 類
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """前向傳播：影像 → 卷積特徵 → 全連接 → 類別 log 機率。"""
        x = self.conv1(x)
        x = F.relu(x) #Conv2d(1→32, 3x3) → ReLU
        x = self.conv2(x)
        x = F.relu(x) #Conv2d(32→64, 3x3) → ReLU
        x = F.max_pool2d(x, 2)  # 空間維度減半，計算量縮小，降維、加速
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # [batch, 64, 12, 12] → [batch, 9216] 變成一維向量做處理
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # log_softmax 搭配 nll_loss 等價於 CrossEntropy，但數值更穩定
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """在訓練集上跑一個 epoch。

    Args:
        args: argparse 物件，內含 log_interval、dry_run 等設定。
        model: 要訓練的 nn.Module 模型。
        device: 'cuda' 或 'cpu'；資料與模型必須在同一個 device。
        train_loader: 訓練資料的 DataLoader。
        optimizer: 已綁定 model.parameters() 的優化器。
        epoch: 目前是第幾個 epoch（從 1 開始），只用於印 log。
    """
    model.train()  # 切到 train 模式，啟用 Dropout 等行為
    for batch_idx, (data, target) in enumerate(train_loader):
        # 把資料搬到 GPU（或保持在 CPU）
        data, target = data.to(device), target.to(device)
        # 標準四步走：歸零梯度 → 預測 → 算 loss → 反向 → 更新權重
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) #比較準確度，算出損失
        loss.backward()
        optimizer.step()
        # 每隔 log_interval 個 batch 印一次進度
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break  # --dry-run 只跑一個 batch，用於快速驗證程式跑得起來


def test(model, device, test_loader):
    """在測試集上評估準確率，不更新權重。

    Args:
        model: 已訓練的 nn.Module 模型。
        device: 同 train()。
        test_loader: 測試資料的 DataLoader。
    """
    model.eval()  # 切到 eval 模式：關閉 Dropout、固定 BatchNorm 統計
    test_loss = 0
    correct = 0
    # no_grad() 區塊內不會建立計算圖，省記憶體、加速
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # reduction='sum' 是為了之後再除以總樣本數得平均
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # argmax 取機率最高的類別；keepdim=True 保留維度方便比對
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset) #算出平均 loss

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    """主程式：解析參數 → 準備資料 → 建模型 → 訓練 → 測試 → 存模型。
       用 argparse 來讓你在執行程式時可以從命令列傳參數"""
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    # gamma：每個 epoch 結束時把 lr 乘上這個值，逐步降低學習率
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    # 固定亂數種子，讓實驗結果可重現
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    args = parser.parse_args()

    # 自動偵測加速器（CUDA / MPS / 等），加 --no-accel 可強制用 CPU
    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    # 用 GPU 時開 pin_memory 與多 worker，加快資料搬到 GPU 的速度
    if use_accel:
        accel_kwargs = {'num_workers': 1, #多工的subprocess數量，0 表示在主 process 讀取
                        'persistent_workers': True,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    # 影像預處理：轉為 tensor，並用 MNIST 統計過的 mean/std 做標準化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 第一次跑會自動下載 MNIST 到 ../data/ 資料夾
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    # Adadelta：自適應學習率優化器，對初始 lr 較不敏感
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # StepLR：每個 epoch 結束把 lr 乘以 gamma（0.7），逐步降低學習率
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # 只有加 --save-model 才會把權重存成 mnist_cnn.pt
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
