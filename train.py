"""
    训练部分函数
"""
import numpy as np
import torch


def evaluate(model, dataloader, criterion, device):
    """评估"""
    model.eval()
    total_loss = 0
    all_preds = []  # 存储所有预测值
    all_labels = []  # 存储所有标签值
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            all_preds.append(out.view(-1).cpu().numpy())
            all_labels.append(data.y.view(-1).cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.concatenate(all_preds), np.concatenate(all_labels)  # 返回验证平均损失，所有预测值，所有标签值


def run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, model_weights_path):
    """
        训练循环
        保存验证损失最低的模型
    """
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}] train_loss: {avg_train_loss:.4f}, val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_weights_path)
            print(f"[INFO] 最佳模型已保存，val_loss: {best_val_loss:.4f}")

    return model
