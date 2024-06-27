import torch
import time
from torch import nn
from conv_net import DeepConvModel
from data import prepare_train_data


def train_epoch(model, optimizer, loss_func, data_loader, device, start, epoch):
    correct = 0
    total = 0
    start_epoch = time.time()
    losses = []
    accuracies = []
    model.train()
    for i, (X_batch, y_batch) in enumerate(data_loader):
        total += y_batch.shape[0]
        # print(X_batch.shape, y_batch)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)

        correct += (y_pred.max(dim=-1)[1] == y_batch).sum().item()
        accuracy = correct / total * 100
        loss = loss_func(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        current_time = time.time()
        print(f'\r--train-- Epoch: {epoch}, Iteration: {i + 1}/{len(data_loader)}, '
              f'Loss: {loss.item()}, Accuracy: {round(accuracy, 2)}%, '
              f'Time/iteration: {round((current_time - start_epoch) / (i + 1), 2)}s, '
              f'Total time: {round(current_time - start, 2)}s', end='')

        losses.append(loss.item())
        accuracies.append(accuracy)
    print()

    return losses, accuracies


def train(model_name, epochs=10, batch_size=128):
    data = prepare_train_data(batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    deepconvmodel = DeepConvModel().to(device)
    optimizer = torch.optim.SGD(deepconvmodel.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    start = time.time()

    train_losses, train_accuracies = [], []

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(deepconvmodel, optimizer, loss_fn, data, device, start, epoch)
        train_losses.extend(train_loss)
        train_accuracies.extend(train_accuracy)

    torch.save(deepconvmodel.state_dict(), f"models/{model_name}.pth")

    print("Finished Training")
    print("Max training accuracy: {:.2f}%".format(max(train_accuracies)))
