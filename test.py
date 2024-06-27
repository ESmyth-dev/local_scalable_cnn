import statistics
import torch
import time
from torch import nn
from conv_net import DeepConvModel
from data import prepare_test_data


def test_epoch(model, loss_func, data_loader, device, start, epoch):
    correct = 0
    total = 0
    start_epoch = time.time()
    losses = []
    accuracies = []
    model.eval()
    for i, (X_batch, y_batch) in enumerate(data_loader):
        total += y_batch.shape[0]
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)

        correct += (y_pred.max(dim=-1)[1] == y_batch).sum().item()
        accuracy = correct / total * 100
        loss = loss_func(y_pred, y_batch)

        current_time = time.time()
        print(f'\r--val-- Epoch: {epoch}, Iteration: {i + 1}/{len(data_loader)}, '
              f'Loss: {loss.item()}, Accuracy: {round(accuracy, 2)}%, '
              f'Time/iteration: {round((current_time - start_epoch) / (i + 1), 2)}s, '
              f'Total time: {round(current_time - start, 2)}s', end='')

        losses.append(loss.item())
        accuracies.append(accuracy)
    print()

    return losses, accuracies


def test(model_name, epochs=1, batch_size=128):
    data = prepare_test_data(batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    deepconvmodel = DeepConvModel().to(device)
    deepconvmodel.load_state_dict(torch.load(f"models/{model_name}.pth"))
    deepconvmodel.eval()

    loss_fn = nn.CrossEntropyLoss()
    start = time.time()

    test_losses, test_accuracies = [], []

    for epoch in range(epochs):
        test_loss, test_accuracy = test_epoch(deepconvmodel, loss_fn, data, device, start, epoch)
        test_losses.extend(test_loss)
        test_accuracies.extend(test_accuracy)

    print("Finished testing")
    print("Average testing accuracy: {:.2f}%".format(statistics.mean(test_accuracies)))

