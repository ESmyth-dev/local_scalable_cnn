import statistics
import torch
import time
from torch import nn
from conv_net import DeepConvModel
from data import prepare_test_data


def multi_test_epoch(models, loss_func, data_loader, device, start):
    avg_correct = 0
    total = 0
    start_epoch = time.time()
    losses = []
    avg_accuracies = []
    individual_accuracies = []
    individual_correct = []
    for i in range(len(models)):
        individual_accuracies.append([])
        individual_correct.append(0)

    for i, (X_batch, y_batch) in enumerate(data_loader):
        total += y_batch.shape[0]
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        preds = []
        for model in models:
            preds.append(model(X_batch))

        stacked_tensors = torch.stack(preds, dim=0)
        y_pred = torch.mean(stacked_tensors, dim=0)

        for j in range(len(preds)):
            individual_correct[j] += (preds[j].max(dim=-1)[1] == y_batch).sum().item()
            individual_accuracies[j].append( individual_correct[j] / total * 100)


        avg_correct += (y_pred.max(dim=-1)[1] == y_batch).sum().item()
        avg_accuracy = avg_correct / total * 100
        loss = loss_func(y_pred, y_batch)

        current_time = time.time()
        # print(f'\r--val-- Iteration: {i + 1}/{len(data_loader)}, '
        #       f'Loss: {loss.item()}, Accuracy: {round(avg_accuracy, 2)}%, '
        #       f'Time/iteration: {round((current_time - start_epoch) / (i + 1), 2)}s, '
        #       f'Total time: {round(current_time - start, 2)}s', end='')

        losses.append(loss.item())
        avg_accuracies.append(avg_accuracy)
    print()

    return losses, avg_accuracies, individual_accuracies


def multi_test(model_names, epochs=1, batch_size=128):
    data = prepare_test_data(batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    models = []
    for model_name in model_names:
        model = DeepConvModel().to(device)
        model.load_state_dict(torch.load(f"models/{model_name}.pth"))
        model.eval()
        models.append(model)

    loss_fn = nn.CrossEntropyLoss()
    start = time.time()

    test_loss, avg_test_accuracy, individual_test_accuracy = multi_test_epoch(models, loss_fn, data, device, start)
    # print(avg_test_accuracy)
    # print(individual_test_accuracy)

    print("Finished testing")
    print("Average of average testing accuracy: {:.2f}%".format(statistics.mean(avg_test_accuracy)))
    print ("average of each individual accuracy: ")
    for i in range(len(individual_test_accuracy)):
        print(f"{i}: {statistics.mean(individual_test_accuracy[i])}")
