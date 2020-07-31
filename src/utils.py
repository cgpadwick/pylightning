import matplotlib.pyplot as plt
import psutil
from sklearn.metrics import confusion_matrix, accuracy_score
import torch


def display_grid_data(data_loader, classmap, cmap='gray', figsize=(50, 50),
                      title_size=40, pad=3.0, ncols=4):
    for batch, lab in data_loader:
        nimages = batch.size()[0]
        nrows = nimages // ncols
        f, axarr = plt.subplots(nrows, ncols, figsize=figsize)
        for idx in range(nimages):
            col = idx % ncols
            row = idx // ncols
            img = batch[idx, :].numpy().squeeze()
            axarr[row, col].imshow(img, cmap=cmap)
            axarr[row, col].set_xticks([])
            axarr[row, col].set_yticks([])
            axarr[row, col].set_title(
                classmap[lab.numpy()[idx]], size=title_size)
        f.tight_layout(pad=pad)
        break


def model_predictions(data_loader, model):
    predictions = []
    labels = []
    for batch, lab in data_loader:
        pred = model(batch)
        pred = torch.argmax(pred, dim=1).numpy()
        predictions += list(pred)
        labels += list(lab.numpy())

    return labels, predictions


def measure_accuracy(labels, predictions, all_possible_labels):
    cm = confusion_matrix(labels, predictions, labels=all_possible_labels)
    print('Conf Matrix:')
    print(cm)

    print('\nAccuracy Score:')
    print(accuracy_score(labels, predictions))


def get_num_cpus(logical=False):
    return psutil.cpu_count(logical)
