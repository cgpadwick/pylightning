import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import torch

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10

sns.set()


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


def measure_accuracy(labels, predictions, classes, plot=True):
    cm = confusion_matrix(labels, predictions, labels=list(classes.keys()))
    df = pd.DataFrame(data=cm,
                      index=classes.values(),
                      columns=classes.values())
    acc = accuracy_score(labels, predictions)

    if plot:
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(df, annot=True, fmt="d", linewidths=.5, ax=ax)

    return df, acc


def get_num_cpus(logical=False):
    return psutil.cpu_count(logical)


def embeddable_image(data):
    img_data = data * 255
    img_data = img_data.astype(np.uint8)
    image = Image.fromarray(img_data, mode='L').resize((64, 64), Image.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()


def plot_interactive_embedding(embedding_data,
                               orig_data,
                               labels,
                               classes,
                               palette=Spectral10,
                               title='Plot Title',
                               width=600,
                               height=600):

    output_notebook()

    df = pd.DataFrame(embedding_data, columns=('x', 'y'))
    df['example'] = [str(classes[x]) for x in labels]
    df['image'] = list(map(embeddable_image, orig_data))

    factors = []
    for i in classes.values():
        factors.append(str(i))
    datasource = ColumnDataSource(df)
    color_mapping = CategoricalColorMapper(factors=factors,
                                           palette=palette)

    plot_figure = figure(
        title=title,
        width=width,
        height=height,
        tools=('pan, wheel_zoom, reset')
    )

    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 16px; color: #224499'>Example:</span>
            <span style='font-size: 18px'>@example</span>
        </div>
    </div>
    """))

    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        color=dict(field='example', transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )
    show(plot_figure)


def plot_autoencoder_results(batch, logits, ncols=8):
    nimages = batch.size()[0]
    nrows = nimages // ncols
    f, axarr = plt.subplots(nrows, ncols, figsize=(20, 20))
    imgidx = 0
    for row in range(nrows):
        colidx = 0
        while colidx < ncols - 1:
            img = batch[imgidx, :].numpy().squeeze()
            reconstructed_img = logits[imgidx, ].detach().numpy().squeeze()
            axarr[row, colidx].imshow(img, cmap='gray')
            axarr[row, colidx].set_xticks([])
            axarr[row, colidx].set_yticks([])
            axarr[row, colidx].set_title('Image', size=20)
            colidx = colidx + 1
            axarr[row, colidx].imshow(reconstructed_img, cmap='gray')
            axarr[row, colidx].set_xticks([])
            axarr[row, colidx].set_yticks([])
            axarr[row, colidx].set_title('Reconstruction', size=20)
            colidx = colidx + 1
            imgidx += 1
    f.tight_layout(pad=0.1, w_pad=0.0)
