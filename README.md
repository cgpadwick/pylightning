# Install the environment

```
virtualenv pl -p python3.6
source pl/bin/activate
pip install pytorch-lightning
pip install torchvision
pip install matplotlib
pip install sklearn
pip install jupyter
```

# Running The Code

Start up jupyter with the following command:

`jupyter notebook`

Navigate to the src directory and run the notebooks.

* mnist_warmup_pytorch.ipynb - a notebook with a pure pytorch implementation of an MNIST solver using a traditional NN
* mnist_warmup.ipynb - same network as above but using pytorch-lightning 
* mnist_convnet.ipynb - CNN implementation using pytorch-lightning

