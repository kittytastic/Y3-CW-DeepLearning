### Export to python
```
jupytext --set-formats ipynb,py:percent Pegasus.ipynb
jupytext --to notebook notebook.py
```

conda install jupyterlab ipykernel numpy scipy pytorch cudatoolkit=11.0 matplotlib tqdm umap-learn pandas  -c conda-forge

https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
https://syncedreview.com/2019/06/06/going-beyond-gan-new-deepmind-vae-model-generates-high-fidelity-human-faces/
https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed