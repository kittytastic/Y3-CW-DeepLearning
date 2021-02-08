import umap
import pandas as pd
import umap.plot
import matplotlib.pyplot as plt
import torchvision
import numpy as np

#######################################################
#                     Latent space                    #
#######################################################

def PlotLatentSpace(model,  train_iterator, device, class_names, point_count=1000):

    acc_labels = None
    acc_vals = None

    for i in range(point_count//256):
        # sample x from the dataset
        x,l = next(train_iterator)
        x,t = x.to(device), l.to(device)

        z = model.encode(x).cpu()

        latent_space = z.detach().numpy()
        labels = np.array(l)
        
        latent_space = np.squeeze(latent_space)

        if not acc_labels is None:
            acc_labels = np.concatenate((acc_labels, labels), axis=0)
            acc_vals = np.concatenate((acc_vals, latent_space))
        else:
            acc_labels = labels
            acc_vals = latent_space


    df = pd.DataFrame(data=acc_vals)

    mapper = umap.UMAP(n_neighbors=15).fit(df)
    df['labels'] = acc_labels
    df['labels'] = df['labels'].map(dict((i, val) for i, val in enumerate(class_names)))
    umap.plot.points(mapper, labels=df["labels"])

def PlotCustomLatentSpace(model, datasets, class_labels, latent_size, device):
    full_df =  pd.DataFrame()
    data_df = pd.DataFrame()
    labels_df = pd.DataFrame()

    data = np.empty([0, latent_size])
    labels = []
    indexes = []
    for i in range(len(datasets)):
        ds = datasets[i]
        l = class_labels[i]
        ds, l = ds.to(device), l

        z = model.encode(ds).detach().cpu().numpy()
        z = np.squeeze(z)
        data = np.concatenate((data, z), axis=0)
        labels+=[l]*len(ds)
        indexes += list(range(len(ds)))
    
    
    mapper = umap.UMAP(n_neighbors=15).fit(data)
    labels_df['labels']=labels
    labels_df['index']=indexes
    p = umap.plot.interactive(mapper, labels=labels_df['labels'], hover_data = labels_df, point_size=2)
    umap.plot.show(p)

#######################################################
#                       Other                         #
#######################################################
def PlotLoss(loss_array, loss_type="Loss"):
    plt.plot(loss_array)
    plt.ylabel(loss_type)
    plt.xlabel('Epoch')
    plt.show()

def PlotAllLoss(losses, loss_names):
    fig, axs = plt.subplots(len(losses), sharex=True, gridspec_kw={'hspace': 0})
    for i in range(len(losses)):
        axs[i].plot(losses[i])
        axs[i].set_ylabel(loss_names[i])
    plt.xlabel('Epoch')
    plt.show()

def plotTensor(tensor):
    plt.rcParams['figure.dpi'] = 175
    plt.grid(False)
    plt.axis('off')
    plt.imshow(torchvision.utils.make_grid(tensor).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.show()