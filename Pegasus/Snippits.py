# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
# ---

# %%
#####################################
#          Play with latent space   #
#####################################
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display

def LatentSpacePlayground(model):
    def on_slider_change(b):
        slider_vals = [s.value for s in sliders]
        
        with output:
            output.clear_output()
            ts = torch.zeros(1,latent_size)
            ts[0] = torch.FloatTensor(slider_vals)
            ts = ts.to(device)
            genImg = model.decode(ts)
            plotTensor(genImg)

            
    sliders = []
    row_size = 16
    vb = []
    for i in range(int(math.ceil(latent_size/row_size))):
        left = latent_size - row_size*i
        hb = []
        for j in range(min(row_size, left)):
            v = i * row_size + j
            slider = widgets.FloatSlider(description='LV %d'%v, continuous_update=False, orientation='vertical', min=-2, max=2)
            sliders.append(slider)
            sliders[-1].observe(on_slider_change, names='value')
            hb.append(slider)
        
        vb.append(widgets.HBox(hb))


    slider_bank = widgets.VBox(vb)
    output = widgets.Output()

    return (slider_bank, output)


#display(*LatentSpacePlayground(Vres))

# %%
def HorseBirdTensors(count=batch_size):
    hc = 0
    bc = 0

    horses = torch.zeros(count, image_channels, image_size, image_size, requires_grad=False)
    birds = torch.zeros(count, image_channels, image_size, image_size, requires_grad=False)

    while hc<count or bc<count:
        x,t = next(train_iterator)
        for i in range(len(t)):
            cn = class_names[t[i].item()]
            if cn == "horse" and hc<count:
                horses[hc] = x[i]
                hc+=1
            if cn == "bird" and bc<count:
                birds[bc] = x[i]
                bc+=1
    return horses, birds

def GetTensorOfClass(class_name, count=batch_size):
    if class_name not in class_names:
        raise Exception("%s is not in classes"%class_name)

    imgClass = torch.zeros(count, image_channels, image_size, image_size, requires_grad=False)
    ic = 0

    while ic<count:
        x,t = next(train_iterator)
        for i in range(len(t)):
            cn = class_names[t[i].item()]
            if cn == class_name and ic<count:
                imgClass[ic] = x[i]
                ic+=1
    return imgClass

def SeeHB(model):
    horses, birds = HorseBirdTensors(count=128)

    plotTensor(horses)
    plotTensor(birds)

def TryPegasus(model, width=8, rows=8):
    horses, birds = HorseBirdTensors(count=rows)

    gpu_horses = horses.to(device)
    gpu_birds = birds.to(device)
    z_horses = model.full_encode(gpu_horses)
    z_birds = model.full_encode(gpu_birds)
    
    
    z_amalgum = torch.zeros(rows*width, latent_size)
    for i in range(rows):
        cm = width-1
        for j in range(width):
            z_amalgum[i*width+j] = z_horses[i]*j/cm + z_birds[i]*(1-j/cm)

    z_amalgum = z_amalgum.to(device)
    pegasus_decoded = model.decode(z_amalgum)
    plotTensor(pegasus_decoded)

    plotTensor(horses)
    plotTensor(birds)
   

#TryPegasus(Vres)

# %%
def plotImageFrom(images, ds):
    t = torch.zeros(len(images), image_channels, image_size, image_size, requires_grad=False)
    for i in range(len(images)):
        t[i] = ds[images[i]]
    plotTensor(t)

'''
horses, birds = HorseBirdTensors(count=1000)
planes = GetTensorOfClass('airplane', 1000)
cats = GetTensorOfClass('cat', 1000)
PlotCustomLatentSpace(Vres, [birds, cats], ['birds', 'cat'])
'''

# %%
'''
from sklearn.cluster import KMeans

cluster_count = 15
birds_gpu = birds.to(device)
birds_encodeed = Vres.full_encode(birds_gpu).detach().cpu().numpy()
birds_encodeed = np.squeeze(birds_encodeed)
kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(birds_encodeed)
print(kmeans.cluster_centers_.shape)
centers = kmeans.cluster_centers_
centers = torch.FloatTensor(centers)
centers = centers.to(device)
out_t = Vres.decode(centers)
plotTensor(out_t)
'''

# %%
'''
groups = [[] for i in range(cluster_count)]
group_lengths = [0] * cluster_count
group_i = [0] * cluster_count
predictions = kmeans.predict(birds_encodeed[:100])
for i in range(len(predictions)):
    group_lengths[predictions[i]] += 1

for i in range(len(group_lengths)):
    groups[i] = torch.zeros(group_lengths[i], image_channels, image_size, image_size, requires_grad=False)

for i in range(len(predictions)):
    g = predictions[i]
    groups[g][group_i[g]] = birds[i]
    group_i[g]+=1

for i in range(len(group_lengths)):
    plotTensor(groups[i])
'''

# %%
#### Help pick good thing of class

horse_i = 7
bird_l = 2

birds100 = torch.zeros(300, image_channels, image_size, image_size, requires_grad=False)
birds_id = [None]*300
bc = 0
i=0
while bc<300:
    img, label = test_ds[i]
    if label == horse_i:
        birds100[bc] = img
        birds_id[bc] = i
        bc +=1
    i+=1




good_birds_img = torch.zeros(len(good_birds), image_channels, image_size, image_size, requires_grad=False)
good_horses_img = torch.zeros(len(good_horses), image_channels, image_size, image_size, requires_grad=False)
for i in range(len(good_birds)):
    good_birds_img[i] = test_ds[birds_id[good_birds[i]]][0]

for i in range(len(good_horses)):
    good_horses_img[i] = test_ds[birds_id[good_horses[i]]][0]

#plotTensor(birds100[0:100])
#plotTensor(good_birds_img)
#plotTensor(good_horses_img)

# %%
########## Get union of classes

def getIntersectingClasses(classes_a, classes_b, clusters, threshold=0.1):

    intersecting = []
    a_cluster = []
    b_cluster = []
    for i in range(clusters):
        a = len(classes_a[i])
        b = len(classes_b[i])
        
        diffrence = abs(a-b)
        magnitude = abs(a+b)

        if diffrence/magnitude < threshold:
            intersecting.append(i)
        else:
            if a>b:
                a_cluster.append(i)
            else:
                b_cluster.append(i)

    return intersecting, a_cluster, b_cluster

def unionIntersection(model, data, cc):
    a = data[0]
    b = data[1]
    data_in_clusters = clusterClasses(model, data, clusters=cc)
    aAndb, aNotb, bNota = getIntersectingClasses(data_in_clusters[0], data_in_clusters[1], cc)
    print(aAndb)
    print(aNotb)
    print(bNota)

    for cluster in aAndb:
        printImageFromCluster(a, data_in_clusters[0], cluster,max_plot=8)
        printImageFromCluster(b, data_in_clusters[1], cluster,max_plot=8)

    print("-------------------------------")
    print("-            A Not B          -")
    print("-------------------------------")
    for cluster in aNotb:
        printImageFromCluster(a, data_in_clusters[0], cluster,max_plot=8)
    
    print("-------------------------------")
    print("-            B Not A          -")
    print("-------------------------------")
    for cluster in bNota:
        printImageFromCluster(b, data_in_clusters[1], cluster,max_plot=8)


# %%
unionIntersection(Vfcres, [birds, ships], 15)

# %%
###### Resnet only encoder

import ResNet
import importlib
importlib.reload(ResNet)
from ResNet import ResNet18Encoder, ResNet18Decoder

vae_enc_old = ResNet18Encoder()
vae_dec_old = ResNet18Decoder(
    latent_dim=latent_size,
    input_height=image_size
)
Vres_old = VAE(vae_enc_old, vae_dec_old).to(device)
#elo_loss, kl_loss, recon_loss = TrainModel(Vres, 0)
#PlotAllLoss([elo_loss, kl_loss, recon_loss], ["EBLO", "KL", "Recon"])
#PlotLoss(elo_loss)
