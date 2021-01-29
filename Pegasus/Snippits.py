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
