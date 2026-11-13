"""Headless matched observed/prediction/residual plots with shared scaling."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def residual_vmax(pairs):
    if not pairs: raise ValueError('empty comparison')
    return max(float(np.max(np.abs(np.asarray(obs)-np.asarray(pred)))) for obs,pred in pairs) or 1e-12


def save_panel(path,observed,predictions,labels,title,vmax):
    fig,axes=plt.subplots(2,len(predictions)+1,figsize=(3*(len(predictions)+1),6),dpi=100,squeeze=False)
    axes[0,0].imshow(observed,cmap='gray',vmin=0,vmax=1); axes[0,0].set_title('Observed'); axes[1,0].axis('off')
    for j,(pred,label) in enumerate(zip(predictions,labels),1):
        axes[0,j].imshow(pred,cmap='gray',vmin=0,vmax=1); axes[0,j].set_title(label)
        axes[1,j].imshow(np.abs(np.asarray(observed)-pred),cmap='magma',vmin=0,vmax=vmax); axes[1,j].set_title('|Residual|')
    for ax in axes.flat: ax.axis('off')
    fig.suptitle(title); fig.subplots_adjust(left=.02,right=.98,top=.87,bottom=.03,wspace=.08,hspace=.1); fig.savefig(path,dpi=100); plt.close(fig)


def save_gif(frames,path,duration_ms=150):
    images=[]
    for frame in frames:
        with Image.open(frame) as image: images.append(image.convert('RGB'))
    if not images or len({x.size for x in images})!=1: raise ValueError('GIF frames must have a common size')
    images[0].save(path,save_all=True,append_images=images[1:],duration=duration_ms,loop=0,optimize=False,disposal=2)
