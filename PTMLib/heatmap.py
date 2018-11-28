# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:04:45 2018

@author: BT
"""

######## This script creates a heatmap / table from PTM profile data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from matplotlib import rc
#from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap,LogNorm
#from matplotlib.colors import LogNorm

fontsize = 12
axeswidth = 1.5
#figSize = (4.5,3)

# Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
rc('font',**{'family':'sans-serif','sans-serif':['Arial'],'size': fontsize,'weight':'bold'})
plt.rcParams['axes.labelsize'] = fontsize 
plt.rcParams['axes.labelweight'] = 'bold' 
plt.rcParams['axes.linewidth'] = axeswidth





## Load data file
df = pd.read_csv( 'Milk_3_Store_Comparision_Python.csv')


## First row is actually masses of the PTMs, lets keep those apart:

masses = df.iloc[0]
masses = masses.iloc[2:]

df = df.iloc[1:]

## Average by class:

#groupedClass = df.groupby(['Class']).agg(['mean','std'])
groupedClass = df.groupby(['Class']).agg(['mean'])
groupedClassSD = df.groupby(['Class']).agg(['std'])

groupedClass = groupedClass.transpose()
groupedClassSD = groupedClassSD.transpose()


numPTM = groupedClass.astype(bool).sum(axis=0).as_matrix().reshape((1, -1))

groupedClass = groupedClass.as_matrix()
groupedClassSD = groupedClassSD.as_matrix()

groupedClass = np.concatenate( [numPTM, groupedClass] )
groupedClassSD = np.concatenate( [[[0, 0 ,0 ,0 ,0 ,0]], groupedClassSD] )

#%%
ClassLabels  = df['Class Names'].unique()
PTMLabels = masses.index






##
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="",addgrid=False, **kwargs):

    if not ax:
        ax = plt.gca()


    vm = data[1:,:].max()*1.01

    # Plot the heatmap
    im = ax.imshow(data, vmax=vm, **kwargs)
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.15, shrink=0.5, aspect=10)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center",
             rotation_mode="default")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)


    if addgrid:
        
        shift1 = 1
        shift2 = 0.5
        lw = axeswidth 
    else:
        shift1 = 0
        shift2 = 0
        lw =0
    

    ax.set_xticks(np.arange(data.shape[1]+shift1)-shift2, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+shift1)-shift2, minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=lw)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data, datasd,
                     textcolors=["black", "white"],
                     threshold=None, **textkw):

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)


    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if i > 0:
                    
                if data[i, j]:
                    kw.update(color='black')
                    text = im.axes.text(j, i,
                                        '{:.1g} ± {:.1g}'.format( data[i, j], datasd[i, j] ),
                                        horizontalalignment='center',
                                        verticalalignment='center')
                    texts.append(text)
            else:
                kw.update(color='black')
                text = im.axes.text(j, i,
                                    '{:.0f}'.format( data[i, j] ),
                                    horizontalalignment='center',
                                    verticalalignment='center')
                texts.append(text)

    return texts



def createHeatmap( groupedClass, groupedClassSD , ylabels, ClassLabels, outputname, figsize=(12,9)  ):
    colors = [ (255/255, 0/255, 0/255), (253/255, 150/255, 195/255), (199/255, 94/255, 71/255), (253/255, 250/255, 1/255), (0/255, 176/255, 240/255)]  # R -> G -> B
    cm = LinearSegmentedColormap.from_list( 'PTM_List', colors, N=255)
    cm.set_bad(color='white')
    cm.set_over(color='white')
    
    
    fig, ax = plt.subplots(figsize=figsize)
        
    im, cbar = heatmap(groupedClass, ylabels, ClassLabels, ax=ax,
                       cmap=cm, cbarlabel="AvP",norm=LogNorm(),addgrid=True, aspect='auto')
    
    annotate_heatmap(im, data=groupedClass, datasd=groupedClassSD)
    
    fig.tight_layout()
    plt.show()
    
    fig.savefig(outputname + '.png', dpi=400, bbox_inches='tight', transparent=True)
    fig.savefig(outputname+ '.eps', dpi=400, bbox_inches='tight', transparent=True)


def createVennDiagram( Classes, ClassLabels, outputname, figsize=(12,9) ):
    from matplotlib_venn import venn3
    
#    data = groupedClass.astype(bool)
    fig, ax = plt.subplots(figsize=figsize)

    Classes = Classes.reset_index()
    l1 = Classes[Classes[1] > 0].index.values.astype(int)
    l2 = Classes[Classes[2] > 0].index.values.astype(int)
    l3 = Classes[Classes[3] > 0].index.values.astype(int)
    
    # Custom text labels: change the label of group A
    v=venn3(subsets = [set(l1), set(l2), set(l3)] , set_labels = ClassLabels )
#    v.!get_label_by_id('A').set_text('My Favourite group!')
    plt.show()

    
    fig.savefig(outputname + '.png', dpi=400, bbox_inches='tight', transparent=True)
    fig.savefig(outputname+ '.eps', dpi=400, bbox_inches='tight', transparent=True)