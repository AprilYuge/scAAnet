import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib
from matplotlib.colors import Normalize

def getURL(term, gmt):
    with open(gmt, 'r') as file:
        for line in file:
            if term in line:
                line = line.strip().split('\t')
                return line[1]
            
def convertTermREACTOME(term, gmt):
    url = getURL(term, gmt)
    file = urllib.request.urlopen(url)
    
    # Locate the line that contains the decription
    mark = 0
    for line in file:
        decoded_line = line.decode("utf-8")
        if mark == 1:
            info = decoded_line
            break
        if 'Brief description' in decoded_line:
            mark = 1
    
    # Get term description
    start = info.find("<td>") + len("<td>")
    end = info.find("</td>")
    substring = info[start:end]
    
    return substring

def convertTermGO(term, gmt):
    url = getURL(term, gmt)
    file = urllib.request.urlopen(url)
    
    # Locate the line that contains the external link
    mark = 0
    for line in file:
        decoded_line = line.decode("utf-8")
        if mark == 1:
            info = decoded_line
            break
        if 'External links' in decoded_line:
            mark = 1
    
    # Get the external link
    start = info.find(">http:") + 1
    end = info.find("</a></td>")
    url = info[start:end]
    
    file = urllib.request.urlopen(url)
    for line in file:
        decoded_line = line.decode("utf-8")          
        if 'Term Details for &quot;' in decoded_line:
            info = decoded_line
            break
            
    # Get term description
    start = info.find("Term Details for &quot;") + len("Term Details for &quot;")
    end = info.find("&quot; (GO:")
    substring = info[start:end]
    
    return substring


def unique(seq):
    """Remove duplicates from a list in Python while preserving order.
    :param seq: a python list object.
    :return: a list without duplicates while preserving order.
    """

    seen = set()
    seen_add = seen.add
    """
    The fastest way to sovle this problem is here
    Python is a dynamic language, and resolving seen.add each iteration
    is more costly than resolving a local variable. seen.add could have
    changed between iterations, and the runtime isn't smart enough to rule
    that out. To play it safe, it has to check the object each time.
    """

    return [x for x in seq if x not in seen and not seen_add(x)]

def dotplot(res, title, gmt, go=True, cutoff=0.05, n_terms=10, figsize=(6,5.5), 
            cmap='viridis_r', savefile=False, saveplot=False): 
    
    column = 'Adjusted P-value'
    colname = column
    res = res.sort_values(by=colname)
    df = res[res[colname] < cutoff]
    
    # Convert term names to brief descriptions
    if go:
        for i in range(df.shape[0]):
            df['Term'].iloc[i] = convertTermGO(df['Term'].iloc[i], gmt)
    else:
        for i in range(df.shape[0]):
            df['Term'].iloc[i] = convertTermREACTOME(df['Term'].iloc[i], gmt)
    
    if savefile:
        # Save enrichment results sorted by adjusted p values to file
        df.to_excel(savefile)
        
    if df.shape[0] > n_terms:
        df = df.iloc[:n_terms]
    df = df.iloc[::-1]
        
    df = df.assign(logAP=lambda x: - x[colname].apply(np.log10))
    colname='logAP'
    
    temp = df['Overlap'].str.split("/", expand=True).astype(int)
    df = df.assign(Hits=temp.iloc[:,0], Background=temp.iloc[:,1])
    df = df.assign(Hits_ratio=lambda x:x.Hits / x.Background)
    
    # x axis values
    x = df.loc[:, colname].values
    xlabel = "-log$_{10}$(%s)"%column
    hits_ratio = df['Hits_ratio'].round(2).astype('float')
    # y axis index and values
    y = [i for i in range(0,len(df))]
    ylabels = df['Term'].values
    
    # control the size of scatter and legend marker
    levels = numbers = np.sort(df.Hits.unique())
    norm = Normalize()
    min_width, max_width = np.r_[20, 100] * plt.rcParams["lines.linewidth"]
    norm.clip = True
    if not norm.scaled():
        norm(np.asarray(numbers))
    size_limits = norm.vmin, norm.vmax
    scl = norm(numbers)
    widths = np.asarray(min_width + scl * (max_width - min_width))
    if scl.mask.any():
        widths[scl.mask] = 0
    sizes = dict(zip(levels, widths))
    df['sizes'] = df.Hits.map(sizes)
    area = df['sizes'].values
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(x=x, y=y, s=area, edgecolors='face', c=hits_ratio,
                    cmap=cmap)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.yaxis.set_major_locator(plt.FixedLocator(y))
    ax.yaxis.set_major_formatter(plt.FixedFormatter(ylabels))
    ax.set_yticklabels(ylabels, fontsize=16)
    ax.grid()
    
    # colorbar
    cax=fig.add_axes([0.95,0.20,0.03,0.22])
    cbar = fig.colorbar(sc, cax=cax,)
    cbar.ax.tick_params(right=True)
    cbar.ax.set_title('Hits\nRatio',loc='left', fontsize=12)
    
    idx = [area.argmax(), np.abs(area - area.mean()).argmin(), area.argmin()]
    idx = unique(idx)
    label = df.iloc[idx, df.columns.get_loc('Hits')]
    
    handles, _ = ax.get_legend_handles_labels()
    legend_markers = []
    for ix in idx: 
        legend_markers.append(ax.scatter([],[], s=area[ix], c='b'))
    ax.legend(legend_markers, label, title='Hits')
    ax.set_title(title, fontsize=20, fontweight='bold')
    
    if saveplot:
        # Save plot to file
        plt.savefig(saveplot, dpi=500, bbox_inches="tight")
        plt.close()