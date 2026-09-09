import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import networkx as nx
import os
BASE=os.path.dirname(os.path.abspath(__file__))
W=BASE
OUT=f"{W}/figures_300dpi"; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans'})
INT_CMAP=matplotlib.colormaps['Purples']; IMP_CMAP=matplotlib.colormaps['Greens']
RESP=['sdm','ldm','gdm']
def save(fig,b):
    fig.savefig(f"{OUT}/{b}.png",dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(f"{OUT}/{b}.pdf",bbox_inches='tight',facecolor='white'); plt.close(fig)
def load(r):
    df=pd.read_csv(f"{W}/csv/vivi_v2_{r}.csv",index_col=0); df.columns=df.index; return df
def reorder(df):
    o=np.argsort(np.diag(df.values))[::-1]; return df.iloc[o,o],[df.index[i] for i in o]

# ---- GLOBAL shared scales across the three outputs ----
alldiag=[]; alloff=[]
for r in RESP:
    M=load(r).values; alldiag.append(np.diag(M))
    off=M.copy(); np.fill_diagonal(off,np.nan); alloff.append(off[~np.isnan(off)])
alldiag=np.concatenate(alldiag); alloff=np.concatenate(alloff)
NIMP=Normalize(alldiag.min(),alldiag.max())          # shared Vimp
NINT=Normalize(0.0,alloff.max())                     # shared Vint (0..global max)
INT_THR=0.30*alloff.max()                            # common absolute edge threshold
print("shared Vimp",round(alldiag.min(),4),round(alldiag.max(),4),"| shared Vint 0",round(alloff.max(),4),"| thr",round(INT_THR,4))

# ---- Figure 2 (unchanged, regenerated) ----
def fig2():
    e=np.load(f"{W}/eval_v2.npz"); pred,act=e['pred_te'],e['act_te']
    order=['gdm','ldm','sdm']; idx={'sdm':0,'ldm':1,'gdm':2}
    fig,axes=plt.subplots(1,3,figsize=(13,4.7))
    for ax,name in zip(axes,order):
        k=idx[name]; a,p=act[:,k],pred[:,k]
        r2=1-np.sum((a-p)**2)/np.sum((a-a.mean())**2); rmse=np.sqrt(np.mean((a-p)**2))
        ax.scatter(a,p,s=26,facecolor='#2b2b2b',edgecolor='none',alpha=0.45)
        hi=max(a.max(),p.max())*1.05; ax.plot([0,hi],[0,hi],color='red',ls='--',lw=1.3)
        ax.set_xlim(-0.05*hi,hi); ax.set_ylim(-0.05*hi,hi); ax.set_aspect('equal','box')
        ax.set_title(f"{name.upper()}  (R² = {r2:.2f}, RMSE = {rmse:.2f} t ha⁻¹)",fontsize=12,fontweight='bold')
        ax.set_xlabel("Simulated by JULES-crop (t ha⁻¹)")
        ax.grid(True,color='#ececec'); ax.set_axisbelow(True)
        for s in ax.spines.values(): s.set_visible(False)
    axes[0].set_ylabel("Emulator prediction (t ha⁻¹)")
    fig.suptitle("Observed (JULES-crop) vs predicted (neural emulator)",fontsize=15,x=0.09,ha='left')
    fig.tight_layout(rect=[0,0,1,0.95]); save(fig,"fig2_obs_pred")

def heatmap(r):
    df,labs=reorder(load(r)); M=df.values; n=len(labs)
    rgba=np.zeros((n,n,4))
    for i in range(n):
        for j in range(n):
            rgba[i,j]=IMP_CMAP(NIMP(M[i,j])) if i==j else INT_CMAP(NINT(M[i,j]))
    fig=plt.figure(figsize=(7.6,6.4))
    gs=gridspec.GridSpec(2,2,width_ratios=[1,0.05],height_ratios=[1,1],wspace=0.06,hspace=0.35)
    ax=fig.add_subplot(gs[:,0]); ax.imshow(rgba,aspect='equal')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labs,rotation=90,fontsize=10); ax.set_yticklabels(labs,fontsize=10)
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(np.arange(-.5,n,1),minor=True); ax.set_yticks(np.arange(-.5,n,1),minor=True)
    ax.grid(which='minor',color='white',lw=1.0); ax.tick_params(which='both',length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(f"VIVID heatmap — {r.upper()}",fontsize=14,pad=26)
    c1=fig.add_subplot(gs[0,1]); cb1=fig.colorbar(cm.ScalarMappable(NINT,INT_CMAP),cax=c1); cb1.set_label("Interaction (Vint)",fontsize=10)
    c2=fig.add_subplot(gs[1,1]); cb2=fig.colorbar(cm.ScalarMappable(NIMP,IMP_CMAP),cax=c2); cb2.set_label("Importance (Vimp)",fontsize=10)
    save(fig,f"heat_{r}")

def network(r):
    df,labs=reorder(load(r)); M=df.values; n=len(labs); diag=np.diag(M)
    G=nx.Graph()
    for i,l in enumerate(labs): G.add_node(l,imp=diag[i])
    for i in range(n):
        for j in range(i+1,n):
            if M[i,j]>=INT_THR: G.add_edge(labs[i],labs[j],w=M[i,j])
    G.remove_nodes_from([x for x,dd in dict(G.degree()).items() if dd==0])
    if G.number_of_nodes()==0: return
    pos=nx.spring_layout(G,seed=7,k=1.3/np.sqrt(G.number_of_nodes()),iterations=250)
    fig=plt.figure(figsize=(8.6,6.2))
    gs=gridspec.GridSpec(2,2,width_ratios=[1,0.04],height_ratios=[1,1],wspace=0.04,hspace=0.35)
    ax=fig.add_subplot(gs[:,0]); ax.axis('off')
    ecol=[INT_CMAP(NINT(G.edges[e]['w'])) for e in G.edges]; ew=[1.5+7*NINT(G.edges[e]['w']) for e in G.edges]
    nx.draw_networkx_edges(G,pos,ax=ax,edge_color=ecol,width=ew)
    ncol=[IMP_CMAP(NIMP(G.nodes[x]['imp'])) for x in G.nodes]; nsz=[350+2600*NIMP(G.nodes[x]['imp']) for x in G.nodes]
    nx.draw_networkx_nodes(G,pos,ax=ax,node_color=ncol,node_size=nsz,edgecolors='grey',linewidths=0.6)
    nx.draw_networkx_labels(G,pos,ax=ax,font_size=10)
    ax.set_title(f"VIVID interaction network — {r.upper()}",fontsize=14); ax.margins(0.18)
    c1=fig.add_subplot(gs[0,1]); cb1=fig.colorbar(cm.ScalarMappable(NINT,INT_CMAP),cax=c1); cb1.set_label("Interaction (Vint)",fontsize=10)
    c2=fig.add_subplot(gs[1,1]); cb2=fig.colorbar(cm.ScalarMappable(NIMP,IMP_CMAP),cax=c2); cb2.set_label("Importance (Vimp)",fontsize=10)
    save(fig,f"net_{r}")

fig2()
for r in RESP: heatmap(r); network(r)
print("done:",sorted(os.listdir(OUT)))
