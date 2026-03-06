import numpy as np
import matplotlib.pyplot as plt 
import pickle 
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import colorConverter
from matplotlib import patheffects
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle

colorlist = ['#1565C0', '#FFFFFF', '#E6B86A']
mycmap = LinearSegmentedColormap.from_list('mycmap', colorlist, N=256)
colors = np.array([
    [230/255, 184/255, 106/255],
    [21/255, 101/255, 192/255]
])
torch.set_default_dtype(torch.float64)

markers = np.array(['o', 's'])

with open('results/toydataset.pkl', 'rb') as file:
    dataset = pickle.load(file)

X1 = dataset['X1'][:,:2]
y1 = dataset['y1']
X2 = dataset['X2'][:,:2]
y2 = dataset['y2']

Xall = np.concatenate([X1, X2], axis=0)
yall = np.concatenate([y1, y2], axis=0)

def plot_dataset(ax, X, y, client): 
    visible_points1 = (X[:, 0] >= -6) & (X[:, 0] <= 7) & (X[:, 1] >= -3) & (X[:, 1] <= 3) & (y == 0)
    visible_points2 = (X[:, 0] >= -6) & (X[:, 0] <= 7) & (X[:, 1] >= -3) & (X[:, 1] <= 3) & (y == 1)

    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb):
        return '#{:02X}{:02X}{:02X}'.format(*rgb)

    def mix_colors(hex1, hex2, ratio=0.5):
        rgb1 = hex_to_rgb(hex1)
        rgb2 = hex_to_rgb(hex2)
        mixed = tuple(round(rgb1[i] * (1 - ratio) + rgb2[i] * ratio) for i in range(3))
        return rgb_to_hex(mixed)

    myc = ['#6AC1DE', '#CF7A7A']
    myc1 = [mix_colors(c, '#FFFFFF', 0.3) for c in myc]
    myc2 = [mix_colors(c, '#000000', 0.3) for c in myc]

    mycf = (myc1, myc2)[client]

    myedge = [colorConverter.to_rgba('white', alpha=.5), colorConverter.to_rgba('white', alpha=.5)]
    #filled_minus = MarkerStyle("_", fillstyle="full")
    ax.scatter(X[visible_points1, 0], X[visible_points1, 1], marker='s', c=mycf[0], 
        s=140, edgecolors=myedge[0], clip_on=True, linewidths=0.5)
    ax.scatter(X[visible_points2, 0], X[visible_points2, 1], marker='o', c=mycf[1], 
        s=140, edgecolors=myedge[1], clip_on=True, linewidths=0.5)

    #for xi, yi in zip(X[visible_points2, 0], X[visible_points2, 1]):
    #    ax.add_patch(Rectangle((xi - 0.25, yi - 0.03), 0.5, 0.06, facecolor=myc[1], edgecolor=myedge[1], linewidth=0.5))  # Adjust width and height


def plot_classifier(ax, m, S, X, y, title='',data=True, bayes=True, client=0,it=0,pos=None): 
    Nplot = 100
    xaxis = np.linspace(-6, 7, Nplot)
    yaxis = np.linspace(-3, 2, Nplot)
    xx, yy = np.meshgrid(xaxis, yaxis)
    Xtest = torch.Tensor(np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1), np.ones((Nplot*Nplot,1))], axis=1))

    if bayes == True: 
        Cs = torch.linalg.cholesky(S) 
        CsInv = torch.linalg.inv(Cs.T)
        num_samples = 1000
        for i in range(num_samples):
            theta = m + torch.linalg.matmul(torch.randn(3), CsInv)
            pred_logits = Xtest @ theta 

            if i == 0:
                probs = 1.0 / (1.0 + torch.exp(pred_logits))
            else:
                probs += 1.0 / (1.0 + torch.exp(pred_logits))

        probs = probs.detach().numpy()
        probs /= float(num_samples)
    else:
        pred_logits_sgd = Xtest @ m
        probs_sgd = 1.0 / (1.0 + torch.exp(pred_logits_sgd))
        probs = probs_sgd.detach().numpy()

    ax.set_xlim([-6, 7])
    ax.set_ylim([-3, 2])

    #colors = np.array([[1, 0, 0], [0, 0, 1]])
    #ax.contourf(xaxis, yaxis, probs.reshape(Nplot, Nplot), levels=50, cmap=mycmap)
    if bayes:
        contour = ax.contour(xaxis, yaxis, probs.reshape(Nplot, Nplot), [0.25, 0.5, 0.75], linewidths=[1,5,1], 
                colors=['gray', 'black', 'gray'], linestyles=['dashed', 'solid', 'dashed'],zorder=1)

        # labels = ax.clabel(
        #     contour, 
        #     levels=[0.25, 0.5, 0.75],
        #     fmt={0.5: f'{it+1}', 0.25: '', 0.75: ''},  # Label format for the level 0.5
        #     inline=True,        # Inline label with the contour line
        #     fontsize=24,        # Font size of the label
        #     colors='black',     # Text color
        #     inline_spacing=10,   # Space around the label
        #     use_clabeltext=True,
        #     manual=False,
        #     zorder=10
        # )
    
    else: 
        contour = ax.contour(xaxis, yaxis, probs.reshape(Nplot, Nplot), [0.5], linewidths=5, 
                colors=['black'], linestyles=['solid'], zorder=1)
        # labels = ax.clabel(
        #     contour, 
        #     levels=[0.5],
        #     fmt={0.5: f'{it+1}'},  # Label format for the level 0.5
        #     inline=True,        # Inline label with the contour line
        #     fontsize=24,        # Font size of the label
        #     colors='black',     # Text color
        #     inline_spacing=15,   # Space around the label
        #     use_clabeltext=True,
        #     manual=False,
        #     zorder=10
        # )

    # for label in labels:
    #     label.set_path_effects([
    #         patheffects.withStroke(linewidth=3, foreground="white"),
    #         patheffects.Normal()
    #     ])
        


with open('results/illu_bayes_admm_iterates.pkl', 'rb') as file:
    bayes_iterates = pickle.load(file)

with open('results/illu_admm_iterates.pkl', 'rb') as file:
    admm_iterates = pickle.load(file)

it = 1

for client in range(2): 
    fig, ax = plt.subplots(figsize=(13/1.5,4.5))

    Xs = [X1,X2]
    ys = [y1,y2]

    plot_dataset(ax, Xs[client], ys[client], client=client)
    for it in range(2): 
        plot_classifier(ax, 
                        bayes_iterates[it + 1]['m_clients'][client], 
                        bayes_iterates[it + 1]['S_clients'][client], 
                        Xs[client], 
                        ys[client], 
                        f'bay_client1_it{it}', 
                        data=(it==0), 
                        bayes=True,
                        client=client,
                        it=it)


    for spine in ax.spines.values():
        spine.set_edgecolor('lightgray') 
        spine.set_linewidth(1)      

    ax.grid(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    #ax.axis('equal')
    plt.tight_layout()
    plt.savefig(f'results/illu_client{client}_bayes.pdf')
    plt.close()

    fig, ax = plt.subplots(figsize=(13/1.5,4.5))
    plot_dataset(ax, Xs[client], ys[client], client=client)

    for it in range(0, 5, 2): 
        plot_classifier(ax, 
                        admm_iterates[it+1]['m_clients'][client], 
                        None, 
                        Xs[client], 
                        ys[client], 
                        f'admm_client1_it{it}', 
                        data=(it==0), 
                        bayes=False,
                        client=client,
                        it=it)
        
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgray') 
        spine.set_linewidth(1)      

    ax.grid(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    #ax.axis('equal')
    plt.tight_layout()
    plt.savefig(f'results/illu_client{client}_admm.pdf')
    plt.close()


fig, ax = plt.subplots(figsize=(13/1.5,4.5))

for client in range(2):
    plot_dataset(ax, Xs[client], ys[client], client=client)

for it in range(0,5,2): 
    plot_classifier(ax, 
                    admm_iterates[it + 1]['m_server'], 
                    None, 
                    Xs[client], 
                    ys[client], 
                    f'admm_client1_it{it}', 
                    data=(it==0), 
                    bayes=False,
                    client=client,
                    it=it)

for spine in ax.spines.values():
    spine.set_edgecolor('lightgray') 
    spine.set_linewidth(1)      

ax.grid(False)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

#ax.axis('equal')
plt.tight_layout()
plt.savefig(f'results/illu_server_admm.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(13/1.5,4.5))

for client in range(2):
    plot_dataset(ax, Xs[client], ys[client], client=client)

for it in range(2): 
    plot_classifier(ax, 
                    bayes_iterates[it + 1]['m_server'], 
                    bayes_iterates[it + 1]['S_server'], 
                    Xs[client], 
                    ys[client], 
                    f'bayes_client1_it{it}', 
                    data=(it==0), 
                    bayes=True,
                    client=client,
                    it=it)

for spine in ax.spines.values():
    spine.set_edgecolor('lightgray') 
    spine.set_linewidth(1)      

ax.grid(False)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

#ax.axis('equal')
plt.tight_layout()
plt.savefig(f'results/illu_server_bayes.pdf')
plt.close()



#plot_classifier_Bayes(bayes_iterates[it]['admm_m'][1], bayes_iterates[it]['admm_S'][1], X2, y2, f'bay_client2_it{it}')
#plot_classifier_Bayes(bayes_iterates[it]['mbar'], bayes_iterates[it]['Sbar'], Xall, yall, f'bay_server_it{it}')
