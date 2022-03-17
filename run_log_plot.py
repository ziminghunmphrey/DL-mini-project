import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import argparse

def generate_plots(args):
    model = args.model
    dataset = args.dataset

    xx = np.load(f'{model}_xx_{dataset}.npy')
    yy = np.load(f'{model}_yy_{dataset}.npy')
    zz = np.load(f'{model}_zz_{dataset}.npy')

    zz = np.log(zz)

    plt.figure(figsize=(10, 10))
    plt.contour(xx, yy, zz)
    plt.savefig(f'results/{model}_log_contour_{dataset}.png', dpi=100)
    plt.close()

    ## 3D plot
    fig, ax = plt.subplots(subplot_kw={'projection' : '3d'})
   '' ax.set_axis_off()''
    surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    plt.savefig(f'results/{model}_log_surface_{dataset}.png', dpi=100,
                format='png', bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    ax.set_axis_off()

    def init():
        ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return fig,

    def animate(i):
        ax.view_init(elev=(15 * (i // 15) + i % 15) + 0., azim=i)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return fig,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=100, interval=20, blit=True)

    anim.save(f'results/{model}_log_surface_{dataset}.gif',
              fps=15,  writer='imagemagick')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()
    generate_plots(args)