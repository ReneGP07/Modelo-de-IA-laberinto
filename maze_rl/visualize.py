
import os
os.environ.setdefault("MPLBACKEND", "TkAgg")

import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def pump(fig=None, fps=30):
    if fig is not None:
        fig.canvas.draw_idle(); fig.canvas.flush_events()
    plt.pause(1.0/max(1,fps))

def dibujar_laberinto_en_ax(ax, lab):
    ax.clear()
    ax.imshow(lab, cmap="gray_r", interpolation="nearest")
    h,w = lab.shape
    ax.add_patch(Rectangle((-0.5,-0.5), w, h, fill=False, linewidth=6))
    ax.set_axis_off()
