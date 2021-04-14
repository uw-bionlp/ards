

import seaborn as sns

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



def font_adjust(font_size=10, font_family='serif', font_type='Times New Roman'):

    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.serif'] = [font_type] + plt.rcParams['font.serif']

    params = {'axes.labelsize': font_size, 'axes.titlesize':font_size, 'legend.fontsize': font_size, 'xtick.labelsize': font_size, 'ytick.labelsize': font_size}
    mpl.rcParams.update(params)
