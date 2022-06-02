""" Present an interactive function explorer with slider widgets.

Scrub the sliders to change the properties of the ``model`` curve, or
type into the title text box to update the title of the plot.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve model_kor.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/sliders

in your browser.

"""

import math as mt

import numpy as np
import caracnl

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput
from bokeh.plotting import figure


# Computed frequency (x-axis)
N = 51  # Number of data points
f = np.logspace(-1, 1, N)  # Frequency MHz
omega = f*2*np.pi * 1e6  # Angular pulsation rad/s

# Linear parameters
A_prot = 1e9
R_korb = 1.3
tau_m = 2.4e-11
tau_s = 5e-7

tau = np.sqrt(tau_m * tau_s)


def func_J(omega, tau, tau_s):
    return 2*mt.gamma(0.5)*tau / (1+(omega*tau_s)**2)**(1/4)*np.cos(0.5*np.arctan(omega*tau_s))


def my_model(omega, A_prot, tau_s, tau_m, R_korb):
    N_s = 0.1
    tau = np.sqrt(tau_s*tau_m)
    return A_prot * N_s * (3*func_J(omega, tau, tau_s)+7*func_J(660*omega, tau, tau_s) + R_korb)


# Model evaluation
val = my_model(omega, A_prot, tau_s, tau_m,  R_korb)

# Figure initialization
source = ColumnDataSource(data=dict(x=np.log10(f), y=val))
plot = figure(width=500, plot_height=500, title=None, tools="crosshair,pan,reset,save,wheel_zoom")
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)


A_prot = 1e9
R_korb = 1.3
tau_m = 2.4e-11
tau_s = 5e-7


# Set up widgets


A_PROT = Slider(title=r"A_prot []", value=np.log10(A_prot), start=8, end=10, step=0.1)
TAU_S = Slider(title=r"$\tau_s$ [s]", value=np.log10(tau_s), start=-12, end=-6, step=0.1)
TAU_M = Slider(title=r"$\tau_m$ [s]", value=np.log10(tau_m), start=-12, end=-6, step=0.1)
R_KORB = Slider(title=r"R_korb []", value=R_korb, start=0, end=2, step=0.1)

text = TextInput(title="title",
                 value=rf'A_prot={A_prot:.1e}\n tau_s={tau_s:.1e}\n tau_m={tau_m:.1e}\nR={R_korb:.2f}')


widgets = [text, A_PROT, TAU_S, TAU_M, R_KORB]
sliders = [A_PROT, TAU_S, TAU_M, R_KORB]


# Set up callbacks
# noinspection PyUnusedLocal
def update_title(attrname, old, new):
    plot.title.text = text.value


text.on_change('value', update_title)


# noinspection PyUnusedLocal
def update_data(attrname, old, new):
    # Get the current slider values

    A_prot = 10**A_PROT.value
    tau_m = 10**TAU_M.value
    tau_s = 10**TAU_S.value
    R_korb = R_KORB.value

    # text.value = rf"A_prot={A_prot:.1e}\n tau_s={tau_s:.1e}\n tau_m={tau_m:.1e}\nR={R_korb:.2f}"

    val = np.log10(my_model(omega, A_prot, tau_s, tau_m, R_korb))

    source.data = dict(x=f, y=val)


for slider in sliders:
    slider.on_change('value', update_data)


# Set up layouts and add to document
inputs = column(*widgets)
# plots = column(plot_real, plot_imag, plot_comp)

curdoc().add_root(row(inputs, plot, width=1000))
curdoc().title = "Model JP"
