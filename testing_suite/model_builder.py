""" Present an interactive function explorer with slider widgets.

Scrub the sliders to change the properties of the ``model`` curve, or
type into the title text box to update the title of the plot.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve model_builder.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/sliders

in your browser.

"""


import numpy as np
import caracnl

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput
from bokeh.plotting import figure


# Computed frequency (x-axis)
N = 51  # Number of data points
f = np.linspace(60, 68, N)  # Frequency MHz
omega = f*2*np.pi * 1e6  # Angular pulsation rad/s

# Linear parameters
Q_c = 10**1.3  # Coupling quality factor
Q_lin = 10**3 * Q_c  # Linear quality factor
omega_0 = 64e6 * 2 * np.pi  # Resonant angular pulsation rad/s
P_vna = 10**(1/10)  # Read-out power mW

# Non-linear parameters
x1, P_cnl1 = 2., 10**(20/10)  # Model parameter 1sr non-linearity
x2, P_cnl2 = 2., 10**(20/10)  # Model parameter 2nd non-linearity

n1, d1 = caracnl.diophantine_approx(x1, 20)  # real -> int / int
n2, d2 = caracnl.diophantine_approx(x2, 20)

# Model construction
params = [caracnl.models.NonlinearModelParameters(n1, d1, P_cnl1),
          caracnl.models.NonlinearModelParameters(n2, d2, P_cnl2)]
model = caracnl.models.get_model(Q_c, Q_lin, omega_0, params)

# Model evaluation
s11 = model.evaluate(omega, P_vna)


# Figure initialization
# 1st figure: Real(s11)
source_real = ColumnDataSource(data=dict(x=f, y=20 * np.log10(np.real(s11))))
plot_real = figure(width=250, plot_height=250, title=None, tools="crosshair,pan,reset,save,wheel_zoom",
                   x_range=[60, 68], y_range=[-40., 1])
plot_real.line('x', 'y', source=source_real, line_width=3, line_alpha=0.6)
plot_real.scatter('x', 'y', source=source_real, marker="o", size=3)

# 2nd figure: Angle(s11)
source_imag = ColumnDataSource(data=dict(x=f, y=np.angle(s11)))
plot_imag = figure(width=250, plot_height=250, title=None, tools="crosshair,pan,reset,save,wheel_zoom",
                   x_range=[60, 68], y_range=[-np.pi, np.pi])
plot_imag.line('x', 'y', source=source_imag, line_width=3, line_alpha=0.6)
plot_imag.scatter('x', 'y', source=source_imag, marker="o", size=3)

# 3rd figure: Smith chart
source_comp = ColumnDataSource(data=dict(x=np.real(s11), y=np.imag(s11)))
plot_comp = figure(width=250, plot_height=250, aspect_scale=1., title=None, tools="crosshair,pan,reset,save,wheel_zoom",
                   x_range=[0, 1], y_range=[-0.6, 0.6])
plot_comp.line('x', 'y', source=source_comp, line_width=3, line_alpha=0.6)
plot_comp.scatter('x', 'y', source=source_comp, marker="o", size=3)


# Set up widgets
text = TextInput(title="title", value='Non-linear model')
frequency = Slider(title=r"Frequency [MHz]", value=64, start=60, end=68, step=0.2)
power = Slider(title=r"Power [dBm]", value=1., start=-35, end=20, step=0.2)
coupling = Slider(title=r"Coupling [log10]", value=1.3, start=0, end=3, step=0.1)
q_factor = Slider(title=r"Linear Q [log10]", value=3., start=-2, end=4, step=0.1)
model1_x = Slider(title=r"Model 1: exponent", value=2., start=0., end=2, step=0.05)
model1_P = Slider(title=r"Model 1: P_cnl [dBm]", value=20, start=-10., end=20., step=1)
model2_x = Slider(title=r"Model 2: exponent", value=2., start=0., end=2, step=0.05)
model2_P = Slider(title=r"Model 2: P_cnl [dBm]", value=20, start=-10, end=20, step=1)

widgets = [text, frequency, power, coupling, q_factor, model1_x, model1_P, model2_x, model2_P]
sliders = [frequency, power, coupling, q_factor, model1_x, model1_P, model2_x, model2_P]


# Set up callbacks
# noinspection PyUnusedLocal
def update_title(attrname, old, new):
    plot_real.title.text = text.value


text.on_change('value', update_title)


# noinspection PyUnusedLocal
def update_data(attrname, old, new):
    # Get the current slider values
    omega_0 = frequency.value * 2*np.pi * 1.e6
    P_vna = 10**(power.value/10)
    Q_c = 10**coupling.value
    Q_lin = 10**q_factor.value * Q_c
    n1, d1 = caracnl.diophantine_approx(model1_x.value, 20)
    P_cnl1 = 10 ** (model1_P.value / 10)
    n2, d2 = caracnl.diophantine_approx(model2_x.value, 20)
    P_cnl2 = 10 ** (model2_P.value / 10)

    # Generate the new curve
    params = [caracnl.models.NonlinearModelParameters(n1, d1, P_cnl1),
              caracnl.models.NonlinearModelParameters(n2, d2, P_cnl2 / 10)]
    model = caracnl.models.get_model(Q_c, Q_lin, omega_0, params)
    s11 = model.evaluate(omega, P_vna)

    source_real.data = dict(x=f, y=20 * np.log10(np.real(s11)))
    source_imag.data = dict(x=f, y=np.angle(s11))
    source_comp.data = dict(x=np.real(s11), y=np.imag(s11))


for slider in sliders:
    slider.on_change('value', update_data)


# Set up layouts and add to document
inputs = column(*widgets)
# plots = column(plot_real, plot_imag, plot_comp)

curdoc().add_root(row(inputs, plot_real, plot_imag, plot_comp, width=1000))
curdoc().title = "Non-linear characterization model"
