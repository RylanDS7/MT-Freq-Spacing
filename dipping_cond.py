# Code by Rylan Stutters - github.com/RylanDS7

# SimPEG functionality
from simpeg import maps
from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import model_builder
from pymatsolver import Pardiso

# discretize functionality
from discretize import TreeMesh, TensorMesh
from discretize.utils import mkvc, active_from_xyz

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


rx_pts = np.linspace(-3000, 3000, 10)
rx_locs = np.zeros((10, 2))
rx_locs[:, 0] = rx_pts

dh = 25 # fine cell size

dom_width_x = 100000.0  # 100 km
dom_width_z = 100000.0  # 100 km

nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))

hx = [(dh, nbcx)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hz], x0="CC", diagonal_balance=True)


# Coarse refinement over the whole domain first
mesh.refine_box(
    [-50000, -50000],
    [50000, 0],
    levels=3,
    finalize=False
)

# Medium refinement in the core region
mesh.refine_box(
    [-7500, -15000],
    [7500, 0],
    levels=7,
    finalize=False
)

# Finer refinement within the rxs area
mesh.refine_box(
    [-3250, -12500],
    [3250, 0],
    levels=-3,
    finalize=False
)

refine_pts = np.zeros((len(rx_locs), 2))
for i, pt in enumerate(rx_locs):
    refine_pts[i] = [pt[0], pt[1]]
mesh.refine_points(refine_pts, padding_cells_by_level=[3, 2, 2], finalize=False)

mesh.finalize()
print(f"Cell Count: {mesh.n_cells}")


background_conductivity = 0.01
cond_conductivity = 0.1
air_conductivity = 1e-8

conductivity_model = air_conductivity * np.ones(mesh.nC)

earth_inds = mesh.cell_centers[:,1] < 0
conductivity_model[earth_inds] = background_conductivity

cond_pts = np.array([
    [ 2439,  -9939], [ 2500,  -9506], [ 2183,  -8768], [ 1530,  -7835],
    [  646,  -6854], [ -335,  -5970], [-1268,  -5317], [-2006,  -5000],
    [-2439,  -5061], [-2500,  -5494], [-2183,  -6232], [-1530,  -7165],
    [ -646,  -8146], [  335,  -9030], [ 1268,  -9683], [ 2006, -10000]
])

cond_indicies = model_builder.get_indices_polygon(mesh, cond_pts)

conductivity_model[cond_indicies] = cond_conductivity

# CHECKPOINT
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# plot_image outputs (v_type='CC' assumes values live at cell centers)
out = mesh.plot_image(
    conductivity_model,
    ax=ax,
    v_type="CC",
    cmap="viridis",
    grid=True,  # Set to True to overlay the mesh grid lines
    grid_opts={"color": "w", "alpha": 0.2, "linewidth": 0.5},
)

cb = plt.colorbar(out[0], ax=ax, orientation='vertical')
cb.set_label('Conductivity (S/m)')

# plot a zoomed in cross section
ax.set_xlim([-3500, 3500])
ax.set_ylim([-12000, 100])
plt.title(f"Conductivity Model")
plt.show()