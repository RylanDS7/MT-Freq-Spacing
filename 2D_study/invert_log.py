# code by Rylan Stutters - github.com/RylanDS7

from simpeg import maps, data, optimization, regularization, inverse_problem, directives, inversion, data_misfit, utils
from discretize import TreeMesh
import discretize
import numpy as np
from pymatsolver import Pardiso
from simpeg.electromagnetics import natural_source as nsem
import matplotlib.pyplot as plt

from simpeg.utils import model_builder

te_dpred = np.load('data/te_dpred.npy')
tm_dpred = np.load('data/tm_dpred.npy')
freqs = np.load('data/freqs.npy')

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

te_src_list = []
tm_src_list = []

for f in freqs: 
    te_rx_list = [
        nsem.receivers.Impedance(rx_locs, orientation="xy", component="real"),
        nsem.receivers.Impedance(rx_locs, orientation="xy", component="imag"),
    ]
    tm_rx_list = [
        nsem.receivers.Impedance(rx_locs, orientation="yx", component="real"),
        nsem.receivers.Impedance(rx_locs, orientation="yx", component="imag"),
    ]

    te_src_list.append(nsem.sources.Planewave(te_rx_list, frequency=f))
    tm_src_list.append(nsem.sources.Planewave(tm_rx_list, frequency=f))

te_survey = nsem.survey.Survey(te_src_list)
te_data = data.Data(te_survey, te_dpred)

tm_survey = nsem.survey.Survey(tm_src_list)
tm_data = data.Data(tm_survey, tm_dpred)

active_cells = discretize.utils.mesh_utils.active_from_xyz(mesh, rx_locs)
actmap = maps.InjectActiveCells(
    mesh, active_cells=active_cells, value_inactive=np.log10(1/1e-8)
)
expmap = maps.ExpMap()
recipmap = maps.ReciprocalMap()

background_cond = 0.001
m0 = (np.ones(mesh.nC) * np.log10(1/background_cond))[active_cells]


# create the simulation
te_sim = nsem.simulation.Simulation2DElectricField(
    mesh,
    survey=te_survey,
    sigmaMap=recipmap * expmap * actmap,
    solver=Pardiso
)

tm_sim = nsem.simulation.Simulation2DMagneticField(
    mesh,
    survey=tm_survey,
    sigmaMap=recipmap * expmap * actmap,
    solver=Pardiso
)


# TE mode
te_dmisfit = data_misfit.L2DataMisfit(data=te_data, simulation=te_sim)
te_data.standard_deviation = np.abs(te_data.dobs) * 0.05

# TM mode
tm_dmisfit = data_misfit.L2DataMisfit(data=tm_data, simulation=tm_sim)
tm_data.standard_deviation = np.abs(tm_data.dobs) * 0.05

# assign the weights
te_dmisfit.W = 1. / te_data.standard_deviation
tm_dmisfit.W = 1. / tm_data.standard_deviation
dmisfit_combo = te_dmisfit + tm_dmisfit

coolingFactor = 2
coolingRate = 2
beta0_ratio = 1e0

# Map for a regularization
regmap = maps.IdentityMap(nP=int(active_cells.sum()))

reg = regularization.WeightedLeastSquares(mesh, active_cells=active_cells, mapping=regmap)

reg.alpha_s = 0
reg.alpha_x = 1
reg.alpha_z = 1

opt = optimization.ProjectedGNCG(maxIter=5, upper=8, lower=-3)
invProb_tetm = inverse_problem.BaseInvProblem(dmisfit_combo, reg, opt)
beta = directives.BetaSchedule(
    coolingFactor=coolingFactor, coolingRate=coolingRate
)
betaest = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
target = directives.TargetMisfit()

directiveList = [
    beta, 
    betaest, 
    target,
]

inv = inversion.BaseInversion(
    invProb_tetm, directiveList=directiveList)
opt.remember('xc')

# print("Testing")
# te_m0_dpred = te_sim.dpred(m0)

# for i in np.arange(len(rx_locs)):
#     plt.plot(te_dpred.reshape(len(freqs), len(rx_locs), 2)[:, i, 0])
#     plt.plot(te_m0_dpred.reshape(len(freqs), len(rx_locs), 2)[:, i, 0], c='orange')
#     plt.title(f"TE Receiver {i} Real")
#     plt.show()

# for i in np.arange(len(rx_locs)):
#     plt.plot(te_dpred.reshape(len(freqs), len(rx_locs), 2)[:, i, 1])
#     plt.plot(te_m0_dpred.reshape(len(freqs), len(rx_locs), 2)[:, i, 1], c='orange')
#     plt.title(f"TE Receiver {i} Imag")
#     plt.show()

# tm_m0_dpred = tm_sim.dpred(m0)

# for i in np.arange(len(rx_locs)):
#     plt.plot(tm_dpred.reshape(len(freqs), len(rx_locs), 2)[:, i, 0])
#     plt.plot(tm_m0_dpred.reshape(len(freqs), len(rx_locs), 2)[:, i, 0], c='orange')
#     plt.title(f"TM Receiver {i} Real")
#     plt.show()

# for i in np.arange(len(rx_locs)):
#     plt.plot(tm_dpred.reshape(len(freqs), len(rx_locs), 2)[:, i, 1])
#     plt.plot(tm_m0_dpred.reshape(len(freqs), len(rx_locs), 2)[:, i, 1], c='orange')
#     plt.title(f"TM Receiver {i} Imag")
#     plt.show()

# Run Inversion
minv_tetm = inv.run(m0)

cond_est = recipmap * expmap * actmap * minv_tetm
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# log conductivity
model = (cond_est)
print(np.log10(model))
model[~active_cells] = np.nan
clim = [(-1), (-3)]

dat = mesh.plot_image(
    np.log10(model),
    ax=ax,
    # grid=True,
    clim=clim,
    pcolor_opts={"cmap": "viridis"}
)

ax.set_title('Log Conductivity')
plt.colorbar(
    dat[0],
    cmap='viridis', 
    label=r'Log Conductivity ($\Omega$m^-1)',   
    shrink=0.6
).ax.tick_params(labelsize=14)

ax.set_aspect('equal')
ax.plot(
    rx_locs[:, 0],
    rx_locs[:, 1], 'k.'
)
ax.set_title("TE+TM mode - Weighted Least Squares Inversion")
ax.set_xlabel("easting (m)")
ax.set_ylabel("elevation (m)")
ax.set_xlim([-3500, 3500])
ax.set_ylim([-12000, 100])
plt.show()
# fig.savefig('out/dipping_cond_final_model.png')

J_matrix = te_sim.getJ(minv_tetm)

cell_sensitivity = np.sqrt(np.sum(J_matrix**2, axis=0))

# Plot the sensitivity mapped back onto your inversion mesh
plotmap = maps.InjectActiveCells(
    mesh, active_cells=active_cells, value_inactive=0
)
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
mesh.plot_image(plotmap * cell_sensitivity, ax=ax, grid=True)
ax.set_title("Total Sensitivity per Cell at Inversion Final State")
plt.show()