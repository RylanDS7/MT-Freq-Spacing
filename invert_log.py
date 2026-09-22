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
    mesh, active_cells=active_cells, value_inactive=np.log(1/1e-8)
)
expmap = maps.ExpMap()
recipmap = maps.ReciprocalMap()

background_cond = 0.01
m0 = (np.ones(mesh.nC) * np.log(1/background_cond))[active_cells]


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

reg.alpha_s = 1e-8
reg.alpha_x = 1
reg.alpha_z = 1

opt = optimization.ProjectedGNCG(maxIter=20, upper=np.log(1/1e-5), lower=np.log(1/100))
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

# Run Inversion
minv_tetm = inv.run(m0)

