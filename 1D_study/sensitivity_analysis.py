# Code by Rylan Stutters - github.com/RylanDS7

# SimPEG functionality
from simpeg import maps
from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import model_builder
from pymatsolver import Pardiso

# discretize functionality
from discretize import TensorMesh

import numpy as np
import matplotlib.pyplot as plt
import math


def generate_halfspace(rho):
    cell_widths = np.append(np.logspace(2, 1, 100), 5.0 * np.ones(50))
    mesh = TensorMesh([cell_widths], origin="N")

    rho_log = np.log(rho) # log of 100 Ohm-m
    model = rho_log * np.ones(mesh.nC)

    return model, mesh


def build_sim(mesh, freqs):
    rx_loc = np.array([-0.1])

    rx_list = [
            nsem.receivers.Impedance(
                locations_e=rx_loc,
                orientation="xy",
                component="real",
            ),
            nsem.receivers.Impedance(
                locations_e=rx_loc,
                orientation="xy",
                component="imag",
            ),
    ]

    src_list = []
    for f in freqs:
        src_list.append(
            nsem.sources.Planewave(
                receiver_list=rx_list,
                frequency=f,
            )
        )

    survey = nsem.Survey(src_list)

    mapping = maps.ExpMap()

    sim = nsem.Simulation1DElectricField(
        mesh,
        survey=survey,
        rhoMap=mapping,
        solver=Pardiso,
    )

    return sim


def plot_mesh_quantity(plot_q, sim, label='Quantity'):
    mesh = sim.mesh
    mapping = maps.ExpMap()

    fig, ax = plt.subplots(figsize=(6, 10))

    ax.step(np.append(plot_q, plot_q[-1]), mesh.nodes_x, where='post', color='blue', lw=2)

    for node in mesh.nodes_x:
        ax.axhline(node, color='gray', linestyle='--', alpha=0.3)

    ax.set_ylabel('Distance/Depth (m)')
    ax.set_xlabel(label)
    ax.set_title(f'1D {label}')
    ax.grid(True, alpha=0.1)
    plt.show()


# DOESNT WORK BECAUSE getJ IS NOT IMPLEMENTED FOR SIMPEG 1DSIM
def plot_J(m, sim):
    J = sim.getJ(m)
    mesh = sim.mesh

    cell_sensitivity = np.sqrt(np.sum(J**2, axis=0))

    active_cells = sim.active_cells
    plot_map = maps.InjectActiveCells(mesh, active_cells=active_cells, value_inactive=0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    mesh.plot_image(plot_map * cell_sensitivity, ax=ax, grid=True)
    ax.set_title("Total Sensitivity per Cell")
    plt.show()


def skin_depth(f, m, sim):
    mesh = sim.mesh
    widths = np.flip(mesh.h[0])
    rho = np.flip(sim.rhoMap * m)

    depth = 0
    A = 1
    for i, w in enumerate(widths):
        if A <= 1 / math.e:
            break

        delta = 503 * np.sqrt(rho[i] / f)
        A = A * np.exp(-w / delta)

        depth += w

    return depth



def plot_delta_sensitivity(m, sim):
    n_freq = len(freqs)

    weights = np.zeros(mesh.nC)
    for f in freqs:
        depth = skin_depth(f, m, sim)
        seen_cells_mask = mesh.cell_centers > - depth
        seen_cells = np.where(seen_cells_mask)[0]

        weight = 1 / (n_freq * len(seen_cells))
        weights[seen_cells] += weight

    plot_mesh_quantity(weights, sim, label='Skin Depth Weight')

    



model, mesh = generate_halfspace(10)
freqs = [100, 8.88096, 3.06581, 1.53673, 0.92058, 0.6124, 0.43663, 0.32693, 0.25392, 0.20289, 0.16583, 0.13807, 0.11674, 0.1]
sim = build_sim(mesh, freqs)

# plot_mesh_quantity(sim.rhoMap * model, sim)

plot_delta_sensitivity(model, sim)