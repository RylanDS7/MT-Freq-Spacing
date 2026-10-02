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
    cell_widths = np.append(np.logspace(3, 1, 150), 10.0 * np.ones(100))
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


def plot_mesh_quantity(plot_q, sim, label='Quantity', ax=None):
    mesh = sim.mesh
    mapping = maps.ExpMap()

    if ax == None:
        plot = True
        fig, ax = plt.subplots(figsize=(6, 10))
    else:
        plot = False

    ax.step(np.append(plot_q, plot_q[-1]), mesh.nodes_x, where='post', color='blue', lw=2)

    for node in mesh.nodes_x:
        ax.axhline(node, color='gray', linestyle='--', alpha=0.3)

    ax.set_ylabel('Distance/Depth (m)')
    ax.set_xlabel(label)
    ax.set_ylim((-10000, 1))
    ax.set_title(f'1D {label}')
    ax.grid(True, alpha=0.1)

    if plot:
        plt.show()

    return ax


def compare_sensitivities(m, sim, title="", subfig=None):
    if subfig == None:
        fig, ax = plt.subplots(1, 2, figsize=(15, 10))
        axes = ax.flatten()
    else:
        axes = subfig.subplots(1, 2)

    axes[0] = plot_J(m, sim, ax=axes[0])
    axes[1] = plot_delta_sensitivity(m, sim, ax=axes[1])

    if subfig == None:
        fig.suptitle(title)
        plt.show()
    else:
        subfig.suptitle(title)
    
    return axes


def compare_models(m_list, sim_list, title_list):
    fig = plt.figure(layout="constrained")
    subfigs = fig.subfigures(nrows=len(m_list), ncols=1)

    for i, subfig in enumerate(subfigs):
        compare_sensitivities(m_list[i], sim_list[i], title=title_list[i], subfig=subfig)

    plt.show()


def plot_J(m, sim, ax=None):
    J = sim.getJ(m)

    cell_sensitivity = np.sqrt(np.sum(J**2, axis=0))

    plot_mesh_quantity(cell_sensitivity, sim, label="J Sensitivities", ax=ax)


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



def plot_delta_sensitivity(m, sim, ax=None):
    freqs = sim.survey.frequencies
    n_freq = len(freqs)

    weights = np.zeros(mesh.nC)
    for f in freqs:
        depth = skin_depth(f, m, sim)
        seen_cells_mask = mesh.cell_centers > - depth
        seen_cells = np.where(seen_cells_mask)[0]

        weight = 1 / (n_freq * len(seen_cells))
        weights[seen_cells] += weight

    plot_mesh_quantity(weights, sim, label='Skin Depth Weight', ax=ax)


def freqs_2_skin_depths(freqs, rho):
    return 503 * np.sqrt(rho / np.array(freqs))




model, mesh = generate_halfspace(10)

model_list = []
sim_list = []
title_list = []

freqs_skin_spaced = [100, 8.88096, 3.06581, 1.53673, 0.92058, 0.6124, 0.43663, 0.32693, 0.25392, 0.20289, 0.16583, 0.13807, 0.11674, 0.1]
sim = build_sim(mesh, freqs_skin_spaced)
# compare_sensitivities(model, sim, title=f"10 Ohmm Halfspace Skin Depth Spaced Frequencies")
sim_list.append(sim)
model_list.append(model)
title_list.append(f"10 Ohmm Halfspace Skin Depth Spaced Frequencies")

freq_med = [np.median(freqs_skin_spaced)]
sim = build_sim(mesh, freq_med)
# compare_sensitivities(model, sim, title=f"10 Ohmm Halfspace f={freq_med[0]}Hz")
sim_list.append(sim)
model_list.append(model)
title_list.append(f"10 Ohmm Halfspace f={freq_med[0]}Hz")

freqs_log_spaced = np.logspace(2, -1, 14)
sim = build_sim(mesh, freqs_log_spaced)
# compare_sensitivities(model, sim, title=f"10 Ohmm Halfspace Log Spaced Frequencies")
sim_list.append(sim)
model_list.append(model)
title_list.append(f"10 Ohmm Halfspace Log Spaced Frequencies")


compare_models(model_list, sim_list, title_list)


print(f"Skin Depth Spaced Skin Depths: {freqs_2_skin_depths(freqs_skin_spaced, 10)}")
print(f"Median Freq Skin Depths: {freqs_2_skin_depths(freq_med, 10)}")
print(f"Log Spaced Skin Depths: {freqs_2_skin_depths(freqs_log_spaced, 10)}")