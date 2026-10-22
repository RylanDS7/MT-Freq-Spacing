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
from matplotlib.backends.backend_pdf import PdfPages
import math


def generate_halfspace(rho):
    cell_widths = np.append(np.logspace(3, 1, 150), 10.0 * np.ones(100))
    mesh = TensorMesh([cell_widths], origin="N")

    rho_log = np.log(rho) # log of 100 Ohm-m
    model = rho_log * np.ones(mesh.nC)

    return model, mesh


def build_sim(mesh, freqs, rho_back=100):
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
            nsem.sources.PlanewaveXYPrimary(
                receiver_list=rx_list,
                frequency=f,
            )
        )

    survey = nsem.Survey(src_list)

    mapping = maps.ExpMap()

    sim = nsem.Simulation1DPrimarySecondary(
        mesh,
        survey=survey,
        rhoMap=mapping,
        sigmaPrimary=np.zeros(mesh.nC) + 1 / rho_back,
        solver=Pardiso,
    )

    return sim


def plot_mesh_quantity(plot_q, sim, label='Quantity', ax=None, depth=10000):
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
    ax.set_ylim((-depth, 0))
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


def single_method_diff(m1, m2, s1, s2, method, depth=10000, show=True):
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    axes = ax.flatten()

    if method == 'skin':
        w1 = delta_sesitivity_weights(m1, s1)
        w2 = delta_sesitivity_weights(m2, s2)

        axes[0] = plot_mesh_quantity(w1, s1, label='Skin Depth Weight Model 1', ax=axes[0], depth=depth)
        axes[1] = plot_mesh_quantity(w2, s2, label='Skin Depth Weight Model 2', ax=axes[1], depth=depth)
        axes[2] = plot_mesh_quantity(w2 - w1, s1, label="Skin Depth Weight Model 2 - Model 1", ax=axes[2], depth=depth)


    if method == 'J':
        plotJ1 = get_plottingJ(m1, s1, normalized=False)
        plotJ2 = get_plottingJ(m2, s2, normalized=False)

        axes[0] = plot_mesh_quantity(plotJ1, s1, label='J Model 1 (Norm)', ax=axes[0], depth=depth)
        axes[1] = plot_mesh_quantity(plotJ2, s2, label='J Model 2 (Norm)', ax=axes[1], depth=depth)
        axes[2] = plot_mesh_quantity(plotJ2 - plotJ1, s1, label="J Model 2 - Model 1 (Norm)", ax=axes[2], depth=depth)

    if show:
        plt.show()
    else:
        return fig, axes
        


def single_sim_diff(m, s, depth=10000):
    # J normalization assumed
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    axes = ax.flatten()

    w = delta_sesitivity_weights(m, s)
    plotJ = get_plottingJ(m, s, normalized=True)

    axes[0] = plot_mesh_quantity(w, s, label='Skin Depth Weight', ax=axes[0], depth=depth)
    axes[1] = plot_mesh_quantity(plotJ, s, label='Normalized J Sensitivities', ax=axes[1], depth=depth)
    axes[2] = plot_mesh_quantity(plotJ - w, s, label='Normalized J - Skin Depth Weight', ax=axes[2], depth=depth)

    plt.show()


def get_plottingJ(m, sim, normalized=False):
    J = sim.getJ(m)

    cell_sensitivity = np.sqrt(np.sum(J**2, axis=0))
    if normalized:
        cell_sensitivity = normalize_J(cell_sensitivity)

    return cell_sensitivity


def plot_J(m, sim, ax=None, normalized=False):
    cell_sensitivity = get_plottingJ(m, sim, normalized)
    
    plot_mesh_quantity(cell_sensitivity, sim, label="J Sensitivities", ax=ax)


# TODO: NEEDS TO NORMALIZE BY AREA DEPENDANCE
def normalize_J(J):
    normJ = J / np.sum(J)
    return normJ


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

def delta_sesitivity_weights(m, sim):
    freqs = sim.survey.frequencies
    n_freq = len(freqs)

    weights = np.zeros(m.nC)
    for f in freqs:
        depth = skin_depth(f, m, sim)
        seen_cells_mask = m.cell_centers > - depth
        seen_cells = np.where(seen_cells_mask)[0]

        weight = 1 / (n_freq * len(seen_cells))
        weights[seen_cells] += weight

    return weights


def plot_delta_sensitivity(m, sim, ax=None):
    weights = delta_sesitivity_weights(m, sim)

    plot_mesh_quantity(weights, sim, label='Skin Depth Weight', ax=ax)


def freqs_2_skin_depths(freqs, rho):
    return 503 * np.sqrt(rho / np.array(freqs))


def get_skin_depth_samples(rho, fmax, fmin, Nfreq):
    delta_min = 503 * np.sqrt(rho / fmax)
    delta_max = 503 * np.sqrt(rho / fmin)

    delta = np.linspace(delta_min, delta_max, Nfreq)
    freqs = rho * (503 / delta)**2

    return freqs


def resistivity_sweep(upper_rho, lower_rho, Nmodels, method='J', title='rho_sweep.pdf'):
    rhos = np.logspace(np.log10(lower_rho), np.log10(upper_rho), Nmodels)

    fmax = 10
    fmin = 0.001
    Nfreq = 12

    skin_sims = []
    log_sims = []
    models = []
    for r in rhos:
        model, mesh = generate_halfspace(r)
        models.append(model)

        freqs_skin = get_skin_depth_samples(r, fmax, fmin, Nfreq)
        freqs_log = np.logspace(np.log10(fmin), np.log10(fmax), Nfreq)

        sim_skin = build_sim(mesh, freqs_skin, rho_back=r)
        sim_log = build_sim(mesh, freqs_log, rho_back=r)

        skin_sims.append(sim_skin)
        log_sims.append(sim_log)

    with PdfPages(title) as pdf:
        cover_fig = plt.figure(figsize=(15, 10))

        cover_title = "Resisitivity Sweep - J Plotting Skin Depth Spaced Freqs vs Log Spaced Freqs"
        cover_fig.text(0.5, 0.9, cover_title, va="top", ha="center", fontsize=20, wrap=False)

        freq_text = f"Log Spaced Freqs = {freqs_log}"
        cover_fig.text(0.5, 0.7, freq_text, va="top", ha="center", fontsize=16, wrap=False)

        rho_text = f"Resistivities = {rhos}"
        cover_fig.text(0.5, 0.5, rho_text, va="top", ha="center", fontsize=16, wrap=False)

        pdf.savefig(cover_fig)
        plt.close(cover_fig)

        for i, m in enumerate(models):
            fig, axes = single_method_diff(m, m, skin_sims[i], log_sims[i], method='J', depth=10000, show=False)
            fig.suptitle(f"{rhos[i]} Ohm m Halfspace Model")
            axes[0].set_title("Skin Depth Spaced J")
            axes[1].set_title("Log Spaced J")
            axes[2].set_title("Log Spaced J - Skin Depth Spaced J")
            pdf.savefig(fig)
            plt.close(fig)




# model, mesh = generate_halfspace(10)

# model_list = []
# sim_list = []
# title_list = []

# freqs_skin_spaced = [100, 8.88096, 3.06581, 1.53673, 0.92058, 0.6124, 0.43663, 0.32693, 0.25392, 0.20289, 0.16583, 0.13807, 0.11674, 0.1]
# sim = build_sim(mesh, freqs_skin_spaced)
# # compare_sensitivities(model, sim, title=f"10 Ohmm Halfspace Skin Depth Spaced Frequencies")
# sim_list.append(sim)
# model_list.append(model)
# title_list.append(f"10 Ohmm Halfspace Skin Depth Spaced Frequencies")

# freq_med = [np.median(freqs_skin_spaced)]
# sim = build_sim(mesh, freq_med)
# # compare_sensitivities(model, sim, title=f"10 Ohmm Halfspace f={freq_med[0]}Hz")
# sim_list.append(sim)
# model_list.append(model)
# title_list.append(f"10 Ohmm Halfspace f={freq_med[0]}Hz")

# freqs_log_spaced = np.logspace(2, -1, 14)
# sim = build_sim(mesh, freqs_log_spaced)
# # compare_sensitivities(model, sim, title=f"10 Ohmm Halfspace Log Spaced Frequencies")
# sim_list.append(sim)
# model_list.append(model)
# title_list.append(f"10 Ohmm Halfspace Log Spaced Frequencies")


# # compare_models(model_list, sim_list, title_list)

# single_method_diff(model_list[0], model_list[2], sim_list[0], sim_list[2], method='skin', depth=3000)

# single_sim_diff(model_list[0], sim_list[0], depth=3000)


resistivity_sweep(1000, 10, 10)