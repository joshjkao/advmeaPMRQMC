import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import sparse


def c_tau_beta(h, o, beta, tau):
    """
    C(\\tau) = \\langle O(\\tau) O \\rangle - \\langle O \\rangle ^2
    """
    evals, evecs = np.linalg.eigh(h)

    z = np.sum(np.exp(-beta * evals))

    corr = (
        np.sum(
            [
                np.exp(-(beta - tau) * evals[i])
                * np.exp(-tau * evals[j])
                * np.abs(np.dot(np.conj(evecs[:, i].T), np.dot(o, evecs[:, j]))) ** 2
                for i in range(len(evals))
                for j in range(len(evals))
            ]
        )
        / z
    )

    avgO = (
        np.sum(
            [
                np.exp(-beta * evals[i])
                * np.dot(np.conj(evecs[:, i].T), np.dot(o, evecs[:, i]))
                for i in range(len(evals))
            ]
        )
        / z
    )

    gTau = corr - (avgO) ** 2
    if gTau.imag < 1e-8:
        gTau = gTau.real

    return gTau


def Hdiag_kint_beta(h, o, beta, tau):
    """
    C(\\tau) = \\langle O(\\tau) O \\rangle
    """
    evals, evecs = np.linalg.eigh(h)

    z = np.sum(np.exp(-beta * evals))

    corr = (
        np.sum(
            [
                np.exp(-(beta - tau) * evals[i])
                * np.exp(-tau * evals[j])
                * np.abs(np.dot(np.conj(evecs[:, i].T), np.dot(o, evecs[:, j]))) ** 2
                for i in range(len(evals))
                for j in range(len(evals))
            ]
        )
        / z
    )

    gTau = corr
    if gTau.imag < 1e-8:
        gTau = gTau.real

    return gTau


def m_beta_chi_k(h, o, beta, k):
    """
    Integrates tau^k*c_tau_beta over tau from [0, beta/2].
    """
    result, error = scipy.integrate.quad(
        lambda tau: tau**k * c_tau_beta(h, o, beta, tau), 0, beta / 2
    )
    return result, error


def calc_Hdiag_kint(h, o, beta, k):
    result, error = scipy.integrate.quad(
        lambda tau: tau**k * Hdiag_kint_beta(h, o, beta, tau), 0, beta / 2
    )
    return result, error


def calc_approx_gs_gap_k(h, o, beta, k):
    """
    approximate the gap by numerically integrating the correlators
    and taking their ratio
    """
    m_0, _error_0 = m_beta_chi_k(h, o, beta, k)
    m_1, _error_1 = m_beta_chi_k(h, o, beta, k + 1)
    return m_0, m_1, m_0 / m_1


# symbolically calculate the k-th moment M_k^{\\beta/2}
def moment_estimator(k, o, evals):
    n = len(evals)
    return math.factorial(k) * np.sum(
        [np.abs(o[0, j]) ** 2 / (evals[j] - evals[0]) ** (k + 1) for j in range(1, n)]
    )


# k-th moment without the factorial in front
def Q_k(k, o, evals):
    return moment_estimator(k, o, evals) / math.factorial(k)


# ratio between the k-th and (k+1)-th moments
def R_k(k, o, evals):
    return Q_k(k, o, evals) / Q_k(k + 1, o, evals)


I2 = sparse.csr_matrix([[1, 0], [0, 1]], dtype=complex)
X2 = sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
Y2 = sparse.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
Z2 = sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)

PAULI_DICT = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}


def text_to_hamiltonian_matrix(
    input_data: str, num_qubits=None, sparse_output: bool = False
):
    """
    Parses a Pauli Hamiltonian string or text file into a matrix representation.
    Assumes 1-based qubit indexing matching Qiskit's right-to-left tensor order:
    |qN qN-1 ... q1>  --->  P_N (x) P_{N-1} (x) ... (x) P_1
    """
    # 1. Read input data from path or string
    try:
        with open(input_data, "r") as f:
            lines = f.readlines()
    except (OSError, ValueError):
        lines = input_data.strip().splitlines()

    # 2. Parse lines into coefficient and qubit-operator pairs
    parsed_terms = []
    max_qubit_idx = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        tokens = line.split()
        coeff = float(tokens[0])
        ops = {}

        for i in range(1, len(tokens), 2):
            q_idx = int(tokens[i])
            p_char = tokens[i + 1].upper()
            ops[q_idx] = p_char
            max_qubit_idx = max(max_qubit_idx, q_idx)

        parsed_terms.append((coeff, ops))

    # Infer or validate total qubit count N
    N = max_qubit_idx if num_qubits is None else num_qubits
    if num_qubits is not None and max_qubit_idx > num_qubits:
        raise ValueError(
            f"Max qubit index in input ({max_qubit_idx}) exceeds num_qubits ({num_qubits})."
        )

    if N == 0:
        return np.array([[0.0]])

    dim = 2**N
    H = sparse.csr_matrix((dim, dim), dtype=complex)

    # 3. Construct tensor products in Qiskit order (qN x qN-1 x ... x q1)
    for coeff, ops in parsed_terms:
        term_mat = PAULI_DICT[ops.get(N, "I")]
        for q in range(N - 1, 0, -1):
            term_mat = sparse.kron(term_mat, PAULI_DICT[ops.get(q, "I")], format="csr")

        H += coeff * term_mat

    # Drop imaginary part if Hamiltonian is purely real
    if np.allclose(H.data.imag, 0):
        H = H.real

    return H if sparse_output else H.toarray()


def run_numerics(
    h_file,
    o_file,
    k_values,
    beta_values,
    pt_format=False,
    h_gamma_file=None,
    gamma_values=None,
):
    h_base = text_to_hamiltonian_matrix(h_file)
    o = text_to_hamiltonian_matrix(o_file)

    # Load H_gamma if provided
    h_gamma = (
        text_to_hamiltonian_matrix(h_gamma_file) if h_gamma_file is not None else None
    )

    # Determine (beta, gamma) pairs
    if gamma_values is None:
        bg_pairs = [(beta, 0.0) for beta in beta_values]
    elif len(gamma_values) == len(beta_values):
        # 1D PT schedule pairing: replica i has (beta_i, gamma_i)
        bg_pairs = list(zip(beta_values, gamma_values))
    else:
        # Full grid evaluation: all beta x gamma combinations
        bg_pairs = [(beta, gamma) for gamma in gamma_values for beta in beta_values]

    # Group betas by unique gamma values to optimize sparse.linalg.eigsh calls
    gamma_to_betas = defaultdict(list)
    for beta, gamma in bg_pairs:
        gamma_to_betas[gamma].append(beta)

    data_rows = []

    # Loop over each unique gamma value
    for gamma, betas_for_gamma in gamma_to_betas.items():
        # Construct H_eff(gamma) = H_base + gamma * H_gamma
        if h_gamma is not None and gamma != 0.0:
            h_eff = h_base + gamma * h_gamma
        else:
            h_eff = h_base

        # Diagonalize effective Hamiltonian for the current gamma
        evals, evecs = sparse.linalg.eigsh(h_eff)
        o_rotated = evecs.conj().T @ o @ evecs

        for this_k in k_values:
            exact_bound = R_k(this_k, o_rotated, evals)

            for beta in betas_for_gamma:
                # Pass effective Hamiltonian h_eff to numerical integration
                num_den_estimate = calc_approx_gs_gap_k(h_eff, o, beta, this_k)

                moment = num_den_estimate[0] / math.factorial(this_k)
                if this_k % 2 == 0:
                    moment *= -1

                k_plus_one_moment = num_den_estimate[1] / math.factorial(this_k + 1)
                if this_k % 2 == 1:
                    k_plus_one_moment *= -1

                estimate = num_den_estimate[2]

                # Factorial rescaling: (k+1)! / k! = k + 1
                estimate *= this_k + 1

                Hdiag_kint, _ = calc_Hdiag_kint(h_eff, o, beta, this_k)
                Hdiag_kint /= math.factorial(this_k)
                if this_k % 2 == 0:
                    Hdiag_kint *= -1

                if not pt_format:
                    data_rows.append(
                        {
                            "k": this_k,
                            "exact_bound": exact_bound,
                            "beta": beta,
                            "gamma": gamma,
                            "M_k^{beta/2}": moment,
                            "R_k": estimate,
                        }
                    )
                else:
                    data_rows.append(
                        {
                            "beta": beta,
                            "gamma": gamma,
                            "name": f"measure_Hdiag_kint_{this_k}",
                            "exact": Hdiag_kint,
                        }
                    )
                    data_rows.append(
                        {
                            "beta": beta,
                            "gamma": gamma,
                            "name": f"ratio_k{this_k}",
                            "exact": estimate,
                        }
                    )
                    data_rows.append(
                        {
                            "beta": beta,
                            "gamma": gamma,
                            "name": f"susceptibility_k{this_k}",
                            "exact": moment,
                        }
                    )
                    if this_k == k_values[-1]:
                        data_rows.append(
                            {
                                "beta": beta,
                                "gamma": gamma,
                                "name": f"susceptibility_k{this_k + 1}",
                                "exact": k_plus_one_moment,
                            }
                        )
                        Hdiag_kint_plus_one, _ = calc_Hdiag_kint(
                            h_eff, o, beta, this_k + 1
                        )
                        Hdiag_kint_plus_one /= math.factorial(this_k + 1)
                        if this_k % 2 == 1:
                            Hdiag_kint_plus_one *= -1
                        data_rows.append(
                            {
                                "beta": beta,
                                "gamma": gamma,
                                "name": f"measure_Hdiag_kint_{this_k + 1}",
                                "exact": Hdiag_kint_plus_one,
                            }
                        )

    gap_estimates = pd.DataFrame(data_rows)
    return gap_estimates


import re
from collections import defaultdict


def parse_simulation_output(file_path_or_text, is_file=True):
    if is_file:
        with open(file_path_or_text, "r") as f:
            text = f.read()
    else:
        text = file_path_or_text

    metadata = {}

    # 1. Parse Simulation Parameters & General Statistics
    params_match = re.search(
        r"Parameters:\s*beta\s*=\s*(?P<beta>[\d.]+),\s*Tsteps\s*=\s*(?P<Tsteps>\d+),\s*steps\s*=\s*(?P<steps>\d+)",
        text,
    )
    if params_match:
        metadata["beta"] = float(params_match.group("beta"))
        metadata["Tsteps"] = int(params_match.group("Tsteps"))
        metadata["steps"] = int(params_match.group("steps"))

    mpi_match = re.search(r"Number of MPI processes:\s*(\d+)", text)
    if mpi_match:
        metadata["num_mpi_processes"] = int(mpi_match.group(1))

    mc_updates = re.search(r"Total number of MC updates\s*=\s*(\d+)", text)
    if mc_updates:
        metadata["total_mc_updates"] = int(mc_updates.group(1))

    # 2. Parse q, sgn(W), and Non-k Observables
    q_mean = re.search(r"Total mean\(q\)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    q_max = re.search(r"Total max\(q\)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if q_mean:
        metadata["mean_q"] = float(q_mean.group(1))
    if q_max:
        metadata["max_q"] = float(q_max.group(1))

    sgn_mean = re.search(
        r"Total mean\(sgn\(W\)\)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text
    )
    sgn_std = re.search(
        r"Total std\.dev\.\(sgn\(W\)\)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text
    )
    if sgn_mean:
        metadata["mean_sgn_w"] = float(sgn_mean.group(1))
    if sgn_std:
        metadata["std_sgn_w"] = float(sgn_std.group(1))

    h_diag = re.search(
        r"Total of observable #1: H_\{diag\}\s*\n"
        r"Total mean\(O\) = (?P<mean>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\n"
        r"Total std\.dev\.\(O\) = (?P<std>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        text,
    )
    if h_diag:
        metadata["h_diag_mean"] = float(h_diag.group("mean"))
        metadata["h_diag_std"] = float(h_diag.group("std"))

    # 3. Parse Timing Data
    cpu_time = re.search(r"Total elapsed cpu time\s*=\s*([\d.]+)\s*seconds", text)
    wall_time = re.search(r"Wall-clock time\s*=\s*([\d.]+)\s*seconds", text)
    if cpu_time:
        metadata["cpu_time_seconds"] = float(cpu_time.group(1))
    if wall_time:
        metadata["wall_time_seconds"] = float(wall_time.group(1))

    # 4. Parse k-indexed Observables Block
    block_pattern = re.compile(
        r"Total of (?:derived )?observable(?: #\d+)?: (?P<obs>\S+)\s*\n"
        r"Total mean\(O\) = (?P<mean>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\n"
        r"Total std\.dev\.\(O\) = (?P<std>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    target_pattern = re.compile(
        r"^(measure_Hdiag_kint_|susceptibility_k|ratio_k)(\d+)$"
    )

    data = defaultdict(dict)

    for match in block_pattern.finditer(text):
        obs_name = match.group("obs")
        mean_val = float(match.group("mean"))
        std_val = float(match.group("std"))

        k_match = target_pattern.match(obs_name)
        if k_match:
            prefix, k_str = k_match.groups()
            k = int(k_str)

            if "measure_Hdiag" in prefix:
                metric = "measure_Hdiag_kint"
            elif "susceptibility" in prefix:
                metric = "susceptibility"
            else:
                metric = "ratio"

            data[k][f"{metric}_mean"] = mean_val
            data[k][f"{metric}_std"] = std_val

    df = (
        pd.DataFrame.from_dict(data, orient="index")
        .rename_axis("k")
        .reset_index()
        .sort_values("k")
    )

    # Attach metadata dictionary directly to the DataFrame attributes as well
    df.attrs = metadata

    return df, metadata


def validate_pt(path):
    path = Path(path)
    observables = path / "pmrqmc_pt_observables.csv"
    hamiltonian = path / "H.txt"
    target_hamiltonian_gamma = path / "H_gamma.txt"
    hamiltonian_gamma = (
        target_hamiltonian_gamma if target_hamiltonian_gamma.exists() else None
    )
    test_operator = path / "O.txt"

    sim_data = pd.read_csv(observables)
    beta_values = np.unique(sim_data["beta"])
    gamma_values = np.unique(sim_data["gamma"])
    k_max = len(np.unique(sim_data[sim_data["name"].str.contains("ratio")]["name"]))
    k_values = [i for i in range(k_max)]
    theor = run_numerics(
        hamiltonian,
        test_operator,
        k_values,
        beta_values,
        True,
        h_gamma_file=hamiltonian_gamma,
        gamma_values=gamma_values,
    )

    merged = pd.merge(sim_data, theor, "left", on=["beta", "gamma", "name"])
    merged["zscore"] = (merged["mean"] - merged["exact"]) / merged["stdev"]

    return merged


if __name__ == "__main__":
    path = Path(sys.argv[1])
    df = validate_pt(path)
    df.to_csv(path / "validation_out.csv")
