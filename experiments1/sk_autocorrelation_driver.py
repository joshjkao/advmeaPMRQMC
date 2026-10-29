import argparse

from hamiltonian import generate_quantum_sk_hamiltonian
from single_run_autocorrelation_driver import single_run_autocorrelation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Does a single autocorrelation run of sk"
    )

    # Identifiers & Hamiltonian Strings
    parser.add_argument(
        "--name", type=str, default="pt_run", help="Run name/identifier"
    )
    parser.add_argument("--n-qubits", type=int, required=True, help="n qubits")
    parser.add_argument(
        "--hamiltonian-random-seed",
        type=int,
        required=True,
        help="random seed for h",
    )

    # Range Parameters (Floats)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0625,
        help="inverse temperature (beta)",
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="transverse field (gamma)"
    )
    parser.add_argument("--k-max", type=int, required=True, help="maximum k value")
    parser.add_argument(
        "--test-operator-random-seed",
        type=int,
        default=42,
        help="seed to generate the test operator",
    )
    parser.add_argument(
        "--t-steps",
        type=int,
        default=10000,
        help="equilibration steps",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="number of steps",
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    params = parse_args()

    n_qubits = params["n_qubits"]
    h_seed = params["hamiltonian_random_seed"]

    h_str, h_gamma_str = generate_quantum_sk_hamiltonian(n_qubits, 1.0, 1.5, seed=h_seed, split=True)

    params["h_str"] = h_str
    params["h_gamma_str"] = h_gamma_str

    single_run_autocorrelation(params)
