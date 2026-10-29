import argparse

from hamiltonian import generate_quantum_sk_hamiltonian
from pt_validation_driver import full_validation_run


def parse_args():
    parser = argparse.ArgumentParser(description="Does pt optimization for an sk model")

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

    parser.add_argument(
        "--test-operator-random-seed",
        type=int,
        default=42,
        help="seed to generate the test operator",
    )

    # Range Parameters (Floats)
    parser.add_argument(
        "--beta-min",
        type=float,
        default=0.0625,
        help="Minimum inverse temperature (beta)",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        default=16.0,
        help="Maximum inverse temperature (beta)",
    )
    parser.add_argument(
        "--gamma-min", type=float, default=0.5, help="Minimum transverse field (gamma)"
    )
    parser.add_argument(
        "--gamma-max", type=float, default=1.5, help="Maximum transverse field (gamma)"
    )
    parser.add_argument("--k-max", type=int, required=True, help="maximum k value")

    # Execution Parameters (Integers)
    parser.add_argument(
        "--n-temperatures",
        type=int,
        default=10,
        help="Number of PT replicas/schedule points",
    )
    parser.add_argument(
        "--n-procs", type=int, default=40, help="Number of parallel processes"
    )
    parser.add_argument(
        "--n-feedback-loops",
        type=int,
        default=10,
        help="Number feedback optimization iterations",
    )
    parser.add_argument(
        "--schedule-type", type=str, choices=["linear", "linear-geometric", "geometric-linear"], default="linear"
    )

    # Boolean Flags
    parser.add_argument(
        "--measure-speedup",
        action="store_true",
        help="Enable runtime speedup measurement",
    )
    parser.add_argument(
        "--validate-final-run",
        action="store_true",
        help="Run exact diagonalization validation on final outputs",
    )

    parser.add_argument(
        "--autocorr-steps",
        type=int,
        help="timeseries length for autocorrelation calculations",
    )
    parser.add_argument(
        "--t-exchanges-per-iteration",
        type=int,
        help="number of thermalization PT exchanges for each optimization iteration",
    )
    parser.add_argument(
        "--exchanges-per-iteration",
        type=int,
        help="number of PT exchanges for each optimization iteration",
    )

    return vars(parser.parse_args())

if __name__ == "__main__":
    params = parse_args()

    n_qubits = params["n_qubits"]
    h_seed = params["hamiltonian_random_seed"]

    h_str, h_gamma_str = generate_quantum_sk_hamiltonian(n_qubits, 1.0, 1.5, seed=h_seed, split=True)

    params["h_str"] = h_str
    params["h_gamma_str"] = h_gamma_str

    full_validation_run(params)
