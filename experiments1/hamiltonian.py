import numpy as np


def toy_3bit() -> str:
    return """-0.7 1 X 
-0.7 2 X 
1.0 1 Z 2 Z 
-0.7 3 X 
1.0 3 Z 
1.0 2 Z 3 Z"""

def generate_1d_tfi(N: int, J: float = 1.0, h_x: float = 1.0, h_z: float = 0.0, pbc: bool = False) -> str:
    """
    Generates a 1D Transverse Field Ising Hamiltonian text string for QMC input.
    
    Parameters:
    - N: Number of qubits
    - J: Coupling strength (-J * Z_i * Z_j)
    - h_x: Transverse field strength (-h_x * X_i)
    - h_z: Longitudinal field strength (-h_z * Z_i)
    - pbc: Periodic boundary conditions (True/False)
    """
    lines = []

    # 1. Single-qubit field terms
    for i in range(1, N + 1):
        if h_x != 0.0:
            lines.append(f"{-h_x:.6g} {i} X")
        if h_z != 0.0:
            lines.append(f"{-h_z:.6g} {i} Z")

    # 2. Nearest-neighbor Z-Z interaction terms
    if J != 0.0:
        edges = set()
        for i in range(1, N):
            edges.add((i, i + 1))
        if pbc and N > 2:
            edges.add((1, N))

        for u, v in sorted(edges):
            lines.append(f"{-J:.6g} {u} Z {v} Z")

    return "\n".join(lines)


def generate_2d_tfi(Lx: int, Ly: int, J: float = 1.0, h_x: float = 1.0, h_z: float = 0.0, pbc: bool = False) -> str:
    """
    Generates a 2D Transverse Field Ising Hamiltonian on an Lx x Ly lattice.
    Qubit indexing is row-major: index = r * Ly + c + 1 (1-based).
    
    Parameters:
    - Lx: Rows in the lattice
    - Ly: Columns in the lattice
    - J: Coupling strength (-J * Z_i * Z_j)
    - h_x: Transverse field strength (-h_x * X_i)
    - h_z: Longitudinal field strength (-h_z * Z_i)
    - pbc: Periodic boundary conditions (True/False)
    """
    N = Lx * Ly
    lines = []

    # Helper mapping (row, col) -> 1-based index
    def get_index(r, c):
        return r * Ly + c + 1

    # 1. Single-qubit field terms
    for i in range(1, N + 1):
        if h_x != 0.0:
            lines.append(f"{-h_x:.6g} {i} X")
        if h_z != 0.0:
            lines.append(f"{-h_z:.6g} {i} Z")

    # 2. Nearest-neighbor 2D grid couplings
    if J != 0.0:
        edges = set()
        for r in range(Lx):
            for c in range(Ly):
                u = get_index(r, c)
                
                # Horizontal neighbor (Right)
                if c + 1 < Ly:
                    v = get_index(r, c + 1)
                    edges.add((min(u, v), max(u, v)))
                elif pbc and Ly > 2:
                    v = get_index(r, 0)
                    edges.add((min(u, v), max(u, v)))

                # Vertical neighbor (Down)
                if r + 1 < Lx:
                    v = get_index(r + 1, c)
                    edges.add((min(u, v), max(u, v)))
                elif pbc and Lx > 2:
                    v = get_index(0, c)
                    edges.add((min(u, v), max(u, v)))

        for u, v in sorted(edges):
            lines.append(f"{-J:.6g} {u} Z {v} Z")

    return "\n".join(lines)


def generate_H_from_graph(G, s: float) -> str:
    """
    Returns the Hamiltonian at annealing point s as a formatted string.
    
    H(s) = -(1-s) * sum_i(X_i) + s * sum_{(u,v)}(Z_u * Z_v)
    """
    c_x = -(1.0 - s)  # Coefficient for transverse field terms
    c_z = float(s)    # Coefficient for Max-Cut coupling terms

    # Map graph nodes to 1-based indexing for output readability
    nodes = sorted(list(G.nodes()))
    node_map = {node: idx + 1 for idx, node in enumerate(nodes)}

    lines = []

    # 1. Transverse Field Driver Terms (- (1-s) * X_i)
    for node in nodes:
        qubit = node_map[node]
        lines.append(f"{c_x:.4f} {qubit} X")

    # 2. Max-Cut Interaction Terms (+ s * Z_i * Z_j)
    for u, v in G.edges():
        q1, q2 = node_map[u], node_map[v]
        lines.append(f"{c_z:.4f} {q1} Z {q2} Z")

    return "\n".join(lines)


# def generate_kagome_qmc_file(L, J, h_pin=0.0, delta_J=0.0, filename="kagome_hamiltonian.txt"):
#     """
#     L: Number of unit cells in each direction (L x L)
#     J: Base Heisenberg interaction strength
#     h_pin: Staggered Z-field applied to sublattices to lift degeneracy
#     delta_J: Perturbation added to intra-cell bonds to favor dimerization
#     """
#     # 3 sites per unit cell
#     def site_idx(x, y, basis):
#         # Returns a 1-based integer index for the QMC solver
#         return 1 + basis + 3 * ((x % L) + L * (y % L))
        
#     bonds = []
    
#     # 1. Generate Bonds (Periodic Boundary Conditions)
#     for x in range(L):
#         for y in range(L):
#             s0 = site_idx(x, y, 0)
#             s1 = site_idx(x, y, 1)
#             s2 = site_idx(x, y, 2)
            
#             # Intra-cell bonds (Down-pointing triangles)
#             # We add delta_J here to break spatial symmetry if desired
#             bonds.append((s0, s1, J + delta_J))
#             bonds.append((s1, s2, J + delta_J))
#             bonds.append((s2, s0, J + delta_J))
            
#             # Inter-cell bonds (Up-pointing triangles connecting unit cells)
#             # site 0 of (x,y) connects to site 1 of (x-1,y) and site 2 of (x,y-1)
#             s1_prev_x = site_idx(x - 1, y, 1)
#             s2_prev_y = site_idx(x, y - 1, 2)
            
#             bonds.append((s0, s1_prev_x, J))
#             bonds.append((s0, s2_prev_y, J))
            
#     # Remove duplicate bonds (ensure i < j for consistency)
#     unique_bonds = {}
#     for u, v, weight in bonds:
#         edge = (min(u, v), max(u, v))
#         unique_bonds[edge] = weight
        
#     # 2. Write to File
#     with open(filename, 'w') as f:
#         # Write Pinning Fields (Symmetry Breaking)
#         if h_pin != 0.0:
#             for x in range(L):
#                 for y in range(L):
#                     for b in range(3):
#                         idx = site_idx(x, y, b)
#                         # Staggered field: +h on sublattice 0, -h/2 on sublattices 1 & 2
#                         field = h_pin if b == 0 else -h_pin / 2.0
                        
#                         # Assuming your solver uses 'Z' or 'X' for local fields
#                         f.write(f"{field:.4f} {idx} Z\n")
                        
#         # Write Heisenberg Interactions
#         # Assuming H = J(XX + YY + ZZ) based on your format
#         for (u, v), weight in sorted(unique_bonds.items()):
#             f.write(f"{weight:.4f} {u} X {v} X\n")
#             f.write(f"{weight:.4f} {u} Y {v} Y\n")
#             f.write(f"{weight:.4f} {u} Z {v} Z\n")
            
#     print(f"Generated lattice with {3 * L * L} sites and {len(unique_bonds)} bonds.")
#     print(f"Saved to {filename}")

# def generate_jq_hamiltonian(
#     Lx: int, Ly: int, J: float, Q: float, filename: str = "jq_hamiltonian.txt"
# ):
#     """Generates the sign-problem-free J-Q Hamiltonian file for QMC.

#     Sites are 1-indexed on an Lx x Ly square lattice with periodic boundaries.
#     """

#     def site_idx(x: int, y: int) -> int:
#         return (y % Ly) * Lx + (x % Lx) + 1  # 1-based indexing

#     # Dictionary mapping tuple of ((site, op), ...) -> accumulated coefficient
#     hamiltonian = {}

#     def add_term(coeff: float, pauli_list: list):
#         if abs(coeff) < 1e-12 or not pauli_list:
#             return  # Skip zero terms and pure identity shifts

#         # Sort Pauli operators by site index to canonicalize
#         sorted_paulis = tuple(sorted(pauli_list, key=lambda item: item[0]))
#         hamiltonian[sorted_paulis] = hamiltonian.get(sorted_paulis, 0.0) + coeff

#     def get_P_tilde_terms(i: int, j: int):
#         """Returns non-identity Pauli expansion terms for transformed projector \\tilde{P}_{ij}."""
#         return [
#             (0.25, [(i, "X"), (j, "X")]),
#             (0.25, [(i, "Y"), (j, "Y")]),
#             (-0.25, [(i, "Z"), (j, "Z")]),
#         ]

#     # --- 1. Generate J-terms: -J * \tilde{P}_{ij} ---
#     for x in range(Lx):
#         for y in range(Ly):
#             s = site_idx(x, y)
#             r_neighbor = site_idx(x + 1, y)
#             u_neighbor = site_idx(x, y + 1)

#             for neighbor in (r_neighbor, u_neighbor):
#                 for coeff, paulis in get_P_tilde_terms(s, neighbor):
#                     add_term(-J * coeff, paulis)

#     # --- 2. Generate Q-terms: -Q * (\tilde{P}_{ij}\tilde{P}_{kl} + \tilde{P}_{ik}\tilde{P}_{jl}) ---
#     for x in range(Lx):
#         for y in range(Ly):
#             # Plaquette corner sites:
#             # p1 (x, y)     p2 (x+1, y)
#             # p3 (x, y+1)   p4 (x+1, y+1)
#             p1 = site_idx(x, y)
#             p2 = site_idx(x + 1, y)
#             p3 = site_idx(x, y + 1)
#             p4 = site_idx(x + 1, y + 1)

#             # Two pairs of parallel bonds in a plaquette
#             parallel_bond_pairs = [
#                 ((p1, p2), (p3, p4)),  # Horizontal bonds
#                 ((p1, p3), (p2, p4)),  # Vertical bonds
#             ]

#             for (i, j), (k, l) in parallel_bond_pairs:
#                 terms_ij = get_P_tilde_terms(i, j)
#                 terms_kl = get_P_tilde_terms(k, l)

#                 # Cross-multiply the two bond terms
#                 for c1, p_list1 in terms_ij:
#                     for c2, p_list2 in terms_kl:
#                         add_term(-Q * c1 * c2, p_list1 + p_list2)

#     # --- 3. Write Hamiltonian to file ---
#     with open(filename, "w") as f:
#         for paulis, coeff in hamiltonian.items():
#             if abs(coeff) < 1e-10:
#                 continue
#             pauli_str = " ".join([f"{site} {op}" for site, op in paulis])
#             f.write(f"{coeff:.4f} {pauli_str}\n")

#     print(f"Successfully generated {filename} with {len(hamiltonian)} terms.")

def generate_neel_order_diagonal(
    Lx: int, Ly: int, filename: str = "neel_order_diag.txt"
):
    """Generates the purely diagonal (Z-basis) Néel order parameter O_{Neel}^z.

    Leverages SU(2) symmetry: <M^2> = 3 <(M^z)^2>.
    """
    N = Lx * Ly

    def site_idx(x, y):
        return (y % Ly) * Lx + (x % Lx) + 1

    def eta(x, y):
        return (-1) ** (x + y)

    observable = {}
    coeff_factor = 3.0 / (4.0 * N * N)  # 3 * (1/4) / N^2

    for x1 in range(Lx):
        for y1 in range(Ly):
            s1 = site_idx(x1, y1)
            e1 = eta(x1, y1)
            for x2 in range(Lx):
                for y2 in range(Ly):
                    s2 = site_idx(x2, y2)
                    e2 = eta(x2, y2)
                    sign = e1 * e2

                    if s1 == s2:
                        key = ()  # Identity term Z_i^2 = I
                    else:
                        key = tuple(sorted([s1, s2]))

                    observable[key] = (
                        observable.get(key, 0.0) + coeff_factor * sign
                    )

    lines = []
    for sites, coeff in sorted(
        observable.items(), key=lambda x: (len(x[0]), x[0])
    ):
        if abs(coeff) < 1e-12:
            continue
        if len(sites) == 0:
            lines.append(f"{coeff:+.6f}")
        else:
            lines.append(f"{coeff:+.6f} {sites[0]} Z {sites[1]} Z")

    if filename:
        with open(filename, "w") as f:
            f.writelines(l + "\n" for l in lines)
    return lines


def generate_vbs_order_diagonal(
    Lx: int, Ly: int, filename: str = "vbs_order_diag.txt"
):
    """Generates the purely diagonal (Z-basis) VBS order parameter O_{VBS}^z = (D_x^z)^2 + (D_y^z)^2.

    Leverages SU(2) symmetry by setting B_x(r) = 3 S^z_r S^z_{r+x}.
    """
    N = Lx * Ly

    def site_idx(x, y):
        return (y % Ly) * Lx + (x % Lx) + 1

    observable = {}

    def add_vbs_component(direction: str):
        for x1 in range(Lx):
            for y1 in range(Ly):
                s1 = site_idx(x1, y1)
                s1_next = (
                    site_idx(x1 + 1, y1)
                    if direction == "x"
                    else site_idx(x1, y1 + 1)
                )
                phase1 = ((-1) ** x1) if direction == "x" else ((-1) ** y1)

                for x2 in range(Lx):
                    for y2 in range(Ly):
                        s2 = site_idx(x2, y2)
                        s2_next = (
                            site_idx(x2 + 1, y2)
                            if direction == "x"
                            else site_idx(x2, y2 + 1)
                        )
                        phase2 = (
                            ((-1) ** x2) if direction == "x" else ((-1) ** y2)
                        )

                        coeff = (9.0 / (16.0 * N * N)) * phase1 * phase2

                        # Multiply (Z_s1 Z_s1_next) * (Z_s2 Z_s2_next)
                        counts = {}
                        for s in [s1, s1_next, s2, s2_next]:
                            counts[s] = counts.get(s, 0) + 1

                        # Sites appearing an odd number of times retain a Z operator (since Z^2 = I)
                        active_sites = tuple(
                            sorted([s for s, c in counts.items() if c % 2 == 1])
                        )
                        observable[active_sites] = (
                            observable.get(active_sites, 0.0) + coeff
                        )

    add_vbs_component("x")
    add_vbs_component("y")

    lines = []
    for sites, coeff in sorted(
        observable.items(), key=lambda x: (len(x[0]), x[0])
    ):
        if abs(coeff) < 1e-12:
            continue
        if len(sites) == 0:
            lines.append(f"{coeff:+.6f}")
        else:
            ops_str = " ".join([f"{s} Z" for s in sites])
            lines.append(f"{coeff:+.6f} {ops_str}")

    if filename:
        with open(filename, "w") as f:
            f.writelines(l + "\n" for l in lines)
    return lines

def generate_quantum_sk_hamiltonian(num_spins, J_scale, gamma, seed=None):
    """
    Generates a string representation of the Quantum SK model Hamiltonian.
    
    Args:
        num_spins (int): Total number of spins (N).
        J_scale (float): The energy scale for the random interactions. 
        gamma (float): Strength of the transverse magnetic field.
        seed (int): Optional random seed for reproducibility.
        
    Returns:
        str: A multi-line string containing the Hamiltonian terms.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Standard deviation for the Gaussian distribution is J / sqrt(N)
    std_dev = J_scale / np.sqrt(num_spins)
    
    lines = []
    
    # 1. Generate two-body ZZ interactions (all-to-all coupling)
    for i in range(1, num_spins + 1):
        for j in range(i + 1, num_spins + 1):
            # Draw J_ij from a normal distribution
            J_ij = np.random.normal(0, std_dev)
            coeff = -J_ij
            
            # Append to list matching the QMC format
            lines.append(f"{coeff:.4f} {i} Z {j} Z")
            
    # 2. Generate one-body X terms (transverse field)
    for i in range(1, num_spins + 1):
        coeff = -gamma
        lines.append(f"{coeff:.4f} {i} X")

    # Join all lines with a newline character
    return "\n".join(lines)


# def generate_random_z_operator(num_spins, filename="test_operator.txt", seed=None):
#     """
#     Generates a text file for a test operator consisting of a random 
#     weighted sum of single-site Pauli Z operators.
    
#     Args:
#         num_spins (int): Total number of spins (N).
#         filename (str): Output file name.
#         seed (int): Optional random seed for reproducibility.
#     """
#     if seed is not None:
#         np.random.seed(seed)
        
#     with open(filename, 'w') as f:
#         # Generate the one-body random weighted Z terms
#         for i in range(1, num_spins + 1):
#             # Draw a random weight c_i from a uniform distribution [-1.0, 1.0)
#             c_i = np.random.uniform(-1.0, 1.0)
            
#             # Write to file matching the QMC format (e.g., "0.4521 1 Z")
#             f.write(f"{c_i:.4f} {i} Z\n")

#     print(f"Successfully wrote random Z test operator ({num_spins} spins) to {filename}")

def generate_heisenberg_square(Lx, Ly, J=1.0, pbc=False):
    coeff = J / 4.0  # Convert spin-1/2 operators to Pauli matrices
    bonds = []

    def site(x, y):
        return y * Lx + x + 1  # 1-based indexing

    for y in range(Ly):
        for x in range(Lx):
            s1 = site(x, y)

            # Horizontal neighbor (Right)
            if x + 1 < Lx:
                bonds.append((s1, site(x + 1, y)))
            elif pbc and Lx > 2:
                bonds.append((s1, site(0, y)))

            # Vertical neighbor (Up)
            if y + 1 < Ly:
                bonds.append((s1, site(x, y + 1)))
            elif pbc and Ly > 2:
                bonds.append((s1, site(x, 0)))

    lines = []
    for s1, s2 in bonds:
        for op in ["X", "Y", "Z"]:
            lines.append(f"{coeff:.2f} {s1} {op} {s2} {op}")

    return "\n".join(lines)

