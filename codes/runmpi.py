"""
This module provides functions for calculating the variances of HPFE under depolarizing noise by Pauli CRM according to Eqs. (S143-145) in arXiv:2511.22509

The code to caulate the data for Pauli CRM in Fig. 3. V_star_val (variance) is the result, one should further use Eq.(2) to get the final circuit cost.
Author: YZY
"""

import numpy as np
from itertools import product
import os, time, datetime
import pandas as pd
from mpi4py import MPI


def T_expectation(p):
    if p == 0: return 1.0
    elif p in (1,2): return 1/np.sqrt(2)
    else: return 0.0

def tensor_expectation(pauli):
    val = 1.0
    for p in pauli:
        val *= T_expectation(p)
        if val == 0: break
    return val


def conjugate_by_cz_chain(label):
    n = len(label)
    new_label = list(label)
    add_z = [0]*n
    for i in range(n):
        if label[i] in (1,2):
            if i-1 >= 0: add_z[i-1] ^= 1
            if i+1 < n:  add_z[i+1] ^= 1
    for i in range(n):
        if add_z[i]:
            if new_label[i] == 0: new_label[i] = 3
            elif new_label[i] == 1: new_label[i] = 2
            elif new_label[i] == 2: new_label[i] = 1
            elif new_label[i] == 3: new_label[i] = 0
    return tuple(new_label)


def generate_magic_P_set(n):
    P_list, c_list = [], []
    for P in product(range(4), repeat=n):
        P_tilde = conjugate_by_cz_chain(P)
        if all(p != 3 for p in P_tilde) and not all(p == 0 for p in P_tilde):
            P_list.append(P)
            c_list.append(tensor_expectation(P_tilde))
    return np.array(P_list), np.array(c_list, dtype=complex)


def commutes_sitewise(Pi, Pj):
    for a, b in zip(Pi, Pj):
        if a != 0 and b != 0 and a != b:
            return False
    return True

def weight(Pi, Pj):
    return sum(a == b and a != 0 for a, b in zip(Pi, Pj))


def compute_V_star_mpi(n, verbose=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    d = 2**n
    P_list, c_list = None, None

    
    if rank == 0:
        P_list, c_list = generate_magic_P_set(n)

   
    P_list = comm.bcast(P_list, root=0)
    c_list = comm.bcast(c_list, root=0)
    num_P = len(P_list)

    
    counts = [num_P // size + (1 if i < num_P % size else 0) for i in range(size)]
    starts = [sum(counts[:i]) for i in range(size)]
    ends = [starts[i] + counts[i] for i in range(size)]

    local_start, local_end = starts[rank], ends[rank]

 
    local_sum = 0.0
    for i in range(local_start, local_end):
        Pi, ci = P_list[i], c_list[i]
        for j, (Pj, cj) in enumerate(zip(P_list, c_list)):
            if not commutes_sitewise(Pi, Pj):
                continue
            w = weight(Pi, Pj)
            val = (3**w / d**2) * (np.abs(ci)**2) * (np.abs(cj)**2)
            local_sum += val
            if verbose and rank==0:
                print(f"Pi={Pi}, Pj={Pj}, ci={ci:.4g}, cj={cj:.4g}, w={w}, val={val:.4g}")

   
    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

   
    if rank == 0:
        V_star = total_sum - (1 - 1/d)**2
        return np.real(V_star)
    else:
        return None


if __name__ == "__main__":
    output_dir = r"xxx"  # User defined
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run_time_record.txt")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    for n in range(1, 12):
        if rank == 0:
            start = time.time()

        V_star_val = compute_V_star_mpi(n, verbose=False)

        if rank == 0:
            elapsed = time.time() - start
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            csv_name = f"Vstar_n={n}_{timestamp}.csv"
            csv_path = os.path.join(output_dir, csv_name)

            pd.DataFrame([[n, V_star_val]], columns=["n", "V_star"]).to_csv(csv_path, index=False)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} | n={n} | V_star={V_star_val:.8e} | time={elapsed:.2f}s | file={csv_name}\n")
