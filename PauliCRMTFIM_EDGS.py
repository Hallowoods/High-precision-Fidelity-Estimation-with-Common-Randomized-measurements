# from tqdm import tqdm
# import numpy as np
# from itertools import product
# from joblib import Parallel, delayed
# from tqdm import tqdm
# import numpy as np
# from itertools import product

# # 单比特Pauli矩阵
# I = np.array([[1, 0], [0, 1]], dtype=complex)
# X = np.array([[0, 1], [1, 0]], dtype=complex)
# Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
# Z = np.array([[1, 0], [0, -1]], dtype=complex)
# paulis = [I, X, Y, Z]
# pauli_labels = ['I','X','Y','Z']

# def tensor_pauli(labels):
#     op = paulis[pauli_labels.index(labels[0])]
#     for label in labels[1:]:
#         op = np.kron(op, paulis[pauli_labels.index(label)])
#     return op

# def shared_nontrivial_sites(labels1, labels2):
#     count = 0
#     for a,b in zip(labels1, labels2):
#         if a==b and a!='I':
#             count += 1
#     return count

# def generate_all_nontrivial_paulis(n):
#     all_labels = product(['I','X','Y','Z'], repeat=n)
#     return [list(lbl) for lbl in all_labels if any(l != 'I' for l in lbl)]

# def generate_commuting_Pj(Pi_labels):
#     n = len(Pi_labels)
#     options = []
#     for i in range(n):
#         if Pi_labels[i] == 'I':
#             options.append(['I','X','Y','Z'])
#         else:
#             options.append([Pi_labels[i],'I'])
#     return [list(lbl) for lbl in product(*options) if any(l != 'I' for l in lbl)]

# def V_star_optimized(sigma,nqubit):
#     n = nqubit
#     d = 2**n
#     I = np.eye(d)
#     O = sigma - I/d
#     pauli_list = generate_all_nontrivial_paulis(n)
#     result = 0.0

#     # tqdm 进度条
#     for Pi_labels in tqdm(pauli_list, desc="Processing Pi"):
#         Pi = tensor_pauli(Pi_labels)
#         tr_OPi = np.trace(O @ Pi)
#         Pj_candidates = generate_commuting_Pj(Pi_labels)
#         for Pj_labels in Pj_candidates:
#             if Pj_labels <= Pi_labels:
#                 continue
#             Pj = tensor_pauli(Pj_labels)
#             tr_OPj = np.trace(O @ Pj)
#             w = shared_nontrivial_sites(Pi_labels, Pj_labels)
#             result += (3**w / d**2) * (tr_OPi)**4  # 注意你原来只用了 tr_OPi **4

#     result -= np.trace(O @ O)**2
#     return np.real(result)


# def V_star_contrib(Pi_labels, O, d):
#     contrib = 0.0
#     Pi = tensor_pauli(Pi_labels)
#     tr_OPi = np.trace(O @ Pi)
#     Pj_candidates = generate_commuting_Pj(Pi_labels)
#     for Pj_labels in Pj_candidates:
#         if Pj_labels <= Pi_labels:
#             continue
#         w = shared_nontrivial_sites(Pi_labels, Pj_labels)
#         contrib += (3**w / d**2) * (tr_OPi)**4  # 保持原来的公式
#     return contrib

# def V_star_optimized_parallel(sigma, nqubit, n_jobs):
#     n = nqubit
#     d = 2**n
#     I_full = np.eye(d)
#     O = sigma - I_full/d
#     pauli_list = generate_all_nontrivial_paulis(n)

#     # 使用 joblib 并行计算
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(V_star_contrib)(Pi_labels, O, d) for Pi_labels in tqdm(pauli_list, desc="Processing Pi")
#     )
#     result = sum(results)
#     result -= np.trace(O @ O)**2
#     return np.real(result)

# # def V_optimized(O, rho):
# #     n = int(np.log2(O.shape[0]))
# #     d = 2**n
# #     pauli_list = generate_all_nontrivial_paulis(n)
# #     result = 0.0
# #     for Pi_labels in pauli_list:
# #         Pi = tensor_pauli(Pi_labels)
# #         tr_OPi = np.trace(O @ Pi)
# #         Pj_candidates = generate_commuting_Pj(Pi_labels)
# #         for Pj_labels in Pj_candidates:
# #             if Pj_labels <= Pi_labels:
# #                 continue
# #             Pj = tensor_pauli(Pj_labels)
# #             tr_OPj = np.trace(O @ Pj)
# #             tr_rho_PiPj = np.trace(rho @ Pi @ Pj)
# #             w = shared_nontrivial_sites(Pi_labels, Pj_labels)
# #             result += (3**w/d**2) * tr_OPi * tr_OPj * tr_rho_PiPj
# #     result -= np.trace(O @ rho)**2
# #     return np.real(result)



# # =========================
# # Example
# n = 8
# import pennylane as qml
# layout = rf'1x{n}'
# print(layout)
# H2datasets = qml.data.load("qspin", sysname="Ising", periodicity="closed", lattice="chain", layout= layout ,attributes=["ground_states",'hamiltonians','parameters','order_params'])[0]
# # print(H2datasets.ground_states[1])
# # O = X
# # R = Z
# # print("V_*(O,R) optimized =", V_star_optimized(O,R))
# # print("V(O,R) optimized =", V_optimized(O,R))

# psi = np.array(H2datasets.ground_states[30
#                                 ],dtype=complex)
# # np.outer(psi, np.conj(psi))
# sigma = np.outer(psi, np.conj(psi))
# bbV = V_star_optimized_parallel(sigma,n,10)
# print(bbV)
# # print(V_star_optimized_parallel(sigma,n,njobs=10))

from tqdm import tqdm
import numpy as np
from itertools import product
from joblib import Parallel, delayed
import pandas as pd
import pennylane as qml
from datetime import datetime
import os

# ========================
# 单比特 Pauli 矩阵
# ========================
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
paulis = [I, X, Y, Z]
pauli_labels = ['I','X','Y','Z']

def tensor_pauli(labels):
    op = paulis[pauli_labels.index(labels[0])]
    for label in labels[1:]:
        op = np.kron(op, paulis[pauli_labels.index(label)])
    return op

def shared_nontrivial_sites(labels1, labels2):
    return sum(a == b and a != 'I' for a, b in zip(labels1, labels2))

def generate_all_nontrivial_paulis(n):
    all_labels = product(['I','X','Y','Z'], repeat=n)
    return [list(lbl) for lbl in all_labels if any(l != 'I' for l in lbl)]

def generate_commuting_Pj(Pi_labels):
    n = len(Pi_labels)
    options = []
    for i in range(n):
        if Pi_labels[i] == 'I':
            options.append(['I','X','Y','Z'])
        else:
            options.append([Pi_labels[i], 'I'])
    all_candidates = [list(lbl) for lbl in product(*options)]
    return [lbl for lbl in all_candidates if any(l != 'I' for l in lbl)]


# ========================
# 局部计算函数（仅查表）
# ========================
def V_star_contrib(Pi_labels, d, p, tr_OP_dict, tr_Delta_dict, tr_rhoPi_dict):
    contribv_star_1 = 0.0
    contribv_star_2 = 0.0
    contribv = 0.0

    Pi_key = tuple(Pi_labels)
    tr_OPi = tr_OP_dict[Pi_key]
    tr_DeltaPi = tr_Delta_dict[Pi_key]
    tr_rhoPi = tr_rhoPi_dict[Pi_key]

    Pj_candidates = generate_commuting_Pj(Pi_labels)
    for Pj_labels in Pj_candidates:
        Pj_key = tuple(Pj_labels)
        w = shared_nontrivial_sites(Pi_labels, Pj_labels)

        tr_OPj = tr_OP_dict[Pj_key]
        tr_DeltaPj = tr_Delta_dict[Pj_key]
        tr_rhoPj = tr_rhoPi_dict[Pj_key]
        tr_rhoPiPj = tr_rhoPi * tr_rhoPj  # ≈ <ψ|PiPj|ψ>，若Pi与Pj对易可直接相乘

        contribv_star_1 += (3**w / d**2) * (tr_OPi * tr_DeltaPi * tr_OPj * tr_DeltaPj)
        contribv_star_2 += (3**w / d**2) * (tr_OPi * tr_OPj * tr_rhoPi * tr_rhoPj)
        contribv +=  (3**w / d**2) * (tr_OPi * tr_OPj * tr_rhoPiPj)

    return np.real([contribv_star_1, contribv_star_2, contribv])


# ========================
# 并行主函数
# ========================
def V_star_optimized_parallel(psi, nqubit, n_jobs, p):
    d = 2**nqubit
    pauli_list = generate_all_nontrivial_paulis(nqubit)

    # 预计算所有 Pauli 期望值（仅一次）
    tr_rhoPi_dict = {}
    for lbl in tqdm(pauli_list, desc=f"Precomputing ⟨ψ|P|ψ⟩ for n={nqubit}"):
        val = np.vdot(psi, tensor_pauli(lbl) @ psi)
        tr_rhoPi_dict[tuple(lbl)] = val

    tr_OP_dict = {lbl: val * (1 - 1/d) for lbl, val in tr_rhoPi_dict.items()}
    tr_Delta_dict = {lbl: val * p * (1 - 1/d) for lbl, val in tr_rhoPi_dict.items()}

    # 并行计算贡献
    results = Parallel(n_jobs=n_jobs)(
        delayed(V_star_contrib)(Pi_labels, d, p, tr_OP_dict, tr_Delta_dict, tr_rhoPi_dict)
        for Pi_labels in tqdm(pauli_list, desc=f"Computing contributions for n={nqubit}")
    )

    results = np.array(results)
    contribv_star_1_sum, contribv_star_2_sum, contribv_sum = results.sum(axis=0)

    contribv_star_1_sum -= (p * (1 - 1/d))**2
    contribv_star_2_sum -= (1 - 1/d)**2
    contribv_sum        -= (1 - 1/d)**2

    return np.real([contribv_star_1_sum, contribv_star_2_sum, contribv_sum])


# ========================
# 主循环：只筛选 h=1.000, 2.000, 4.000
# ========================
n_list = range(8, 11)
base_folder = r"C:\Users\22968\Desktop\TFIM"
save_dir = r"C:\Users\22968\Desktop\TFIM\PauliCRM_EDGS"
os.makedirs(save_dir, exist_ok=True)
p = 0.01
target_h_values = {"h1.000", "h2.000", "h4.000"}

for n in n_list:
    folder_n = os.path.join(base_folder, f"1DTFIM_nqubit={n}_EDGS_info")
    if not os.path.exists(folder_n):
        print(f"❌ Folder not found, skipping n={n}: {folder_n}")
        continue

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"Vstar_results_n={n}_time={timestamp}_EDGS_p={p}.csv")

    file_list = sorted(
        [f for f in os.listdir(folder_n)
         if f.endswith(".npz") and any(tag in f for tag in target_h_values)]
    )

    if not file_list:
        print(f"⚠️ No matching files found for n={n}")
        continue

    for idx, fname in enumerate(tqdm(file_list, desc=f"Processing n={n}")):
        fpath = os.path.join(folder_n, fname)
        data = np.load(fpath)
        psi = data["psi"]

        vals = V_star_optimized_parallel(psi, n, n_jobs=10, p=p)

        results.append({
            "filename": fname,
            "h": float(data["h"]),
            "index": idx,
            "nqubit": n,
            "V_starDelta": vals[0],
            "V_starrho": vals[1],
            "VarianceStd": vals[2],
        })

    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"✅ Results saved to {filename}")
