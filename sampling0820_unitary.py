"""
upperboundterms_k.py

This module provides functions for calculating upper bound terms(including cross-characteristic, characteristic, variance, and their amplified version, sample complexity) for snk states with different k. Both for CRM and THR protocol.

Pauli noise is assumed.

Author: YZY
Date: 2025-04-01
"""


import time
import numpy as np
# import paddle
import matplotlib.pyplot as plt
import paddle as paddle
import paddle_quantum  as pq
from paddle_quantum.qinfo import trace_distance
from paddle_quantum.qinfo import shadow_trace
from paddle_quantum.state import to_state
from paddle_quantum.backend import Backend
from paddle_quantum import Hamiltonian
from paddle_quantum.qinfo import random_hamiltonian_generator
from paddle_quantum.trotter import  get_1d_heisenberg_hamiltonian
from paddle_quantum.intrinsic import _type_fetch, _type_transform
from paddle_quantum.ansatz import Circuit
from datetime import datetime  # 导入 datetime 模块
import multiprocessing as mp
from paddle_quantum.ansatz import Circuit
from paulichannel_utils import generate_channel
import pandas as pd
import pickle
import random
import csv
from scipy import stats
from itertools import product
import json
from multiprocessing import Pool
import sys, os
sys.path.append(r"c:\Users\Yzy\Desktop\TEMP0820")
import numpy as np
from scipy.linalg import expm

def random_single_qubit_rotation():
    """
    生成一个随机单比特旋转矩阵 U (2x2 unitary)
    旋转轴为随机单位向量，角度在 [0,π] 均匀采样
    """
    # 随机生成球面方向 (均匀分布)
    phi = np.random.uniform(0, 2*np.pi)    # 方位角
    costheta = np.random.uniform(-1, 1)    # cosθ 均匀分布
    sintheta = np.sqrt(1 - costheta**2)
    axis = np.array([sintheta*np.cos(phi), sintheta*np.sin(phi), costheta])

    # 随机角度 θ ∈ [0, π]
    theta = np.random.uniform(0, np.pi)

    # Pauli 矩阵
    X = np.array([[0,1],[1,0]],dtype=complex)
    Y = np.array([[0,-1j],[1j,0]],dtype=complex)
    Z = np.array([[1,0],[0,-1]],dtype=complex)
    paulis = [X,Y,Z]

    # 旋转算符 U = exp(-i θ/2 * (n·σ))
    H = axis[0]*X + axis[1]*Y + axis[2]*Z
    U = expm(-1j * theta/2 * H)

    return U
import functools

def generate_random_unitary_noise(nqubit):
    """
    生成 nqubit 的随机旋转噪声（单位矩阵）
    返回整体 U (2^n x 2^n)
    """
    # 每个 qubit 独立生成旋转矩阵
    single_qubit_rotations = [random_single_qubit_rotation() for _ in range(nqubit)]

    # Kronecker product 得到整体噪声
    U = functools.reduce(np.kron, single_qubit_rotations)
    return U
def apply_unitary_noise(rho, U):
    """
    在密度矩阵 rho 上施加噪声 U
    """
    return U @ rho @ U.conj().T

def generate_pauli_group(nqubit):
    """生成 nqubit 的 Pauli Group，并将其写入 JSON 和 Pickle 文件"""
    pauli_labels = ['x', 'y', 'z']
    pauli_group = []
    json_path = f"pauli_group_{nqubit}qubit.json"
    pkl_path = f"pauli_group_{nqubit}qubit.pkl"
    
    for pauli_ops in product(['I'] + pauli_labels, repeat=nqubit):
        formatted_pauli = [1]  # 系数固定为 1
        pauli_str = []
        
        # 格式化 Pauli 操作，并添加对应 qubit 的位置
        for idx, op in enumerate(pauli_ops):
            if op != 'I':
                pauli_str.append(f"{op}{idx}")
        
        # 如果有非 'I' 操作，则将其加入
        if pauli_str:
            formatted_pauli.append(','.join(pauli_str))  # Pauli 操作按逗号分隔
        
        # 如果没有 Pauli 操作 (只有 'I' 操作)，则跳过
        elif not pauli_str and pauli_ops != tuple(['I'] * nqubit):
            continue
        
        # 将每个操作加入 Pauli group
        pauli_group.append(formatted_pauli)

    # 写入 JSON 文件
    with open(json_path, "w") as f:
        json.dump(pauli_group, f, indent=4)

    # 写入 Pickle 文件
    with open(pkl_path, "wb") as f:
        pickle.dump(pauli_group, f)
    del pauli_group[0]
    return pauli_group  # 返回生成的列表

def load_pauli_group(nqubit):
    """从 JSON 和 Pickle 文件中读取 Pauli Group"""
    json_path = f"pauli_group_{nqubit}qubit.json"
    pkl_path = f"pauli_group_{nqubit}qubit.pkl"
    
    with open(json_path, "r") as f:
        pauli_group_json = json.load(f)

    with open(pkl_path, "rb") as f:
        pauli_group_pickle = pickle.load(f)
    
    return pauli_group_json, pauli_group_pickle

def generate_random_state(nqubit,type='random',k=0):
    if type == 'random':
        state = pq.state.random_state(nqubit)
    elif type == 'w_state':
        state = pq.state.w_state(nqubit)
    elif type == 'ghz_state':
        state = pq.state.ghz_state(nqubit)
    elif type =='bell_state':
        state = pq.state.bell_state(nqubit)
    elif type == 'zero_state':
        state = pq.state.zero_state(nqubit)
    elif type == 's_nk_state':
        if k !=0:
            cir = Circuit(nqubit)
            cir.h(range(k))
            cir.t(range(k))
            state = cir()
        else:
            state = pq.state.zero_state(nqubit)
    state = _type_transform(state, "density_matrix").numpy()
    np.save('state.npy', state)      
    return state


# def save_pauli_channel(channelinfo, filename):
    '''write-in Pauli Channel file'''
    with open(filename, 'wb') as file:
        pickle.dump(channelinfo, file)  # 将 channelinfo 保存到文件中

# def load_pauli_channel(filename):
    '''Read the Pauli Channel file'''

    with open(filename, 'rb') as file:
        channelinfo = pickle.load(file)  # 从文件中读取 channelinfo
    return channelinfo


# def generate_random_pauli_channel(nqubit,magnitude = 0.1,channel_num = 0):
    channelinfo = []
    existing_pauli_strings = set()  # 用于跟踪已生成的 Pauli 字符串
    maxallowed = 4**nqubit - 1
    if channel_num == 0:    #default:random
        # number_of_string = random.randint(1, maxallowed)
        number_of_string = maxallowed
    else:
        number_of_string = channel_num
    

    while len(channelinfo) < number_of_string:  # 生成指定数量的通道信息
        coefficient = random.uniform(0, 1)  # 在 [0, 1] 区间生成实数
        num_pauli = random.randint(1, nqubit)  # 随机生成 Pauli 字符串的数量
        indices = random.sample(range(nqubit), num_pauli)  # 随机选择不重复的索引
        pauli_string = ','.join(f'{random.choice("xyz")}{i}' for i in sorted(indices))  # 生成 Pauli 字符串
        
        if pauli_string not in existing_pauli_strings:  # 检查字符串是否已存在
            existing_pauli_strings.add(pauli_string)  # 添加到集合中
            channelinfo.append([coefficient, pauli_string])  # 添加到 channelinfo 列表
    sum_p = sum(ele[0] for ele in channelinfo)
    for ele in channelinfo:
        ele[0] = ele[0]/sum_p*magnitude  
        save_pauli_channel(channelinfo, 'pauli_channel.pkl')  # 保存到文件
    return channelinfo,magnitude

def caculate_characters_CRM(state,sigma,nqubit):
    '''Calculate cross chatacteristic functions of rho and sigma and charcteristic functions, output as a product of them'''
   
    pauli_group = generate_pauli_group(nqubit=nqubit) #loading Pauli group first
    # print('pauligroup:',pauli_group)
    delta = state-sigma
    d = 2**nqubit
    I = np.identity(d)
    delta_vec = []
    sigma_vec = []
    cross_vec = []
    ob = sigma-I/d
    for i in range(len(pauli_group)):
        # print(i)
        # print(pauli_group[i])
        ele = pauli_group[i]
        ele[1] = pq.qinfo.pauli_str_to_matrix([[1,ele[1]]],nqubit).numpy()
        # pauli =pq.qinfo.pauli_str_to_matrix([[1,pauli_group[i][1]]],nqubit)
        # print(pauli)
        delta_vec.append(np.real(np.trace(delta@ele[1])))  
        sigma_vec.append(np.real(np.trace(ob@ele[1])))
        cross_vec.append(np.real(np.trace(delta@ele[1]@ob@ele[1])))
    delta_vec = np.array(delta_vec)
    sigma_vec = np.array(sigma_vec)
    cross_vec = np.array(cross_vec)

    
    cross_character = np.sum(delta_vec*sigma_vec*cross_vec)

    character = np.dot(delta_vec**2,sigma_vec**2)
    overlap = np.trace(delta@state)
    return character,cross_character,overlap


def caculate_characters_THR(rho,sigma,nqubit):
    '''Calculate cross chatacteristic functions of rho and sigma and charcteristic functions, output as a product of them'''
   
    pauli_group = generate_pauli_group(nqubit=nqubit) #loading Pauli group first
    # print('pauligroup:',pauli_group)
    sigma_vec = []
    cross_vec = []
    state_vec = []
    d = 2**nqubit
    I = np.identity(d)
    ob = sigma-I/d
    ob_vec = []
    for i in range(len(pauli_group)):
        # print(i)
        # print(pauli_group[i])
        ele = pauli_group[i]
        ele[1] = pq.qinfo.pauli_str_to_matrix([[1,ele[1]]],nqubit).numpy()
        # pauli =pq.qinfo.pauli_str_to_matrix([[1,pauli_group[i][1]]],nqubit)
        # print(pauli) 
        state_vec.append(np.real(np.trace(rho@ele[1])))
        ob_vec.append(np.real(np.trace(ob@ele[1])))
        cross_vec.append(np.real(np.trace(rho@ele[1]@ob@ele[1])))
    ob_vec = np.array(ob_vec)
    cross_vec = np.array(cross_vec)
    state_vec = np.array(state_vec)
    cross_character = np.sum(state_vec*ob_vec*cross_vec)

    character = np.dot(state_vec**2,ob_vec**2)
    return character,cross_character



def compute_for_magnitude(channel, state, nqubit):
    """单个进程的计算任务"""
    d = 2**nqubit
    # channel, mag = generate_random_pauli_channel(nqubit, magnitude=magnow,channel_num=3)
    unitary  = generate_random_unitary_noise(nqubit)
    state_noised = apply_unitary_noise(state,unitary)
    state_noised = _type_transform(state_noised, "density_matrix").numpy()
    
    character, cross_character,overlap = caculate_characters_CRM(state_noised, state, nqubit)
    # fidelity = np.real(np.trace(state_noised @ state))
    # variance_u = ((d+1)/(d*(d+2)))*(np.real(character)+np.real(cross_character))-(( np.real(fidelity)-1)**2)/(d+2)
    character2, cross_character2 = caculate_characters_THR(state_noised, state, nqubit)
    # amp_character = np.real(character/(1-np.real(fidelity))**2)
    # amp_cross = np.real(cross_character/(1-fidelity)**2)


    # variance_0 = (d+1)/(d*(d+2))*(np.real(character2)+np.real(cross_character2)+(1-1/d)**2+(fidelity-1/d)*(-1/d))-((np.real(fidelity)-1/d)**2)/(d+2)
    # partone_crm = np.real(variance_u/(1 - np.real(fidelity))**2)
    # partone_thr= np.real(variance_0/(1 - np.real(fidelity))**2)
    # parttwo = np.real(((-fidelity**2+d*(2*fidelity+1)/(d+2)-variance_0)/(1 - np.real(fidelity))**2))
    # variance = variance_u + 1/N_m*(-fidelity**2+d*(2*fidelity+1)/(d+2)-variance_0)
    # thrifty_variance = variance_0+1/N_m*(-fidelity**2+d*(2*fidelity+1)/(d+2)-variance_0)
    # amp_thrifty_variance = np.real(thrifty_variance/(1-fidelity)**2)
    # amp_variance= np.real(variance/(1-fidelity)**2)
    # sample = int(136*np.log(200)*variance*16/(1-fidelity)**2)
    # sample_thr = int(136*np.log(200)*thrifty_variance*16/(1-fidelity)**2)
    #thrifty ones
    

    return [character, cross_character,character2, cross_character2]


import os
import csv
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm   # pip install tqdm


def generate_haar_state(nqubit):
    """生成一个 nqubit Haar 随机纯态 (密度矩阵形式, numpy array)"""
    state = pq.state.random_state(nqubit)
    state_dm = _type_transform(state, "density_matrix").numpy()
    return state_dm


def analyze_rho_sigma(rho, sigma,nqubit):
    """计算 Delta 的特征值性质、2-范数平方、以及 inverse infidelity"""
    Delta = rho - sigma
    eigvals = np.linalg.eigvalsh(Delta)  # 厄米矩阵
    min_eig = np.min(eigvals)
    min_eig_term = 4 * (min_eig ** 2)

    norm2_sq = np.trace(Delta @ Delta).real
    overlap = np.trace(rho @ sigma).real
    infid = (1.0 - overlap) 
    infid2 = infid**2
    character, cross_character,overlap = caculate_characters_CRM(rho,sigma,nqubit)
    character2, cross_character2 = caculate_characters_THR(rho, sigma, nqubit)
    return infid,infid2,min_eig_term, norm2_sq,character, cross_character,character2, cross_character2


def worker(args):
    idx, nqubit, chan_num = args
    sigma = generate_haar_state(nqubit)
    unitary  = generate_random_unitary_noise(nqubit)
    rho = apply_unitary_noise(sigma,unitary)
    rho = _type_transform(rho, "density_matrix").numpy()
    infid, infid2, min_eig_term, norm2_sq, character, cross_character, character2, cross_character2 = analyze_rho_sigma(
        rho, sigma, nqubit
    )
    return [
        idx,
        infid,
        infid2,
        min_eig_term,
        norm2_sq,
        np.real(character),
        np.real(cross_character),
        np.real(character2),
        np.real(cross_character2),
    ]



def main(nqubit=2, num_samples=2000, nproc=10,num_chan=0):
    """主程序：并行生成样本，保存结果到 CSV，并显示进度"""
    # 输出目录
    out_dir = rf"C:\Users\Yzy\Desktop\TEMP0820\results0820\uni_noise"
    os.makedirs(out_dir, exist_ok=True)

    # 文件名：包含 nqubit 和时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"data_statstic_n={nqubit}_{timestamp}_unitary_num_sample_{num_samples}.csv")

    args_list = [(i,nqubit,num_chan) for i in range(num_samples)]

    results = []
    with Pool(processes=nproc) as pool:
        for res in tqdm(pool.imap(worker, args_list), total=num_samples, desc=f"n={nqubit}"):
            results.append(res)

    # 保存到 CSV
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "infid", "infid2", "4*min_eig^2", "||Delta||_2^2",'character','cross_character','character2', 'cross_character2'])

        writer.writerows(results)

    print(f"已保存 {num_samples} 条结果到 {out_file}")


if __name__ == "__main__":
    nqubit_list = [1,2,3,4]
    num_samples = 20000
    magnitude = 1
    nproc = 7
    chan_num = 1  # 每次生成的 Pauli 通道数量

    for nqubit in nqubit_list:
        print(f"开始计算 nqubit = {nqubit}")
        main(nqubit=nqubit, num_samples=num_samples, nproc=nproc,num_chan=chan_num)




