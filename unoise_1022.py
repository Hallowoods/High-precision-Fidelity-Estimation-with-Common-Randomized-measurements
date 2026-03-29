"""
unitary_noise_upperboundterms.py

This module provides functions for calculating upper bound terms(including cross-characteristic, characteristic, variance, and their amplified version, sample complexity) for snk states with different k. Both for CRM and THR protocol.

Local unitary noise(phase gate) is assumed.

Author: YZY
Date: 2025-04-01
"""

import math
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
from paddle_quantum.channel.common import MixedUnitaryChannel


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


import numpy as np

def generate_theta_vector(nqubit, mag):
    """
    生成 nqubit 个 theta 值的向量，每个 theta_i = arccos(sqrt(v_i))，v_i >=0, sum(v_i)=mag
    返回 2*theta_i 的向量
    """
    # Step 1: 随机生成 nqubit 个 [0,1] 的数
    v = np.random.rand(nqubit)
    beta = np.log2(mag)
    # Step 2: 归一化使得 sum(v) = mag
    v = v / np.sum(v) * beta
    
    # Step 3: 每个分量开根号再 arccos
    theta = np.arccos(2*2**v-1)

    # Step 4: 返回 2*theta
    return 2 * theta

# # 示例
# nqubit = 5
# mag = 2.0
# theta_vec = generate_theta_vector(nqubit, mag)
# print(theta_vec)

def generate_random_unitary_channel_rotation2(nqubit, mag, num_sites=1):
    channel_info = []
    sites = random.sample(range(0, nqubit), num_sites)
    # vec = np.random.randn(3)
    # axis = vec / np.linalg.norm(vec)
    # axis = fix_axis
    # 在 [thetamag, thetamag2] * pi 范围内随机选择旋转角

    thetalist = generate_theta_vector(num_sites, mag)
    for i  in range(len(sites)):
        site = sites[i]
        theta = thetalist[i]
        # 随机生成旋转轴（单位向量）
        vec = np.random.randn(3)
        axis = vec / np.linalg.norm(vec)
        
        # 在 [thetamag, thetamag2] * pi 范围内随机选择旋转角
        # theta = random.uniform(thetamag, thetamag2) 
        # print('axis:', axis)
        # print('theta:', theta)
        # theta = random.uniform(0, 1) * 2 * np.pi
        # 保存信息：[qubit index, rotation angle, rotation axis]
        channel_info.append([site, theta, axis.tolist()])  # axis 转成 list 方便保存

    filename = 'unitary_channel(local_rotation).pkl'
    with open(filename, 'wb') as file:
        pickle.dump(channel_info, file)
    
    return channel_info    


def axis_to_u3_params(theta, axis):
    """
    Convert rotation (axis, theta) -> U3 parameters (θ', φ, λ)
    so that U3(θ', φ, λ) = exp(-i theta/2 * (n·σ))
    consistent with Paddle Quantum's U3 definition:
        U3(θ, φ, λ) = [[cos θ/2, -exp(i λ) sin θ/2],
                        [exp(i φ) sin θ/2, exp(i(φ+λ)) cos θ/2]]
    
    Args:
        theta: rotation angle
        axis: array-like, 3D unit vector of rotation axis [nx, ny, nz]
    
    Returns:
        tuple (theta_u3, phi, lam)
    """
    axis = np.array(axis, dtype=float)
    if np.linalg.norm(axis) == 0:
        raise ValueError("Rotation axis cannot be zero vector.")
    axis = axis / np.linalg.norm(axis)
    nx, ny, nz = axis

    # Build rotation matrix R = exp(-i theta/2 * n·σ)
    cos_term = np.cos(theta/2)
    sin_term = np.sin(theta/2)
    R = np.array([[cos_term - 1j * nz * sin_term, -1j * (nx - 1j * ny) * sin_term],
                  [-1j * (nx + 1j * ny) * sin_term, cos_term + 1j * nz * sin_term]])

    # Extract U3 parameters from R
    theta_u3 = 2 * np.arccos(np.abs(R[0,0]))
    
    # Avoid division by zero
    if np.sin(theta_u3/2) < 1e-12:
        phi = 0.0
        lam = np.angle(R[1,1]) - np.angle(R[0,0])
    else:
        phi = np.angle(R[1,0])
        lam = np.angle(-R[0,1])

    return [theta_u3, phi, lam]

def generate_unitary_channel_rotation2(state, nqubit, channelinfo):
    cir = Circuit(nqubit)
    state = to_state(state, backend=Backend.DensityMatrix)
    prodtest = 1
    statep  = _type_transform(state, "density_matrix").numpy()
    s_x = 1/np.sqrt(2)
    s_y = 1/np.sqrt(2)
    s_z = 0
    s_vec = np.array([s_x, s_y, s_z])
    for info in channelinfo:
        site = info[0]
        theta = info[1]
        axis = info[2]
        print('axis::',axis,'   theta:',theta)
        u3_params = axis_to_u3_params(theta, axis)
        cir.u3([site], param=u3_params)  # 使用 u3 门
        p_j = np.cos(theta/2)**2 + (np.dot(axis, s_vec))**2 * np.sin(theta/2)**2
        prodtest *= p_j
    
    state_after = cir.forward(state)
    # /statep  = _type_transform(state, "density_matrix").numpy()
    state_afterp = _type_transform(state_after, "density_matrix").numpy()
    print('fid_inst:',np.trace(statep@state_afterp),'  fid_theory:',prodtest)
    return state_after

def caculate_characters_CRM(state,sigma,nqubit):
    '''Calculate cross chatacteristic functions of rho and sigma and charcteristic functions, output as a product of them'''
    pauli_group = generate_pauli_group(nqubit=nqubit) #loading Pauli group first
    # print('pauligroup:',pauli_group)
    delta = state-sigma
    delta_vec = []
    ob_vec = []
    cross_vec=[]
    d= 2**nqubit
    I = np.identity(d)
    ob = sigma-I/d
    for i in range(len(pauli_group)):
        # print(i)
        # print(pauli_group[i])
        ele = pauli_group[i]
        ele[1] = pq.qinfo.pauli_str_to_matrix([[1,ele[1]]],nqubit).numpy()
        # pauli =pq.qinfo.pauli_str_to_matrix([[1,pauli_group[i][1]]],nqubit)
        # print(pauli)
        delta_vec.append(np.trace(delta@ele[1]))  
        ob_vec.append(np.trace(ob@ele[1]))
        cross_vec.append(np.trace(delta@ele[1]@ob@ele[1]))
    delta_vec = np.array(delta_vec)
    ob_vec = np.array(ob_vec)
    cross_vec = np.array(cross_vec)

    
    cross_character = np.sum(delta_vec*ob_vec*cross_vec)

    character = np.dot(delta_vec**2,ob_vec**2)
    return character,cross_character


def caculate_characters_THR(state,sigma,nqubit):
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
        state_vec.append(np.trace(state@ele[1]))
        ob_vec.append(np.trace(ob@ele[1]))
        cross_vec.append(np.trace(state@ele[1]@ob@ele[1]))
    ob_vec = np.array(ob_vec)
    cross_vec = np.array(cross_vec)
    state_vec = np.array(state_vec)
    cross_character = np.sum(state_vec*ob_vec*cross_vec)

    character = np.dot(state_vec**2,ob_vec**2)
    return character,cross_character

    


def compute_for_magnitude(magnow ,state, nqubit, noisetype='rotation', N_m=None, errorsites=1):
    """单个进程的计算任务"""
    d = 2**nqubit
    if noisetype == 'rotation':
        channel = generate_random_unitary_channel_rotation2(nqubit, magnow, num_sites=errorsites)
        state_noised = generate_unitary_channel_rotation2(state, nqubit, channel)


    
    state_noised = _type_transform(state_noised, "density_matrix").numpy()
    
    # fidelity
    fidelity = np.real(np.trace(state_noised @ state))

    # 在这里动态设置 N_m
    if (1 - fidelity) != 0:
        N_m = math.ceil(20 / (1 - fidelity) ** 2)
        # N_m = math.ceil(d / (1 - fidelity) ** 2)
    else:
        N_m = 1e20  # 防止除零，可以给一个很大的数

    # CRM
    character, cross_character = caculate_characters_CRM(state_noised, state, nqubit)
    variance_u = ((d+1)/(d*(d+2))) * (np.real(character) + np.real(cross_character)) - (1 - fidelity)**2 / (d+2)

    # THR
    character2, cross_character2 = caculate_characters_THR(state_noised, state, nqubit)
    variance_0 = (d+1)/(d*(d+2)) * (np.real(character2) + np.real(cross_character2)) - (fidelity - 1/d)**2 / (d+2)

    amp_character = np.real(character / (1 - fidelity) ** 2)
    amp_cross = np.real(cross_character / (1 - fidelity) ** 2)

    variance = variance_u + 1 / N_m * (-fidelity**2 + d*(2*fidelity + 1) / (d+2) - variance_0)
    thrifty_variance = variance_0 + 1 / N_m * (-fidelity**2 + d*(2*fidelity + 1) / (d+2) - variance_0)

    amp_variance = np.real(variance / (1 - fidelity) ** 2)
    amp_thrifty_variance = np.real(thrifty_variance / (1 - fidelity) ** 2)

    sample = int(68 * np.log(200) * 16* variance / (1 - fidelity) ** 2) if (1 - fidelity) ** 2 != 0 else -1
    sample_thr = int(68 * np.log(200) * 16*thrifty_variance / (1 - fidelity) ** 2) if (1 - fidelity) ** 2 != 0 else -1
    partone_crm = np.real(variance_u)
    parttwo  = np.real(-fidelity**2+d*(2*fidelity+1)/(d+2)-variance_0)
    partone_thr = np.real(variance_0)
    print(r'infid^-1={}'.format(1 / (1 - np.real(fidelity))))
    return [
        1 / (1 - fidelity),
        np.real(character),
        np.real(cross_character),
        sample,
        np.real(variance),
        amp_variance,
        amp_character,
        amp_cross,
        np.real(thrifty_variance),
        amp_thrifty_variance,
        sample_thr,N_m,partone_crm,parttwo,partone_thr
    ]



def main(nqubit, k,errorsites,theta,statetype,noisetype='rotation',):

    # folder_path = rf"C:\Users\Administrator\Desktop\0808test2\test"
    folder_path = r"C:\Users\Administrator\Desktop\0808test2\testwithd1103"
    

    
    # full_path = f"{folder_path}\\{filename}"
    # errorsites = nqubit
    start_time = time.time()
    current_time = datetime.now().strftime("%m%d_%H%M")
    N_m = None
    # N_m = 10000

    noisetype = 'rotation'
    if statetype == 'snk':
        filename = f'Unoise_nqubit={nqubit}_k={k}_NM={N_m}_errorsites={errorsites}_type_{noisetype}_{current_time}.csv'
        if k != 0:
            cir = Circuit(nqubit)
            cir.h(range(k))
            cir.t(range(k))
            state = cir()
        else:
            state = pq.state.zero_state(nqubit)
    
    elif statetype=='ghz':
        filename = f'Unoise_nqubit={nqubit}_ghz_errorsites={errorsites}_type_{noisetype}_{current_time}.csv'
        state = pq.state.ghz_state(nqubit)
    elif statetype =='w':
        filename = f'Unoise_nqubit={nqubit}_W_errorsites={errorsites}_type_{noisetype}_{current_time}.csv'
        state = pq.state.w_state(nqubit)
    elif statetype == 'snk_theta':
        filename = f'Unoise_nqubit={nqubit}_k={k}_NM={N_m}_theta={theta}_errorsites={errorsites}_type_{noisetype}_{current_time}.csv'
        if k != 0:
            cir = Circuit(nqubit)
            cir.h(range(k))
            cir.p(range(k), param=theta,param_sharing=True)
            state = cir()
        else:
            state = pq.state.zero_state(nqubit)
    full_path = f"{folder_path}\\{filename}"
    state = _type_transform(state, "density_matrix").numpy()

    N = 200 # 样本数量

    # 1/x 在 10^-5 到 1 之间对数均匀分布
    # log_inv_x = np.linspace(-0.01,0, N)
    # log_inv_x = np.linspace(-5, 0, N)
    # log_inv_x = np.array([-5.0 for _ in range(N)])

    # 计算对应的 x 值
    # magnitudelist = 10 ** log_inv_x
    # magnitudelist = np.linspace(0.8,1,N)
    # state = _type_transform(state, "density_matrix").numpy()
    bx = np.linspace(0,3,N)
    magnitudelist = 1-10**(-bx)
    # 使用 Pool 进行并行计算
    with Pool(processes=10) as pool:  # 10 个进程并行
        results = pool.starmap(compute_for_magnitude, [(magnitudelist[i], state, nqubit,noisetype,N_m,errorsites) for i in range(len(magnitudelist)-1)])

    # 统一写入 CSV 文件
    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['infid^{-1}', f'{k}_Character', f'{k}_Cross_Character', 'Sample#','Variance','Variance(amp)',f"{k}_Character(amp)",f"{k}_Cross_Character(amp)",'Variance_thrifty','Variance_thrifty(amp)','Sample#_thrifty', 'N_m', 'PartOne_CRM', 'PartTwo', 'PartOne_THR'])
        writer.writerows(results)

    print(f'Data written to {full_path}')
    
    # 画图
    # plot_results(filename,k)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'运行时间: {hours}时{minutes}分{seconds}秒')
    return full_path

if __name__ == "__main__":
    klist = [1,7]
    n = 7
    for k in klist:
        filename = main(nqubit=n, k=k,theta=0,errorsites=n,noisetype='rotation',statetype='snk')# 画
        # plot_results(filename,k)
        # plot_results('shadow_norm_snk_state_nqubit=5_k=5_Nm=1000000_0325_1931.csv',k)
        variance_amp_values = []

        # 读取 CSV 文件
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # 读取表头
            
            # 找到 'Variance(amp)' 在第几列
            index = header.index('Variance(amp)')

            for row in reader:
                try:
                    variance_amp_values.append(float(row[index]))  # 读取并转换为浮点数
                except ValueError:
                    pass  # 如果转换失败（例如空值），就跳过

        # 计算平均值
        if variance_amp_values:
            mean_variance_amp = np.mean(variance_amp_values)
            print(f"Variance(amp) 平均值: {mean_variance_amp}")
            # print(collection_axis)

        else:
            print("没有有效的 Variance(amp) 数据")
            # print(collection_axis)

