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



def save_pauli_channel(channelinfo, filename):
    '''write-in Pauli Channel file'''
    with open(filename, 'wb') as file:
        pickle.dump(channelinfo, file)  # 将 channelinfo 保存到文件中

def load_pauli_channel(filename):
    '''Read the Pauli Channel file'''

    with open(filename, 'rb') as file:
        channelinfo = pickle.load(file)  # 从文件中读取 channelinfo
    return channelinfo


def generate_random_pauli_channel(nqubit,magnitude = 0.1,channel_num = 0):
    channelinfo = []
    existing_pauli_strings = set()  # 用于跟踪已生成的 Pauli 字符串
    maxallowed = 4**nqubit - 1
    if channel_num == 0:    #default:random
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
    d = 2**nqubit
    pauli_group = generate_pauli_group(nqubit=nqubit) #loading Pauli group first
    # print('pauligroup:',pauli_group)
    delta = state-sigma
    sigma -= np.identity(d)/d
    delta_vec = []
    sigma_vec = []
    cross_vec = []
    for i in range(len(pauli_group)):
        print(i)
        print(pauli_group[i])
        ele = pauli_group[i]
        ele[1] = pq.qinfo.pauli_str_to_matrix([[1,ele[1]]],nqubit).numpy()
        delta_vec.append(np.trace(delta@ele[1]))  
        sigma_vec.append(np.trace(sigma@ele[1]))
        cross_vec.append(np.trace(delta@ele[1]@sigma@ele[1]))
    delta_vec = np.array(delta_vec)
    sigma_vec = np.array(sigma_vec)
    cross_vec = np.array(cross_vec)

    
    cross_character = np.sum(delta_vec*sigma_vec*cross_vec)

    character = np.dot(delta_vec**2,sigma_vec**2)
    return character,cross_character


def caculate_characters_THR(state,sigma,nqubit):
    '''Calculate cross chatacteristic functions of rho and sigma and charcteristic functions, output as a product of them'''
    d = 2**nqubit
    pauli_group = generate_pauli_group(nqubit=nqubit) #loading Pauli group first
    # print('pauligroup:',pauli_group)
    state_vec = []
    cross_vec = []
    traceles_sigma_vec = []
    traceless_sigma = sigma-np.identity(d)/d
    for i in range(len(pauli_group)):
        print(i)
        print(pauli_group[i])
        ele = pauli_group[i]
        ele[1] = pq.qinfo.pauli_str_to_matrix([[1,ele[1]]],nqubit).numpy()
        # pauli =pq.qinfo.pauli_str_to_matrix([[1,pauli_group[i][1]]],nqubit)
        # print(pauli) 
        state_vec.append(np.trace(state@ele[1]))
        traceles_sigma_vec.append(np.trace(traceless_sigma@ele[1]))
        cross_vec.append(np.trace(state@ele[1]@traceless_sigma@ele[1]))
    state_vec = np.array(state_vec)
    traceles_sigma_vec = np.array(traceles_sigma_vec)
    cross_vec = np.array(cross_vec)

    
    cross_character = np.sum(traceles_sigma_vec*state_vec*cross_vec)

    character = np.dot(state_vec**2,traceles_sigma_vec**2)
    return character,cross_character



def compute_for_magnitude(magnow, state, nqubit,N_m =100):
    """单个进程的计算任务"""
    d = 2**nqubit
    channel, mag = generate_random_pauli_channel(nqubit, magnitude=magnow,channel_num=3)
    state_noised, l1_norm = generate_channel(state, nqubit, channel)
    state_noised = _type_transform(state_noised, "density_matrix").numpy()
    
    character, cross_character = caculate_characters_CRM(state_noised, state, nqubit)
    fidelity = np.trace(state_noised @ state)
    variance_u = ((d+1)/(d*(d+2)))*(np.real(character)+np.real(cross_character))-(1 - np.real(fidelity))**2/(d+2)
    character2, cross_character2 = caculate_characters_THR(state_noised,state,nqubit)
    amp_character = np.real(character/(1-fidelity)**2)*16
    amp_cross = np.real(cross_character/(1-fidelity)**2)*16


    variance_0 = (d+1)/(d*(d+2))*(np.real(character2)+np.real(cross_character2))-(1 - np.real(fidelity))**2/(d+2)

    variance = variance_u + 1/N_m*(-fidelity**2+d*(2*fidelity+1)/(d+2)-variance_0)
    thrifty_variance = variance_0+1/N_m*(-fidelity**2+d*(2*fidelity+1)/(d+2)-variance_0)
    amp_thrifty_variance = np.real(16*thrifty_variance/(1-fidelity)**2)
    amp_variance= np.real(variance/(1-fidelity)**2)*16
    sample = int(16*136*np.log(200)*variance/(1-fidelity)**2)
    sample_thr = int(16*136*np.log(200)*thrifty_variance/(1-fidelity)**2)
    #thrifty ones
    

    return [1 / (1 - np.real(fidelity)), np.real(character), np.real(cross_character), sample,np.real(variance),amp_variance,amp_character,amp_cross,np.real(thrifty_variance),amp_thrifty_variance,sample_thr]



def main_varying_Nm(nqubit, k, magnitude=0.01):
    start_time = time.time()
    current_time = datetime.now().strftime("%m%d_%H%M")
    folder_path = rf"D:\probability-expectation-project\src\N_M as x axis"
    filename = f'shadow_norm_haar_state_nqubit={nqubit}_varyNm_mag={magnitude}_{current_time}.csv'
    full_path = f"{folder_path}\\{filename}"
    
    # log-spaced N_M
    N_M_list = np.unique(np.logspace(0, 7, num=70, dtype=int))  # 从1到1e5，共50个不同N_M
    
    state = pq.state.random_state(nqubit)
    state = _type_transform(state, "density_matrix").numpy()

    
    # 固定 magnitude 生成一次 Pauli 通道（这样所有 N_M 共享）
    channel, _ = generate_random_pauli_channel(nqubit, magnitude=magnitude, channel_num=3)
    state_noised, l1_norm = generate_channel(state, nqubit, channel)
    state_noised = _type_transform(state_noised, "density_matrix").numpy()

    # 预先计算 Character 和 Fidelity，避免重复
    character, cross_character = caculate_characters_CRM(state_noised, state, nqubit)
    character2, cross_character2 = caculate_characters_THR(state_noised,state, nqubit)
    fidelity = np.real(np.trace(state_noised @ state))
    d = 2**nqubit
    variance_0 = (d+1)/(d*(d+2))*(np.real(character2)+np.real(cross_character2))-(fidelity-1/d)**2/(d+2)

    results = []

    for N_m in N_M_list:
        variance_u = ((d+1)/(d*(d+2)))*(np.real(character)+np.real(cross_character))-(1 - fidelity)**2/(d+2)
        variance = variance_u + 1/N_m*(-fidelity**2 + d*(2*fidelity+1)/(d+2) - variance_0)
        thrifty_variance = variance_0 + 1/N_m*(-fidelity**2 + d*(2*fidelity+1)/(d+2) - variance_0)
        print(('term',-fidelity**2 + d*(2*fidelity+1)/(d+2) -variance_0))
        amp_variance = 16 * variance / (1 - fidelity)**2
        amp_thrifty_variance = 16 * thrifty_variance / (1 - fidelity)**2
        infid_inv = 1/(1 - fidelity)
        sample = int(16 * 68 * np.log(200) * variance / (1 - fidelity)**2)
        sample_thr = int(16 * 68 * np.log(200) * thrifty_variance / (1 - fidelity)**2)

        results.append([N_m, variance, amp_variance, sample, thrifty_variance, amp_thrifty_variance, sample_thr,infid_inv])
    
    # 写入 CSV
    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N_M', 'Variance', 'Variance(amp)', 'Sample#', 'Variance_thrifty', 'Variance_thrifty(amp)', 'Sample#_thrifty','infid^{-1}'])
        writer.writerows(results)

    print(f"结果写入 {full_path}")
    return full_path


if __name__ == "__main__":
    k = 0
    full_path = main_varying_Nm(nqubit=7, k=k)# 画
    # plot_results(filename,k)
    # plot_results('shadow_norm_snk_state_nqubit=5_k=5_Nm=1000000_0325_1931.csv',k)
    variance_amp_values = []

    # 读取 CSV 文件
    with open(full_path, mode='r', newline='') as file:
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
    else:
        print("没有有效的 Variance(amp) 数据")

# if __name__ == "__main__":
#     main(nqubit=3,k=1)
