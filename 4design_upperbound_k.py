"""
4design_upperbound_k.py

This module provides functions for calculating terms in the variance of 4-design CRM and thrifty.

Author: YZY
Date: 2025-04-21
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





def calcualte_terms_vu(state,sigma,nqubit):
    '''Calculate 4 upper terms and their summation that contributes to the expression of v_u'''
    d = 2**nqubit
    delta = state- sigma
    I = np.identity(d)
    operator = sigma-I/d    #traceless part
    term1 = np.real((d**2+3*d+4)/(d*(d+2)*(d+3))*(np.trace(delta @ operator))**2)
    term2 = np.real(2*(1+d)**2/(d*(d+2)*(d+3))*np.trace(delta@delta@operator@operator))
    term3 = np.real((d+1)/(d*(d+3))*np.trace(delta@delta)*np.trace(operator@operator))
    term4 = np.real(-2*(1+d)/(d*(d+2)*(d+3))*np.trace(delta@operator@delta@operator))
    vu = term1+term2+term3+term4
    return vu



def calcualte_terms_vstar(state,sigma,nqubit):
    '''Calculate 4 upper terms and their summation that contributes to the expression of v*'''
    d = 2**nqubit
    I = np.identity(d)
    operator = sigma-I/d    #traceless part
    term1 = np.real((d**2+3*d+4)/(d*(d+2)*(d+3))*(np.trace(state @ operator))**2)
    term2 = np.real(2*(1+d)**2/(d*(d+2)*(d+3))*np.trace(state@state@operator@operator))
    term3 = np.real((d+1)/(d*(d+3))*np.trace(state@state)*np.trace(operator@operator))
    term4 = np.real(-2*(1+d)/(d*(d+2)*(d+3))*np.trace(state@operator@state@operator))
    vstar = term1+term2+term3+term4
    return vstar



def compute_for_magnitude(magnow, state, nqubit):
    """单个进程的计算任务"""
    d = 2**nqubit
    channel, mag = generate_random_pauli_channel(nqubit, magnitude=magnow,channel_num=0)
    state_noised, l1_norm = generate_channel(state, nqubit, channel)
    state_noised = _type_transform(state_noised, "density_matrix").numpy()
    
    vu = calcualte_terms_vu(state_noised, state, nqubit)
    fidelity = np.real(np.trace(state_noised @ state))
    vstar = calcualte_terms_vstar(state_noised,state,nqubit)

    # N_m = d*np.real(1/(1-fidelity)**2)
    N_m = 20*np.real(1/(1-fidelity)**2)
    variance = vu + 1/N_m*(-fidelity**2+d*(2*fidelity+1)/(d+2)-vstar)
    N_m_term = (-fidelity**2+d*(2*fidelity+1)/(d+2)-vstar)
    thrifty_variance = vstar+1/N_m*(-fidelity**2+d*(2*fidelity+1)/(d+2)-vstar)
    amp_thrifty_variance = np.real(thrifty_variance/(1-fidelity)**2)
    amp_variance= np.real(variance/(1-fidelity)**2)
    sample = int(68*np.log(200)*variance*16/(1-fidelity)**2)
    sample_thr = int(68*np.log(200)*thrifty_variance*16/(1-fidelity)**2)
    remain = (-fidelity**2+d*(2*fidelity+1)/(d+2)-vstar)



    
    #thrifty ones

    return [1 / (1 - np.real(fidelity)), sample,np.real(variance),amp_variance,np.real(thrifty_variance),amp_thrifty_variance,sample_thr,np.real(vu),N_m_term, vstar, remain]



def main(nqubit, k):
    start_time = time.time()
    current_time = datetime.now().strftime("%m%d_%H%M")

    # 修改为你的目标路径
    # folder_path = rf"C:\Users\Administrator\Desktop\plottemp\4d_equal_spacing_Pnoise k"
    folder_path = rf"C:\Users\Administrator\Desktop\plottemp\4d_equal_spacing_Pnoise k\alter0226Pnoise"

    filename = f'4design_variance_changing_snk_state_n={nqubit}_k={k}_{current_time}.csv'
    full_path = f"{folder_path}\\{filename}"

    N=40
    log_inv_x = np.linspace(-4.1, 0, N)

    # 计算对应的 x 值
    magnitudelist = 10 ** log_inv_x

    if k != 0:
        cir = Circuit(nqubit)
        cir.h(range(k))
        cir.t(range(k))
        state = cir()
    else:
        state = pq.state.zero_state(nqubit)
    state = _type_transform(state, "density_matrix").numpy()

    with Pool(processes=10) as pool:
        results = pool.starmap(compute_for_magnitude, [(magnow, state, nqubit) for magnow in magnitudelist])

    # 写入 CSV 文件到指定目录
    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['infid^{-1}', 'Sample#','Variance','Variance(amp)','Variance_thrifty','Variance_thrifty(amp)','Sample#_thrifty','vu','N_mterm','vstar','remain'])
        writer.writerows(results)

    print(f'Data written to {full_path}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'运行时间: {hours}时{minutes}分{seconds}秒')
    return full_path


if __name__ == "__main__":
    klist = [1,7]
    # klist = [1,2,3]
    Nm = 1e7
    # N=50
    # log_inv_x = np.linspace(-3, 0, N)

    # # 计算对应的 x 值
    # magnitudelist = 10 ** log_inv_x
    # l = range(2, 11,2)
    # sample_list = [10**int(r) for r in l ]
    for k in klist:
        print(f"Processing k={k}")
        filename = main(nqubit=7, k=k)# 画
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
        else:
            print("没有有效的 Variance(amp) 数据")

# if __name__ == "__main__":
#     main(nqubit=3,k=1)