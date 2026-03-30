import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

# --- 基本设置 ---
plt.rcParams.update({
    'text.usetex': True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    'font.size': 30.6,
    'legend.fontsize': 26.6,
    'font.family': 'Times New Roman',
})

# --- 常数与路径 ---
R = 2e20
p = 0.01
fourDfolder = r"C:\Users\Administrator\Desktop\TFIM\PauliCRM_EDGS4D_ANAL"
PauliCRM_folder = r"C:\Users\Administrator\Desktop\TFIM\PauliCRM_EDGS"
# SRE_folder = r"C:\Users\Administrator\Desktop\TFIM\EDGS_SRE"
SRE_folder = r"C:\Users\Administrator\Desktop\TFIM\EDGS_SRE_DING"

# --- 匹配文件 ---
sre_files = {
    int(re.search(r"L(\d+)\.csv$", f).group(1)): os.path.join(SRE_folder, f)
    for f in os.listdir(SRE_folder)
    if re.search(r"L\d+\.csv$", f)
}
crm_files = {
    int(re.search(r"n=(\d+)_", f).group(1)): os.path.join(PauliCRM_folder, f)
    for f in os.listdir(PauliCRM_folder)
    if f.endswith("p=0.01.csv") and re.search(r"n=(\d+)_", f)
}
fourD_files = {
    int(re.search(r"n=(\d+)_", f).group(1)): os.path.join(fourDfolder, f)
    for f in os.listdir(fourDfolder)
    if f.endswith("p=0.01.csv") and re.search(r"n=(\d+)_", f)
}

# --- 横坐标以 4D CRM 数据为主 ---
n_list_4d = sorted(sre_files.keys())
target_hs = [4.0,2.0,1.0]

data_sre = {h: [] for h in target_hs}
data_crm = {h: [] for h in target_hs}
data_4d = {h: [] for h in target_hs}

# --- 计算 Clifford SRE 数据 ---
for n in n_list_4d:
    if n not in sre_files:
        continue
    d = 2 ** n
    df_sre = pd.read_csv(sre_files[n])
    for h in target_hs:
        idx = (df_sre["h"] - h).abs().idxmin()
        M2_phi = float(df_sre.loc[idx, "val"])*n
        bbV = ((d - 1) / d) ** 2 * (-p ** 2 + 4 * d * p / ((d - 1) * (d + 2))
                                    + (d ** 2 - 3 * d - 2) / ((d - 1) * (d + 2))) + (d ** 2 - 1) / d ** 2
        bbV_starrho = ((1 - p) ** 2 * (2 ** (1 - M2_phi) * (d + 1) - 4)) / (d + 2)
        bbV_starDelta = (p ** 2 * (2 ** (1 - M2_phi) * (d + 1) - 4)) / (d + 2)
        NU_val = np.ceil(5761 * (bbV_starDelta + (bbV - bbV_starrho) / R) / (p * (1 - 1 / d)) ** 2)
        data_sre[h].append((n, NU_val))

# --- 计算 PauliCRM 数据 ---
for n in n_list_4d:
    if n not in crm_files:
        continue
    d = 2 ** n
    df_crm = pd.read_csv(crm_files[n])
    NU_old = np.ceil(68 * np.log2(200) * 16 *
                     (df_crm["V_starDelta"] + (df_crm["VarianceStd"] - df_crm["V_starrho"]) / R)
                     / (p * (1 - 1 / d)) ** 2)
    df_crm["NU_old"] = NU_old
    for h in target_hs:
        idx = (df_crm["h"] - h).abs().idxmin()
        NU_val = float(df_crm.loc[idx, "NU_old"])
        data_crm[h].append((n, NU_val))

# --- 计算 4D CRM 数据 ---
for n in n_list_4d:
    d = 2 ** n
    df_4d = pd.read_csv(fourD_files[n])
    NU_4d = np.ceil(68 * np.log2(200) * 16 *
                    (df_4d["V_starDelta"] + (df_4d["VarianceStd"] - df_4d["V_starrho"]) / R)
                    / (p * (1 - 1 / d)) ** 2)
    df_4d["NU_4d"] = NU_4d
    for h in target_hs:
        idx = (df_4d["h"] - h).abs().idxmin()
        NU_val = float(df_4d.loc[idx, "NU_4d"])
        data_4d[h].append((n, NU_val))

# --- 绘图 ---
scale =6/7
l = 15*scale
plt.figure(figsize=(l, l*0.62))

# --- 颜色区分不同协议 ---
color_dict = {
    'P': 'tab:orange',      # PauliCRM
    'Cl': 'tab:blue',   # Clifford SRE
    '4D': 'tab:green',    # 4D CRM
}

# --- 形状和线型区分不同 h ---
marker_dict = {
    1.0: 'o',
    2.0: 's',
    4.0: '^',
}
linestyle_dict = {
    1.0: '-',
    2.0: '--',
    4.0: ':',
}

# --- 绘制曲线 ---
for h in target_hs:
    marker = marker_dict[h]
    linestyle = linestyle_dict[h]

    # Pauli CRM
    if data_crm[h]:
        ns, vals = zip(*sorted(data_crm[h], key=lambda x: x[0]))
        plt.plot(ns, vals, color=color_dict['P'], linestyle=linestyle, marker=marker,
                 markersize=10, linewidth=2.3, label=fr"$h={h}$ (P)")
for h in target_hs:
    marker = marker_dict[h]
    linestyle = linestyle_dict[h]
    # Clifford SRE
    if data_sre[h]:
        ns, vals = zip(*sorted(data_sre[h], key=lambda x: x[0]))
        plt.plot(
            ns, vals,
            color=color_dict['Cl'],
            linestyle=linestyle,
            marker=marker,
            markersize=9,
            linewidth=2.3,
            label=fr"$h={h}\, (\text{{Cl}}_n)$"
        )
for h in target_hs:
    marker = marker_dict[h]
    linestyle = linestyle_dict[h]
    # 4D CRM
    if data_4d[h]:
        ns, vals = zip(*sorted(data_4d[h], key=lambda x: x[0]))
        plt.plot(ns, vals, color=color_dict['4D'], linestyle=linestyle, marker=marker,
                 markersize=10, linewidth=2.3, label=fr"$h={h}$ (4D)")

# --- 坐标与刻度 ---
plt.xlabel(r"$n$")
plt.ylabel(r"$N_U$")
plt.ylim(0.25, 2.3e5)
plt.xlim(1.7,68)
plt.xscale('log')
plt.yscale('log')
plt.xticks([2, 4, 8, 16, 32, 64], [str(n) for n in [2, 4, 8, 16, 32, 64]])
plt.tick_params(axis='both', which='major', length=6, width=1.2)

# --- 图例 ---
plt.legend(ncol=1, labelspacing=0.1, columnspacing=0.4, loc='lower left')
plt.tight_layout(pad=0.01)
plt.show()

