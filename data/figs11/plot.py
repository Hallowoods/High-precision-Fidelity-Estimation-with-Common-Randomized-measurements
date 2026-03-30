# # ==========================================
# #   TFIM transition: PauliCRM + SRE_DING comparison
# # ==========================================
# import pandas as pd
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import os
# import numpy as np
# import re
# from matplotlib import colormaps

# # ===========================
# # Matplotlib + LaTeX 设置
# # ===========================
# scale = 6 / 7
# mpl.rcParams["text.usetex"] = True
# mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amssymb}"
# plt.rcParams.update({
#     'text.usetex': True,
#     'font.size': 31.6,
#     'legend.fontsize': 28.0,
#     'font.family': 'Times New Roman'
# })

# # ===========================
# # 常数定义
# # ===========================
# p = 0.01
# R = 2e10

# # ===========================
# # 文件夹路径
# # ===========================
# folder_crm = r"C:\Users\Administrator\Desktop\TFIM\plottranstion246"
# folder_sre = r"C:\Users\Administrator\Desktop\TFIM\EDGS_SRE_DING"

# # ===========================
# # 画图准备
# # ===========================
# l = 15 * scale
# plt.figure(figsize=(l, l * 0.618))

# pattern = re.compile(r"n=(\d+)_")

# # ===========================
# # 获取 CRM 文件列表及 n 值（作为基准）
# # ===========================
# file_list_crm = sorted([f for f in os.listdir(folder_crm) if f.endswith(".csv")])
# n_values = []
# for f in file_list_crm:
#     m = pattern.search(f)
#     if m:
#         n_values.append(int(m.group(1)))

# # 反转顺序（若希望从大到小）
# n_values = n_values[::-1]
# file_list_crm = file_list_crm[::-1]

# colors = colormaps.get_cmap("tab10")(np.arange(len(n_values)) % 10)

# # ===========================
# # 绘制 CRM 数据
# # ===========================
# for i, fname in enumerate(file_list_crm):
#     fpath = os.path.join(folder_crm, fname)
#     df = pd.read_csv(fpath)

#     # ⚙️ 跳过 h=0 数据点
#     df = df[np.abs(df["h"]) > 1e-12]

#     n = n_values[i]
#     d = 2 ** n

#     y_val = np.ceil(
#         68 * np.log2(200) * 16 *
#         (df["V_starDelta"] + (df["VarianceStd"] - df["V_starrho"]) / R)
#         / (p * (1 - 1 / d)) ** 2
#     )

#         # 指定不同 n 使用不同的 marker 形状
#     marker_map = {2: "o", 4: "s", 6: "^"}
#     marker = marker_map.get(n, "o")  # 默认 o

#     plt.plot(
#         df["h"], y_val,
#         marker + "-", color=colors[i],
#         alpha=0.8, markersize=10,
#         label=fr"$n={n}$ (P)"
#     )
# # ===========================
# # 绘制 SRE_DING 数据
# # ===========================
# file_list_sre = [f for f in os.listdir(folder_sre) if f.endswith(".csv")]

# for i, n in enumerate(n_values):
#     sre_match = [f for f in file_list_sre if f"L{n}" in f]
#     if not sre_match:
#         print(f"[跳过] 未找到 n={n} 的 SRE 文件。")
#         continue

#     fpath_sre = os.path.join(folder_sre, sre_match[0])
#     df_sre = pd.read_csv(fpath_sre)

#     # ⚙️ 跳过 h=0 数据点
#     df_sre = df_sre[np.abs(df_sre["h"]) > 1e-12]

#     d = 2 ** n
#     NU_vals = []

#     for idx in range(len(df_sre)):
#         # 安全地按行号访问
#         M2_phi = float(df_sre.iloc[idx]["val"]) * n

#         bbV = ((d - 1) / d) ** 2 * (
#             -p ** 2 + 4 * d * p / ((d - 1) * (d + 2))
#             + (d ** 2 - 3 * d - 2) / ((d - 1) * (d + 2))
#         ) + (d ** 2 - 1) / d ** 2

#         bbV_starrho = ((1 - p) ** 2 * (2 ** (1 - M2_phi) * (d + 1) - 4)) / (d + 2)
#         bbV_starDelta = (p ** 2 * (2 ** (1 - M2_phi) * (d + 1) - 4)) / (d + 2)

#         NU_val = np.ceil(
#             5761 * (bbV_starDelta + (bbV - bbV_starrho) / R)
#             / (p * (1 - 1 / d)) ** 2
#         )
#         NU_vals.append(NU_val)

#         # 同样为 SRE 数据设置 marker
#     marker_map = {2: "o", 4: "s", 6: "^"}
#     marker = marker_map.get(n, "s")  # 默认 s

#     plt.plot(
#         df_sre["h"], NU_vals,
#         marker + "--", color=colors[i],
#         alpha=0.8, markersize=9,
#         label=fr"$n={n}$ (Cl)"
#     )


# # ===========================
# # 图形格式
# # ===========================
# plt.xlabel(r"$h$")
# plt.ylabel(r"$N_U$")
# plt.yscale('log')
# plt.ylim(0.9e3, 1.17e5)
# plt.xlim(0.13, 4.08)

# plt.legend(
#     ncol=2,
#     loc="lower right",
#     labelspacing=0.1,         # 图例位置的参考点
#     columnspacing =0.9,
#     bbox_to_anchor=(0.995, 0.005),  # (x, y) 表示相对轴域坐标的偏移
#     borderaxespad=0.1, 
# )
# plt.grid(False)
# plt.tight_layout(pad=0.03)
# plt.show()

# ==========================================
#   TFIM transition: PauliCRM + SRE_DING comparison
# ==========================================
# ==========================================
#   TFIM transition: PauliCRM + SRE_DING comparison
# ==========================================
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import re

# ===========================
# Matplotlib + LaTeX 设置
# ===========================
scale = 6 / 7
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amssymb}"
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 31.6,
    'legend.fontsize': 28.0,
    'font.family': 'Times New Roman'
})

# ===========================
# 常数定义
# ===========================
p = 0.01
R = 2e10

# ===========================
# 文件夹路径
# ===========================
# folder_crm = r"C:\Users\Administrator\Desktop\TFIM\plottranstion246"
folder_crm = r"C:\Users\Administrator\Desktop\TFIM\plottranstion357"
folder_sre = r"C:\Users\Administrator\Desktop\TFIM\EDGS_SRE_DING"

# ===========================
# 画图准备
# ===========================
l = 15 * scale
plt.figure(figsize=(l, l * 0.618))

pattern = re.compile(r"n=(\d+)_")

# ===========================
# 获取 CRM 文件列表及 n 值（作为基准）
# ===========================
file_list_crm = sorted([f for f in os.listdir(folder_crm) if f.endswith(".csv")])
n_values = []
for f in file_list_crm:
    m = pattern.search(f)
    if m:
        n_values.append(int(m.group(1)))

n_values = n_values[::-1]
file_list_crm = file_list_crm[::-1]

# ===========================
# 统一颜色
# ===========================
# ===========================
# 统一颜色（使用 tab10）
# ===========================
color_P  = plt.cm.tab10(1)   # 橙色：PauliCRM (P)
color_Cl = plt.cm.tab10(0)   # 蓝色：SRE_DING (Cl)


# ===========================
# 绘制 CRM 数据 (P)
# ===========================
for i, fname in enumerate(file_list_crm):
    fpath = os.path.join(folder_crm, fname)
    df = pd.read_csv(fpath)

    df = df[np.abs(df["h"]) > 1e-12]

    n = n_values[i]
    d = 2 ** n

    y_val = np.ceil(
        68 * np.log2(200) * 16 *
        (df["V_starDelta"] + (df["VarianceStd"] - df["V_starrho"]) / R)
        / (p * (1 - 1 / d)) ** 2
    )

    # marker_map = {2: "o", 4: "s", 6: "^"}
    marker_map = {3: "o", 5: "s", 7: "^"}
    marker = marker_map.get(n, "o")

    plt.plot(
        df["h"], y_val,
        marker + "-", color=color_P,
        alpha=0.8, markersize=10,
        label=fr"$n={n}$ (P)"
    )

# ===========================
# 绘制 SRE_DING 数据 (Cl)
# ===========================
file_list_sre = [f for f in os.listdir(folder_sre) if f.endswith(".csv")]

for i, n in enumerate(n_values):
    sre_match = [f for f in file_list_sre if f"L{n}" in f]
    if not sre_match:
        print(f"[跳过] 未找到 n={n} 的 SRE 文件。")
        continue

    fpath_sre = os.path.join(folder_sre, sre_match[0])
    df_sre = pd.read_csv(fpath_sre)

    df_sre = df_sre[np.abs(df_sre["h"]) > 1e-12]

    d = 2 ** n
    NU_vals = []

    for idx in range(len(df_sre)):
        M2_phi = float(df_sre.iloc[idx]["val"]) * n

        bbV = ((d - 1) / d) ** 2 * (
            -p ** 2 + 4 * d * p / ((d - 1) * (d + 2))
            + (d ** 2 - 3 * d - 2) / ((d - 1) * (d + 2))
        ) + (d ** 2 - 1) / d ** 2

        bbV_starrho = ((1 - p) ** 2 * (2 ** (1 - M2_phi) * (d + 1) - 4)) / (d + 2)
        bbV_starDelta = (p ** 2 * (2 ** (1 - M2_phi) * (d + 1) - 4)) / (d + 2)

        NU_val = np.ceil(
            5761 * (bbV_starDelta + (bbV - bbV_starrho) / R)
            / (p * (1 - 1 / d)) ** 2
        )
        NU_vals.append(NU_val)

    # marker_map = {2: "o", 4: "s", 6: "^"}
    marker_map = {3: "o", 5: "s", 7: "^"}
    marker = marker_map.get(n, "s")

    plt.plot(
        df_sre["h"], NU_vals,
        marker + "--", color=color_Cl,
        alpha=0.8, markersize=9,
        label=fr"$n={n}\, (\text{{Cl}}_n)$"
    )

# ===========================
# 图形格式
# ===========================
plt.xlabel(r"$h$")
plt.ylabel(r"$N_U$")
plt.yscale('log')
# plt.ylim(0.9e3, 1.17e5)
plt.ylim(1.5e3, 1.3e5)
plt.xlim(0.13, 4.08)

plt.legend(
    ncol=2,
    loc="lower right",
    labelspacing=0.1,
    columnspacing=0.9,
    bbox_to_anchor=(0.995, 0.005),
    borderaxespad=0.1,
)
plt.grid(False)
plt.tight_layout(pad=0.03)
plt.show()
