# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: dev-env (3.13.11.final.0)
#     language: python
#     name: python3
# ---

# %%
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss

# %%
df_1h = pd.read_parquet('/Users/fangsiyu/Desktop/energy/new_data/price/price_dk1_2019-01-01_2026-08-21.parquet')
df_15 = pd.read_parquet('/Users/fangsiyu/Desktop/energy/new_data/price/price15_dk1_2025-09-30_2026-08-21.parquet')
df_1h.head()

# %%
df_15.head()

# %%
df_1h = df_1h.set_index("HourUTC").sort_index()
print(df_1h.index.duplicated().sum(), df_1h.index.to_series().diff().value_counts())

# %%
df_15 = df_15.set_index("TimeUTC").sort_index()
print(df_15.index.duplicated().sum(), df_15.index.to_series().diff().value_counts())

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

df_1h["SpotPriceEUR"].plot(ax=axes[0])

df_1h["SpotPriceEUR"].loc["2023-06"].plot(ax=axes[1])

df_1h["SpotPriceEUR"].loc["2023-06-05":"2023-06-12"].plot(ax=axes[2])  # 一週

plt.show()

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

df_15["DayAheadPriceEUR"].plot(ax=axes[0])

df_15["DayAheadPriceEUR"].loc["2026-06"].plot(ax=axes[1])

df_15["DayAheadPriceEUR"].loc["2026-06-05":"2026-06-12"].plot(ax=axes[2])  # 一週

plt.show()

# %%
g = df_1h["SpotPriceEUR"].resample("W") # ME sould be month, W be week
s = pd.DataFrame({"mean": g.mean(), "std": g.std()}).dropna()
s.plot.scatter(x="mean", y="std", c=s.index.isocalendar().week, cmap="twilight")

# %%
g = df_15["DayAheadPriceEUR"].resample("W") # ME sould be month, W be week
s = pd.DataFrame({"mean": g.mean(), "std": g.std()}).dropna()
s.plot.scatter(x="mean", y="std", c=s.index.isocalendar().week, cmap="twilight")

# %% [markdown]
# ---
# ## dual-resolution pipeline

# %% [markdown]
# 【資料層】
# df_1h   : 2019-01 → 2026-08,聚合到小時(注意:2025-10 之後是四格平均)
# df_15   : 2025-10 → 2026-08,原生 15 分鐘
#
# 【尺度層】
# 在 df_1h 上決定變換(asinh 的尺度參數 c)
# → 同一個 c 套用到 df_15
#
# 【低頻層】- 用 df_1h 估參數
# 年週期傅立葉 → beta_annual
# 週週期傅立葉 → beta_weekly
# 在 2019-2025 整年上估,避免月份分布不均
#
# 【高頻層】- 用 df_15 估
# 日內結構(15 分鐘級的 96 格 profile 或傅立葉)
# 短期自相關(ACF/PACF/AR 階數)
#
# 【殘差】
# r_15 = transform(df_15) - s_annual(t) - s_weekly(t) - s_daily(t)
# 其中前兩個用借來的 beta 求值,第三個用 df_15 自己估

# %% [markdown]
# ---
#
# 這邊需要一個非線性轉換。為什麼要？統計工具的前提。 ACF、ADF、AR 的估計理論都假設變異數穩定。不穩定的話,估計量的標準誤失真,信賴區間不可信。
#
# 「讓離散度不隨水準變化」，不過在這邊我們不用 alpha 作指標，而是用 r 皮爾森積動差相關係數，把他拉成 0

# %%
print(df_1h["SpotPriceEUR"].describe(percentiles=[.001, .01, .05, .25, .5, .75, .95, .99, .999]))

# %%
import numpy as np
import pandas as pd

x = df_1h["SpotPriceEUR"]

for c in [10, 20, 30, 40, 50, 61.5, 73.4, 100.0]:
    p = pd.Series(np.arcsinh(x.values / c), index=x.index)
    g = p.resample("W")
    s = pd.DataFrame({"mean": g.mean(), "std": g.std()}).dropna()
    r = np.corrcoef(s["mean"], s["std"])[0, 1]
    print(c, round(r, 4), round(s["std"].std(), 4))

# %%
# 這是最佳化套件，但是沒必要在噪音上過度優化
from scipy.optimize import brentq

def r_of_c(c):
    p = pd.Series(np.arcsinh(x.values / c), index=x.index)
    g = p.resample("W")
    s = pd.DataFrame({"mean": g.mean(), "std": g.std()}).dropna()
    return np.corrcoef(s["mean"], s["std"])[0, 1]

c_opt = brentq(r_of_c, 20, 40)
print(c_opt)

# %%
C_PRICE = 30.0

# %%
p_1h = pd.Series(np.arcsinh(df_1h["SpotPriceEUR"].values / C_PRICE),
                 index=df_1h.index)
p_15 = pd.Series(np.arcsinh(df_15["DayAheadPriceEUR"].values / C_PRICE),
                 index=df_15.index)

# %%
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# p_1h.resample("ME").mean().plot(ax=axes[0])

# g = p_1h.resample("W")
# s = pd.DataFrame({"mean": g.mean(), "std": g.std()}).dropna()
# s.plot.scatter(x="mean", y="std", c=range(len(s)), cmap="viridis", ax=axes[1])


# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

p_1h.resample("ME").mean().plot(ax=axes[0])
axes[0].axhline(0, color="k", lw=0.5)
axes[0].set_title("monthly mean (asinh scale)")

sc = axes[1].scatter(s["mean"], s["std"], c=s.index.year, cmap="plasma", s=25)
plt.colorbar(sc, ax=axes[1], label="year")
axes[1].set_xlabel("weekly mean")
axes[1].set_ylabel("weekly std")

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

p_15.resample("ME").mean().plot(ax=axes[0])

g = p_15.resample("W")
s = pd.DataFrame({"mean": g.mean(), "std": g.std()}).dropna()
s.plot.scatter(x="mean", y="std", c=range(len(s)), cmap="viridis", ax=axes[1])


plt.show()

# %%
