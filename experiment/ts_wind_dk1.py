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
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# %%
df = pd.read_parquet('/Users/fangsiyu/Desktop/energy/new_data/weather/weather_dk1_2019-01-01_2026-08-21.parquet')
df.head()

# %%
df.columns

# %% [markdown]
# ## step 1, make sure its time series
#
# 1. sort
# 2. duplicated
# 3. gap check

# %%
df = df.set_index("hour_utc").sort_index()
# 這邊的作法是下索引，然後sort

# %%
df.index.duplicated().sum()


# %%
df.index.to_series().diff().value_counts()
# 如果頻率是 1 小時，理想上只會看到一個 0 days 01:00:00。跑出別的值就代表有跳號或重複

# %%
err_df = df.drop(df.index[1])

err_df.sort_index()

err_df.duplicated().sum()

err_df.index.to_series().diff().value_counts()
# # 如果頻率是 1 小時，理想上只會看到一個 0 days 01:00:00。跑出別的值就代表有跳號或重複

# %% [markdown]
# ## step 2, plot

# %%
df["wind_speed_100m"].plot(figsize=(14,4))          # 全期


# %%
df["wind_speed_100m"].loc["2023-06"].plot(figsize=(14,4))        # 一個月

# %%
df["wind_speed_100m"].loc["2023-06-05":"2023-06-12"].plot(figsize=(14,4))  # 一週

# %% [markdown]
# ## step 3, alpha

# %%
# Q2 波動幅度穩定嗎？
# 這邊有一個重點是上時間的色，時間區段的先後也重要

g = df["wind_speed_100m"].resample("W") # ME sould be month, W be week
s = pd.DataFrame({"mean": g.mean(), "std": g.std()}).dropna()
s.plot.scatter(x="mean", y="std", c=s.index.isocalendar().week, cmap="twilight")


# %%
# 要注意這邊數據不能是 0 才能作這log

g = df["wind_speed_100m"].resample("W")
s = pd.DataFrame({"mean": g.mean(), "std": g.std()}).dropna()

alpha = np.polyfit(np.log(s["mean"]), np.log(s["std"]), 1)[0]
print(alpha)

# ME 0.6509423549512899
# W 0.6063960778014535

# %%
# 這邊是轉換後驗證，看能不能變成 0

w = np.sqrt(df["wind_speed_100m"])

g2 = w.resample("W")
s2 = pd.DataFrame({"mean": g2.mean(), "std": g2.std()}).dropna()
alpha2 = np.polyfit(np.log(s2["mean"]), np.log(s2["std"]), 1)[0]
print(alpha2)

# %% [markdown]
# ### note: form now on, we have 'w' serve as the new col for the 'wind_speed_100m'

# %%
w = w[w.index.year <= 2025]

# %% [markdown]
# ---
# ## step 4, plot

# %%
w.groupby(w.index.month).mean().plot.bar()

# %%
w.groupby(w.index.hour).mean().plot()      # 日週期


# %%
w.groupby(w.index.dayofyear).mean().plot() # 年週期


# %%
w.groupby([w.index.month, w.index.hour]).mean().unstack().T.plot(legend=False)

# 日這邊是我全部有的天數按照小時平均？如果是這邊圖形看到大概8到9有明顯的波谷，19時有最大的波峰。日週期大概4.91到5。年週期一直都是波動，但是可以看到一年的150~200 這邊會是最底的波谷，0~50 出現最高的波峰。你按照月份的大概有7條是幾乎平移在底下，有另外5條是幾乎相似在上方，但彼此都是不完全重疊的


# %% [markdown]
# ---
#
# ## step 5, each potion
#
# 找到成分 → 移除 → 用新的 r 跑 ACF/PACF/ADF
#
# 但不是「比例大就要移除」
# 這是關鍵。判準有兩條,比例只是第一條:
#
# 第一條:這個成分佔比夠大嗎?
# 11.7% 比 6.9% 好,通過。
#
# 第二條:這個成分是確定性的嗎?能不能外推算出？
# 也就是——給你一個未來的時間點,你能不能算出它的值?
#
# 這一條 year 過不了。會把一個隨機成分當成確定性成分移除,然後宣稱剩下的是定態的。這正是統計上要避免的事。「3 月的平均比全年高 0.4」→ 這個數字對 2027 年 3 月仍然適用,因為 3 月的物理成因每年重複。「2022 年的平均比整體高 0.2」→ 這個數字對 2027 年完全沒用
#
#
# 為什麼要剔除？
# 清理乾淨才能滿足定態、區分開已知未知在這邊是季節（查表就可知）加上殘差（交給模型學），所以要砍掉的要確保我可以再現不然就弄丟了

# %%
# 拆開計算個成分比例

# 這是因為 var 有一重要性質，可以互相相加，且所有不同的子項可以家總成母相，所以這計算總相 w 作為分母，去檢驗分子 r 占比（var(w) = var(r 這邊是非季節) + var(s 這邊是季節但是未知)）
# std 是沒辦法相加的

# 計算參差 r 

# 只扣月份
r_m = w - w.groupby(w.index.month).transform("mean")
# 只扣小時  
r_h = w - w.groupby(w.index.hour).transform("mean")
# 只扣年  
r_y = w - w.groupby(w.index.year).transform("mean")
# 兩者都扣(含交互)
r_mh = w - w.groupby([w.index.month, w.index.hour]).transform("mean")
# 兩者都扣(含交互)
r_ym = w - w.groupby([w.index.year, w.index.month]).transform("mean")

# ! 這邊多作一個傅利葉
doy = w.index.dayofyear
X = np.column_stack([np.sin(2*np.pi*k*doy/365.25) for k in [1,2,3]] +
                    [np.cos(2*np.pi*k*doy/365.25) for k in [1,2,3]])
X = np.column_stack([np.ones(len(w)), X])      # 加常數項

beta, *_ = np.linalg.lstsq(X, w.values, rcond=None)
fit = X @ beta                                  # 配適值,相當於 s_t
r_f = w - fit                                   # 殘差

for name, res in [("month", r_m), ("hour", r_h), ("year", r_y), ("month&hour", r_mh), ("year&month", r_ym), ("fourier(doy)", r_f)]:
    print(name, round(1 - res.var()/w.var(), 5))


# %%
# 判定要踢掉誰後，要決定怎麼留下查表或是一個傅利葉參數，這邊選傅利葉，因為會有多比ts 資料以及不用存一堆table
# 下面簡化了找最最佳 K 的過程

def build_annual_fourier(index, K=3):
    doy = index.dayofyear.values
    cols = [np.ones(len(index))]
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * doy / 365.25))
        cols.append(np.cos(2 * np.pi * k * doy / 365.25))
    return np.column_stack(cols)


K = 3
X = build_annual_fourier(w.index, K)
beta, *_ = np.linalg.lstsq(X, w.values, rcond=None)

seasonal = pd.Series(X @ beta, index=w.index)   # 季節成分
fourier_r = w - seasonal                                # 殘差

print("R2:", round(1 - fourier_r.var() / w.var(), 5)) 
print("beta:", beta.round(4))


# %%
# AIC/BIC 在 n 上萬時對階數選擇無效:懲罰項(2k 或 k·ln n)相對配適改善太輕,
# 且沒有轉折
# 會單調偏好更大的 K。此處僅作記錄,實際選擇看樣本外 MSE。
n = len(w)
for K in [1, 2, 3, 4, 6, 8]:
    X = build_annual_fourier(w.index, K)
    beta, *_ = np.linalg.lstsq(X, w.values, rcond=None)
    sse = ((w.values - X @ beta) ** 2).sum()
    k = X.shape[1]
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    print(K, k, round(aic, 1), round(bic, 1))

# %%
pd.DataFrame({
    "fit": seasonal.groupby(seasonal.index.month).mean(),
    "actual": w.groupby(w.index.month).mean()
}).plot()

# %% [markdown]
# 樣本外 MSE
#
# 雜訊是隨機的,下一批資料的雜訊會不一樣。所以模型花在擬合雜訊上的努力,不但沒用,還會在新資料上造成額外誤差。把資料切成兩塊:訓練集:用來估beta，測試集:模型沒看過,用來算誤差

# %%
cut = w.index[int(len(w) * 0.8)]
tr, te = w[w.index < cut], w[w.index >= cut]

for K in [1, 2, 3, 4, 6, 8]:
    Xtr = build_annual_fourier(tr.index, K)
    beta, *_ = np.linalg.lstsq(Xtr, tr.values, rcond=None)
    Xte = build_annual_fourier(te.index, K)
    print(K, round(((te.values - Xte @ beta) ** 2).mean(), 5))

# %%
np.save("fourier_beta_dk1.npy", beta)

# %%
# 這是重要的一張圖，說明弱定態的第一個條件成立、第二條件成立、

g3 = fourier_r.resample("W")
s3 = pd.DataFrame({"mean": g3.mean(), "std": g3.std()}).dropna()
s3.plot.scatter(x="mean", y="std", c=s3.index.isocalendar().week, cmap="twilight")

# %%
# 這邊反而留下所有「去季節化」之前的痕跡

# g3 = w.resample("W")
# s3 = pd.DataFrame({"mean": g3.mean(), "std": g3.std()}).dropna()
# s3.plot.scatter(x="mean", y="std", c=s3.index.isocalendar().week, cmap="twilight")

# %% [markdown]
# ---

# %% [markdown]
# ## step 6. ACF, PACF
#
# 有用過員資料還看，沒處理過的幾乎都進不去區間

# %%
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(fourier_r, lags=168)
plot_acf(fourier_r, lags=720)

# %%
from statsmodels.tsa.stattools import acf
a = acf(fourier_r, nlags=720)
print(a[[1, 6, 12, 24, 48, 72, 168, 336, 720]].round(3))

# %% [markdown]
# 顯著這邊幾乎都是顯著，因為計算的代寬只要樣本大幾乎所有點位都會超過
#
# 白噪音:ACF 全部是 0,沒有慣性,沒有可預測部分,沒東西可模型化。
#
# ACF是此時跟過去的lag相關係數。整個檢測目的，依照順序回答。左邊欄位是 cor
#
# 第一，有沒有東西預測？全部信賴區建沒有，這就是噪音ACF 全部是 0。
#
# 第二，記憶有多長？看半衰期多少有效範圍。
#
# 第三，衰減的形狀是什麼?平滑指數衰減 → AR 類模型。在某個 lag 突然截斷 → MA 類模型。衰減極慢不收斂 → 可能有單位根。

# %%
from statsmodels.tsa.stattools import pacf
p = pacf(fourier_r, nlags=20)
print(p.round(3))

# 看的是從第幾個 lag 開始掉到接近 0。
# ag 1、2 是大數字,lag 3 開始就是 0.00x 的量級。斷崖在 2 和 3 之間從 0.152 掉到 0.007,差 20 倍。
# 所以 p=2

# %%
# # 這是強制計算有沒有在區間代裡面的方式

# from statsmodels.tsa.stattools import pacf
# import numpy as np

# vals, ci = pacf(fourier_r, nlags=20, alpha=0.05)
# sig = (ci[:, 0] > 0) | (ci[:, 1] < 0)
# print(np.where(sig)[0])

# %% [markdown]
# ---
# ## step 7, ADF & KPSS

# %%
# PACF 的 lag 2 值恰好等於 ϕ2，但是 ϕ1 還需要多一步計算 
# 在 AR(2) 裡, ϕ1會因為 ϕ2的存在而不同
# 但是不去計算，下面的ADF 還是會計算

from statsmodels.tsa.ar_model import AutoReg
m = AutoReg(fourier_r, lags=2, old_names=False).fit()
print(m.params.round(4))

# %%
# ADF 的臨界值大約在 −3.43(1% 水準) 你的 −32.81 遠遠低於它
from statsmodels.tsa.stattools import adfuller
res = adfuller(fourier_r, regression="c", autolag="AIC")
print(res[0], res[1])
print(res[4])


# %%
from statsmodels.tsa.stattools import kpss
s, p, lags, crit = kpss(fourier_r, regression="c", nlags="auto")
print(s, p)
print(lags, crit)

# %% [markdown]
# 這邊得出 ADF 拒絕 單位根 KPSS 接受定態

# %% [markdown]
#
