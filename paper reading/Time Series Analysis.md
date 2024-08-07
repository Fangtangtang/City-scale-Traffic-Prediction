# Time Series Analysis
[link](https://www.machinelearningplus.com/time-series/time-series-analysis-python/)

## Plot of a Time Series
时间序列有一定变化趋势，对交通流量显然有明显日周期趋势
hourly time series and follows a certain repetitive pattern every day
![image-202405072](./plot/time_series_Figure_1.png)

对日流量求和，发现有按星期的变化趋势

**(todo:月变化趋势？初步来看并不显著)**
![image-202405073](./plot/time_series_Figure_2.png)
Any time series may be split into the following components:
- Base Level
- Trend
- Seasonality
- Error

## 时间序列分解
$$
TimeSeries= F(Base Level, Trend , Seasonality , Error)
$$
可以将$F$建模为加性、乘性（或其他更复杂的变化形式）

*期望在预处理过程中将时间序列的成分做一个更好的分解？*

以下为对单点的数据（以小时为粒度，取前792小时(33天)，缺省值用邻近值补位）做乘性和加性分解使得结果（更多图像：`./plot`）
![image-202405082](./plot/decompose_mul_hourly.png)
![image-202405081](./plot/decompose_add_hourly.png)

分解后，trend还有明显的周期性信息，对trend按日计算，再做一次分解
![image-202405092](./plot/decompose_mul_daily.png)
![image-202405091](./plot/decompose_add_daily.png)


感觉上乘性分解和加性差距不大，似乎乘性略好一点//目标应该是把周期性信息充分提取，[所以resid越随机越好](https://www.machinelearningplus.com/time-series/time-series-analysis-python/#:~:text=The%20multiplicative%20decomposition%2C%20however%2C%20looks%20quite%20random%20which%20is%20good.)？？

**数据中存在0，如果不对0做处理，不能使用乘性分解**

**由于padding方式很不合适，在有padding段的数据上会出问题。**也许考虑使用别的简单模型对padding数据做个预测？
```
一个奇怪的想法：
先用一种简朴的方法补全，然后一轮轮迭，拿前一次结果对缺省的位置做预测，作为后面的数据。设计一种衡量指标来筛选。
```
![image-202405093](./plot/fatal_padding_sample.png)

*这里暂时仅考虑了每天的周期性，从trend中大概能感觉出还有一个按周的周期性，考虑对周信息也做类似操作？*

*另一个想法是手写一个分解，由两个周期序列和一个trend序列*

## How to treat missing values in a time series?
- 均值（似乎不是很好）
- 前向、反向
- knn
- 周期序列
- 用模型预测
- ……


# ARIMA(AutoRegressive Integrated Moving Average)
[link](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)

basic idea: the information in the past values of the time series can alone be used to predict the future values

for **non-seasonal** time series.
for seasonal time series, we need to add seasonal terms.

## terms
```python
# p: the order of the AR term. 自回归部分的阶数，表示模型中考虑的过去时间步的数量
# d: the number of differencing required to make the time series stationary. 差分的阶数，表示对原始时间序列进行差分的次数，使其平稳
# q: the order of the MA term. 滑动平均部分的阶数，表示模型中考虑的预测误差的数量
ARIMA(p=10, d=1, q=8)
```

Predicted Yt = Constant + Linear combination Lags of Y (upto p lags) + Linear Combination of Lagged forecast errors (upto q lags)
$$
Y_t=c+(\beta_1Y_{t-1}+\beta_2Y_{t-2}+\cdots+\beta_pY_{t-p})+(\theta_1\epsilon_{t-1}+\theta_2\epsilon_{t-2}+\cdots+\theta_q\epsilon_{t-q})+\epsilon_t
$$

simple usage: `from statsmodels.tsa.arima_model import ARIMA`


### AR(Auto Regressive)
linear regressive model

$$
Y_t=\alpha_{ar}+\beta_1Y_{t-1}+\beta_2Y_{t-2}+\cdots+\beta_pY_{t-p}+\epsilon
$$

### I(Integrated)
对原始时间序列做差分处理。

差分可以消除时间序列的季节性和趋势，使得模型更容易建立和预测。

### MA(Moving Average)
滑动平均部分，指模型使用过去时间步的预测误差来预测未来时间步的观测值。MA 模型中的滑动平均项表示当前观测值与前几个预测误差之间的线性关系。

