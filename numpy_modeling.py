#%%
'''
numpy 구현
'''
import numpy as np
import matplotlib.pyplot as plt
#%%
# 학습률, 반복횟수 지정
learning_rate = 1e-4
iter = 10000

# 학습데이터(원랜 쪼개기)
x = 5*np.random.rand(50,1) # 랜덤 값(0 ~ 5 사이의 실수)으로 만든 50개의 수 
y = 3*x + 5*np.random.rand(50,1) # 각 x의 3을 곱한 값과 랜덤 값(0 ~ 5 사이의 실수)의 합
# x = np.array([[8.70153760], [3.90825773], [1.89362433], [3.28730045], [7.39333004], [2.98984649], [2.25757240], [9.84450732], [9.94589513], [5.48321616]])
# y = np.array([[5.64413093], [3.75876583], [3.87233310], [4.40990425], [6.43845020], [4.02827829], [2.26105955], [7.15768995], [6.29097441], [5.19692852]])
#%%
# 현재 가중치로 테스트 데이터 예측
def prediction(a,b,x):
    equation = x*a + b
    return equation
# 오차로 가중치 갱신
def update_ab(a,b,x,error,lr):
    delta_a = -(lr*(2/len(error))*(np.dot(x.T, error)))
    delta_b = -(lr*(2/len(error))*np.sum(error))
    return delta_a, delta_b


# 경사하강법
def gradient_descent(x,y,iters=iter):
    a = np.zeros((1,1))
    b = np.zeros((1,1))

    for i in range(iters):
        error = y - prediction(a,b,x)
        a_delta, b_delta = update_ab(a,b,x,error,lr=learning_rate)
        a-= a_delta
        b-= b_delta
    return a, b

def plot_graph(x,y,a,b):
    y_pred=a[0,0]*x+b
    plt.scatter(x,y)
    plt.plot(x, y_pred)

def main():
    a, b = gradient_descent(x,y,iters=iter)
    print(f'a:{a}, b:{b}')
    plot_graph(x,y,a,b)
    return a,b
main()
# %%
'''
sicitlearn 구현
'''
import matplotlib as mpl
from sklearn.linear_model import LinearRegression

# 모델 클래스 불러오기
lr = LinearRegression()
# 학습데이터 학습
lr.fit(x,y)
# 테스트
predicted = lr.predict(x)
#그래프 시각화
fig, ax = plt.subplots(1,2, figsize=(16, 7))
ax[0].scatter(x,y)
ax[1].scatter(x,y)
# x와 예측 데이터 값으로 그래프 그리기
ax[1].plot(x, predicted, color='b')
ax[0].set_xlabel('x');ax[0].set_ylabel('y');ax[1].set_xlabel('x');ax[1].set_ylabel('y')



# %%
