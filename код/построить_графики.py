import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out = Path('/home/ubuntu/learning_curves_seminar/assets')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'figure.dpi':150})

# 1 exact toy curve
n = np.array([5,10,20,50,100,200])
L = 1 + 1/n
fig, ax = plt.subplots(figsize=(7,4.2))
ax.plot(n,L,'o-',color='#1f77b4',label=r'Теория: $L(n)=1+1/n$')
ax.axhline(1,color='#d62728',ls='--',label='Непреодолимый floor = 1')
ax.set(xlabel='Размер обучающей выборки n',ylabel='Ожидаемый MSE',title='1. Точная learning curve для константной регрессии')
ax.legend(); fig.tight_layout(); fig.savefig(out/'01_exact_curve.png',bbox_inches='tight'); plt.close(fig)

# 2 bias/variance conceptual
n = np.array([10,20,40,80,160,320])
train_simple=np.array([4.2,4.0,3.9,3.85,3.82,3.80])
test_simple=np.array([4.8,4.5,4.25,4.05,3.95,3.90])
train_complex=np.array([0.35,0.55,0.85,1.20,1.55,1.80])
test_complex=np.array([8.4,6.0,4.7,3.9,3.45,3.25])
fig, ax = plt.subplots(figsize=(7,4.2))
ax.plot(n,test_simple,'o-',label='Простая модель: test')
ax.plot(n,train_simple,'o--',label='Простая модель: train')
ax.plot(n,test_complex,'s-',label='Гибкая модель: test')
ax.plot(n,train_complex,'s--',label='Гибкая модель: train')
ax.set_xscale('log',base=2); ax.set(xlabel='Размер n (логарифмическая шкала)',ylabel='MSE',title='2. Недообучение и высокая дисперсия: схема')
ax.legend(ncol=2,fontsize=9); fig.tight_layout(); fig.savefig(out/'02_under_overfit.png',bbox_inches='tight'); plt.close(fig)

# 3 mean with uncertainty bands (deterministic)
n=np.array([10,20,40,80,160,320])
mean=np.array([5.8,4.7,4.0,3.55,3.30,3.18])
sd=np.array([1.5,1.0,.72,.50,.36,.28])
se=sd/np.sqrt(25)
fig, ax=plt.subplots(figsize=(7,4.2))
ax.plot(n,mean,'o-',color='#2ca02c',label='Средняя test loss по 25 повторам')
ax.fill_between(n,mean-1.96*se,mean+1.96*se,color='#2ca02c',alpha=.2,label='≈95% CI среднего')
ax.errorbar(n,mean,yerr=sd,fmt='none',ecolor='#555',alpha=.45,capsize=3,label='±1 SD между повторами')
ax.set_xscale('log',base=2); ax.set(xlabel='Размер n (логарифмическая шкала)',ylabel='Test loss',title='3. Среднее качество и разброс повторных обучений')
ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(out/'03_uncertainty.png',bbox_inches='tight'); plt.close(fig)

# 4 one noisy trajectory vs average
n=np.array([10,20,40,80,160,320])
paths=np.array([[6.0,4.0,3.9,3.4,3.1,3.0],[5.2,5.0,4.2,3.8,3.5,3.2],[7.0,4.2,3.5,3.8,3.0,3.3],[4.9,5.4,4.4,3.2,3.4,3.1],[6.3,4.8,4.0,3.6,3.2,3.0]])
mean=paths.mean(0)
fig, ax=plt.subplots(figsize=(7,4.2))
for p in paths: ax.plot(n,p,color='#999',alpha=.65,lw=1.4)
ax.plot(n,mean,'o-',color='#111',lw=2.5,label='Средняя траектория')
ax.set_xscale('log',base=2); ax.set(xlabel='Размер n (логарифмическая шкала)',ylabel='Test loss',title='4. Одна вложенная траектория не равна математическому ожиданию')
ax.legend(); fig.tight_layout(); fig.savefig(out/'04_paths_vs_mean.png',bbox_inches='tight'); plt.close(fig)
