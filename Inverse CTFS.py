import numpy as np
import matplotlib.pyplot as plt

# 정의: 표준 sinc 함수 (sin(x)/x, x=0일 때 sinc(0)=1로 정의)
def sinc(x):
    return np.where(x == 0, 1.0, np.sin(x)/x)

# 파라미터 설정
N1 = 25  # 촘촘한 샘플 수
N2 = 4   # 덜 촘촘한 샘플 수
bw = np.pi  # 대역폭: 첫 번째 영점까지가 π로 가정

# 주파수 영역 샘플링: [-bw, bw] 범위에서 샘플
# 촘촘한 신호의 주파수 샘플
omega1 = np.linspace(-bw, bw, N1, endpoint=False)
X1 = sinc(omega1)

# 덜 촘촘한 신호의 주파수 샘플
omega2 = np.linspace(-bw, bw, N2, endpoint=False)
X2 = sinc(omega2)

# 시간 영역으로의 복원 (단순 inverse transform 근사)
t = np.linspace(-1, 1, 1000)  # 시간축 예제
Delta_omega1 = (2*bw)/N1
Delta_omega2 = (2*bw)/N2

x1 = np.zeros_like(t, dtype=complex)
for i, w in enumerate(omega1):
    x1 += X1[i] * np.exp(1j*w*t) * Delta_omega1/(2*np.pi)

x2 = np.zeros_like(t, dtype=complex)
for i, w in enumerate(omega2):
    x2 += X2[i] * np.exp(1j*w*t) * Delta_omega2/(2*np.pi)

# 복원된 신호의 magnitude 계산
x1 = np.abs(x1)
x2 = np.abs(x2)

# 플로팅
fig, axes = plt.subplots(2, 2, figsize=(12,8))

# 주파수 영역 플롯 (촘촘한 샘플)
axes[0,0].stem(omega1, X1)
axes[0,0].set_title(f'Frequency Domain (N={N1})')
axes[0,0].set_xlabel('ω')
axes[0,0].set_ylabel('X(ω)')

# 시간 영역 플롯 (복원신호, N=20) - magnitude
axes[0,1].plot(t, x1)
axes[0,1].set_title(f'Time Domain Signal Magnitude (N={N1})')
axes[0,1].set_xlabel('t')
axes[0,1].set_ylabel('|x(t)|')

# 주파수 영역 플롯 (덜 촘촘한 샘플)
axes[1,0].stem(omega2, X2)
axes[1,0].set_title(f'Frequency Domain (N={N2})')
axes[1,0].set_xlabel('ω')
axes[1,0].set_ylabel('X(ω)')

# 시간 영역 플롯 (복원신호, N=5) - magnitude
axes[1,1].plot(t, x2)
axes[1,1].set_title(f'Time Domain Signal Magnitude (N={N2})')
axes[1,1].set_xlabel('t')
axes[1,1].set_ylabel('|x(t)|')

plt.tight_layout()
plt.show()
