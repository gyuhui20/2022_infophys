# vgg19_bn / transfer learning

#필요한 라이브러리 임포트
import numpy as np
np.set_printoptions(suppress=True, precision=4) #넘파이 부동소수점 자릿수 표시
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False #마이너스 기호 정상 출력
import torch
from torch import tensor
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from IPython.display import display
import warnings
warnings.simplefilter('ignore')

#공통함수 불러오기

#데이터 다운로드
#깃허브 레포지토리를 갖고 있는 경우 사진을 어떻게 다운로드해?

#데이터 압축 해제
