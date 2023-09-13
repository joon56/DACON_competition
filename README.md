# Satellite Image Building Segmentation

DACON에서 주최하는 satellite image building segmentation contest에 참여한 팀의 작업 내용입니다.

## 팀 구성 및 역할

| 팀원   | 역할            |
|--------|-----------------|
| 이동우 | 팀 총괄         |
| 정유라 | 모델 개발       |
| 지성현 | 데이터 전처리   |
| 최훈   | 데이터 전처리   |
| 유민준 | 데이터 전처리 + 학습 |

## 작업 내용

- [데이터 다운로드](https://dacon.io/competitions/official/236092/overview/description)

### 1. 라이브러리 임포트

우리 팀은 다양한 라이브러리를 사용하여 작업을 진행했습니다. 아래는 주요 라이브러리들의 일부입니다:

```python
import glob
import os
import cv2
import pandas as pd
import numpy as np
import gc
from skimage.color import lab2lch, rgb2lab
from skimage.exposure import rescale_intensity
from skimage.morphology import disk
from sklearn.cluster import KMeans
```

### 2. RLE 디코딩 함수 (`rle_decode`)

Run Length Encoding (RLE) 형식의 마스크를 디코딩하여 이미지 형식으로 변환하는 함수를 구현했습니다. 이 함수는 주어진 RLE 문자열을 해독하여 해당 마스크의 2차원 배열을 반환합니다.

```python
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
```

### 3. 이미지 분할 함수 (`cut_image`)

주어진 이미지를 작은 서브 이미지로 분할하는 함수를 구현했습니다. 이 함수는 큰 이미지를 여러 개의 작은 이미지로 분할하여 리스트로 반환합니다.

```python
def cut_image(image, subimage_size=256):
    subimages = []
    for i in range(0, image.shape[0], subimage_size):
        for j in range(0, image.shape[1], subimage_size):
            subimage = image[i:i+subimage_size, j:j+subimage_size]
            subimages.append(subimage)
    return subimages
```

## 대회 피드백

> 여기에는 대회에 대한 피드백을 작성하면 됩니다.
