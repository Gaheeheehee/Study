# Chapter 06

# 기저 - Basis

## 6.1 좌표계 - Coordinate system

### 6.1.1 데카르트의 생각

![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Frans_Hals_-_Portret_van_Ren%C3%A9_Descartes.jpg/300px-Frans_Hals_-_Portret_van_Ren%C3%A9_Descartes.jpg)

1618년 프랑스의 수학자 르네 데카르트(René Descartes)는 기하학을 접근하는 방식을 완전히 바꾼 개념을 발견하였다. 일화에 따르면 데카르트는 침대에 누워 방의 천장 모서리 주위를 날고 있는 파리를 보고 있다가 기하학에 대한 훌륭한 생각이 떠올랐다고 한다. *(역시 천재는 생각하는 자체가 다른듯...)* 데카르트는 파리의 위치는 두 개의 숫자, 즉 파리 근처 두 개의 벽으로부터 파리까지의 거리로 기술할 수 있다는 것을 깨달았고,  두 벽이 수직이 아니라도 이것이 사실이라는 것을 알게 되었다. 또한 데카르트는 기하학적 분석을 대수적으로 접근할 수 있음을 알게 되었다. *(엄청나다...)* 



### 6.1.2 좌표표현 - Coordinate representation

위에서 얘기한 파리의 위치를 특정하는 두 개의 숫자를 *좌표(Coordinates)* 라고 한다. 벡터공간 $\mathcal{V}$ 에 대한 *좌표계* 는 $\mathcal{V}$ 의 생성자([4.2.3 참고](http://nbviewer.jupyter.org/github/ExcelsiorCJH/Study/blob/master/LinearAlgebra/CodingTheMatrix/Chap04%20-%20The%20Vector%20Space/Chap04-The_Vector_Space.ipynb)) $a_1, ..., a_n$에 의해 명시된다. $\mathcal{V}$ 내의 모든 벡터 $v$는 아래와 같이 생성자의 선형결합으로 나타낼 수 있다.

$$v = \alpha_1 a_1 + \cdots + \alpha_n a_n$$

따라서, $v$는 계수들의 벡터 $[\alpha_1, ..., \alpha_n]$ 에 의해 나타낼 수 있다. 이러한 계수들을 *좌표* 라 하고 벡터 $[\alpha_1, \cdots , \alpha_n]$ 은 $a_1, ..., a_n$ 에 대한 $v$ 의 *좌표표현* 이라고 한다. <br />

하지만 점에 대한 좌표 할당만으로는 충분하지 않다. 각 점에 대한 좌표 할당은 정확하게 한 가지 방식으로 이루어 져야 한다. 이를 위해서는 생성자 $a_1, ..., a_n$를 잘 선택해야 한다. 이 부분은 [6.7.1 좌표표현의 존재와 유일성]()에서 설명할 것이다. <br />

***Example 6.1.1*** : 벡터 $v=[1, 3, 5, 3]$은 $1[1,1,0,0]+2[0,1,1,0]+3[0,0,1,1]$ 와 동일하다. 따라서 $v$의 벡터 $[1,1,0,0],[0,1,1,],[0,0,1,1]$ 에 대한 좌표표현은 $[1, 2, 3]$ 이다.

***Example 6.1.3*** : $GF(2)$ 상의 벡터에 대해 알아보자. 벡터 $[0,0,0,1]$ 이 벡터 $[1,1,0,1],[0,1,0,1],[1,1,0,0]$에 대한 좌표표현은 아래와 같다.

$$[0,0,0,1] = 1[1,1,0,1] + 0[0,1,0,1] + 1[1,1,0,0]$$

따라서, $[0,0,0,1]$의 좌표표현은 $[1,0,1]$ 이다.



### 6.1.3 좌표표현과 행렬-벡터 곱셈

좌표를 왜 벡터로 나타낼까? 좌표표현을 행렬-벡터 및 벡터-행렬 곱셈의 선형결합 정의의 관점에서 보도록 하자.  좌표축이 $a_1,...,a_n$ 이라고 하고, 이 좌표축을 열벡터로 보고, 행렬 $A$를 나타내면 $A =\begin{bmatrix}  &  &  \\ a_1 & \cdots & a_n \\  &  &  \end{bmatrix} $ 로 나타낼 수 있고, 이 행렬의 열들은 생성자를 나타낸다. 

- "$u$ 는 $a_1, ..., a_n$ 에 대한 $v$ 의 좌표표현이다." 라는 것을 행렬-벡터 방정식으로 다음과 같이 쓸 수 있다.

$$Au = v$$

- 그러므로, 좌표표현 $u$ 에서 표현할 벡터를 나타내려면 $A$와 $u$를 곱한다.
- 또한, 벡터 $v$ 에서 그 좌표표현을 얻으려면 행렬-벡터 방정식 $Ax = v$ 를 풀면 된다. $A$ 의 열들은 $\mathcal{V}$ 에 대한 생성자들이고 $v$ 는 $\mathcal{V}$에 속하므로 방정식은 적어도 하나의 해를 가져야 한다.



## 6.2 손실압축(Lossy compression) 들여다 보기

좌표표현의 한 가지 응용으로 손실압축에 대해 알아보자. 예를 들어 많은 수의 $2000 \times 1000$ 흑백이미지를 저장한다고 하면, 이러한  이미지는 $D$-벡터에 의해 표현될 수 있다. 여기서 $D=\{0,1,...,1999\} \times \{0,1,...,999\}$ 이다. 이러한 흑백이미지를 컴팩트하게 (compactly) 저장한다고 할 때 다음과 같은 3가지 방안을 생각해 볼 수 있다. 

### 6.2.1 Strategy 1: 벡터를 가장 가까운 스파스 벡터로 대체하기

벡터를 가장 가까운 $k$-스파스 벡터로 대체하는 것을 생각해보자. 이러한 압축 방법은 원래의 이미지 정보에 대한 손실이 있으므로 *손실압축* 이라고 한다. 아직까지는 벡터들 사이의 거리를 구하는 방법을 배우지 않았으므로 단순하게 이미지의 픽셀에서 값의 크기가 큰 $k$개의 원소를 제외한 나머지 원소를 모두 $0$ 으로 대체하여 압축할 수 있다. 아래의 예제는 매트릭스 영화의 한 장면을 $k$-Sparse로 압축한 예제이다. 

***Example 6.2.2*** : 

```python
# 이미지 파일 불러오기
img = Image.open('./images/img01.png')
img = img.convert('L')
img = np.asarray(img, dtype='float32')
print(img.shape)  # (256 x 512) 이미지 행렬

min_img_top10p = min(sorted(img.reshape(-1).tolist(), reverse=True)[:13108])  # = 92
# k-sparse 이미지 행렬 만들기
# 상위 10%를 제외한 나머지 값은 0으로 대체하기
img_sparse = [pix  if pix >= min_img_top10p else 0 for pix in img.reshape(-1).tolist()]
img_sparse = np.array(img_sparse)
# 원래의 이미지 행렬로 바꿔주기
img_sparse = img_sparse.reshape(256, -1)
print(img_sparse.shape)

fig, axs = plt.subplots(1, 2, figsize=(25, 5))
fig.subplots_adjust(hspace = .5, wspace=.5)

img_list = [img, img_sparse]
title_list = ['original', 'sparse']

for i, img in enumerate(img_list):
    axs[i].imshow(img ,cmap='Greys_r')
```

위의 결과 이미지는 많은 개수의 픽셀이 $0$ 으로 대체되기 때문에 원래의 이미지와 많이 다르다. 



### 6.2.2 Strategy 2: 이미지 벡터를 좌표표현으로 표현하기

또 다른 방법은 원래의 이미지에 fidelity를 없애는 것이다. 

- 이미지를 압축하려고 하기 전에 벡터들의 컬렉션 $a_1, ..., a_n$ 을 선택한다. 
- 다음에, 각 이미지 벡터에 대해 그 벡터의 $a_1, ..., a_n$에 대한 좌표표현 $u$를 찾아 그것을 저장한다. 
- 좌표표현으로부터 원래 이미지를 복원하기 위해 대응하는 선형결합을 계산한다. 

하지만 이 방법은 $2000 \times 1000$ 이미지 벡터가 $a_1, ..., a_n$ 의 선형결합으로 표현될 수 있어야 한다. 즉, $\mathbb{R}^D=Span \{a_1,...,a_n\}$ 이어야 한다.  따라서, 위를 만족하는 벡터들의 수는 적지 않아 압축을 하기에는 무리가 있다. (Example 6.2.3 참고)



### 6.2.3 Strategy 3: 하이브리드 방식

앞의 두 방안(Strategy1, 2) 좌표표현과 가장 가까운 $k$-스파스 벡터를 결합하는 방법이 있다. 

- *Step 1* : 벡터 $a_1,..., a_n$ 을 선택한다.
- *Step 2* : 압축하고자 하는 각 이미지에 대해, 대응하는 벡터 $v$ 를 정하고, $a_1,...,a_n$ 에 대한 좌표표현 $u$를 찾는다.
- *Step 3* : 다음에, $u$ 를 가장 가까운 $k$-스파스 벡터 $\tilde{u}$ 로 대체한다.
- *Step 4* : $\tilde{u}$ 로 부터 원래 이미지를 복원하기 위해 $a_1,...,a_n$ 의 대응하는 선형결합을 계산한다. 

*Step 1* 에서 벡터 $a_1,...,a_n$을 선택하는 방법은 11장에서 다룬다. 따라서, 11장 까지 배우고 난 다음 다시 풀도록 하겠다.. ㅜㅜ <br />

이 방법으로 압축하면 아래와 같은 결과를 얻을 수 있다고 한다.

![](./images/img02.png)



## 6.3 생성자 집합을 찾기 위한 두 개의 Greedy 알고리즘

이번 절에서는 아래의 물음의 답을 찾기 위한 두 개의 알고리즘을 고려해 본다.

*주어진 벡터공간 $\mathcal{V}$에 대해 $Span$ 이 $\mathcal{V}$ 와 동일하게 되는 최소 개수의 벡터들은 무엇인가?*

### 6.3.1 Grow 알고리즘

위의 질문에서 어떻게 최소 개수의 벡터들을 구할 수 있을까? 라는 질문에 생각할 수 있는 방법은 *Grow* 알고리즘과 *Shrink* 알고리즘이 있다. 먼저 Grow 알고리즘에 대해 알아보자. <br />

Grow 알고리즘은 특별한 알고리즘이 아니라 벡터를 추가하다가 더이상 추가할 벡터가 없을 때, 종료되는 알고리즘을 의미한다. (*소제목이 Grow 알고리즘이길래 대단한 알고리즘인 줄 알았다는...*) Grow 알고리즘을 의사코드(pseudocode)로 나타내면 다음과 같다.

```
def Grow(V)
	B = 0
	repeat while possible:
		find a vector v in V that is not in Span B, and put it in B
```

위에서 설명한 대로 이 알고리즘은 더이상 추가할 벡터가 없을 때, 즉, $B$ 가 $\mathcal{V}$ 의 $Span$ 일 때 종료된다. <br />

***Example 6.3.1*** : $\mathbb{R}^3$ 에 대한 생성자들의 집합을 선택하는 데 Grow 알고리즘을 사용해 보자. [4.2.5 표준생성자](http://nbviewer.jupyter.org/github/ExcelsiorCJH/Study/blob/master/LinearAlgebra/CodingTheMatrix/Chap04%20-%20The%20Vector%20Space/Chap04-The_Vector_Space.ipynb)에서 $\mathbb{R}^n$에 대한 표준 생성자에 대해 알아보았다. Grow 알고리즘을 사용하면 첫 번째 이터레이션(iteration)에서 집합 $B$에 $[1,0,0]$ 추가한다. 그런다음 $[0,1,0]$ 은 $Span \{[1,0,0]\}$에 포함되지 않으므로 $[0,1,0]$을 추가한다. 마지막으로 $[0,0,1]$을 $B$ 에 추가한다. 임의의 벡터 $v=[\alpha_1, \alpha_2, \alpha_3] \in \mathbb{R}^3$ 은 아래와 같이  선형결합으로 나타낼 수 있으므로 $Span (e_1, e_2, e_3)$ 내에 있다. 

$$v = \alpha_1 e_1 + \alpha_2 e_2 + \alpha_3 e_3$$

그러므로 $B$ 에 추가할 벡터 $v \in \mathbb{R}^3$ 는 더이상 없기 때문에 Grow 알고리즘은 종료된다.



###6.3.2 Shrink 알고리즘

이번에는 Grow 알고리즘과는 반대라고 할 수 있는 Shrink 알고리즘에 대해 알아보자. 

```
def Shrink(V)
	B = some finite set of vectors that span V
	repeat while possible:
		find a vector v in B such taht Span (B - {v}) = V, and remove v from B
```

위의 의사코드에서 알 수 있듯이 Shrink 알고리즘은 Span 집합에서 더이상 제거할 벡터가 없을 때 종료된다. 아래의 예제를 보자. <br />

***Example 6.3.2*** : 처음 집합 $B$는 아래와 같은 벡터들로 구성되어있다고 하자.

$$\begin{matrix} v_1 & = & [1,0,0] \\ v_2 & = & [0,1,0] \\ v_3 & = & [1,2,0] \\ v_4 & = & [3,1,0] \end{matrix}$$

$v_4 = 3v_1 + v_2$ 이므로, 첫 번째 이터레이션에 $B$ 에서 $v_4$를 제거한다. 두번 째 이터레이션에서 $v_3 = v_1 + 2v_2$ 이므로 $B$ 에서 $v_3$를 제거한다.  따라서 $B = \{v_1, v_2\}$ 가 되고 Span$B = \mathbb{R}^2$ 이 되며, 알고리즘은 종료된다. 



### 6.3.3 Greedy 알고리즘이 실패하는 경우
