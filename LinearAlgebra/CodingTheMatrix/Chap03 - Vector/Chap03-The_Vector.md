# Chap03

# 벡터 - Vector

## 3.1 벡터란 무엇인가?

*벡터*란 단어는 "vehere(운반하다)"라는 뜻의 라틴어에서 유래되었다. 어떤 것을 한 장소에서 다른 곳으로 이동하는 벡터의 **방향성**을 내포하고 있다. 
한 벡터의 모든 원소는 *하나의 필드* ([Chap02](https://github.com/ExcelsiorCJH/Study/blob/master/LinearAlgebra/CodingTheMatrix/Chap02%20-%20The%20Field/Chap02-The_Field.ipynb) 참고)에서 나와야 한다.

- **Definition 1** : 필드 $F$와 양의 정수 $n$에 대해, $F$에 속하는 $n$개의 원소를 가지는 벡터를 $F$상의 $n$-*벡터*라고 한다. $F$상의 $n$-벡터들의 집합은 $F^{n}$으로 나타낸다. <br />예를 들어, 아래의 $\mathbb{R}$(실수) 상의 4-벡터들의 집합을 $\mathbb{R}^{4}$라고 쓴다.
  $$[3.14, 2.17, -1.0, 2.0]$$

위의 4-벡터 집합을 함수로 생각하면  $\mathbb{R}^{4}$ 를 함수의 집합에 대한 표기법으로 해석할 수 있다. 따라서, 위의 4-벡터는 사실상 함수라고 할 수 있다.

$$ 0\mapsto 3.14\\ 1\mapsto 2.17\\ 2\mapsto -1.0\\ 3\mapsto 2.0$$



## 3.2 벡터는 함수이다.

위의 예제를 통해 알 수 있듯이 벡터는 함수로 나타낼 수 있다.

- **Definition 2** : 유한 집합 $D$와 필드 $F$에 대해, $F$상의 $D$-*벡터*는 $D$에서 $F$로의 함수이다.

### 3.2.1 파이썬의 딕셔너리를 이용한 벡터 표현

파이썬의 딕셔너리(Dictionary) 타입은 정의역(Domain) $\mapsto$ 치역(Image)의 형태로 벡터를 표현하는 데 유용하다. 위의 예제를 딕셔너리를 이용하면 아래와 같이 쓸 수 있다.

`{0: 3.14, 1: 2.17, 2: -1.0, 3: 2.0}`

### 3.2.2 Sparsity

대부분의 원소값이 $0$인 벡터를 ***Sparse vector***(희소 벡터)라고 한다. $0$이 아닌 언소의 수가 $k$개인 벡터는 $k$-*sparse* 라고 한다. $k$-*sparse* 벡터는 $k$에 비례하는 공간을 사용하여 표현할 수 있다. 예를 들어 여러 문서로 구성된 단어들의 모음을 $f: Words \mapsto \mathbb{R}$ 을 벡터로 나타내려고 하면 필요한 공간은 모든 문서를 구성하는 총 단어의 수에 비례한다.

## 3.3 벡터로 무엇을 표현할 수 있는가?

다양한 데이터들에 대해 벡터로 나타낼 수 있다. 

1. **이진 문자열(binary string)** : $n$-비트 이진 문자열 `10111011`을 $GF(2)$상의 $n$-벡터, `[1, 0, 1, 1, 1, 0, 1, 1]`로 표현할 수 있다. 

2. **속성(attribute)** : 예를 들어, 소비자에 관한 데이터를 딕셔너리 형태의 벡터로 표현할 수 있다. 이러한 벡터를 이용하여 머신러닝 모델에 적용할 수 있다.

   ```python
   Jane = {'age': 30, 'education_level': 16, 'income': 85000}
   ```

3. **확률분포** : 아래와 같이 유한한 확률 분포는 벡터로 나타낼 수 있다.

   ```python
   {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}
   ```

4. **이미지** : 예를 들어, `1024 x 768` 크기의 흑백 이미지는 집합 $\left\{ { (i,j) }|{ 0\le i<1024,0\le j<768 } \right\} $ 에서 실수 $\mathbb{R}$로의 함수로 볼 수 있고, 벡터로 불 수 있다. 

5. **공간상의 점**: 벡터를 이용하여 2차원 뿐만아니라 3차원 이상의 다차원의 공간의 점을 나타낼 수 있다.

   ```python
   # 2차원 공간상의 점
   import numpy as np
   import plotly.offline as offline 
   import plotly.graph_objs as go

   # jupyter notebook 에서 출력 
   offline.init_notebook_mode(connected=True)

   L = np.array([[2,2],[3,2],[1.75,1],[2,1],[2.25,1],[2.5,1],[2.75,1],[3,1],[3.25,1]])
   x = L[:, 0]
   y = L[:, 1]

   def plot(x, y):
       '''plotly를 이용해 plotting 함수 구현'''
       trace = go.Scatter(
                   x = x,
                   y = y,
                   mode = 'markers')

       layout = go.Layout(
           showlegend=False,
           xaxis=dict(
               rangemode='tozero',
               autorange=False
           ),
           yaxis=dict(
               rangemode='tozero',
               autorange=True
           )
       )

       data = [trace]
       fig = go.Figure(data=data, layout=layout)
       return offline.iplot(fig)

   plot(x, y)
   ```

   ```python
   # 3차원 공간상의 점
   x, y, z = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 10).transpose()

   trace1 = go.Scatter3d(
       x=x,
       y=y,
       z=z,
       mode='markers',
       marker=dict(
           size=12,
           line=dict(
               color='rgba(217, 217, 217, 0.14)',
               width=0.5
           ),
           opacity=0.8
       )
   )

   x2, y2, z2 = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 10).transpose()
   trace2 = go.Scatter3d(
       x=x2,
       y=y2,
       z=z2,
       mode='markers',
       marker=dict(
           color='rgb(127, 127, 127)',
           size=12,
           symbol='circle',
           line=dict(
               color='rgb(204, 204, 204)',
               width=1
           ),
           opacity=0.9
       )
   )
   data = [trace1, trace2]
   layout = go.Layout(
       margin=dict(
           l=0,
           r=0,
           b=0,
           t=0
       )
   )
   fig = go.Figure(data=data, layout=layout)

   offline.iplot(fig)
   ```



## 3.4 벡터 덧셈

### 3.4.1 평행이동과 벡터 덧셈

벡터의 평행이동은 벡터($v$)에 더하는 함수 $f(v)=v_{0} + v$ 에 의해 평행이동을 할 수 있다.

- **Definition 3** : $n$-벡터들의 덧셈은 대응하는 원소들의 덧셈으로 정의된다.

  $$[u_1, u_2, ..., u_n] + [v_1, v_2, ...,v_n] = [u_1+v_1, u_2+v_2,...,u_n+v_n]$$

모든 필드 $F$ ($\mathbb{R}, \mathbb{C}$ 등)는 $0$을 원소로 가진다. 그렇기 때문에 $F$상의 $D$-벡터들로 구성된 집합 $F^{D}$는 반드시 영벡터를 가진다. **영벡터**는 모든 원소의 값이 $0$인 벡터를 말하며 $\mathbb{0}$ 으로 표기한다. <br />따라서, 함수 $f(v) = v + 0$에 의한 평행이동은 그 결과가 입력과 동일한 평행이동이다.

#### Task 3.4.3

[1, 2]를 아래의 리스트 `L`의 각각의 벡터에 더하여 얻어진 점들을 그래프로 그려보자. 

```python
# Task 3.4.3
L = [[2,2],[3,2],[1.75,1],[2,1],[2.25,1],[2.5,1],[2.75,1],[3,1],[3.25,1]]
L = np.array(L)
L_add = L + [1, 2]
x = L_add[:, 0]
y = L_add[:, 1]
# plot(x, y)
```

### 3.4.2 벡터 덧셈의 결합성과 교환성 

필드(체)에서 덧셈의 두 가지 성질은 *결합성(associativity)*과 *교환성(commutativity)*이다. 

- **Proposition** : 임의의 벡터 $u, v, w$에 대해 다음의 성질이 성립한다.

$$(u+v)+w=u+(v+w)\\ u+v=vu$$

### 3.4.3 벡터를 화살표로 표현하기

필드$\mathbb{R}$상의 $n$-벡터들은 $\mathbb{R}^{n}$의 화살표로 나타낼 수 있다. 예를 들어, $2$-벡터 $[3, 1.5]$는 꼬리가 원점에 있고 화살표가 $(3, 1.5)$에 있는 화살표로 나타낼 수 있다.

```python
%matplotlib inline
import matplotlib.pyplot as plt

ax = plt.axes()
ax.arrow(0, 0, 3.0, 1.5, head_width=0.1, head_length=0.1)
plt.ylim([0, 10])
plt.xlim([0, 10])
```

또한 $\mathbb{R}$상의 벡터들의 덧셈을 화살표를 사용하여 보여줄 수 있다. 

![](./images/vector_add.PNG)

## 3.5 스칼라 - 벡터 곱셈

[Chap02-필드](https://render.githubusercontent.com/view/ipynb?commit=e5e626bfc6259c98589c42b21e7c0207ce918d45&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f457863656c73696f72434a482f53747564792f653565363236626663363235396339383538396334326232316537633032303763653931386434352f4c696e656172416c67656272612f436f64696e675468654d61747269782f4368617030322532302d2532305468652532304669656c642f4368617030322d5468655f4669656c642e6970796e62&nwo=ExcelsiorCJH%2FStudy&path=LinearAlgebra%2FCodingTheMatrix%2FChap02+-+The+Field%2FChap02-The_Field.ipynb&repository_id=116745719&repository_type=Repository#2.4.3-%EC%96%91%EC%9D%98-%EC%8B%A4%EC%88%98%EB%A1%9C-%EB%B3%B5%EC%86%8C%EC%88%98-%EA%B3%B1%ED%95%98%EA%B8%B0)에서 *스케일링(Scaling)*은 복소평면에서 입력된 복소수를 양의 실수 $r$과 곱하는 함수 $f(z)=r\cdot z$로 나타낼 수 있었다. 이처럼 벡터에 대해서도 스칼라-벡터 곱(scalar-vector multiplication)에 의해 벡터를 스케일링 할 수 있다. 벡터에서 필드 원소(e.g. 숫자)는 *스칼라(scalar)*라 하며, 그 이유는 곱셈을 통해 벡터를 스케일링 하는데 사용할 수 있기 때문이다.

-  **Definition 4** : 벡터 $v$와 스칼라 $\alpha$의 곱셈은 $v$의 원소 각각을 $\alpha$와 곱하는 것으로 정의된다.

  $$\alpha [v_1,v_2,...,v_n] = [\alpha v_1, \alpha v_2,..., \alpha v_n]$$

#### Task 3.5.4

`L`내의 벡터들을 $0.5$만큼 스케일링한 결과와 $-0.5$만큼 스케일링한 결과를 그래프로 그려보자.

```python
L = [[2,2],[3,2],[1.75,1],[2,1],[2.25,1],[2.5,1],[2.75,1],[3,1],[3.25,1]]
L = np.array(L)

L1 = L * 0.5
L2 = L * (-0.5)

trace1 = go.Scatter(x=L1[:, 0],
                    y=L1[:, 1],
                    mode = 'markers')

trace2 = go.Scatter(x=L2[:, 0],
                    y=L2[:, 1],
                    mode = 'markers')

layout = go.Layout(
        showlegend=False,
        xaxis=dict(
            rangemode='tozero',
            autorange=True
        ),
        yaxis=dict(
            rangemode='negative',
            autorange=True
        )
    )

data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
# offline.iplot(fig)
```

### 3.5.1 화살표 스케일링하기

$\mathbb{R}$상의 벡터를 양의 실수로 스케일링 하는 것은 벡터의 방향을 바꾸지 않고 화살표의 길이만 변경한다. 아래의 예제 코드는 위의 $[3, 1.5]$의 벡터를 2배한 화살표이다. 음의 실수를 곱하게 되면 벡터의 방향이 반대가 된다.

```python
ax = plt.axes()
ax.arrow(0, 0, 3.0*2, 1.5*2, head_width=0.1, head_length=0.1)
plt.ylim([0, 10])
plt.xlim([0, 10])
```

### 3.5.2 스칼라-벡터 곱셈의 결합성

벡터를 스칼라와 곱한 다음에 그 결과를 또 다른 스칼라와 곱하는 것은 아래와 같이 단순화 할 수 있다.

- **Proposition** (Associativity) : $\alpha (\beta v)=(\alpha \beta)v$ 

  ![](./images/proof01.PNG)

### 3.5.3 원점을 지나는 선분

하나의 벡터와 스칼라 곱을 통해 스케일링하여 원점을 지나는 선분을 만들 수 있다. 아래의 예제는 벡터 $[3, 2]$를 스케일링하여 선분을 만드는 예시이다.

```python
# [3, 2] 벡터를 10등분으로 스케일링
vecs = [[3 * (i/10), 2 * (i/10)] for i in range(11)]
vecs = np.array(vecs)
x = vecs[:, 0]
y = vecs[:, 1]
plot(x, y)
```

```python
# [3, 2] 벡터를 100등분으로 스케일링
vecs = [[3 * (i/100), 2 * (i/100)] for i in range(101)]
vecs = np.array(vecs)
x = vecs[:, 0]
y = vecs[:, 1]
plot(x, y)
```

### 3.5.4 원점을 지나는 직선

위의 예제에서 선분을 확장하여 양수의 스칼라와 음수의 스칼라를 곱하여 스케일링 하게 되면 원점을 지나는 직선을 만들 수 있다.

```python
vecs = [[3 * (i/10), 2 * (i/10)] for i in range(-10, 11)]
vecs = np.array(vecs)
x = vecs[:, 0]
y = vecs[:, 1]
plot(x, y)
```

```python
vecs = [[3 * (i/100), 2 * (i/100)] for i in range(-100, 101)]
vecs = np.array(vecs)
x = vecs[:, 0]
y = vecs[:, 1]
plot(x, y)
```

## 3.6 벡터 덧셈과 스칼라 곱셈 결합하기

### 3.6.1 원점을 지나지 않는 선분과 직선

위의 예제에서 $[x, y] \mapsto [x+0.5,y+1]$평행이동을 적용하게 되면 아래의 그림처럼 그래프가 그려진다.

```python
vecs = [[3 * (i/100), 2 * (i/100)] for i in range(101)]
vecs = np.array(vecs)
vecs_trns = [[3 * (i/100) + 0.5, 2 * (i/100) + 1] for i in range(101)]
vecs_trns = np.array(vecs_trns)

trace1 = go.Scatter(x=vecs[:, 0],
                    y=vecs[:, 1],
                    mode = 'markers',
                    name = 'original')

trace2 = go.Scatter(x=vecs_trns[:, 0],
                    y=vecs_trns[:, 1],
                    mode = 'markers',
                    name = 'translation')

layout = go.Layout(
        showlegend=False,
        xaxis=dict(
            rangemode='tozero',
            autorange=True
        ),
        yaxis=dict(
            rangemode='negative',
            autorange=True
        ),
        annotations=[
            dict(
                x=3,
                y=2,
                xref='x',
                yref='y',
                text='Orignial',
                showarrow=True,
                arrowhead=7
            ),
            dict(
                x=3.5,
                y=3,
                xref='x',
                yref='y',
                text='Translation',
                showarrow=True,
                arrowhead=7
            ),
        ]
    )

data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig)
```

### 3.6.2 스칼라-벡터 곱셈과 벡터 덧셈의 분배 법칙

아래의 성질은 필드에 대한 분배법칙 $x(y+z)=xy+xz$ 에서 비롯된다.

- **Proposition** (*벡터 덧셈에 대한 스칼라-벡터 곱의 분배* ):  $\alpha (u+v)=\alpha u + \alpha v$  
- **Proposition** (*스칼라 덧셈에 대한 스칼라-벡터 곱의 분배* ): $(\alpha + \beta)u=\alpha u + \beta u$

![](./images/proof2.PNG)

### 3.6.3 블록결합(Convex combination) 들여다 보기

$[0, 5, 1]$ 와 $[3.5,3]$을 잇는 선분을 이루는 점들의 집합에 대한 표현식은 $\left\{ \alpha [3, 2]+[0.5, 1] : \alpha \in \mathbb{R}, 0\le \alpha \le  \right\} $ 이다. 이를 다음과 같이 더 나은 식으로 표현할 수 있다. 

$$\begin{eqnarray} \alpha [3,2]+[0.5,1] & = & \alpha ([3.5,3]-[0.5,1)+[0.5,1] \\  & = & \alpha [3.5,3]-\alpha [0.5,1]+[0.5,1] \\  & = & \alpha [3.5,3]+(1-\alpha )[0.5,1] \\  & = & \alpha [3.5,3]+\beta [0.5,1] \end{eqnarray}$$ 

$$\therefore \quad \left\{ \alpha [3.5,3]+\beta [0.5,1]:\alpha ,\beta \in \mathbb{ R },\alpha ,\beta \ge 0, \alpha + \beta = 1 \right\}$$

$\alpha u + \beta v$형태의 표현식은 $u$와 $v$의 블록결합이라고 한다. 위의 예를 통해 임의의 $\mathbb{R}$상의 $n$-벡터들의 쌍 $u, v$에 대해 아래와 같이 말할 수 있다.

- **Proposition** : $u$-$v$ 선분은 $u$와 $v$의 블록결합들의 집합으로 구성된다. 

#### Task 3.6.9

파이썬 함수, `segment(pt1, pt2)`를 작성해 보자. `pt1=[3.5,3], pt2=[0.5,1]`일 경우, 리턴 결과인 100개의 점을 그래프로 그려보자

```python
def segment(pt1, pt2):
    pt1 = [[pt1[0] * i/100, pt1[1] * i/100] for i in range(101)]
    pt2 = [[pt2[0] * (1-(i/100)), pt2[1] * (1-(i/100))] for i in range(101)]
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    result = pt1 + pt2
    x = result[:, 0]
    y = result[:, 1]
    return x, y
```

```python
pt1 = [3.5, 3]
pt2 = [0.5, 1]

x, y = segment(pt1, pt2)
plot(x, y, autorange=False)
```

#### Example 3.6.10 

이미지를 나타내는 벡터들의 쌍에 대한 블록결합을 고려해 보자. 이미지 예로는 설현의 이미지를 이용하였다.



