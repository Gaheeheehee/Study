# Chap 07

# 차원 - Dimension

## 7.1 기저의 크기

### 7.1.1 Morphing 보조 정리와 그 응용

***Lemma (Morphing Lemma)*** : $\mathcal{V}$ 는 벡터공간이라고 하자. $S$ 는 $\mathcal{V}$ 에 대한 생성자들의 집합이라 하고, $B$ 는 $\mathcal{V}$ 에 속하는 벡터들로 구성된 선형독립인 집합(즉, 기저)이라고 하면, $|S| \ge |B|$ 이다.  <br />

***Theorem (Basis Theorem)*** : $\mathcal{V}$ 는 벡터공간이라 하고, $\mathcal{V}$ 에 대한 모든 기저(basis)는 동일한 크기를 가진다.

- **Proof** : $B_1$ 과 $B_2$ 는 $\mathcal{V}$ 에 대한 두 기저라고 하자. $S=B_1$ 과 $B=B_2$ 를 위의 *Morphing Lemma* 에 적용하면 $|B_1| \ge |B_2|$ 라고 할 수 있다. $S = B_2$ 와 $B=B_1$ 을 적용하면 $|B_2| \ge |B_1|$ 이다. 이 둘의 부등식을 결합하면 $|B_1| = |B_2|$ 를 얻을 수 있다. 


***Theorem*** : $\mathcal{V}$ 는 벡터공긴이라고 하면, $\mathcal{V}$ 에 대한 생성자들의 집합이 $\mathcal{V}$ 에 대한 생성자들로 구성된 *가장 작은 집합* 이 되는 필요충분 조건은 이 집합이 $\mathcal{V}$ 에 대한 기저인 것이다.

- **Proof** : $T$ 는 $\mathcal{V}$ 에 대한 생성자들의 집합이라고 하자. 그렇다면, 증명해야 하는 것은 
  - (1) 만약 $T$ 가 $\mathcal{V}$ 에 대한 기저이면 $T$ 는 $\mathcal{V}$ 에 대한 생성자들로 구성된 가장 작은 집합이다.
  - (2) 만약 $T$ 가 $\mathcal{V}$ 에 대한 기저가 아니면 생성자들로 구성된 $T$ 보다 더 작은 집합이 존재한다.


1. $T$ 를 기저라고 하고, $S$ 는 $\mathcal{V}$ 에 대한 생성자들로 구성된 가장 작은 집합이라고 하자. 위의 *Morphing Lemma* 에 의하면, $|T| \le |S|$ 이고, 따라서 $T$ 또한 생성자들의 가장 작은 집합이다.
2. $T$ 는 기저가 아니라고 해보자. 기저는 *생성자들로 구성된 선형독립* 인 집합이다. 그러므로 $T$ 는 기저가 아니라 했으니, $T$ 는 생성자들로 구성된 선형종속인 집합이다.  [6.5.4의 Lemma](http://nbviewer.jupyter.org/github/ExcelsiorCJH/Study/blob/master/LinearAlgebra/CodingTheMatrix/Chap06%20-%20The%20Basis/Chap06-The_Basis.ipynb#6.5.4-일차독립-및-종속의-성질)에 따르면 $T$ 내에 다른 벡터들의 생성에 속하는 일부 벡터들이 있다. 그러므로 *[Superfluous-Vector Lemma](http://nbviewer.jupyter.org/github/ExcelsiorCJH/Study/blob/master/LinearAlgebra/CodingTheMatrix/Chap06%20-%20The%20Basis/Chap06-The_Basis.ipynb#6.5.1-Superfluous-Vector-보조정리)* 에 의해, $T$ 에서 제거하면 $\mathcal{V}$ 에 대한 생성자들의 집합이 되는 일부 벡터가 존재한다. 따라서 $T$ 는 생성자들로 구성된 가장 작은 집합이 아니다.




*7.1.2 생략*

## 7.2 차원과 랭크 - Dimension and Rank

### 7.2.1 정의 및 예제

***Definition*** : 벡터공간의 *차원* 은 그 벡터공간에 대한 기저의 크기로 정의한다.  벡터공간 $\mathcal{V}$ 의 차원은 $\dim \mathcal{V}$ 로 표현한다. <br />

- ***Example 7.2.2*** : $\mathbb{R}^3$ 에 대한 하나의 기저는 표준 기저 $\{[1,0,0],[0,1,0],[0,0,1]\}$ 이다. 그러므로 $\mathbb{R}^3$ 의 차원은 기저의 크기인 3, 즉 $\dim \mathcal{V}=3$ 이다. 
- ***Example 7.2.3*** : 좀 더 일반적으로, 임의의 필드 $F$ 와 유한집합 $D$ 에 대해, $F^D$ 에 대한 하나의 기저는 표준기저이고 이것은 $|D|$ 벡터들로 구성되므로, $F^D$의 차원은 $|D|$ 이다.


<br />

***Definition*** : 벡터들의 집합 $S$ 의 랭크(rank)를 Span $S$ 의 차원이라 정의한다. $S$ 의 랭크는 rank $S$ 로 나타낸다.

- ***Example 7.2.6*** : 벡터 $[1, 0, 0], [0,2,0],[2,4,0]$ 은 선형종속이다. 그러므로 이 벡터들의 랭크는 $3$보다 작다. 이들 중 임의의 두 벡터는 세 벡터들의 Span에 대한 기저를 형성한다. 따라서 랭크는 $2$ 이다.


<br />

***Proposition*** : 벡터들로 구성된 임의의 집합 $S$ 에 대해, rank $S \le |S|$  <br />

***Definition*** : 행렬 $M$에 대해, $M$ 의 *행랭크* 는 그 행렬의 행의 랭크이고, $M$ 의 *열랭크* 는 그 행렬의 열의 랭크이다. 즉, $M$의 행랭크는 Row $M$의 차원이고, $M$ 의 열랭크는 Col $M$의 차원이다.

- ***Example 7.2.10*** : 

$$
M=\begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 2 & 4 & 0 \end{bmatrix}
$$

- 이 행렬의