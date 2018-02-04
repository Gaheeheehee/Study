# Chap04 - 선형 탐색법(리니어 서치)

## 1. 선형 탐색법 개념 이해하기

예를 들어 아래의 표와 같이  5개의 칸으로 나누어진 상자(0~4 라고 적힌)가 있고, 숫자가 적힌 공이 상자에 하나씩 들어있다고 해보자. <br />

이 상자들 중 *5라고 적힌* *공* 을 선형 탐색법을 이용하여 찾으려고 한다. 방법은 매우 간단하다! 왼쪽에서 부터 오른쪽( 상자 0 $\rightarrow$ 4 방향) 방향으로 순서대로 하나씩 확인해 가면 된다.    

|  index   |  0   |  1   |  2   |  3   |  4   |
| :------: | :--: | :--: | :--: | :--: | :--: |
| 숫자가 적힌 공 |  4   |  2   |  3   |  5   |  1   |

이렇듯 선형 탐색법은 매우 간단 하지만 단점이 있다. 만약 찾는 공이 앞쪽에 있으면 (위의 예시의 경우 4라고 적힌 공) 짧은 시간에 탐색할 수 있지만, 만약 찾고자 하는 공이 뒤에 있거나 없는 경우 탐색하는데 시간이 오래 걸리게 된다. 따라서, 선형 탐색법은 이해하기에는 쉬운 알고리즘이지만, 효율은 좋지 못하다.



## 2. 선형 탐색법 알고리즘

### 2.1 순서도(Flow Chart) 로 작성하기

위에서 선형 탐색법의 개념을 이해 했으니, 이제 순서도(flow chart)를 이용해서 선형탐색법을 나타내 보자.

```flow
st=>start: Start
op1=>operation: i = 0
cond1=>condition: array[i]=5
op2=>operation: print 'i번째 요소 일치'
op3=>operation: i = i + 1
cond2=>condition: i < 5
op4=>operation: print 'Not Found'
e=>end

st->op1->cond1->op3->cond2
cond1(no)->op3
cond1(yes, right)->op2->e
cond2(yes, right)->cond1
cond2(no)->op4->e
```



### 2.2 의사코드로 나타내기(Pseudocode)

이번에는 의사코드(pseudocode)를 이용해 선형 탐색법 알고리즘을 나타내 보자.

```
input: array[5] = [4, 2, 3, 5, 1]

let i = 0
for each array[i] in array:
	if array[i] is 5:
		print "i 번째의 요소가 일치" then finish
	i = i + 1
	else:
		print 'Nout Found'
```



### 2.3 선형 탐색 알고리즘을 파이썬으로 나타내기

순서도와 의사코드로 선형 탐색법 알고리즘을 나타냈으니, 이번에는 파이썬을 이용해서 실제로 동작하도록 구현 해보자. <br />

선형 탐색 알고리즘 구현은 `linear_search` 라는 함수로 구현하였다. 코드는 아래와 같다.

```python
def linear_search(array, target):
    '''Linear Search Algorithm
        - input : list of numbers
        - return: 
            if exist return list index
            if doesn't exist return Not Found '''
    for i, num in enumerate(array):
        if num is target:
            return i
    return 'Not Found'
```

```python
array = [4, 2, 3, 5, 1]

idx = linear_search(array, 5)

print('{} 번째의 index와 일치'.format(idx))
'''
>>> 3 번째의 index와 일치
'''
```

