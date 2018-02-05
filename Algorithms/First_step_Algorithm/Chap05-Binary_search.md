# Chap05 - 이진 탐색법 Binary Search

## 1. 이진 탐색법 개념 이해하기

이진 탐색법은 탐색의 대상인 데이터가 미리 오름차순 또는 내림차순으로 정렬되어 있는 경우에 사용할 수 있는 알고리즘이다. <br />

예를 들어, 7개의 칸으로 나누어진 상자(0 ~ 6이라고 적힌)가 있다고 하자. 그리고 이 상자 안에는 각각 숫자가 적힌 공이 하나씩 들어 있다. 단, 이 상자에 들어 있는 공은 오름차순으로 정렬되어 들어 있다.  <br />

이를 표로 나타내면 다음과 같다.

|  index   |  0   |  1   |  2   |  3   |  4   |  5   |  6   |
| :------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 숫자가 적힌 공 |  11  |  13  |  17  |  19  |  23  |  29  |  31  |

이제 *17 이라고 적힌 공* 을 이진 탐색법으로 찾아보자. 이진 탐색법은 탐색하는 *범위를 반으로 나누어* 즉, 둘로 나누어 절반씩 좁혀 나가면서 탐색을 진행하는 알고리즘이다. 

1. **가운데에 있는 공의 숫자를 살펴본다.**
   - 위의 표에서 가운데에 있는 공은 index가 3인 19라고 적힌 공이다. 
   - 위의 표처럼 공의 숫자는 오름차순으로 정렬 되어 있기 때문에 17 이 적힌 공은 19 라고 적힌 공 보다 앞쪽 즉, 왼쪽에 위치하게 된다.
2. **다시 한 번 가운데 공의 숫자를 살펴본다.**
   - 17 은 19보다 앞쪽에 있는 것을 알고 있기 때문에 19의 왼쪽에서 위 1번과 같은 과정을 거치면 된다.
   - 아래의 표와 같이 19의 왼쪽에서 가운데 공을 살펴보면 13 이라고 적힌 공이다.
   - 그렇게 되면, 17은 13보다 크므로 오른쪽에 위치하게 된다.
   - 따라서, 우리가 찾고자 하는 17이 적힌 공을 찾게 되므로 탐색은 종료된다.
|  index   |  0   |  1   |  2   |   3    |   4    |   5    |   6    |
| :------: | :--: | :--: | :--: | :----: | :----: | :----: | :----: |
| 숫자가 적힌 공 |  11  |  13  |  17  | ~~19~~ | ~~23~~ | ~~29~~ | ~~31~~ |



## 2. 이진 탐색법의 알고리즘

이진 탐색법은 크게 다음과 같은 처리로 구성된다.

- 가운데 요소를 선택하는 처리
- 가운데 데이터와 원하는 데이터를 비교하는 처리
- 탐색 범위를 절반으로 좁히는 처리

그럼, 이진 탐색법에 필요한 처리를 하나씩 살펴 보도록 하자.



### 1) 가운데 요소를 선택하는 처리

배열에서 가운데 요소를 알아내는 방법은 그 배열의 **index** 를 이용하면 된다. 위의 표에서 상자를 배열로 본다면 이 배열의 맨 앞 index는 0이고, 맨 마지막 index는 6이다. <br />

맨 앞의 index를 `head` 라는 변수로 정의하고, 맨 마지막 index를 `tail`이라는 변수에 정의한다. 그런 다음 가운데 index를 구하는 식은 다음과 같다.
$$
center = \frac{head + tail}{2}
$$
여기서 가운데 index는 `center = (head + tail)/2 = 3` 이 된다. <br />

만약 array의 개수가 6, 8 개와 같이 짝수개이면 어떻게 해야 할까? array 개수가 8개인 경우 가운데 index는 $center = \frac{0 + 7}{2} = 3.5$ 가 되어버린다. 이럴 경우에는 소수를 정수로 바꿔주는 *반올림, 올림, 버림*  을 이용하여 해결할 수 있다.



### 2) 가운데 요소와 원하는 데이터 비교하기

위의 1)에서 가운데 요소를 알아냈으니, 이제 이 요소가 우리가 찾고자 하는 데이터와 일치하는지를 확인해야 한다.

- 가운데 데이터와 원하는 데이터가 일치하는 경우
  - 이럴 경우에는 원하는 데이터를 찾았으므로 탐색이 종료된다.
- 가운데 데이터와 원하는 데이터가 일치하지 않는 경우
  - 일치하지 않는 경우는 찾고자 하는 데이터가 가운데 요소보다 작거나, 크거나 두 가지 경우 중 하나이다. 
  - 따라서, 탐색 범위를 절반으로 좁히는 처리로 이동한다.



### 3) 탐색 범위를 절반으로 좁히기

탐색 범위를 절반으로 좁히기 위한 방법은 두 가지 경우가 있다.

- 원하는 데이터가 가운데 데이터보다 큰 경우
  - 원하는 데이터가 큰 경우는 가운데 데이터의 오른쪽 부분을 절반으로 좁힌다. 
  - `head = center + 1` 이 된다. 즉, 아래와 같이 `head`와 `tail`의 값이 변한다. 

|         |  처음  |     다음     |
| :-----: | :--: | :--------: |
| head의 값 |  0   | center + 1 |
| tail의 값 |  6   | 6(변하지 않음)  |

- 원하는 데이터가 가운데 데이터 보다 작은경우
  - 원하는 데이터가 큰 경우는 가운데 데이터의 왼쪽 부분을 절반으로 좁힌다. 

|         |  처음  |     다음     |
| :-----: | :--: | :--------: |
| head의 값 |  0   | 0 (변하지 않음) |
| tail의 값 |  6   | center - 1 |

## 4. 이진 탐색법 알고리즘을 순서도로 나타내기

위에서 공부한 이진 탐색 알고리즘을 순서도(Flow chart)로 나타내 보자.

```flow
st=>start: Start
op1=>operation: head = 0, tail = len(array) - 1
cond1=>condition: head <= tail
op2=>operation: print 'Not Found'
op3=>operation: center = (head + tail) / 2
cond2=>condition: array[center] = target
op4=>operation: print 'center번째의 요소가 일치'
cond3=>condition: array[center] < target
op5=>operation: tail = center - 1
op6=>operation: head = center + 1
e=>end

st->op1->cond1->op3->cond2->cond3->op5->e
cond1(no)->op2->e
cond1(yes)->op3
cond2(no)->cond3
cond2(yes, right)->op4->e
cond3(yes)->op6->cond1
cond3(no)->op5->cond1
```



## 5. 이진 탐색 알고리즘을 의사코드로 나타내기

이번에는 이진 탐색 알고리즘을 의사코드로 나타내 보자.

```
input: array

let head = 0
let tail = len(array) - 1

while tail - head >= 0:
	center = (head + tail) / 2
	if array[center] == target:
		print 'center 번째의 요소가 일치' then finish
	else if array[center] < target:
		head = center + 1
	else if array[center] > target:
		tail = center - 1

if Not Found:
	print 'Not Found'
```



## 6. 이진 탐색 알고리즘을 파이썬 코드로 나타내기

파이썬을 이용하여 이진 탐색 알고리즘을 나타내 보자.

```python
def binary_search(array, target):
    '''Binary Search Algorithm
        - input : list of numbers
        - return :
            if exist return list(target) index
            if doesn't exist return Not Found'''
    array.sort()
    head = 0
    tail = len(array) - 1
    
    while tail - head >= 0:
        center = int((head + tail) / 2)
        if array[center] == target:
            return center
        elif array[center] < target:
            head = center + 1
        elif array[center] > target:
            tail = center - 1
    return 'Not Found'
```

```python
array = [13, 11, 19, 17, 29, 31, 23]
idx = binary_search(array, 17)
print('{} 번째의 index와 일치'.format(idx))
'''
>>> 2 번째의 index와 일치
'''
```

