# Chapter01 - 파이썬 데이터 모델

데이터 모델은 일종의 프레임워크로서, *파이썬을 설명하는 것* 이라고 할 수 있으며, 시퀀스(sequences), 반복자(iterators), 함수(functions), 클래스(class), 콘텍스트 관리자 등 언어 자체의 구성단위에 대한 인터페이스를 공식적으로 정의한다. <br />파이썬은 *특별 메소드(매직 메소드, magic method)* 를 호출해서 기본적인 객체 연산을 수행한다. 특별 메소드는 `__getitem__()` 처럼 이중 언더바를 가지고 있다. 예를 들어, `obj[key]`형태의 구문은 `__getitem__()` 특별 메소드가 지원한다. `__getitem__()`과 같은 메소드를 읽을때에는 *던더(dunder) - getitem*이라고 부르는 것을 선호한다고 한다. 던더는 더블 언더바(double under)를 줄인 말이다. 따라서, 특별 메소드를 ***던더 메소드*** 라고도 한다.



## 1.1 파이썬 카드 한 벌

아래의 [예제 1-1]은 간단한 코드지만, 특별 메서드 `__getitem__()`과 `__len__()`만으로도 괜찮은 기능을 구현할 수 있다는 것을 보여준 예시이다. 이 코드는 카드(스페이드, 다이아몬드, 클로버, 하트) 한 벌을 나타내는 클래스`FrenchDeck` 이다.

```python
## 예제 1-1 : namedtuple을 이용한 카드 한 벌 클래스
import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()
    
    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]
        
    def __len__(self):
        return len(self._cards)
    
    def __getitem__(self, position):
        return self._cards[position]
```



[예제 1-1] 코드는 `collections.namedtuple()` 을 이용해서 구현 하였다. `collections.namedtuple()`에 대해서는 [여기](http://excelsior-cjh.tistory.com/entry/collections-%EB%AA%A8%EB%93%88-namedtuple?category=966334)를 참고하면 된다. <br />`FrenchDeck` 클래스는 `__len__` 메소드를 정의해 줌으로써  `len()` 함수를 통해 카드의 수를 반환 한다. 

```python
>>> deck = FrenchDeck()
>>> len(deck)
52
```



또한, `__getitem__` 정의를 통해, `obj[key]` 형태로 특정 카드를 읽을 수 있다. 예를 들어, `deck[0]`은 첫 번째 카드 이며, `deck[-1]`은 마지막 카드를 가져온다.

```python
>>> print(deck[0])
Card(rank='2', suit='spades')

>>> print(deck[-1])
Card(rank='A', suit='hearts')
```



이렇듯, 특별 메소드를 통해 파이썬 데이터 모델을 사용할 때의 두 가지 장점이 있다.

- 사용자가 표준 연산을 수행하기 위해 클래스 자체에서 구현한 임의 메서드명을 암기할 필요가 없다.
- 파이썬 표준 라이브러리에서 제공하는 기능을 별도로 구현할 필요 없이 바로 사용할 수 있다.



`__getitem__()` 메소드는 `self._cards`를 호출 하므로, 슬라이싱(slicing)도 당연히 지원한다. 

```python
>>> print(deck[:3])
[Card(rank='2', suit='spades'), Card(rank='3', suit='spades'), Card(rank='4', suit='spades')]

>>> print(deck[12::13])
[Card(rank='A', suit='spades'), Card(rank='A', suit='diamonds'), Card(rank='A', suit='clubs'), Card(rank='A', suit='hearts')]
```



`__getitem__()` 특별 메소드를 구현함으로써 deck을 리스트 형태 처럼 반복할 수도 있다.

```python
>>> for card in deck:
    	print(card)
Card(rank='2', suit='spades')
Card(rank='3', suit='spades')
Card(rank='4', suit='spades')
Card(rank='5', suit='spades')
Card(rank='6', suit='spades')
Card(rank='7', suit='spades')
...

>>> for card in reversed(deck):
        print(card)
Card(rank='A', suit='hearts')
Card(rank='K', suit='hearts')
Card(rank='Q', suit='hearts')
Card(rank='J', suit='hearts')
Card(rank='10', suit='hearts')
...
```



이 외에도 `in` 이나 정렬(sort)를 적용할 수도 있다. 교재에서는 정렬을 카드 (에이스가 제일 높고, 숫자가 같은 경우 스페이드 > 하트 > 다이아몬드 > 클로버 순) 순으로 정렬하는 코드를 아래와 같이 작성 했다. 

```python
# in 사용 예시
>>> Card('Q', 'hearts') in deck
True

# 정렬
>>> for card in sorted(deck, key=spades_high):
        print(card)

Card(rank='2', suit='clubs')
Card(rank='2', suit='diamonds')
Card(rank='2', suit='hearts')
Card(rank='2', suit='spades')
Card(rank='3', suit='clubs')
Card(rank='3', suit='diamonds')
...
```

 

`FrenchDeck` 클래스에 `__len__()`과 `__getitem__()` 특별 메소드를 구현함으로써 `FrenchDeck`은 마치 파이썬 시퀀스(ex. `list`) 처럼 작동하므로 반복, 슬라이싱 등이 가능하다. 



## 1.2 특별 메서드는 어떻게 사용되나?





