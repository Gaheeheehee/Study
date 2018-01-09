from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def mysum(request, numbers):
    # numbers = "1/2/12/123/12331/12313/12313"
    # request: HttpRequest
    numbers = list(map(lambda s: int(s or 0), numbers.split('/')))
    sum_numbers = sum(numbers)
    return HttpResponse(sum_numbers)


def hello(request, name, age):
    return HttpResponse('안녕하세요. {}. {}살이시네요'.format(name, age))
