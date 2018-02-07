import os
from django.conf import settings
from django.shortcuts import redirect, render
from django.http import HttpResponse, JsonResponse
from .forms import PostForm
from .models import Post


def post_new(request):
    if request.method == 'POST':
        form = PostForm(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False)
            post.ip=request.META['REMOTE_ADDR']
            post.save()
            return redirect('/dojo/')  # namespace:name
    else:
        form = PostForm()
    return render(request, 'dojo/post_form.html', {
        'form': form,
    })


def mysum(request, numbers):
    # numbers = "1/2/12/123/12331/12313/12313"
    # request: HttpRequest
    numbers = list(map(lambda s: int(s or 0), numbers.split('/')))
    sum_numbers = sum(numbers)
    return HttpResponse(sum_numbers)


def hello(request, name, age):
    return HttpResponse('안녕하세요. {}. {}살이시네요'.format(name, age))


def post_list1(request):
    '''FBV(Function Based View): 직접 문자열로 HTML 형식 응답하기'''

    name = '최종현'
    return HttpResponse('''
            <h1>AskDjango</h1>
            <p>{name}</p>
            <p>Django 공부 열심히 할게요</p>
            '''.format(name=name))


def post_list2(request):
    '''FBV2: 템플릿을 통해 HTML형식 응답하기'''

    name = '최종현'
    return render(request, 'dojo/post_list.html', {'name': name})


def post_list3(request):
    '''FBV3: JSON 형식 응답하기'''

    return JsonResponse({
        'message': '안녕 파이썬 & 장고',
        'itmes': ['파이썬', '장고', 'Celery', 'Azure', 'AWS'],
    },json_dumps_params={'ensure_ascii': False})


def excel_download(request):
    '''FBV4: Excel 다운로드 응답하기'''

    # filepath = 'D:/Users/cjh/dev/Study/Study/Django/askdjango/test.xlsx'
    filepath = os.path.join(settings.BASE_DIR, 'test.xlsx')
    filename = os.path.basename(filepath) # 파일명 가져오기
    with open(filepath, 'rb') as f:
        response = HttpResponse(f, content_type='application/vnd.ms-excel')
        # 필요한 응답헤더 세팅
        response['Content-Disposition'] = 'attachment; filename="{}"'.format(filename)
        return response
