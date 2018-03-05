import os
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from .forms import PostForm
from .models import Post


class DetailView(object):
    '이전 FBV를 CBV버전으로 컨셉만 간단히 구현. 같은 동작을 수행'
    def __init__(self, model):
        self.model = model

    def get_object(self, *args, **kwargs):
        return get_object_or_404(self.model, id=kwargs['id'])

    def get_template_name(self):
        return '{}/{}_detail.html'.format(self.model._meta.app_label, self.model._meta.model_name)

    def dispatch(self, request, *args, **kwargs):
        return render(request, self.get_template_name(), {
            self.model._meta.model_name: self.get_object(*args, **kwargs),
        })

    @classmethod
    def as_view(cls, model):
        def view(request, *args, **kwargs):
            self = cls(model)
            return self.dispatch(request, *args, **kwargs)
        return view


post_detail = DetailView.as_view(Post)

# # STEP 2. 함수를 통해 뷰 생성
# def generate_view_fn(model):
#     def view_fn(request, id):
#         instance = get_object_or_404(model, id=id)
#         instance_name = model._meta.model_name
#         template_name = '{}/{}_detail.html'.format(
#             model._meta.app_label, instance_name)
#         return render(request, template_name, {
#             instance_name: instance,
#         })
#     return view_fn

# post_detail = generate_view_fn(Post)


# # STEP 1. FBV
# def post_detail(request, id):
#     post = get_object_or_404(Post, id=id)
#     return render(request,'dojo/post_detail.html', {
#         'post': post,
#     })

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

# form 수정기능
def post_edit(request, id):
    post = get_object_or_404(Post, id=id)
    if request.method == 'POST':
        form = PostForm(request.POST, request.FILES, instance=post)
        if form.is_valid():
            post = form.save(commit=False)
            post.ip=request.META['REMOTE_ADDR']
            post.save()
            return redirect('/dojo/')  # namespace:name
    else:
        form = PostForm(instance=post)
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
