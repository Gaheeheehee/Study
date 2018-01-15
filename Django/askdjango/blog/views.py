# blog/views.py
from django.http import Http404
from django.shortcuts import render, get_object_or_404
from .models import Post

def post_list(request):
    qs = Post.objects.all()  # 이 시점에는 DB 액세스가 이루어 지지 않음 

    q = request.GET.get('q', '')  # q가 있는경우 가져오고, 없는 경우  ''(빈 문자열) 반환
    if q:
        qs = qs.filter(title__icontains=q)

    return render(request, 'blog/post_list.html', {
        'post_list': qs,
        'q': q,
    })  # appname/html 이름


def post_detail(request, id):
    # try:
    #     post = Post.objects.get(id=id)
    # except Post.DoesNotExist:
    #     raise Http404

    post = get_object_or_404(Post, id=id)

    return render(request, 'blog/post_detail.html', {
        'post': post,
    })
