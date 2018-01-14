# blog/views.py
from django.shortcuts import render
from .models import Post

def post_list(request):
    qs = Post.objects.all()  # 이 시점에는 DB 액세스가 이루어 지지 않음 
    return render(request, 'blog/post_list.html', {
        'post_list': qs,   
    })  # appname/html 이름

