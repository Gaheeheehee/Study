# blog/models.py
from django.db import models


class Post(models.Model):
    title = models.CharField(max_length=100, verbose_name='제목',
                             help_text='포스팅 제목을 입력해주세요. 최대 100자 내외')  # 길이 제한이 있는 문자열
    content = models.TextField(verbose_name='내용')               # 길이 제한이 없는 문자열
    created_at = models.DateTimeField(auto_now_add=True)  # auto_now_add: 최초 생성 날짜 기록
    updated_at = models.DateTimeField(auto_now=True)  # auto_now: 갱신이 될때마다 자동 저장
