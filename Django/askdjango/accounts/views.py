# accounts/views.py
from django.shortcuts import render


def profile(request):
    print("이게 뭐지? >>>", request)
    return render(request, 'accounts/profile.html')
