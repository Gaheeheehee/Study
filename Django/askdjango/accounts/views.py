# accounts/views.py
from django.conf import settings
from django.shortcuts import redirect, render
from django.contrib.auth.forms import UserCreationForm


def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            return redirect(settings.LOGIN_URL)  # default : "/accounts/login/"

    else:
        form = UserCreationForm()
        return render(request, 'accounts/signup_form.html', {
            'form': form,
        })

def profile(request):
    print("이게 뭐지? >>>", request)
    return render(request, 'accounts/profile.html')
