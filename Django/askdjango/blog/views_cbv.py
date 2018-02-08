from django import forms
from django.views.generic import CreateView, ListView
from .models import Post


# blog/forms.py
class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = '__all__'

class PostCreateView(CreateView):
    model = Post
    form_class = PostForm


post_new = PostCreateView.as_view()


class PostListView(ListView):
    model = Post
    paginate_by = 10

    def get_template_names(self):
        return 'blog/post_list2.html'


post_list2 = PostListView.as_view()