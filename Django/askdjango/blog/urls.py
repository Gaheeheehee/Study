# blog/urls.py
from django.urls import path
from . import views
from . import views_cbv

# django2.0 부터는 namespace 지정 후 
# 해당 app/urls.py에 app_name을 명시 해줘야함
app_name = 'blog'
urlpatterns = [
    path('', views.post_list, name='post_list'),
    path('detail/<int:id>/', views.post_detail, name='post_detail'),

    path('new/', views.post_new, name='post_new'),
    path('<int:id>/edit/', views.post_edit, name='post_edit'),

    path('cbv/new/', views_cbv.post_new),
    path('cbv/blog/', views_cbv.post_list2, name='post_list2'),
]
