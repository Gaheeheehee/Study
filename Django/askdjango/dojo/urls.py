# dojo/urls.py
from django.urls import path, re_path
from . import views
from . import views_cbv  # Class Based View import

app_name = 'dojo'
urlpatterns = [
    # re_path(r'^sum/(?P<x>\d+)/$', views.mysum),
    # re_path(r'^sum/(?P<x>\d+)/(?P<y>\d+)/$', views.mysum),
    # re_path(r'^sum/(?P<x>\d+)/(?P<y>\d+)/(?P<z>\d+)/$', views.mysum),
    re_path(r'^sum/(?P<numbers>[\d/]+)/$', views.mysum),
    # re_path(r'^hello/(?P<name>[ㄱ-힣]+)/(?P<age>\d+)/$', views.hello)
    path('hello/<str:name>/<int:age>/', views.hello), # django2.0의 path 설정
    # path('sum/<int:x>/', views.mysum),  # django2.0의 path 설정
    # path('sum/<int:x>/<int:y>/', views.mysum),  # django2.0의 path 설정
    # path('sum/<int:x>/<int:y>/<int:z>/', views.mysum),  # django2.0의 path 설정
    path('list1/', views.post_list1),
    path('list2/', views.post_list2),
    path('list3/', views.post_list3),
    path('excel/', views.excel_download),

    path('cbv/list1/', views_cbv.post_list1),
    path('cbv/list2/', views_cbv.post_list2),
    path('cbv/list3/', views_cbv.post_list3),
    path('cbv/excel/', views_cbv.excel_download),
]
