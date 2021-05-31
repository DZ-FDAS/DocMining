from django.urls import path

from . import views

urlpatterns = [
    path('index', views.index, name='index'),
    path("MainFun", views.MainFun, name='MainFun'),
    path("fileDisplay", views.fileDisplay, name='fileDisplay'),
]