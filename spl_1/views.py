from django.shortcuts import render


# Create your views here.
import os
import django
from django.urls import path

#os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mytestsite.settings")
#django.setup()
from spl_1.models import Binary_tree_1
# 根据id获取对应数据
from django.http import JsonResponse

def getProductById(request):
    if request.method == "GET":
        mod = Binary_tree_1.objects  # 获取DProduct模型的Model操作对象
        ProductList = mod.filter(product_id=Binary_tree_1.id).values()
        ProductList = list(ProductList)  # 转化为列表属性
        return JsonResponse({'data':ProductList})
