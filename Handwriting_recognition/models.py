# from django.db import models

# # Create your models here.
from pyexpat import model
from time import time
from turtle import title
from unicodedata import category, name
from django.db import models

from django.db import models
# from view_table.models import ViewTable

# Create your models here.
class upload_img(models.Model):
    請輸入員工編號 = models.CharField(max_length=200)
    選擇檔案 = models.ImageField(upload_to="static/upload_img")

class emp_table(models.Model):
    time = models.CharField(max_length=100)
    category = models.CharField(max_length=100)
