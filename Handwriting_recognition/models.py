# from django.db import models

# # Create your models here.
from pyexpat import model
from turtle import title
from django.db import models

# Create your models here.
class upload_img(models.Model):
    請輸入員工編號 = models.CharField(max_length=200)
    選擇檔案 = models.ImageField(upload_to="static/upload_img")


