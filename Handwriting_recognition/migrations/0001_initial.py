# Generated by Django 3.2.5 on 2022-08-18 13:35

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='upload_img',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('請輸入員工編號', models.CharField(max_length=200)),
                ('選擇檔案', models.ImageField(upload_to='static/upload_img')),
            ],
        ),
    ]
