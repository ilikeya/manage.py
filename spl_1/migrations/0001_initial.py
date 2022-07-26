# Generated by Django 2.2.5 on 2022-07-22 07:52

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Users',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('uese_name', models.CharField(max_length=255, null=True, verbose_name='用户名')),
                ('phone_number', models.CharField(max_length=20, verbose_name='手机号')),
                ('password', models.CharField(max_length=255, verbose_name='密码')),
                ('register_date', models.DateTimeField(auto_now_add=True, verbose_name='注册时间')),
            ],
        ),
    ]
