# Generated by Django 2.2.5 on 2022-07-22 09:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('spl_1', '0003_auto_20220722_1628'),
    ]

    operations = [
        migrations.CreateModel(
            name='Binary_tree_1',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('img', models.ImageField(upload_to='static/pictures/')),
                ('frontanswer', models.CharField(max_length=255, null=True, verbose_name='层次遍历')),
                ('firstanswer', models.CharField(max_length=255, verbose_name='前序遍历')),
                ('middleanswer', models.CharField(max_length=255, verbose_name='中序遍历')),
                ('endanswer', models.CharField(max_length=255, verbose_name='后序遍历')),
                ('explain', models.CharField(max_length=255, verbose_name='备注')),
            ],
        ),
        migrations.DeleteModel(
            name='Users',
        ),
    ]
