# Generated by Django 2.2.5 on 2022-07-22 10:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('spl_1', '0004_auto_20220722_1719'),
    ]

    operations = [
        migrations.AlterField(
            model_name='binary_tree_1',
            name='img',
            field=models.BinaryField(verbose_name='图片'),
        ),
    ]
