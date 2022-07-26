from django.db import models
from io import BytesIO
import matplotlib.pyplot as plt
import os
import datetime#注意要输入这个模块
from matplotlib import pyplot as plt
import os
import datetime
import hashlib

# encoding: utf-8
"""
This module provides access to binary tree visualization
"""
import turtle
import math

class HTNode:
    def __init__(self, value=None, weight=0, parent=None, left_child=None, right_child=None):
        self.value = value
        self.weight = weight
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
nalue_1=[]
nalue_2=[]
nalue_3=[]
class ShowHuffmanBiTree(object):
    """显示哈夫曼树"""
    def __init__(self, root, d_hor=4, d_vec=8, radius=1.5, figsize =(11, 9)):
        """
            对所展示二叉树的一些元素参数的设置
            :param d_hor: 节点与节点的水平距离
            :param d_vec: 节点与节点的垂直距离
            :param radius: 节点的半径
            :param figsize: 画布大小，用一个元祖表示画布width和high,单位为inch
            """
        self.root = None
        self.d_hor = d_hor
        self.d_vec = d_vec
        self.radius = radius
        self.figsize = figsize
        self.root = root

    nalue_1=''
    def PostOrderTree(self,root):  # 前序遍历
          if root != None:
            nalue_1.append(root.value)
            nalue_1.append(',')
            self.PostOrderTree(root.left_child)
            self.PostOrderTree(root.right_child)

    def InOrderTree(self,root):
        if root != None:
            self.InOrderTree(root.left_child)
            nalue_3.append(root.value)
            nalue_3.append(',')
            self.InOrderTree(root.right_child)

    def PreOrderTree(self,root):
        if root != None:
            self.PreOrderTree(root.left_child)
            self.PreOrderTree(root.right_child)
            nalue_2.append(root.value)
            nalue_2.append(',')


    def get_left_width(self, root):
        """获得根左边宽度，也是根的左子孙节点数"""
        return self.get_width(root.left_child)

    def get_right_width(self, root):
        """获得根右边宽度，也是根的右子孙节点数"""
        return self.get_width(root.right_child)

    def get_width(self, root):
        """获得树的宽度，也是该树的节点数。使用的是中序遍历方式"""
        if root:
            return self.get_width(root.left_child) + 1 + self.get_width(root.right_child)
        else:
            return 0

    def get_height(self, root):
        """获得二叉树的高度, 使用后序遍历"""
        if root:
            return max(self.get_height(root.left_child), self.get_height(root.right_child)) + 1
        else:
            return 0

    def get_w_h(self, root):
        """获得树的宽度和高度"""
        w = self.get_width(root)
        h = self.get_height(root)
        return w, h

    def __draw_a_node(self, x, y, value, ax):#增加权值
        """画一个节点"""
        c_node = plt.Circle((x, y), radius=self.radius, color="#65DDFF")
        ax.add_patch(c_node)
        plt.text(x, y, value, ha='center', va='center', fontsize=25, )#增加权值

    def __draw_a_edge(self, x1, y1, x2, y2):
        """画一条边"""
        x = (x1, x2)
        y = (y1, y2)
        plt.plot(x, y, 'g-')



    def __create_win(self, root):
        """创建窗口"""
        # WEIGHT： 树宽，HEIGHT： 树高
        WEIGHT, HEIGHT = self.get_w_h(root)
        # WEIGHT：树宽 + 1
        WEIGHT = (WEIGHT+1)*self.d_hor
        # HEIGHT = 树高+1
        HEIGHT = (HEIGHT+1)*self.d_vec
        # print(WEIGHT, HEIGHT)

        # fig = plt.figure(figsize=(a, b), dpi=dpi)
        # 设置图形的大小，a 为图形的宽， b 为图形的高，单位为英寸
        # dpi 为设置图形每英寸(inch)的点数
        # 1点（英美点）=0.3527毫米=1/72英寸（Office里面的点）。
        # 线条，标记，文本等大多数元素都有以磅(point即点)为单位的大小。因1inch = 72point,则72dp/inch=1dp/point、144dp/inch=2dp/point
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)  # 表示整个figure分成1行1列，共1个子图，这里子图在第一行第一列

        plt.xlim(0, WEIGHT)  # 设定x座标轴的范围，当前axes上的座标轴。
        plt.ylim(0, HEIGHT)  # 设定y座标轴的范围，当前axes上的座标轴。
        x = (self.get_left_width(root) + 1) * self.d_hor  # x, y 是第一个要绘制的节点坐标，由其左子树宽度决定
        y = HEIGHT - self.d_vec
        return fig, ax, x, y

    def __print_tree_by_preorder(self, root, x, y, ax):
        """通过先序遍历打印二叉树"""

        if not root:
            # 根节点为空返回
            return

        # 画节点
        self.__draw_a_node(x, y, root.value, ax)

        # 画左右分支
        lx = rx = 0
        ly = ry = y - self.d_vec
        if root.left_child:
            lx = x - self.d_hor * (self.get_right_width(root.left_child) + 1)   # x-左子树的右边宽度
            self.__draw_a_edge(x, y, lx, ly)
            # print(root.left_child, (lx, ly))
        if root.right_child:
            rx = x + self.d_hor * (self.get_left_width(root.right_child) + 1)   # x-右子树的左边宽度
            # print(root.right_child, (rx, ry))
            self.__draw_a_edge(x, y, rx, ry)

        # 递归打印
        self.__print_tree_by_preorder(root.left_child, lx, ly, ax)
        self.__print_tree_by_preorder(root.right_child, rx, ry, ax)

    def show_BiTree_1(self):
        """可视化二叉树"""
        _, ax, x, y = self.__create_win(self.root)
        self.__print_tree_by_preorder(self.root, x, y, ax)
        plt.show()

    def save_BiTree_1(self):
        for i in range(0, 1, 1):
            _, ax, x, y = self.__create_win(self.root)
            self.__print_tree_by_preorder(self.root, x, y, ax)
            name_list = ["one", "two", "three"]  # 先把图片名字存储成列表
            files_name = ["file_1_", "file_2_", "file_3_"]  # 先把文件夹名字存储成列表

            # 指定图片保存路径
            global figure_save_path
            figure_save_path = files_name[i]  # 给文件夹赋予名字

            mk_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')  # 这里是引用时间
            mkfigure_time = name_list[i] + mk_time  # 注意这里在下面是用作图片名字的，里面也用了列表使每个文件夹的名字不一样
            # 指定图片保存路径
            global figure_save
            figure_save = mkfigure_time  # 给图片赋予名字

            if not os.path.exists(figure_save_path):
                os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
    

            plt.savefig(os.path.join(figure_save_path, figure_save))  # 分别创建文件夹，分别储存命名图片
            #plt.show()
            
def md5(a):
    md5 = hashlib.md5()
    md5.update(a.encode("utf-8"))
    return md5.hexdigest()[:5]

class HuffmanTree:
    def __init__(self):
        self.leaf_nodes = []  # 用于保存创建好的叶子结点
        self.root = None

    def creat_HuffmanTree(self, elements_seq):
        """
        创建哈夫曼树：由n个带权叶子结点构成的所有二叉树种带权路径长度最短的二叉树。
        算法思想：
            (1) 用给定的n个权值{w1,w2,...,wn}构造n棵二叉树并构造成的森林F={T1,T2,...,Tn},其中每一棵树
                Ti(1<=i<=n)都只有一个权值为wi的根节点，其左右子树均为空。
            (2) 找最小树；在森林F中选择两颗根结点权值最小的二叉树，作为一棵新二叉树的左右子树，标记新二
                叉树的根结点权值为其左右子树的根结点权值之和。
            (3) 从F中删除被选中的那两颗二叉树，同时把新构成的二叉树加入到森林F中
            (4) 判断：重复(2)(3)操作，直到森林中只有一颗二叉树为止，此时得到的这颗二叉树就是哈夫曼树。
        :param elements_seq:
        :return:
        """
        while elements_seq:  # 构造n棵只有一个权值为wi的根节点，其左右子树均为空
            element = elements_seq.pop()
            leaf_node = HTNode(value=element[0], weight=element[1])
            self.leaf_nodes.append(leaf_node)

        key = lambda node: node.weight

        temp_nodes = self.leaf_nodes.copy()  # 森林F
        while len(temp_nodes) > 1:
            temp_nodes.sort(key=key, reverse=True)  # 根据结点的权值进行排序，按降序排序
            left_child = temp_nodes.pop()  # 获取最小树
            right_child = temp_nodes.pop()  # 获取最小树
            parent_weight = right_child.weight + left_child.weight  # 计算双亲结点的权值，即新二叉树根的权值left_child

            id_ = md5(f'{left_child.value}{right_child.value}')  # 默认的双亲结点标识，只是为了区别创建哈夫曼树过程中的所创双亲结点
            parent = HTNode(value=parent_weight, weight=parent_weight, left_child=left_child, right_child=right_child)  # 创建双亲结点
            left_child.parent = parent  # 为左孩子结点设置双亲
            right_child.parent = parent  # 为有孩子结点设置双亲
            temp_nodes.append(parent)  # 将新创建的二叉树放入森林F

        self.root = temp_nodes.pop()

    def huffman_code(self):
        """
        哈夫曼编码是前缀编码。前缀编码中的任一编码都不是其它编码的前缀(左字串)，避免了非前缀编码中分隔符的使用

        哈夫曼编码是最优前缀编码。对于使用频度最高的字符其编码长度最短，而频度最低的字符其编码长度最长。因此对
        于n个字符，分别以它们的使用频度作为叶子全职来构造哈夫曼树，则该树对应的哈夫曼编码能使各种报文(由这n种字
        符构成的文本)对应的二进制串的平均长度最短。

        哈夫曼编码：按规定向左的分支标记为1，向右的分支标记为0



        :return: 哈夫曼编码集
        """
        code_set ={}
        for leaf in self.leaf_nodes:
            code = []
            q = leaf
            while q.parent:
                if q.parent.left_child is q:
                    code.insert(0,'0')
                else:
                    code.insert(0,'1')
                q = q.parent
            code = ''.join(code)
            #print("哈夫编码：")
            #print(code)
            code_set[leaf.value] = code
        return code_set
# 需要创建应用
# Create your models here.
# 数据表名 应用名_类名

class Binary_tree_1(models.Model):
    img = models.BinaryField('图片')
    frontanswer = models.CharField('层次遍历', max_length=255, null=True)
    firstanswer = models.CharField('前序遍历', max_length=255)
    middleanswer = models.CharField('中序遍历', max_length=255)
    endanswer = models.CharField('后序遍历', max_length=255)
    explain = models.CharField('备注', max_length=255)

    '''class Meta:
        db_table = 'tree_1'  # 指明数据库表名
        verbose_name = '图书'  # 在admin站点中显示的名称
        verbose_name_plural = verbose_name  # 显示的复数名称'''

    '''def __str__(self):
        """定义每个数据对象的显示信息"""
        return self.explain'''


# CharField和DateTimeField----数据类型
# max_length----最大长度
# auto_now_add----默认时间为创建时间

# 类名.objects.create(字段=值,字段=值)
#批量绘制图片+批量存储
#将图片存储在指定文件夹（附带创建文件夹）+用循环创建文件名
def log(request):
    if request.method == "POST":
        try:
            string_list = [
                'a','b','c','d','e','f','g','h','l','m','m','l','n','o','p','q','r','s','t','u','v','w','x','y','z'
            ]
            number_list = [
                '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'
            ]
            strings_str =""
            for pop in string_list:
                strings = request.POST.get(pop)
                strings_str = strings_str +strings
                strings = request.POST.get(',')
                strings_str = strings_str + strings
                for pip in number_list:
                    strings = request.POST.get(pip)
                    strings_str = strings_str + strings
            param_list = []
            param_list = strings_str.split(",")
            param = []
            j = 1
            for pop in param_list:
                if j%2 == 1 :
                   test_list =[]
                   test_list.append(pop)
                else :
                    if j%2 == 0 :
                        pop = int(pop)
                        test_list.append(pop)
                        param.append(tuple(test_list))
                j = j+1
            #param = [('a',3),('b',12),('c',6),('d',1),('e', 11),('f',2)]
            result=[str(i) for i in param]
            d=''.join(result)
            #print(d)
            huffman_tree = HuffmanTree()
            huffman_tree.creat_HuffmanTree(param)

            ShowHuffmanBiTree(huffman_tree.root).save_BiTree_1()
            ShowHuffmanBiTree(huffman_tree.root).PostOrderTree(huffman_tree.root)
            #print("前序遍历:")
            result=[str(i) for i in nalue_1]
            a=''.join(result)
            #print(a)
            ShowHuffmanBiTree(huffman_tree.root).PreOrderTree(huffman_tree.root)
            #print("后序遍历:")
            result=[str(i) for i in nalue_2]
            b=''.join(result)
            #print(b)
            ShowHuffmanBiTree(huffman_tree.root).InOrderTree(huffman_tree.root)
            #print("中序遍历:")
            result=[str(i) for i in nalue_3]
            c=''.join(result)
            #print(c)
            f = figure_save_path+"\\"+figure_save+".png"

            #print(f)
            with open(f, "rb") as file:
                image = file.read()
            Binary_tree_1.objects.create(img=image, frontanswer=d, firstanswer=a, middleanswer=c,endanswer=b, explain='binarytree')
        except:{}
