# Python初级语法详解 - C++转Python指南

> 本文档针对C++转Python的选手，梳理Python特有的常用语法和操作，帮助快速上手Python开发。

---

## 目录

1. [Python与C++核心差异](#1-python与c核心差异)
2. [数据结构操作](#2-数据结构操作)
3. [常用内置函数](#3-常用内置函数)
4. [列表推导式与生成器表达式](#4-列表推导式与生成器表达式)
5. [字符串操作](#5-字符串操作)
6. [字典操作](#6-字典操作)
7. [集合操作](#7-集合操作)
8. [NumPy/PyTorch常用操作](#8-numpytorch常用操作)
9. [文件与路径操作](#9-文件与路径操作)
10. [异常处理](#10-异常处理)

---

## 1. Python与C++核心差异

### 1.1 变量与类型

```python
# Python: 动态类型，无需声明类型
x = 10          # int
x = "hello"     # str (可以改变类型)
x = [1, 2, 3]   # list

# C++: 静态类型
# int x = 10;
# x = "hello";  // 编译错误
```

### 1.2 变量作用域

```python
# Python: 函数作用域，没有块作用域
if True:
    x = 10
print(x)  # 10，可以访问

# C++: 块作用域
# if (true) {
#     int x = 10;
# }
# cout << x;  // 编译错误
```

### 1.3 引用与拷贝

```python
# Python: 变量是引用
a = [1, 2, 3]
b = a           # b是a的引用，不是拷贝
b.append(4)
print(a)        # [1, 2, 3, 4] a也被修改了

# 深拷贝 vs 浅拷贝
import copy
a = [[1, 2], [3, 4]]
b = a.copy()            # 浅拷贝
c = copy.deepcopy(a)    # 深拷贝

# C++对比
# vector<int> a = {1, 2, 3};
# vector<int> b = a;  // 拷贝，不是引用
```

### 1.4 None vs nullptr

```python
# Python: None
x = None
if x is None:       # 使用 is 比较
    print("x is None")

# C++: nullptr
# int* x = nullptr;
# if (x == nullptr) { ... }
```

### 1.5 布尔值

```python
# Python: True/False (首字母大写)
flag = True
if flag:
    print("true")

# C++: true/false
# bool flag = true;
```

### 1.6 逻辑运算符

```python
# Python: and, or, not
if a > 0 and b > 0:
    print("both positive")
if not flag:
    print("flag is False")

# C++: &&, ||, !
# if (a > 0 && b > 0) { ... }
```

---

## 2. 数据结构操作

### 2.1 列表 (List) - 类似std::vector

```python
# 创建列表
lst = [1, 2, 3, 4, 5]
lst = list(range(5))        # [0, 1, 2, 3, 4]
lst = [0] * 5               # [0, 0, 0, 0, 0]

# 访问元素
print(lst[0])               # 第一个元素
print(lst[-1])              # 最后一个元素 (Python特有)
print(lst[-2])              # 倒数第二个元素

# 切片 (Python特有，非常强大)
lst = [0, 1, 2, 3, 4, 5]
print(lst[1:4])             # [1, 2, 3] 左闭右开
print(lst[:3])              # [0, 1, 2] 前3个
print(lst[3:])              # [3, 4, 5] 从第3个开始
print(lst[::2])             # [0, 2, 4] 步长为2
print(lst[::-1])            # [5, 4, 3, 2, 1, 0] 反转

# 添加元素
lst.append(6)               # 末尾添加
lst.insert(0, -1)           # 指定位置插入
lst.extend([7, 8, 9])       # 扩展列表

# 删除元素
lst.pop()                   # 删除并返回最后一个
lst.pop(0)                  # 删除并返回指定位置
lst.remove(3)               # 删除第一个值为3的元素
del lst[0]                  # 删除指定位置

# 查找
idx = lst.index(3)          # 查找值为3的索引
count = lst.count(3)        # 统计值为3的个数

# 排序
lst.sort()                  # 原地排序
lst.sort(reverse=True)      # 降序排序
lst.sort(key=lambda x: -x)  # 自定义排序
sorted_lst = sorted(lst)    # 返回新列表

# 反转
lst.reverse()               # 原地反转
reversed_lst = lst[::-1]    # 返回新列表

# 其他操作
len(lst)                    # 长度
sum(lst)                    # 求和
max(lst)                    # 最大值
min(lst)                    # 最小值
3 in lst                    # 是否包含 (返回True/False)
```

### 2.2 元组 (Tuple) - 不可变列表

```python
# 创建元组
t = (1, 2, 3)
t = 1, 2, 3                 # 括号可省略
t = tuple([1, 2, 3])        # 从列表转换

# 解包 (Python特有)
a, b, c = (1, 2, 3)
a, *rest = (1, 2, 3, 4)     # a=1, rest=[2, 3, 4]
first, *middle, last = (1, 2, 3, 4, 5)

# 交换变量 (Python特有)
a, b = b, a

# 多返回值
def get_point():
    return 1, 2             # 返回元组

x, y = get_point()
```

### 2.3 命名元组 (NamedTuple)

```python
from collections import namedtuple
from typing import NamedTuple

# 方式1: collections.namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x, p.y)             # 1 2

# 方式2: typing.NamedTuple (推荐，支持类型注解)
class Point(NamedTuple):
    x: int
    y: int

p = Point(1, 2)
print(p.x, p.y)
```

---

## 3. 常用内置函数

### 3.1 enumerate - 带索引遍历

```python
# Python: enumerate
lst = ['a', 'b', 'c']
for i, val in enumerate(lst):
    print(f"index={i}, value={val}")

# 指定起始索引
for i, val in enumerate(lst, start=1):
    print(f"index={i}, value={val}")

# C++对比
# for (int i = 0; i < lst.size(); i++) {
#     cout << "index=" << i << ", value=" << lst[i] << endl;
# }
```

### 3.2 zip - 并行遍历

```python
# Python: zip
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name}: {age}")

# 转换为列表
pairs = list(zip(names, ages))  # [('Alice', 25), ('Bob', 30), ('Charlie', 35)]

# 解压
names, ages = zip(*pairs)

# C++对比: 需要手动管理索引
# for (int i = 0; i < min(names.size(), ages.size()); i++) { ... }
```

### 3.3 map - 映射

```python
# Python: map
nums = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x**2, nums))
# [1, 4, 9, 16, 25]

# 等价的列表推导式 (更Pythonic)
squares = [x**2 for x in nums]

# C++对比
# vector<int> squares;
# transform(nums.begin(), nums.end(), back_inserter(squares), [](int x) { return x*x; });
```

### 3.4 filter - 过滤

```python
# Python: filter
nums = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda x: x % 2 == 0, nums))
# [2, 4, 6]

# 等价的列表推导式 (更Pythonic)
evens = [x for x in nums if x % 2 == 0]

# C++对比
# vector<int> evens;
# copy_if(nums.begin(), nums.end(), back_inserter(evens), [](int x) { return x % 2 == 0; });
```

### 3.5 reduce - 归约

```python
from functools import reduce

# Python: reduce
nums = [1, 2, 3, 4, 5]
total = reduce(lambda x, y: x + y, nums)  # 15
product = reduce(lambda x, y: x * y, nums)  # 120

# 带初始值
total = reduce(lambda x, y: x + y, nums, 0)

# C++对比
# int total = accumulate(nums.begin(), nums.end(), 0);
```

### 3.6 any/all - 存在/全部

```python
# Python: any/all
nums = [1, 2, 3, 4, 5]
has_even = any(x % 2 == 0 for x in nums)    # True
all_positive = all(x > 0 for x in nums)      # True

# C++对比
# bool has_even = any_of(nums.begin(), nums.end(), [](int x) { return x % 2 == 0; });
# bool all_positive = all_of(nums.begin(), nums.end(), [](int x) { return x > 0; });
```

### 3.7 sorted/sort - 排序

```python
# Python: sorted (返回新列表)
nums = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_nums = sorted(nums)                   # [1, 1, 2, 3, 4, 5, 6, 9]
sorted_nums = sorted(nums, reverse=True)     # 降序

# 自定义排序
students = [('Alice', 85), ('Bob', 90), ('Charlie', 80)]
sorted_students = sorted(students, key=lambda x: x[1])  # 按分数排序
sorted_students = sorted(students, key=lambda x: x[1], reverse=True)

# 多条件排序
data = [(1, 'b'), (1, 'a'), (2, 'b'), (2, 'a')]
sorted_data = sorted(data, key=lambda x: (x[0], x[1]))  # 先按第一个，再按第二个

# list.sort() 原地排序
nums.sort()
nums.sort(reverse=True)

# C++对比
# sort(nums.begin(), nums.end());
# sort(nums.begin(), nums.end(), greater<int>());  // 降序
```

### 3.8 reversed/reverse - 反转

```python
# Python: reversed (返回迭代器)
nums = [1, 2, 3, 4, 5]
reversed_nums = list(reversed(nums))  # [5, 4, 3, 2, 1]

# list.reverse() 原地反转
nums.reverse()

# 切片反转 (最常用)
reversed_nums = nums[::-1]
```

---

## 4. 列表推导式与生成器表达式

### 4.1 列表推导式 (List Comprehension)

```python
# 基本形式
squares = [x**2 for x in range(10)]

# 带条件
evens = [x for x in range(10) if x % 2 == 0]

# 带if-else
result = [x if x % 2 == 0 else -x for x in range(10)]

# 嵌套循环
matrix = [[i*j for j in range(5)] for i in range(5)]

# 展平矩阵
flat = [x for row in matrix for x in row]

# C++对比: 需要手动循环
# vector<int> squares;
# for (int x : range(10)) { squares.push_back(x*x); }
```

### 4.2 字典推导式

```python
# 基本形式
d = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# 带条件
d = {x: x**2 for x in range(10) if x % 2 == 0}

# 交换键值
original = {'a': 1, 'b': 2, 'c': 3}
swapped = {v: k for k, v in original.items()}
```

### 4.3 集合推导式

```python
# 基本形式
s = {x**2 for x in range(10)}  # {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}
```

### 4.4 生成器表达式

```python
# 列表推导式: 立即计算，占用内存
squares_list = [x**2 for x in range(1000000)]

# 生成器表达式: 惰性计算，不占用内存
squares_gen = (x**2 for x in range(1000000))

# 使用
for square in squares_gen:
    if square > 100:
        break

# 作为函数参数
total = sum(x**2 for x in range(100))  # 不需要额外括号
```

---

## 5. 字符串操作

### 5.1 基本操作

```python
s = "Hello, World!"

# 访问
print(s[0])              # 'H'
print(s[-1])             # '!'
print(s[0:5])            # 'Hello'
print(s[::-1])           # 反转

# 长度
print(len(s))            # 13

# 拼接
s1 = "Hello"
s2 = "World"
s3 = s1 + " " + s2       # "Hello World"
s3 = f"{s1} {s2}"        # f-string (推荐)
s3 = "{} {}".format(s1, s2)

# 重复
s = "ab" * 3             # "ababab"

# 成员判断
print("Hello" in s)      # True
```

### 5.2 常用方法

```python
s = "  Hello, World!  "

# 大小写
s.lower()                # "  hello, world!  "
s.upper()                # "  HELLO, WORLD!  "
s.capitalize()           # "  hello, world!  "
s.title()                # "  Hello, World!  "
s.swapcase()             # "  hELLO, wORLD!  "

# 去除空白
s.strip()                # "Hello, World!"
s.lstrip()               # "Hello, World!  "
s.rstrip()               # "  Hello, World!"

# 查找
s.find("World")          # 9，找不到返回-1
s.rfind("o")             # 从右边找
s.index("World")         # 9，找不到抛异常
s.count("l")             # 3

# 替换
s.replace("World", "Python")

# 分割与连接
s = "a,b,c,d"
parts = s.split(",")     # ['a', 'b', 'c', 'd']
s = ",".join(parts)      # "a,b,c,d"

# 判断
s = "hello"
s.startswith("he")       # True
s.endswith("lo")         # True
s.isalpha()              # True (全是字母)
s.isdigit()              # False
s.isalnum()              # True (字母或数字)
s.isspace()              # False (全是空白)

# 格式化
name = "Alice"
age = 25
# f-string (推荐)
print(f"Name: {name}, Age: {age}")
print(f"2 + 3 = {2 + 3}")
print(f"{name:>10}")     # 右对齐，宽度10
print(f"{age:05d}")      # 前导0，宽度5
print(f"{3.14159:.2f}")  # 保留2位小数

# format方法
print("Name: {}, Age: {}".format(name, age))
print("Name: {0}, Age: {1}".format(name, age))
print("Name: {name}, Age: {age}".format(name=name, age=age))
```

### 5.3 多行字符串

```python
# 三引号
s = """
This is a
multi-line
string
"""

# 去除首尾换行
s = """
This is a
multi-line
string
""".strip()

# 行尾续行
s = "This is a very long " \
    "string that spans " \
    "multiple lines"
```

---

## 6. 字典操作

### 6.1 基本操作

```python
# 创建字典
d = {'a': 1, 'b': 2, 'c': 3}
d = dict(a=1, b=2, c=3)
d = dict([('a', 1), ('b', 2), ('c', 3)])
d = {x: x**2 for x in range(5)}  # 字典推导式

# 访问
print(d['a'])             # 1
print(d.get('d'))         # None，不存在返回None
print(d.get('d', 0))      # 0，不存在返回默认值

# 添加/修改
d['d'] = 4                # 添加
d['a'] = 10               # 修改

# 删除
del d['a']                # 删除键
value = d.pop('b')        # 删除并返回值
d.pop('x', None)          # 不存在不报错
d.clear()                 # 清空

# 成员判断
print('a' in d)           # True
print('a' in d.keys())    # True
print(1 in d.values())    # True
```

### 6.2 遍历

```python
d = {'a': 1, 'b': 2, 'c': 3}

# 遍历键
for key in d:
    print(key)
for key in d.keys():
    print(key)

# 遍历值
for value in d.values():
    print(value)

# 遍历键值对
for key, value in d.items():
    print(f"{key}: {value}")

# C++对比
# for (auto& [key, value] : d) { ... }  // C++17
```

### 6.3 常用方法

```python
d = {'a': 1, 'b': 2}

# 获取所有键/值
keys = list(d.keys())     # ['a', 'b']
values = list(d.values()) # [1, 2]
items = list(d.items())   # [('a', 1), ('b', 2)]

# setdefault: 存在则返回，不存在则设置默认值
d = {'a': 1}
d.setdefault('a', 10)     # 返回1，不修改
d.setdefault('b', 2)      # 返回2，添加'b': 2

# update: 合并字典
d1 = {'a': 1, 'b': 2}
d2 = {'b': 3, 'c': 4}
d1.update(d2)             # {'a': 1, 'b': 3, 'c': 4}

# fromkeys: 从序列创建字典
keys = ['a', 'b', 'c']
d = dict.fromkeys(keys)   # {'a': None, 'b': None, 'c': None}
d = dict.fromkeys(keys, 0)  # {'a': 0, 'b': 0, 'c': 0}
```

### 6.4 defaultdict

```python
from collections import defaultdict

# 默认值字典
d = defaultdict(int)      # 默认值为0
d['a'] += 1               # 不需要先初始化
print(d['b'])             # 0

d = defaultdict(list)     # 默认值为[]
d['a'].append(1)

d = defaultdict(set)      # 默认值为set()
d['a'].add(1)

# 普通字典需要先判断
d = {}
if 'a' not in d:
    d['a'] = []
d['a'].append(1)
```

### 6.5 Counter

```python
from collections import Counter

# 计数器
text = "hello world"
counter = Counter(text)
print(counter)            # Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})

# 常用操作
lst = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
counter = Counter(lst)
print(counter.most_common(2))  # [(4, 4), (3, 3)] 最常见的2个
print(counter[2])              # 2
print(list(counter.elements()))  # [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]

# 运算
c1 = Counter('aabbcc')
c2 = Counter('aabbb')
print(c1 + c2)            # Counter({'a': 4, 'b': 5, 'c': 2})
print(c1 - c2)            # Counter({'c': 2})
```

---

## 7. 集合操作

### 7.1 基本操作

```python
# 创建集合
s = {1, 2, 3}
s = set([1, 2, 2, 3, 3])  # {1, 2, 3}，自动去重
s = set()                 # 空集合

# 添加/删除
s.add(4)                  # 添加元素
s.remove(3)               # 删除元素，不存在报错
s.discard(3)              # 删除元素，不存在不报错
s.pop()                   # 删除并返回任意元素
s.clear()                 # 清空

# 成员判断 (O(1)时间复杂度)
print(1 in s)             # True
```

### 7.2 集合运算

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

# 并集
print(a | b)              # {1, 2, 3, 4, 5, 6}
print(a.union(b))

# 交集
print(a & b)              # {3, 4}
print(a.intersection(b))

# 差集
print(a - b)              # {1, 2}
print(a.difference(b))

# 对称差集 (异或)
print(a ^ b)              # {1, 2, 5, 6}
print(a.symmetric_difference(b))

# 子集/超集
print({1, 2}.issubset(a))         # True
print(a.issuperset({1, 2}))       # True
print(a.isdisjoint({5, 6}))       # False (有交集)
```

---

## 8. NumPy/PyTorch常用操作

### 8.1 数组创建

```python
import numpy as np
import torch

# 创建数组
a = np.array([1, 2, 3])
a = np.zeros((3, 4))
a = np.ones((3, 4))
a = np.empty((3, 4))
a = np.arange(0, 10, 2)           # [0, 2, 4, 6, 8]
a = np.linspace(0, 1, 5)          # [0, 0.25, 0.5, 0.75, 1]
a = np.random.randn(3, 4)         # 正态分布

# PyTorch
t = torch.tensor([1, 2, 3])
t = torch.zeros(3, 4)
t = torch.ones(3, 4)
t = torch.randn(3, 4)
```

### 8.2 cumsum - 累积求和

```python
import numpy as np
import torch

# NumPy
a = np.array([1, 2, 3, 4, 5])
print(np.cumsum(a))               # [1, 3, 6, 10, 15]

# 沿轴累积
a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.cumsum(a, axis=0))       # 沿行累积
# [[1, 2, 3],
#  [5, 7, 9]]
print(np.cumsum(a, axis=1))       # 沿列累积
# [[1, 3, 6],
#  [4, 9, 15]]

# PyTorch
t = torch.tensor([1, 2, 3, 4, 5])
print(torch.cumsum(t, dim=0))     # tensor([1, 3, 6, 10, 15])

# vLLM中的应用: 计算序列长度
seq_lens = [3, 5, 2, 4]
cu_seq_lens = [0] + list(np.cumsum(seq_lens))
print(cu_seq_lens)                # [0, 3, 8, 10, 14]
```

### 8.3 diff - 差分

```python
import numpy as np

# NumPy
a = np.array([1, 3, 6, 10, 15])
print(np.diff(a))                 # [2, 3, 4, 5]

# 多阶差分
print(np.diff(a, n=2))            # [1, 1, 1]

# 沿轴差分
a = np.array([[1, 3, 6], [4, 7, 11]])
print(np.diff(a, axis=1))         # 沿列差分
# [[2, 3],
#  [3, 4]]

# vLLM中的应用: 从累积长度恢复原始长度
cu_seq_lens = [0, 3, 8, 10, 14]
seq_lens = np.diff(cu_seq_lens)
print(seq_lens)                   # [3, 5, 2, 4]
```

### 8.4 其他常用操作

```python
import numpy as np
import torch

# reshape
a = np.arange(12)
b = a.reshape(3, 4)
b = a.reshape(-1, 4)              # -1表示自动推断

# transpose
a = np.arange(12).reshape(3, 4)
b = a.T                           # 转置
b = np.transpose(a, (1, 0))       # 指定轴顺序

# squeeze/unsqueeze
a = np.array([[1], [2], [3]])
b = np.squeeze(a)                 # [1, 2, 3] 去除长度为1的维度

# PyTorch
t = torch.tensor([[1], [2], [3]])
t = t.squeeze()                   # [1, 2, 3]
t = t.unsqueeze(0)                # [1, 3] 增加维度
t = t.unsqueeze(-1)               # [3, 1]

# flatten
a = np.arange(12).reshape(3, 4)
b = a.flatten()                   # 展平为一维

# concatenate
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.concatenate([a, b])        # [1, 2, 3, 4, 5, 6]

# stack
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.stack([a, b])              # [[1, 2, 3], [4, 5, 6]]
c = np.stack([a, b], axis=1)      # [[1, 4], [2, 5], [3, 6]]

# split
a = np.arange(12)
b = np.split(a, 3)                # 分成3份
b = np.array_split(a, 5)          # 不均等分割

# where
a = np.array([1, 2, 3, 4, 5])
b = np.where(a > 3, a, 0)         # [0, 0, 0, 4, 5]
idx = np.where(a > 3)             # 返回索引

# argmax/argmin
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(np.argmax(a))               # 5
print(np.argmin(a))               # 1

# argsort
idx = np.argsort(a)               # 返回排序后的索引
print(a[idx])                     # 排序后的数组

# unique
a = np.array([1, 2, 2, 3, 3, 3])
print(np.unique(a))               # [1, 2, 3]
values, counts = np.unique(a, return_counts=True)

# bincount
a = np.array([0, 1, 1, 2, 2, 2, 3])
print(np.bincount(a))             # [1, 2, 3, 1] 每个值出现的次数
```

### 8.5 PyTorch特有操作

```python
import torch

# 设备管理
t = torch.tensor([1, 2, 3])
t = t.to('cuda')                  # 移动到GPU
t = t.to('cpu')                   # 移动到CPU
t = t.cuda()                      # 简写
t = t.cpu()

# 数据类型
t = torch.tensor([1, 2, 3], dtype=torch.float32)
t = t.float()                     # 转为float32
t = t.half()                      # 转为float16
t = t.long()                      # 转为int64
t = t.int()                       # 转为int32

# 梯度
t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
t.requires_grad_(False)           # 关闭梯度

# clone vs detach
a = torch.tensor([1.0, 2.0], requires_grad=True)
b = a.clone()                     # 拷贝，保留梯度
c = a.detach()                    # 拷贝，不保留梯度
d = a.detach().clone()            # 推荐用法

# gather/scatter
# gather: 按索引收集
t = torch.tensor([[1, 2], [3, 4], [5, 6]])
idx = torch.tensor([[0, 1], [1, 0]])
result = torch.gather(t, 0, idx)  # 按行收集

# scatter: 按索引填充
t = torch.zeros(2, 3)
idx = torch.tensor([[0, 1, 2], [2, 1, 0]])
src = torch.tensor([[1, 2, 3], [4, 5, 6]])
t.scatter_(1, idx, src)           # 按列填充

# index_select
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
idx = torch.tensor([0, 2])
result = torch.index_select(t, 0, idx)  # 选择第0行和第2行

# masked_select
t = torch.tensor([1, 2, 3, 4, 5, 6])
mask = t > 3
result = torch.masked_select(t, mask)   # [4, 5, 6]
```

---

## 9. 文件与路径操作

### 9.1 路径操作 (pathlib)

```python
from pathlib import Path

# 创建路径对象
p = Path('/home/user/documents/file.txt')
p = Path('data') / 'train' / 'images'  # 路径拼接

# 路径属性
print(p.name)              # 'file.txt'
print(p.stem)              # 'file'
print(p.suffix)            # '.txt'
print(p.parent)            # Path('/home/user/documents')
print(p.parts)             # ('/', 'home', 'user', 'documents', 'file.txt')

# 路径操作
p.exists()                 # 是否存在
p.is_file()                # 是否是文件
p.is_dir()                 # 是否是目录
p.mkdir(parents=True, exist_ok=True)  # 创建目录
p.unlink()                 # 删除文件
p.rename('new_name.txt')   # 重命名

# 遍历目录
for f in Path('.').iterdir():
    print(f)

for f in Path('.').glob('*.py'):
    print(f)

for f in Path('.').rglob('*.py'):  # 递归
    print(f)
```

### 9.2 文件读写

```python
# 读取文件
with open('file.txt', 'r') as f:
    content = f.read()              # 读取全部
    lines = f.readlines()           # 读取所有行
    line = f.readline()             # 读取一行

# 写入文件
with open('file.txt', 'w') as f:
    f.write('hello\n')
    f.writelines(['line1\n', 'line2\n'])

# 追加
with open('file.txt', 'a') as f:
    f.write('append\n')

# 读写二进制
with open('file.bin', 'rb') as f:
    data = f.read()

with open('file.bin', 'wb') as f:
    f.write(b'\x00\x01\x02')

# JSON
import json
with open('data.json', 'r') as f:
    data = json.load(f)

with open('data.json', 'w') as f:
    json.dump(data, f, indent=2)

# Pickle
import pickle
with open('data.pkl', 'wb') as f:
    pickle.dump(obj, f)

with open('data.pkl', 'rb') as f:
    obj = pickle.load(f)
```

---

## 10. 异常处理

### 10.1 基本语法

```python
# 基本形式
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unknown error: {e}")
else:
    print("No error")       # 没有异常时执行
finally:
    print("Always execute")  # 总是执行

# 抛出异常
raise ValueError("Invalid value")

# 自定义异常
class MyError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

raise MyError("Something went wrong")
```

### 10.2 上下文管理器与异常

```python
# 使用with自动处理异常
with open('file.txt', 'r') as f:
    content = f.read()
# 文件自动关闭，即使发生异常

# 自定义上下文管理器
from contextlib import contextmanager

@contextmanager
def my_context():
    print("Enter")
    try:
        yield "resource"
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        print("Exit")

with my_context() as r:
    print(r)
```

### 10.3 常见异常类型

```python
# 常见异常
ValueError          # 值错误
TypeError           # 类型错误
KeyError            # 键不存在
IndexError          # 索引越界
FileNotFoundError   # 文件不存在
ZeroDivisionError   # 除零错误
AttributeError      # 属性不存在
RuntimeError        # 运行时错误
NotImplementedError # 未实现
StopIteration       # 迭代结束
```

---

## 11. C++ vs Python 快速对照表

| 操作 | C++ | Python |
|------|-----|--------|
| 变量声明 | `int x = 10;` | `x = 10` |
| 常量 | `const int x = 10;` | 无内置常量，约定全大写 |
| 布尔值 | `true`, `false` | `True`, `False` |
| 空值 | `nullptr` | `None` |
| 逻辑运算 | `&&`, `\|\|`, `!` | `and`, `or`, `not` |
| 自增/自减 | `i++`, `i--` | `i += 1`, `i -= 1` |
| 三元运算 | `a ? b : c` | `b if a else c` |
| 循环 | `for (int i = 0; i < n; i++)` | `for i in range(n)` |
| 遍历容器 | `for (auto& x : vec)` | `for x in lst` |
| 字符串 | `std::string s = "hello";` | `s = "hello"` |
| 数组 | `std::vector<int> v = {1, 2, 3};` | `lst = [1, 2, 3]` |
| 映射 | `std::map<string, int> m;` | `d = {}` |
| 集合 | `std::set<int> s;` | `s = set()` |
| 函数定义 | `int foo(int x) { ... }` | `def foo(x) -> int: ...` |
| Lambda | `[](int x) { return x*2; }` | `lambda x: x*2` |
| 类定义 | `class Foo { ... };` | `class Foo: ...` |
| 继承 | `class Bar : public Foo { };` | `class Bar(Foo): ...` |
| 异常 | `try { ... } catch (...) { }` | `try: ... except: ...` |
| 文件读取 | `std::ifstream f("file");` | `with open('file') as f:` |

---

## 12. vLLM代码中的常见模式

### 12.1 类型检查

```python
# isinstance
if isinstance(x, torch.Tensor):
    x = x.numpy()

# 多类型检查
if isinstance(x, (int, float)):
    print("numeric")

# type() vs isinstance()
# type() 不考虑继承
# isinstance() 考虑继承
```

### 12.2 可变参数

```python
# *args: 位置参数元组
def foo(*args):
    for arg in args:
        print(arg)

foo(1, 2, 3)  # args = (1, 2, 3)

# **kwargs: 关键字参数字典
def bar(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

bar(a=1, b=2)  # kwargs = {'a': 1, 'b': 2}

# 混合使用
def func(a, b, *args, **kwargs):
    pass

# 解包
args = [1, 2, 3]
foo(*args)  # 等价于 foo(1, 2, 3)

kwargs = {'a': 1, 'b': 2}
bar(**kwargs)  # 等价于 bar(a=1, b=2)
```

### 12.3 字符串前缀

```python
# f-string: 格式化字符串
name = "Alice"
print(f"Hello, {name}!")

# r-string: 原始字符串（不转义）
path = r"C:\Users\name\file.txt"

# b-string: 字节字符串
data = b"hello"

# u-string: Unicode字符串（Python 3默认）
text = u"你好"

# 组合使用
regex = rf"\d+{name}"
```

### 12.4 下划线约定

```python
# 单下划线前缀: 内部使用（约定，不强制）
class MyClass:
    def _internal_method(self):
        pass

# 双下划线前缀: 名称改写（防止命名冲突）
class MyClass:
    def __private(self):
        pass
    # 实际名称: _MyClass__private

# 双下划线前后缀: 魔术方法
class MyClass:
    def __init__(self):
        pass
    def __str__(self):
        return "MyClass"

# 单下划线: 忽略变量
for _ in range(10):
    print("hello")

a, _, c = (1, 2, 3)  # 忽略中间值
```

---

## 总结

本文档涵盖了C++转Python需要掌握的核心语法差异和常用操作：

| 类别 | 重要程度 | 说明 |
|------|----------|------|
| 列表切片 | ★★★★★ | Python最强大的特性之一 |
| 列表推导式 | ★★★★★ | Pythonic代码必备 |
| 字典操作 | ★★★★★ | Python最常用的数据结构 |
| 内置函数 | ★★★★★ | enumerate/zip/map/filter |
| NumPy/PyTorch | ★★★★★ | AI开发必备 |
| 异常处理 | ★★★★☆ | 与C++差异较大 |
| 文件操作 | ★★★★☆ | with语句很方便 |
| 集合操作 | ★★★☆☆ | 类似C++ set |

掌握这些内容，你就可以流畅地阅读和编写vLLM代码了！
