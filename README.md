# cpd-2020-summer
THUEE Computer Program Design(2) Project (2020 Summer)

清华大学电子工程系 计算机程序设计基础 小学期（2022夏季学期）

## TODO
- [ ] 数据集下载脚本
- [ ] 处理数据
- [ ] 配环境
- [ ] 搭网络
- [ ] 写训练模块
- [ ] 写评估模块
- [ ] 图形界面
- [ ] 人脸检测+标注性别

## 介绍

### 题目

来源：[知乎：如何评价清华大学电子系大一暑假小学期总共 9 个学时的 Python 课程大作业难度？](https://www.zhihu.com/question/471999381)

<img src="./readmeimg/project_demand.jpg" alt="project_demand"  />

### 分析

本质上是一个性别二分类的任务。

### 数据集

- 采用[LFW](http://vis-www.cs.umass.edu/lfw/#download)数据集

- 数据采用"[All images as gzipped tar file](http://vis-www.cs.umass.edu/lfw/lfw.tgz)"

- 标签采用"[Manually verified LFW gender labels, from Mahmoud Afifi, Abdelrahman Abdelhamed.](https://www.dropbox.com/sh/l3ezp9qyy5hid80/AAAjK6HdDScd_1rXASlsmELla?dl=0)"

    - [female_names.txt](https://www.dropbox.com/sh/l3ezp9qyy5hid80/AAA__sZZKZIpic6NeYqUyEc3a/female_names.txt)
    - [male_names.txt](https://www.dropbox.com/sh/l3ezp9qyy5hid80/AAAjK6HdDScd_1rXASlsmELla?dl=0&preview=female_names.txt)


## 环境信息

```
python==3.7
torch==1.8.0
```

## 运行

### 环境

```bash
pip install -r requirements.txt
```

### 数据

```bash
bash init_data.sh
```

### 训练

```bash
python train.py
```

### 评估及测试

```bash
python eval.py
```