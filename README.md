# transE
#### 关于transE：

1、论文原文：[Translating embeddings for modeling multi-relational data](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela)

2、[我的一篇笔记](https://blog.csdn.net/shunaoxi2313/article/details/89766467)

#### 1 训练数据

FB15k. 

其它数据（如WorldNet等），见(https://github.com/thunlp/KB2E)

#### 2. 训练transE

- Tbatch更新：在update_embeddings函数中有一个deepcopy操作，目的就是为了批量更新。这是ML中mini-batch SGD的一个通用的训练知识，在实际编码时很容易忽略。
- 两次更新：update_embeddings函数中，要对correct triplet和corrupted triplet都进行更新。虽然写作$(h,l,t)$和$(h',l,t')$，但两个三元组只有一个entity不同（前面说了，不同时替换头尾实体），所以在每步更新时重叠的实体要更新两次（和更新relation一样），否则就会导致后一次更新覆盖前一次。
- 关于L1范数的求导方法：参考了[刘知远组实现](https://github.com/thunlp/KB2E)中的实现，是先对L2范数求导，逐元素判断正负，为正赋值为1，负则为-1。
- 超参选择：对FB15k数据集，epoch选了1000（其实不需要这么大，后面就没什么提高了），nbatches选了400（训练最快），embedding_dim=50, learning_rate=0.01, margin=1。


 #### 3. 测试
- isFit参数：区分raw和filter。filter会非常慢。
