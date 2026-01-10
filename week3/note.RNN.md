# week3: RNN

## 1. n-gram Language Models

​	n-gram语言模型采用滑动窗口来统计窗口n内的单次出现频率，并近似认为统计频率就是下一个词不同可能性的出现概率。例如"students opened their <u>books</u>"中的"books"在"students opened their"后出现了100次，而“students opened their”短语总共出现了10000次，那么n-gram模型在做推理预测时"books"的概率就是1%。

​	这样做会带来问题：

- 无法预测到训练集中从未出现过的单词组合
- 若续写的短句之前没有在训练集出现过，甚至根本无法预测下一个词。这时减小n的大小可能是一种规避的办法
- n-gram统计模型模型成本较高。增加n或者语料类型，可以增强模型表现，但都会显著增加模型大小。减小n，吐字语义会更加混乱无意。

> [!NOTE]
>
> Y. Bengio, et al. (2000/2003): A Neural Probabilistic Language Model
>
> 这篇论文中，对n-gram模型进行了改进。解决了单词出现频率稀疏的问题，并且不再需要存储所有的n-grams。但它仍然使用固定尺寸的窗口，并且窗口内的每个单词，使用的都是不一样的权重，缺乏对称性，并且这导致窗口并不能开的很大，无法处理任意长度的输入。



## 2. Evaluating Language Models（模型评估）

​	一般模型评估方式是使用困惑度（perplexity），计算方式如下：
$$
perplexity=\prod_{t-1}^{T} (\frac{1}{P_{LM}(x^{(t+1)}|x^{(t)},...,x^{(1)})})^{1/T} \label{1.1} \tag{1.1}
$$
​	它来源于交叉熵损失函数：

![image-20260110111907948](C:\Users\l00855193\AppData\Roaming\Typora\typora-user-images\image-20260110111907948.png)

​	比较奇怪为什么这里的交叉熵并没有做“交叉”，而只是计算了信息量，一般意义上的交叉熵：
$$
H(P,Q)=-\sum_i p(x_i)log(q(x_i)) \label{1.2} \tag{1.2}
$$
​	式$\ref{1.1}$的计算结果越大，说明模型能力越差，这与交叉熵的性质是吻合的。

## 3. Recurrent Neural Networks (RNN)

​	RNN采用了完全一致的权重来进行推理，每一次推理需要保存的只有中间隐藏层状态，如下图所示

![image-20260110114857785](C:\Users\l00855193\AppData\Roaming\Typora\typora-user-images\image-20260110114857785.png)

​	RNN的优点主要是：

- 可以处理任意长度输入
- 可以通过隐藏层状态保留之前推理的信息
- 模型大小不随输入变化，因为W权重在推理时是不变的
- 采用不变权重，有对称性

​	而缺点主要是：

- 迭代计算在训练时会比较慢
- 随着推理步数增加，过往信息会被稀释
