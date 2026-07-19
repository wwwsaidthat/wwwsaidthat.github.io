# 015-Deep Graph Infomax

现在对于一个图$\mathcal{G}$ 

​	其中所有的节点特征表示构成的结合$\mathcal{X} = \{x_1,x_2,...,x_N\}$

​	其中所有的边，也就是节点之间的联系$A\in\mathbb{R}^{N\times N}$

现在要训练一个编码器${E}:\mathbb{R}^{N\times F}\times\mathbb{R}^{N\times N}\rightarrow\mathbb{R}^{N\times F'}$

$E(A,X) = H = \{h_1,h_2,...,h_N\}$

其中$h_i$刻画的是以节点 i 为中心的局部子图区域,而不只是这个节点的原始特征，称为称作**局部区块表征**



出发点是想要用节点的局部特征，捕获到整张图的全部信息，全局的信息用一个全局摘要向量s来表示
$$
s = R(E(X,A))
$$
