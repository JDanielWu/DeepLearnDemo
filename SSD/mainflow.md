SSD整体流程框架，根据这个框架，更能梳理代码流程

st=>start: SSDNet
e=>end
op=>operation: p,loc,layers
op1=>operation: anchors
op2=>operation: g_p, g_loc
op3=>operation: loss
st->op->op1->op2->op3->e

<br/>1.p是预测类别，loc是预测位置，但这个位置是一个变换值，并不是一个具体坐标值  
</br>2.通过layers 获取anchors预测框的信息，然后进行encode得到预测框与真实框
之间的变换关系 g_p g_loc </br>3.这样就可以计算loss函数了。
其中loss函数 还根据正反样本进行了一定的比例筛选。

