# work3 分类与聚类  
## `https://www.kaggle.com/c/predict-west-nile-virus/data`数据集  
## 1.数据分析处理  
   读取数据，进行分析后。首先对数据集中重复数据进行清理；对weather表中缺失值进行处理。对各表中日期时间格式进行规范。  
   train表中重复项进行去除  
   train表中蚊子物种进行拆分  
   根据扑捉站到气象站的距离计算，计算每个扑捉站的距离权重，通过接近度加权天气数据  
   目标干预的效果是在时间和距离两个维度上衰减，关心的参考位置是Trap，计算时间和距离两个维度的喷雾衰减。  
   将train，weather，spray三个表合并成一个表，处理后的数据表存储为csv文件，以供分类和聚类使用。  
## 2.分类聚类  
   分类模型使用`随机森林`  
   ![](https://github.com/michaellee666/work3/blob/master/RandomForest.png)  
   `支持向量机Support Vector Machine`
   ![](https://github.com/michaellee666/work3/blob/master/SVM.png)  
