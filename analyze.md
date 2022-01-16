* LACO_train_data_6_01:早停，6标签0-1，梯度裁剪_5
0.6845roberta_LACO.tsv
![](images\QQ截图20211019210453.png)
* LACO_train_data_6_01:早停，6标签0-1
0.6843
所以加上梯度裁剪效果要好一点
![](images\QQ截图20211019214953.png)
* 测试一下学习率衰减策略有没有效果？
* 
* 测试一下平滑标签有没有效果？\
0.6833roberta_LACO.tsv：早停，6标签0-1，梯度裁剪_5，平滑标签
标签平滑效果不好
![](images\QQ截图20211020095543.png)
* 0.6848roberta_LACO.tsv：早停，6标签0-1，梯度裁剪_5，学习率的调整warmup
![](images\QQ截图20211020112847.png)

* 采用回归方法

* 0.6892regression_robert.tsv:早停，梯度裁剪_5
![](images\QQ截图20211020204904.png)
* 0.6911regression_robert_0warmup.tsv:\
早停，梯度裁剪_5，学习率的调整warmup_0
![](images\QQ截图20211021143153.png)
* 0.6960regression_robert_fgm.tsv:\
早停，学习率的调整warmup_0，fgm
![](images\0.6960QQ截图20211024182635.png)
* 0.6997regression_robert_fgm_剧本id.tsv:\
早停，学习率的调整warmup_0,fgm,加上了剧本id
![](images\0.6997regression_robert_fgm_剧本id.png)
* 0.7002regression_roberta_fgm_剧本id1.tsv:\
早停，学习率的调整warmup_50,fgm,加上了剧本id
![](images\0.7002regression_roberta_fgm1.png)
* 0.7016regression_robert_fgm.tsv：\
早停，学习率的调整warmup_50,fgm,加上了剧本id,将长度小于50的句子拼接前3句
![](images\0.7016.png)
* 0.7019regression_robert_fgm.tsv：\
早停，学习率的调整warmup_50,fgm,加上了剧本id,将长度小于50的句子拼接前3句,将每句长度设置为110
![](images\0.7019.png)
* 0.7023regression_robert_fgm.tsv:\
早停，梯度裁剪_2,学习率的调整warmup_50,fgm,加上了剧本id,将长度小于50的句子拼接前3句,将每句长度设置为110
![](images\0.7023regression_robert_fgm.png)
* 0.7026regression_robert_fgm.tsv:\
早停，梯度裁剪_2,学习率的调整warmup_50,fgm,加上了剧本id,将长度小于70的句子拼接前3句,将每句长度设置为110
![](images\0.7026.png)
* 0.7037regression_robert_fgm.tsv:\
早停，梯度裁剪_2,学习率的调整warmup_50,fgm,加上了剧本id,\
将长度小于70的句子拼接前3句,按照剧本场次划分数据集,将每句长度设置为110
![](images\0.7037.png)
* 0.7038_更换warmup_分层权重衰减regression_robert_fgm.tsv\
早停，梯度裁剪_2,分层权重衰减,学习率的调整warmup_50,fgm,加上了剧本id,\
将长度小于70的句子拼接前3句,按照剧本场次划分数据集,将每句长度设置为110
![](images\0.7038.png)
* 0.7068regression_roformer_fgm_fold_ensemble.tsv：5折训练之后融合
* 0.7044regression_robert_fgm_fold_ensemble.tsv：5折训练之后融合

* 0.7039更改学习率4e-5regression_roformer_fgm.tsv:\
早停，梯度裁剪_2,学习率的调整warmup_50,fgm,加上了剧本id,\
将长度小于70的句子拼接前3句,按照剧本场次划分数据集,将每句长度设置为110
![](images\0.7039更改学习率4e-5regression_roformer_fgm.png)
* 0.6947regression_LACO.tsv:\
早停，梯度裁剪_5，学习率的调整warmup_0
![](images\QQ截图20211021194126.png)
* 0.6952regression_LACO.tsv：\
早停，梯度裁剪_5，学习率的调整warmup_0，去除小于0大于3
![](images\QQ截图20211021210043.png)

'0.7038_更换warmup_分层权重衰减regression_robert_fgm.tsv'
'0.7039更改学习率4e-5regression_roformer_fgm.tsv'融合后0.7070

'0.7032regression_macbert_fgm_fold_ensemble.tsv',
'0.7044regression_robert_fgm_fold_ensemble.tsv',
'0.7068regression_roformer_fgm_fold_ensemble.tsv'融合后0.7073

'0.7068regression_roformer_fgm_fold_ensemble.tsv',
'0.7044regression_robert_fgm_fold_ensemble.tsv',
'0.7070sub_data.tsv',
'0.7073sub_data.tsv'融合后0.7082

比赛总结：
    用了什么方法，看了什么论文，比赛结束了，top方案和你的差距在哪里