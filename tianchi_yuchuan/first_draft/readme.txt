队伍名称:ggplot
ggplot(您现在是一个人参加比赛)
初赛排名/成绩:42/0.88901
队长联系方式(手机/钉钉):18250891997


模型思路
1.构造特征
 a.统计特征
   对于每条渔船的各个变量构造统计特征包括
   ['max','min','median','mean','std','skew','sum',quantile_25,quantile_75,'kurt','mode']
   ['diff_']
 b.map特征
   将每条渔船的数据 从xmax到xmin均分三份 从ymax到ymin均分3份
   得到3*3=9份 对于每份分别提取 num_count,is_point(有没有点存在01变量),v_mean,d_mean 构成4*9=36个变量
2.跑lgb用importnce对特征筛选得到62个特征
3.预测testA并进行标注合并train和testA再跑cv=5的lgb得到最终五个模型


模型
lightgbm


代码加载数据的路径
~/data/hy_round1_testB_20200221/


预测结果输出位置
~/result.csv'


环境
python                    3.7.5
lightgbm                  2.2.3            py37he6710b0_0    defaults
pandas                    0.25.2           py37he6710b0_0    defaults
numpy                     1.17.2           py37haad9e8e_0    defaults
