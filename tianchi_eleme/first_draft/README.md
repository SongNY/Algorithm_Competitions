### 解决方案

#### 数据预处理

- 采用天池论坛上开源的baseline
- 屏蔽训练集每个波次的50%行为 对于只有4次行为屏蔽其后3次行为
- 屏蔽行为每个波次第一个行为标签为1 其余为0 并打乱训练集

#### 特征工程

- 基本特征
  - 波次 wave_index 
  - 行动种类 action_type 
  - 待送单数 unfinished_num_sum 
  - 前一行动种类 last_action_type
  - 天气 weather_grade 分类特征
- 距离相关特征
  - 相对位置 delta_lng;delta_lat
  - 相对位置绝对值 delta_abs_lng;delta_abs_lat
  - 相对距离 delta_abs_lat
  - 高德距离 grid_distance
  - 是否是所有候选行为中最小的高德距离 latest_grid
  - 是否是边界点 lat_max;lat_min;lng_min;lng_max
- 时间相关特征
  - 取单时间-当前时间 delta_pick_time
  - 承诺送达时间-当前时间 delta_deliver_time
  - 是否是所有候选行为中最小的delta_deliver_time latest_deliver
  - 当前时间小时current_hour
  - 当前时间小时分桶 current_hour_bin (111213中午;171819晚上;其他) 分类特征
- 骑手信息特征
  - level;speed;max_loady

#### 模型

- 预测行为
  - 采用预处理后得到的训练集以及特征工程中的特征
  - 使用lightgbm进行分类预测
  - 对testA进行预测后并入训练集
  - 使用train+testA训练最终lightgbm模型 对testB进行预测
- 预测时间 
  - 取训练集中标签为1作为预测时间的训练集 
  - 采用上述全部特征对 expect_time-current_time 使用lightgbm进行回归

### 代码运行说明

- sh main.sh

### 补充说明
- 数据预处理以及特征工程前半部分是采用天池论坛 第一次打比赛啊开源的代码进行修改