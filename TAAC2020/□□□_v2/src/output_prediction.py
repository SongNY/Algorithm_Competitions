
import os

import numpy as np
import pandas as pd
import torch

# jinzhen的结果
jinzhen_output_list = []
for filename in os.listdir("../cache/output/"):
    if filename.startswith("output_"):
        output = torch.load("../cache/output/" + filename)
        jinzhen_output_list.append(output)
jinzhen_output = sum(jinzhen_output_list).float()

# 宁宇的预测结果
ningyu_output = np.load("../tmp_data/model24.npy")
ningyu_output = torch.from_numpy(ningyu_output).float()

# 合并
final_output = jinzhen_output + ningyu_output
# 概率聚合后进行预测
pred_age = final_output.view(-1, 2, 10).sum(1).max(1)[1] + 1
pred_gender = final_output.view(-1, 2, 10).sum(2).max(1)[1] + 1
df = pd.DataFrame({
    "user_id": range(3000001, 4000001),
    "predicted_age": pred_age,
    "predicted_gender": pred_gender
})
df.to_csv("../submission.csv", index=False)
