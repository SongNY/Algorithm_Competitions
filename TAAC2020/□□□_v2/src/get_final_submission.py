
import os

import numpy as np
import pandas as pd


output_dict = {}
for root, _, filename_list in os.walk("../output/"):
    for filename in filename_list:
        if not filename.endswith(".npy"):
            continue
        output = np.load(root + "/" + filename).astype(np.float32)
        output_dict[filename] = output


all_output = sum(output_dict.values())
pred_age = all_output.reshape(-1, 2, 10).sum(1).argmax(1) + 1
pred_gender = all_output.reshape(-1, 2, 10).sum(2).argmax(1) + 1
df = pd.DataFrame({
    "user_id": range(3000001, 4000001),
    "predicted_age": pred_age,
    "predicted_gender": pred_gender
})
df.to_csv("../submission.csv", index=False)


single_output = output_dict["output_trlstm1_1.npy"]
pred_age = single_output.reshape(-1, 2, 10).sum(1).argmax(1) + 1
pred_gender = single_output.reshape(-1, 2, 10).sum(2).argmax(1) + 1
df = pd.DataFrame({
    "user_id": range(3000001, 4000001),
    "predicted_age": pred_age,
    "predicted_gender": pred_gender
})
df.to_csv("../single_model.csv", index=False)
