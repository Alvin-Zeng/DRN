import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


data = pickle.load(open("/home/xuhaoming/Projects/cvpr_baseline/projects/Charades_IoU/total_points.pkl", 'rb'))
data_first_list = torch.cat([item[0] for item in data], dim=0).data.cpu()
data_second_list = torch.cat([item[1] for item in data], dim=0).data.cpu()

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)

# plt.scatter(iou[1:], centerness, alpha=0.7)
plt.scatter(data_second_list, data_first_list, alpha=0.7)
# plt.plot(sorted_locations, sorted_centerness, label='caption_score', color='red', linewidth=2)
plt.xlim(0., 1.)
plt.ylim(0., 1.)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel('IoU', fontsize=10)
plt.ylabel('IoU_scores', fontsize=10)
plt.legend(fontsize=10)

plt.suptitle("IoU_Scores v.s IoU", fontsize=15, VA='center')
plt.savefig('iouScores_IoU_42.36.jpg')
plt.close()