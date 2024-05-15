# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2023-6-16
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2023 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
from tqdm import tqdm
import numpy as np
import pandas as pd


data_dir = 'datasets/NUS-WIDE'
save_dir = 'data/nuswide'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_img_path = os.path.join(data_dir, 'ImageList/TrainImagelist.txt')
test_img_path = os.path.join(data_dir, 'ImageList/TestImagelist.txt')
train_tags1k_path = os.path.join(data_dir, 'NUS_WID_Tags/Train_Tags1k.dat')
test_tags1k_path = os.path.join(data_dir, 'NUS_WID_Tags/Test_Tags1k.dat')
tags81_path = os.path.join(data_dir, 'Concepts81.txt')
tags1k_path = os.path.join(data_dir, 'NUS_WID_Tags/TagList1k.txt')
tags81_dir = os.path.join(data_dir, 'Groundtruth/TrainTestLabels')

train_img_list = [t.strip().replace('\\', '/') for t in open(train_img_path)]
test_img_list = [t.strip().replace('\\', '/') for t in open(test_img_path)]
tags81 = [t.strip() for t in open(tags81_path)]
tags1k = [t.strip() for t in open(tags1k_path)]
train_tags1k_mask = np.loadtxt(train_tags1k_path)
test_tags1k_mask = np.loadtxt(test_tags1k_path)

print('train images number: ', len(train_img_list))
print('test images number: ', len(test_img_list))

unseen_tags = tags81
seen_tags = [t for t in tags1k if t not in unseen_tags]
all_tags = seen_tags + unseen_tags
repead_tagsId = [i for i, t in enumerate(tags1k) if t in unseen_tags]

print('seen labels number: ', len(seen_tags))
print('unseen labels number: ', len(unseen_tags))
print('total labels number: ', len(all_tags))

train_tags1k_mask = np.delete(train_tags1k_mask, repead_tagsId, axis=1)
test_tags1k_mask = np.delete(test_tags1k_mask, repead_tagsId, axis=1)

test_tags81_mask = []
for tag in tags81:
    test_path = os.path.join(tags81_dir, 'Labels_{}_Test.txt'.format(tag))
    test_tags81_mask.append([int(t.strip()) for t in open(test_path)])
    
test_tags81_mask = np.array(test_tags81_mask).transpose()

train_data = []
for img_name, mask in tqdm(zip(train_img_list, train_tags1k_mask)):
    img_path = os.path.join(data_dir, 'Flickr', img_name)
    tags = [seen_tags[i] for i, t in enumerate(mask) if t == 1]
    if len(tags) > 0:
        train_data.append([img_path, '|'.join(tags)])
train_data = pd.DataFrame(train_data, columns=['image_path', 'posi_labels'])
train_data['nega_labels'] = np.nan
    
test_data = []
for img_name, tags1k_mask, tags81_mask in tqdm(zip(test_img_list, test_tags1k_mask, test_tags81_mask)):
    img_path = os.path.join(data_dir, 'Flickr', img_name)
    tags1k = [seen_tags[i] for i, t in enumerate(tags1k_mask) if t == 1]
    tags81 = [unseen_tags[i] for i, t in enumerate(tags81_mask) if t == 1]
    if len(tags81) > 0 or len(tags1k) > 0:
        test_data.append([img_path, '|'.join(tags81), '|'.join(tags1k)])
test_data = pd.DataFrame(test_data, columns=['image_path', 'unseen_posi_labels', 'seen_posi_labels'])
test_data['unseen_nega_labels'] = np.nan
test_data['seen_nega_labels'] = np.nan

train_data.to_csv(os.path.join(save_dir, 'train.txt'), header=True, index=False)
test_data.to_csv(os.path.join(save_dir, 'test.txt'), header=True, index=False)

with open(os.path.join(save_dir, 'label_seen.txt'), 'w') as fw:
    fw.writelines(['{}\n'.format(t) for t in seen_tags])
with open(os.path.join(save_dir, 'label_unseen.txt'), 'w') as fw:
    fw.writelines(['{}\n'.format(t) for t in unseen_tags])

