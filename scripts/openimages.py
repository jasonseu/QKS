# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2023-6-27
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2023 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import json
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool


root_dir = 'datasets/OpenImages'
save_dir = 'data/openimages'
version = '2018_04'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
train_image_path = os.path.join(root_dir, 'train')
train_annotation_path = os.path.join(root_dir, version, 'train/train-annotations-human-imagelabels.csv')
test_image_path = os.path.join(root_dir, 'test')
test_annotation_path = os.path.join(root_dir, version, 'test/test-annotations-human-imagelabels.csv')
label_desc_path = os.path.join(root_dir, version, 'class-descriptions.csv')
trainable_label_path = os.path.join(root_dir, version, 'classes-trainable.txt')
top_idx_path = os.path.join(root_dir, version, 'top_400_unseen.csv')
unseen_label_path = os.path.join(root_dir, version, 'unseen_labels.pkl')


labelmap = {}
for line in open(label_desc_path):
    t = [word.strip(' "\n') for word in line.split(',', 1)]
    labelmap[t[0]] = t[1]

seen_labels = [labelmap[t.rstrip()] for t in open(trainable_label_path)]
seen_labels.sort()

unseen_labels_2594 = pickle.load(open(unseen_label_path, 'rb'))  # 2594 unseen labels
top_unseen_idx = pd.read_csv(top_idx_path, header=None).values[:, 0]
top_unseen_labels = [unseen_labels_2594[t] for t in top_unseen_idx]  # top 400 unseen labels
unseen_labels = [labelmap[t] for t in top_unseen_labels]
unseen_labels.sort()

with open(os.path.join(save_dir, 'label_seen.txt'), 'w') as fw:
    fw.writelines(['{}\n'.format(t) for t in seen_labels])

with open(os.path.join(save_dir, 'label_unseen.txt'), 'w') as fw:
    fw.writelines(['{}\n'.format(t) for t in unseen_labels])
    
my_seen_labels = [t.rstrip() for t in open('data/openimages/label_seen.txt')]
my_unseen_labels = [t.rstrip() for t in open('data/openimages/label_unseen.txt')]

# print(len(set(seen_labels)), len((my_seen_labels)))
# print(set(seen_labels) == set(my_seen_labels))
# print(set(seen_labels) - set(my_seen_labels))
# print(set(my_seen_labels) - set(seen_labels))

# print(len(set(unseen_labels)), len((my_unseen_labels)))
# print(set(unseen_labels) == set(my_unseen_labels))
# print(set(unseen_labels) - set(my_unseen_labels))
# print(set(my_unseen_labels) - set(unseen_labels))


def job_handler(annotations, pid):
    print('Process [{}] has started!'.format(pid))
    pid = str(pid).zfill(2)
    data = []
    for j, (image_id, group) in enumerate(annotations):
        for i in range(100):
            i = str(i).zfill(2)
            image_path = os.path.join(train_image_path, i, '{}.jpg'.format(image_id))
            if not os.path.exists(image_path):
                continue
            posi_labels, nega_labels = [], []
            for _, row in group.iterrows():
                label, confidence = labelmap.get(row['LabelName']), row['Confidence']
                if label in seen_labels:
                    if confidence == 1:
                        posi_labels.append(label)
                    else:
                        nega_labels.append(label)
            if len(set(posi_labels).intersection(set(nega_labels))) == 0:
                data.append([image_path, '|'.join(posi_labels), '|'.join(nega_labels)])
            break
        if (j + 1) % 1000 == 0:
            print("[{}] {}/{} samples have finished!".format(pid, j + 1, len(annotations)))
    print('Process [{}] has finished!'.format(pid))
    
    return data


train_annotation = pd.read_csv(train_annotation_path)
num_workers = 50
annotations = list(train_annotation.groupby('ImageID'))
inds = np.linspace(0, len(annotations), num_workers + 1)
inds = [int(t) for t in inds]
inds = list(zip(inds[:-1], inds[1:]))
slices = [annotations[i: j] for i, j in inds]
jobs = []
pool = Pool(num_workers)
for i in range(num_workers):
    jobs.append(pool.apply_async(job_handler, args=(slices[i], i)))
pool.close()
pool.join()

train_data = []
for job in jobs:
    train_data.extend(job.get())
train_data = pd.DataFrame(train_data, columns=['image_path', 'posi_labels', 'nega_labels'])
train_data.to_csv(os.path.join(save_dir, 'train.txt'), header=True, index=False)


test_annotation = pd.read_csv(test_annotation_path)
test_data = []
for image_id, group in tqdm(test_annotation.groupby('ImageID')):
    image_path = os.path.join(test_image_path, '{}.jpg'.format(image_id))
    if not os.path.exists(image_path):
        continue
    seen_posi_labels, seen_nega_labels = [], []
    unseen_posi_labels, unseen_nega_labels = [], []
    for i, row in group.iterrows():
        label, confidence = labelmap.get(row['LabelName']), row['Confidence']
        if label in seen_labels:
            if confidence == 1:
                seen_posi_labels.append(label)
            else:
                seen_nega_labels.append(label)
        elif label in unseen_labels:
            if confidence == 1:
                unseen_posi_labels.append(label)
            else:
                unseen_nega_labels.append(label)
    posi_labels = seen_posi_labels + unseen_posi_labels
    nega_labels = seen_nega_labels + unseen_nega_labels
    if len(set(posi_labels).intersection(set(nega_labels))) == 0:
        test_data.append([image_path, '|'.join(unseen_posi_labels), '|'.join(unseen_nega_labels),
                          '|'.join(seen_posi_labels), '|'.join(seen_nega_labels)])

test_data = pd.DataFrame(test_data, columns=['image_path', 'unseen_posi_labels', 'unseen_nega_labels',
                                             'seen_posi_labels', 'seen_nega_labels'])
test_data.to_csv(os.path.join(save_dir, 'test.txt'), header=True, index=False)
