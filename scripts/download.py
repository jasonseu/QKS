# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2023-6-26
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2023 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import math
import random
import urllib3
import argparse
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
from multiprocessing import Pool


http = urllib3.PoolManager()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Downloads all image files contained in dataset, if an image fails to download lets skip it.
root_dir = 'datasets/OpenImages'
version = '2018_04'
browser_headers = [
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:44.0) Gecko/20100101 Firefox/44.0"}
]


def job_handler(image_infos, pid, mode):
    pid = str(pid).zfill(2)
    print('Process [{}] has started!'.format(pid))
    image_dir = os.path.join(root_dir, mode)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if mode == 'train':
        image_dir = os.path.join(image_dir, pid)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
    failed_image_infos = []
    for i, image_info in enumerate(image_infos[80000:]):
        image_url, image_name, _ = image_info
        image_path = os.path.join(image_dir, '{}.jpg'.format(image_name))
        if not os.path.exists(image_path):
            cnt = 0
            while cnt < 3:
                try:
                    response = http.request('GET', image_url, headers=random.choice(browser_headers))
                    image_data = response.data
                    image = Image.open(BytesIO(image_data))
                    (w, h) = image.size
                    small_side = min(w, h)
                    large_side = max(w, h)
                    ratio = small_side / 256.0
                    new_size = math.ceil(large_side / ratio)
                    image.thumbnail((new_size, new_size))
                    image.save(image_path, 'JPEG')
                    break
                except:
                    cnt += 1
            if cnt >= 3:
                failed_image_infos.append(image_info)
        if (i + 1) % 200 == 0:
            print("[{}] {}/{} images have finished, {} images have failed!"
                  .format(pid, i + 1, len(image_infos), len(failed_image_infos)))
    print('Process [{}] has finished!'.format(pid))

def main(image_infos, args):
    inds = np.linspace(0, len(image_infos), args.num_workers + 1)
    inds = [int(t) for t in inds]
    inds = list(zip(inds[:-1], inds[1:]))
    slices = [image_infos[l:r] for l, r in inds]
    jobs = []
    pool = Pool(args.num_workers)
    for i in range(args.num_workers):
        jobs.append(pool.apply_async(job_handler, args=(slices[i], i, args.mode)))
    pool.close()
    pool.join()
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validation', 'test'])
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()
    print('load data')
    df_image = pd.read_csv(os.path.join(root_dir, version, args.mode, '{}-images-with-labels-with-rotation.csv'.format(args.mode)))
    image_infos = list(zip(df_image['OriginalURL'].values, df_image['ImageID'], df_image['OriginalMD5']))
    res = main(image_infos, args)
