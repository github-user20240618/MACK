import argparse
import torch
import clip
from PIL import Image
import json
import numpy as np
from pycocotools.coco import COCO
import os
import time

def i2t(_probs, return_ranks=False):
    npts = _probs.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(_probs[index])[::-1]
        
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
    
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(_probs, return_ranks=False):
    npts = _probs.shape[0]
    
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    
    # --> (5N(caption), N(image))
    _probs = _probs.T
    
    for index in range(npts):
        for i in range(5): 
            inds = np.argsort(_probs[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]
    
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def coco_5_fold_1K_Test(Sim): 
    results = []
    for i in range(5):
        start = time.time()
        sim = Sim[i * 1000:(i + 1) * 1000, i * 5000:(i + 1) * 5000]
        end = time.time()
        print("calculate similarity time: {}".format(end - start))
    
        R_i2t = i2t(sim)
        print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % R_i2t)
        R_t2i = t2i(sim)
        print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % R_t2i)
    
        r = R_i2t
        ri = R_t2i
        
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
        results += [list(r) + list(ri) + [ar, ari, rsum]]
    
    print("-----------------------------------")
    print("Mean metrics: ")
    mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
    print("rsum: %.1f" % (mean_metrics[12]))
    print("Average i2t Recall: %.1f" % mean_metrics[10])
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                mean_metrics[:5])
    print("Average t2i Recall: %.1f" % mean_metrics[11])
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                mean_metrics[5:10])
    
    return results



def json_save(content, jf_nm): 
    with open(jf_nm, 'w') as jf:
        json.dump( json.dumps(content), jf )


def json_load(jf_nm): 
    with open(jf_nm, 'r') as jf:
        content = json.loads( json.load(jf) )
    return content



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default='all') ## ["all", ]
    parser.add_argument('--data_name', default='f30k') ## ['f30k', 'coco', ]
    parser.add_argument('--sim_path', default='/raid/xxx/vsepp/43_temp') ## sim matrix path
    parser.add_argument('--sim_name', default='dataset_bbone_sim.npy') ## sim matrix name
    parser.add_argument('--save_path', default='/raid/xxx/vsepp/43_temp') ## json file save path
    parser.add_argument('--save_name', default='dataset_metric.json') ## json file save name
    opt = parser.parse_args()
    
    ## f30k test
    if opt.data_name == 'f30k': 
        f30k_sim = np.load( '{}'.format( os.path.join(opt.sim_path, opt.sim_name) ) )
        assert f30k_sim.shape == (1000, 5000)
        
        result_i2t = i2t(f30k_sim) ## (85.4, 97.1, 98.7, 1.0, 1.556)
        result_t2i = t2i(f30k_sim) ## (65.4, 87.2, 91.7, 1.0, 4.9678)
        
        ##result_i2t == (r1, r5, r10, medr, meanr)
        ##result_t2i == (r1, r5, r10, medr, meanr)
        
        print('{}_{}'.format( result_i2t, result_t2i )) ## (85.3, 97.0, 98.8, 1.0, 1.557)_(65.34, 87.24, 91.7, 1.0, 4.9702)
        
        content = {}
        
        content['1K_Test'] = {}
        content['1K_Test']['i2t_recall_at_1']  = result_i2t[0]
        content['1K_Test']['i2t_recall_at_5']  = result_i2t[1]
        content['1K_Test']['i2t_recall_at_10'] = result_i2t[2]
        content['1K_Test']['t2i_recall_at_1']  = result_t2i[0]
        content['1K_Test']['t2i_recall_at_5']  = result_t2i[1]
        content['1K_Test']['t2i_recall_at_10'] = result_t2i[2]
        content['1K_Test']['recall_sum'] = sum(result_i2t[:3]) + sum(result_t2i[:3])
        content['1K_Test']['mean_recall'] = ( sum(result_i2t[:3]) + sum(result_t2i[:3]) ) / 6.
        
        json_save( content, os.path.join(opt.save_path, opt.save_name) )
        print('{}'.format( content )) ## {'1K_Test': {'i2t_recall_at_1': 85.3, 'i2t_recall_at_5': 97.0, 'i2t_recall_at_10': 98.8, 't2i_recall_at_1': 65.34, 't2i_recall_at_5': 87.24, 't2i_recall_at_10': 91.7, 'recall_sum': 525.38, 'mean_recall': 87.56333333333333}}
    
    ## coco test
    if opt.data_name == 'coco': 
        coco_sim = np.load( '{}'.format( os.path.join(opt.sim_path, opt.sim_name) ) )
        assert coco_sim.shape == (5000, 25000)
        
        result_i2t = i2t(coco_sim) ## (55.22, 78.7, 86.74, 1.0, 6.704)
        result_t2i = t2i(coco_sim) ## (35.44, 60.0, 70.104, 3.0, 24.6332)
        result_i2t_t2i_5_fold_1K = coco_5_fold_1K_Test(coco_sim) ## before mean: [[75.1, 93.5, 97.3, 1.0, 2.124, 53.88, 82.48, 90.82, 1.0, 5.7288, 88.63333333333333, 75.72666666666667, 493.08], [74.0, 91.8, 96.7, 1.0, 2.375, 54.8, 81.24, 89.76, 1.0, 5.528, 87.5, 75.26666666666667, 488.3], [73.8, 93.2, 97.5, 1.0, 2.201, 54.56, 81.14, 90.58, 1.0, 5.6346, 88.16666666666667, 75.42666666666666, 490.78], [72.1, 93.7, 97.4, 1.0, 2.109, 53.22, 80.06, 89.0, 1.0, 5.6944, 87.73333333333335, 74.09333333333333, 485.4800000000001], [75.8, 92.7, 97.1, 1.0, 2.138, 54.32, 80.92, 89.18, 1.0, 6.3332, 88.53333333333335, 74.80666666666667, 490.02000000000004]]
        result_5_fold_1K = tuple(np.array(result_i2t_t2i_5_fold_1K).mean(axis=0).flatten()) ## after mean: (74.2, 93.0, 97.2, 1.0, 2.2, 54.2, 81.2, 89.9, 1.0, 5.8, 88.1, 75.1, 489.5)
        
        ##result_i2t == (r1, r5, r10, medr, meanr)
        ##result_t2i == (r1, r5, r10, medr, meanr)
        ##result_i2t_t2i_5_fold_1K == (i2t: r1, r5, r10, medr, meanr; t2i: r1, r5, r10, medr, meanr; ar, ari, rsum)
        
        print('{}_{}_{}'.format( result_i2t, result_t2i, result_5_fold_1K )) ## (55.22, 78.7, 86.74, 1.0, 6.704)_(35.44, 60.0, 70.104, 3.0, 24.6332)_(74.16, 92.97999999999999, 97.2, 1.0, 2.1894, 54.156000000000006, 81.168, 89.868, 1.0, 5.783799999999999, 88.11333333333334, 75.064, 489.532)
        
        content = {}
        
        content['5K_Test'] = {}
        content['5K_Test']['i2t_recall_at_1']  = result_i2t[0]
        content['5K_Test']['i2t_recall_at_5']  = result_i2t[1]
        content['5K_Test']['i2t_recall_at_10'] = result_i2t[2]
        content['5K_Test']['t2i_recall_at_1']  = result_t2i[0]
        content['5K_Test']['t2i_recall_at_5']  = result_t2i[1]
        content['5K_Test']['t2i_recall_at_10'] = result_t2i[2]
        content['5K_Test']['recall_sum'] = sum(result_i2t[:3]) + sum(result_t2i[:3])
        content['5K_Test']['mean_recall'] = ( sum(result_i2t[:3]) + sum(result_t2i[:3]) ) / 6.
        
        content['5_fold_1K_Test'] = {}
        content['5_fold_1K_Test']['i2t_recall_at_1']  = result_5_fold_1K[0]
        content['5_fold_1K_Test']['i2t_recall_at_5']  = result_5_fold_1K[1]
        content['5_fold_1K_Test']['i2t_recall_at_10'] = result_5_fold_1K[2]
        content['5_fold_1K_Test']['t2i_recall_at_1']  = result_5_fold_1K[5]
        content['5_fold_1K_Test']['t2i_recall_at_5']  = result_5_fold_1K[6]
        content['5_fold_1K_Test']['t2i_recall_at_10'] = result_5_fold_1K[7]
        content['5_fold_1K_Test']['recall_sum'] = sum(result_5_fold_1K[:3]) + sum(result_5_fold_1K[5:8])
        content['5_fold_1K_Test']['mean_recall'] = ( sum(result_5_fold_1K[:3]) + sum(result_5_fold_1K[5:8]) ) / 6.
        
        json_save( content, os.path.join(opt.save_path, opt.save_name) )
        print('{}'.format( content )) ## {'5K_Test': {'i2t_recall_at_1': 55.22, 'i2t_recall_at_5': 78.7, 'i2t_recall_at_10': 86.74, 't2i_recall_at_1': 35.44, 't2i_recall_at_5': 60.0, 't2i_recall_at_10': 70.104, 'recall_sum': 386.204, 'mean_recall': 64.36733333333333}, '5_fold_1K_Test': {'i2t_recall_at_1': 74.16, 'i2t_recall_at_5': 92.97999999999999, 'i2t_recall_at_10': 97.2, 't2i_recall_at_1': 54.156000000000006, 't2i_recall_at_5': 81.168, 't2i_recall_at_10': 89.868, 'recall_sum': 489.532, 'mean_recall': 81.58866666666667}}
    
    '''
    example: 
        python evaluation.py \
        --metric 'all' \
        --data_name 'f30k' \
        --sim_path '/home/user_name/temp_test' \
        --sim_name 'f30k_RN50x16_sim.npy' \
        --save_path '/home/user_name/temp_test' \
        --save_name 'f30k_all.json'

        
        output: 
            f30k_all.json ## 243 Bytes
    
    example: 
        python evaluation.py \
        --metric 'all' \
        --data_name 'coco' \
        --sim_path '/home/user_name/temp_test' \
        --sim_name 'coco_RN50x16_sim.npy' \
        --save_path '/home/user_name/temp_test' \
        --save_name 'coco_all.json'
        
        output: 
            coco_all.json ## 526 Bytes[dtype('float16')] / 561 Bytes[dtype('float32')]
    '''
    
    '''(float32)
    Baseline of the Task of Image-Text Matching(ITM)
    on F30k(Flickr30k) 1K Test
    and COCO(Microsoft COCO) 5 fold 1K Test & 5K Test: 
        [baseline model]
            OpenAI CLIP RN50x16 (https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt)
        [similarity matrix metric]
            cosine(dot product with l2norm)
        [F30k]
            [1K Test]
                R@Sum: 525.54
                mR: 87.590
                i2t R@1/5/10: 85.5 97.0 98.8
                t2i R@1/5/10: 65.32 87.1 91.82
        [COCO]
            [5K Test]
                R@Sum: 386.24
                mR: 64.373
                i2t R@1/5/10: 55.32 78.72 86.66
                t2i R@1/5/10: 35.396 60.012 70.132
            [5 fold 1K Test]
                R@Sum: 489.652
                mR: 81.609
                i2t R@1/5/10: 74.2 93.04 97.22
                t2i R@1/5/10: 54.108 81.216 89.868
    '''
    
    '''(float16)
    Baseline of the Task of Image-Text Matching(ITM)
    on F30k(Flickr30k) 1K Test
    and COCO(Microsoft COCO) 5 fold 1K Test & 5K Test: 
        [baseline model]
            OpenAI CLIP RN50x16 (https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt)
        [similarity matrix metric]
            cosine(dot product with l2norm)
        [F30k]
            [1K Test]
                R@Sum: 525.38
                mR: 87.563
                i2t R@1/5/10: 85.3 97.0 98.8
                t2i R@1/5/10: 65.34 87.24 91.7
        [COCO]
            [5K Test]
                R@Sum: 386.204
                mR: 64.367
                i2t R@1/5/10: 55.22 78.7 86.74
                t2i R@1/5/10: 35.44 60.0 70.104
            [5 fold 1K Test]
                R@Sum: 489.532
                mR: 81.588
                i2t R@1/5/10: 74.16 92.98 97.2
                t2i R@1/5/10: 54.156 81.168 89.868
    '''





























