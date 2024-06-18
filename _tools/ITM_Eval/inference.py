import argparse
import torch
import clip
from PIL import Image
import json
import numpy as np
from pycocotools.coco import COCO
import os
import time

def get_sim(
        bbone = "RN50x16", ## ["RN50x16", "ViT-B/16", ]
        device = "cuda:1" if torch.cuda.is_available() else "cpu", ## 0/1/2/3/4/5/6/7
        root_path = '/raid/xxx/all_data', ## data root path (including all dataset)
        data_name = 'f30k', ## ['f30k', 'coco', ]
        itd = 50, ## parallelism  ## set lower for low GPU memory (if CUDA OOM)
    ): 
    '''
    ## suggested dataset path
    /raid/xxx/all_data/ ## root_path
        f30k/ ## data_name
            images/
            dataset_flickr30k.json
        coco/
            images/
                train2014/
                val2014/
            annotations/
    '''
    model, preprocess = clip.load(bbone, device=device) ## if have not been downloaded, it will be auto-downloaded from the Internet
    '''
    https://github.com/openai/CLIP/blob/main/clip/clip.py#L30C12-L30C12
    _MODELS = {
        "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
        "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
        "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
        "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
        "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
        "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
    }
    '''
    
    annotations_full = {
        'f30k': ['dataset_flickr30k.json'],
        'coco': ['annotations/captions_train2014.json',
                 'annotations/captions_val2014.json'],
    }
    
    
    if data_name == 'f30k': 
        split = 'test' ## split  ## ['train', 'val', 'test']
        data_path = '{}'.format( os.path.join( os.path.join(root_path, data_name), annotations_full[data_name][0]) ) ## .json file path
        root = '{}/images'.format( os.path.join(root_path, data_name) ) ## image file path
        dataset = json.load(open(data_path, 'r'))['images'] ## open .json file
        ids = [] ## get split valid idx
        for i, d in enumerate(dataset):
            if d['split'] == split:
                ids += [(i, x) for x in range(len(d['sentences']))]
        
    
    if data_name == 'coco': 
        path = annotations_full[data_name][1] ## test split
        full_path = os.path.join(os.path.join(root_path, data_name), path)
        coco = COCO(full_path)
        list_a = np.load(os.path.join('{}/coco/annotations'.format( root_path ), 'coco_test_ids.npy')) ## test split
        root = '{}/images/{}'.format( os.path.join(root_path, data_name), 'val2014', ) ## test split
    
    
    max_cap_len = 77
    img_div = 5 ## f30k/coco have 5 times redundance (1 image: 5 captions)
    
    images = []
    texts  = []
    
    if data_name == 'f30k': 
        for index in range(len(ids)): 
            if index%1000 == 0: 
                print('{}/{}=={}%'.format(index, len(ids), 100. * index / len(ids)))
            ann_id = ids[index]
            img_id = ann_id[0]
            caption = dataset[img_id]['sentences'][ann_id[1]]['raw'] ## 1 caption
            imgname = dataset[img_id]['filename'] ## 1 image
            
            if index%img_div==0: ## f30k/coco have 5 times redundance (1 image: 5 captions)
                image = preprocess(Image.open(root+'/'+imgname)).unsqueeze(0) ## CLIP load
                images += [image]
            
            text = clip.tokenize([caption], context_length=max_cap_len, truncate=True)
            texts  += [text]
    
    if data_name == 'coco': 
        for i in range(len(list_a)): 
            if i%1000 == 0: 
                print('{}/{}=={}%'.format(i, len(list_a), 100. * i / len(list_a)))
            
            caption =                coco.anns[ list_a[ i ] ]['caption']
            imgname = coco.loadImgs( coco.anns[ list_a[ i ] ]['image_id'] )[0]['file_name']
            
            if i%img_div==0: ## f30k/coco have 5 times redundance (1 image: 5 captions)
                image = preprocess(Image.open(root+'/'+imgname)).unsqueeze(0) ## CLIP load
                images += [image]
            
            text = clip.tokenize([caption], context_length=max_cap_len, truncate=True)
            texts  += [text]
        
    
    images = torch.stack(images, 0)
    texts  = torch.stack(texts, 0)
    images = images.squeeze() ## f30k: torch.Size([1000, 3, 384, 384]) ; coco: torch.Size([5000, 3, 384, 384])
    texts  = texts.squeeze()  ## f30k: torch.Size([5000, 77]) ;          coco: torch.Size([25000, 77])
    
    
    N = images.shape[0]
    for i in range(0, N, itd): 
        if i%itd == 0: 
            print('{}/{}=={}%'.format(i, N, 100. * i / N))
        _s, _e = i, i+itd
        if _e > N: 
            _e = N
        with torch.no_grad(): 
            image_features = model.encode_image(images[_s:_e].to(device))
            text_features = model.encode_text(texts[5*_s:5*_e].to(device))
            image_embeddings = image_features / image_features.norm(dim=-1, keepdim=True)
            text_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)
        torch.cuda.empty_cache()
        
        if i == 0: 
            acc_image_embeddings = image_embeddings.cpu().numpy()
            acc_text_embeddings = text_embeddings.cpu().numpy()
        else: 
            tmp_image_embeddings = image_embeddings.cpu().numpy()
            tmp_text_embeddings = text_embeddings.cpu().numpy()
            acc_image_embeddings = np.concatenate((acc_image_embeddings, tmp_image_embeddings), axis=0)
            acc_text_embeddings = np.concatenate((acc_text_embeddings, tmp_text_embeddings), axis=0)
        
    
    acc_image_embeddings = torch.from_numpy(acc_image_embeddings).to(device)
    acc_text_embeddings = torch.from_numpy(acc_text_embeddings).to(device)
    acc_sim = acc_image_embeddings.mm(acc_text_embeddings.T)
    acc_sim = acc_sim.cpu().numpy() ## f30k: (1000, 5000) ; coco: (5000, 25000)
    
    return acc_sim




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbone', default='RN50x16') ## ["RN50x16", "ViT-B/16", ]
    parser.add_argument('--device', default='cuda:1') ## 0/1/2/3/4/5/6/7
    parser.add_argument('--data_root_path', default='/raid/xxx/all_data') ## data root path (including all dataset)
    parser.add_argument('--data_name', default='f30k') ## ['f30k', 'coco', ]
    parser.add_argument('--bs', default=50, type=int) ## parallelism  ## set lower for low GPU memory (if CUDA OOM)
    parser.add_argument('--save_path', default='/raid/xxx/vsepp/43_temp') ## sim matrix save path
    parser.add_argument('--save_name', default='dataset_bbone_sim.npy') ## sim matrix save name
    opt = parser.parse_args()
    
    sim = get_sim(
        bbone = opt.bbone, ## ["RN50x16", "ViT-B/16", ]
        device = opt.device if torch.cuda.is_available() else "cpu", ## 0/1/2/3/4/5/6/7
        root_path = opt.data_root_path, ## data root path (including all dataset)
        data_name = opt.data_name, ## ['f30k', 'coco', ]
        itd = opt.bs, ## parallelism  ## set lower for low GPU memory (if CUDA OOM)
    )
    
    full_save_path = os.path.join( opt.save_path, opt.save_name )
    np.save( '{}'.format( full_save_path ), sim )
    
    '''
    example: 
        python inference.py \
        --bbone 'RN50x16' \
        --device 'cuda:1' \
        --data_root_path '/home/user_name/all_data' \
        --data_name 'f30k' \
        --bs 50 \
        --save_path '/home/user_name/temp_test' \
        --save_name 'f30k_RN50x16_sim.npy'
        
        output: 
            f30k_RN50x16_sim.npy ## size: 9.54MB[dtype('float16')] / 19.07MB[dtype('float32')] , shape: (1000, 5000)
    
    example: 
        python inference.py \
        --bbone 'RN50x16' \
        --device 'cuda:1' \
        --data_root_path '/home/user_name/all_data' \
        --data_name 'coco' \
        --bs 50 \
        --save_path '/home/user_name/temp_test' \
        --save_name 'coco_RN50x16_sim.npy'
        
        output: 
            coco_RN50x16_sim.npy ## size: 238.42MB[dtype('float16')] / 476.84MB[dtype('float32')] , shape: (5000, 25000)
    '''























