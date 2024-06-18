








## [README.txt] is the Main README file (for testing MACK). 
## This file focuses on testing MACK to reproduce the experiment results in the paper using our open-sourced codes, models, files, and more with only CPUs. 
## Here, we give the specific instructions to help you to build the environment and excute MACK VLKB testing/Knowledge Inference from scratch. 
## ★ If you want to know how to construct MACK, please see the Supplementary README file [HOW TO REPRODUCE MACK.txt] (for constructing MACK) [Friendship Link]. 

【Do NOT Distribute】
This is a README file of the submitted journal paper TPAMISI-2023-03-0601 of 
the 2023 IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI 2023): 
"Unpaired Image-text Matching via Multimodal Aligned Conceptual Knowledge" 
(Manuscript Type: SI: Large-Scale Multimodal Learning: Universality, Robustness, Efficiency, and Beyond), 
with which the Supplementary Material (as well as source codes, models, datasets, files, etc.) is affiliated. 

The source codes, model files, datasets, VLKB knowledge dictionaries of this paper 
are all in https://github.com/github-user20240618/MACK . 




【Environment】
[Knowledge Construction]
    [1] (python 3) pip install Pillow, pycocotools
        ※ for basic python ability support
    [2] OD(Object Detection) Models ## pre-trained on OD(Object Detection)/VC(Visual Classification) dataset
        ※ for Object Detection: BBox and Feature Extraction
        ★ we will use this tool to extract bboxes and feats of [4] and [5]
        [a] bottom-up-attention.pytorch ## https://github.com/MILVLG/bottom-up-attention.pytorch  ## pre-trained on VG
        ※ for BU(Bottom-Up) BBox and Feature Extraction
            (1) R101 fixed 36 ## (default)/baseline
                extract-bua-caffe-r101-fix36.yaml ## config file
                bua-caffe-frcn-r101_with_attributes_fix36.pth ## model file
            (2) R152 ## BETTER/BEST performance
                bua-caffe-frcn-r152_with_attributes.pth ## config file
                extract-bua-caffe-r152.yaml ## model file
        [b] DETR ## https://github.com/facebookresearch/detr  ## pre-trained on COCO
            (1) COCO PT
        [c] Deformable DETR ## https://github.com/fundamentalvision/Deformable-DETR  ## pre-trained on COCO
            (1) R50
                VG FT: 1600 cls, 300 bbox
                COCO PT: 91 cls, 300 bbox
            (2) R101
                VG FT
        [d] CNN based models ## pre-trained on ImageNet
        ※ we first use BU(Bottom-Up) to extract BBox, and then use CNN to perform Feature Extraction
            (1) resnext50_32x4d ## https://github.com/huggingface/pytorch-image-models/tree/main
            (2) eva02_base_patch14_448 ## https://github.com/huggingface/pytorch-image-models/tree/main
            (3) mobilenetv3_large_100 ## https://github.com/huggingface/pytorch-image-models/tree/main
            (4) CLIP img_enc ## https://github.com/openai/CLIP  ## the image encoder of the original CLIP model
                (i)  RN50x16
                (ii) ViT-B/16
    [3] ITM(Image-text Matching) Models ## pre-trained on Image-text paired dataset
        ※ for ITM(Image-text Matching) Testing/Evaluation and Sim(Similarity) Matrices Construction
        ★ we will use this tool to perform Testing/Evaluation and construct Sim(Similarity) Matrices on [5]
            [a] CLIP ## https://github.com/openai/CLIP  ## pre-trained on 400M
                (i)  RN50x16 ## (default)/baseline
                (ii) ViT-B/16
            [b] ALBEF ## https://github.com/salesforce/ALBEF  ## we use ONLY the dual-stream/dual-path(coarse-grained) model(with ONLY image encoder and text encoder), which means we remove the last 6-layer Multi-modal BERT/Transformer(fine-grained)
                (i)  ALBEF.pth ## pre-trained on 14M
                (ii) ALBEF_4M.pth ## pre-trained on 4M
            [c] VSRN ## https://github.com/KunpengLi1994/VSRN  ## pre-trained on f30k/coco
                (i)  our reproduced version 1
                (ii) our reproduced version 2
            [d] SAEM ## https://github.com/yiling2018/saem  ## pre-trained on f30k/coco
                (i)  our reproduced version 1
                (ii) our reproduced version 2
            [e] UNITER ## https://github.com/ChenRocks/UNITER  ## pre-trained on 9.583M
                (i)  UNITER-base
                (ii) UNITER-large
            [f] OSCAR ## https://github.com/microsoft/Oscar  ## pre-trained on 6.5M
                (i)  OSCAR-base
                (ii) OSCAR-large
            [g] Chinese-CLIP[a Chinese ITM model] ## https://github.com/OFA-Sys/Chinese-CLIP  ## pre-trained on ~200M
                (i)  RN50x16
                (ii) ViT-B/16
            [h] WSD[a Chinese ITM model] ## pre-trained on AIC-ICC  ## No open-source codes/models  ## This is another TPAMI paper of the groups and the authors(Efficient Image and Sentence Matching)  ## https://ieeexplore.ieee.org/abstract/document/9783034/
                (i)  WSD-hard ## slightly better t2i test performance, and is ALREADY Deployed on our Chinese Voice-to-Image Cross-modal Retriaval DEMO System devices
                (ii) WSD-soft
    [4] Datasets for Knowledge Construction ## Image-text paired dataset: Image for Prototype Features , Text for Vocabulary
        [a] Visual Genome v1.2 ## https://homes.cs.washington.edu/~ranjay/visualgenome/api.html Version 1.2 of dataset completed as of August 29, 2016.
        ※ .json files provide sentence and image/bbox info; .jpg files are images
            (0) VG 108K images ## image_data.json, images.zip[extract => VG_100K/], images2.zip[extract => VG_100K_2/]
            (1) VG 3.8M object-word ## objects.json  ## (default)/baseline
            (2) VG 5.4M region-phrase ## region_descriptions.json  ## more bboxes and more words/tokens, which means a RICH vocabulary and a more ROBUST prototype ; BUT bboxes are not so precise as VG 3.8M object.json, which means there are more NOISE from the background pixels/info
        ※ The proposed MACK VLKB(Vision-and-Language Knowledge-Base) Knowledge have 2 key parts: 
            (1) Vocabulary (basic words/tokens) is (are) from tokenized sentences [from .json]
                ★ the Vocabulary files are saved in the directory of vocab_idx_word/ ## (default) my_vg_vocab_prototype_originated_from_pro_vocab_name_list_json_313KB_22_5_19_20_35.json
            (2) Prototype Features are extracted from image object/region bboxes [from .jpg]
                ★ the Prototype Features files are saved in the directory of p_feas_vlkb_word_idx_region_feat/ ## (default) my_p_feas_prototype_26276_240613.npy
            ※ One word/token corresponds to one particular prototype features, ## one-to-one correspondance, just like a REAL Dictionary: a concept is matched with a picture
               so that we can translate a symbolic word (in language) to its visual features (in vision). ## just like querying a REAL Dictionary: from Language to Vision
        [b] ImageNet-21K(IN-21k) ## https://image-net.org/download.php
        ※ this dataset is SUPER LARGE, so you need at least 2TB space to download and extract
        ※ compared to VG 3.8M object-word and VG 5.4M region-phrase, the IN-21k has more images, BUT less number and less precise bbox(ONLY ONE: the whole image), which means more NOISE and lower performance
        ※ the processing method of IN-21k is EASIER than VG, because it has NO BBOX annotation at all(just the whole image), 
           we just need to encode the whole image into ONLY ONE feature vector[CNN is ENOUGH][BU just need 1 BBOX: the width and height of the whole image]
    [5] Datasets for Evaluation(test split ONLY) ## Image-text paired dataset for ITM(Image-text Matching) and reranking task
        ※ Sim(Similarity) Matrices are constructed on these datasets
        ★ the BU(Bottom-Up) Pre-computed Features files are saved in the directory of bu_precomp_feats/ ## (default) test_acc_precomp_features.npy
        ★ the Sim(Similarity) Matrices files are saved in the directory of base_sims/ ## (default) f30k_RN50x16 test embedding_sim.npy
        [a] Flickr30k(F30k) ## https://shannon.cs.illinois.edu/DenotationGraph/
            (1) 1K Test Images ## (default)/baseline
        [b] Microsoft COCO(MSCOCO) ## https://cocodataset.org/#home
            (1) 5K Test Images
        [c] AI Challenger Image Captioning Challenge(AIC-ICC)[a Chinese dataset] ## The Official Website Download is closed, see non-official webpages
            (1) 1K Test Images ## ai_challenger_caption_validation_20170910.zip  ## We use the first 1K images of the last 5K images for test
        ※ ITM(Image-text Matching) are evaluated/tested on these datasets
            ※ including 3 test modes: 
                (A) sim: Sim Matrix ONLY ## for all config, it can evaluate the original testing score of ONLY using the Sim matrix from the pre-trained ITM model
                (B) rank: Knowledge ONLY ## when --top_k=1K/5K, it can evaluate the testing score of ONLY using the MACK Knowledge/Vocabulary/Dictionary
                (C) sim + rerank: first using Sim Matrix for retrieval, and then using Knowledge for reranking ## when --top_k=15, it can evaluate the testing score of BOTH (A)Sim AND (B)MACK
    [6] Java
        ※ basic Java support for [7] and [8]
        ※ we install this environment on Windows[for simplicity], rather than Ubuntu/Linux server, 
           so we don't know whether it can be installed on Ubuntu/Linux
    [7] stanford-postagger-full-2020-11-17 ## https://nlp.stanford.edu/software/tagger.html -> stanford-tagger-4.2.0.zip
        ※ for sentence to word/token tokenization and PoS(Part-of-Speech) Tagging => Unary relationship
        ★ we will use this tool to construct tags file from the Testing Dataset in [5]
        ★ the (Unary) PoS(Part-of-Speech) Tagging files are saved in the directory of tags_NN/ ## (default) tags
    [8] stanford-parser-full-2020-11-17 ## https://nlp.stanford.edu/software/lex-parser.html -> stanford-parser-4.2.0.zip
        ※ for sentence Parsing and word-word(token-token) dependency analysing => Binary relationship
        ★ we will use this tool to construct parses file from the Testing Dataset in [5]
        ★ the (Binary) Parsing dependency analysing files are saved in the directory of parses_JJ/ ## (default) parses

[Knowledge Inference]
    [9] (python 3) pip install python==3.6, numpy, h5py, pickle, torch
        ※ for basic python ability support
        ※ torch CPU ONLY version is ENOUGH ## Knowledge Inference DO NOT NEED GPU(s)
    [10] Codes, Files and more
        ★ the file structure of Codes and Files are as follows: 
            _tools/
                ITM_Eval/ ## how to construct Sim Matrix
                    inference.py ## constructing Sim Matrix for ITM Evaluation using ITM model
                    evaluation.py ## ITM Evaluation using Sim Matrix
                tag_parse/ ## how to construct tags and parses files
                    stanford-tagger-4.2.0.zip
                    stanford-parser-4.2.0.zip
                    coco_parse.py
                    coco_parses
                    coco_tag.py
                    coco_tags
                    coco_test.txt
                    f30k my_saved_test_annotations.txt ## (default)
                    coco my_saved_test_annotations.txt
                    tag_parse_README.txt
            base_sims/ ## the Sim(Similarity) Matrices files
                f30k_RN50x16 test embedding_sim.npy ## (default)
                ...
            bu_precomp_feats/ ## the BU(Bottom-Up) Pre-computed Features files
                test_acc_precomp_features.npy ## (default)  ## download at https://drive.google.com/drive/folders/1dl6KWmEDBdy8eSTbDNuaSMx86KRWImC3?usp=drive_link 
                ...
            p_feas_vlkb_word_idx_region_feat/ ## the Prototype Features files
                my_p_feas_prototype_26276_240613.npy ## (default)  ## download at https://drive.google.com/drive/folders/1NLgn1jtK3qxbyRV_fz-vo8SU_hMbM3zL?usp=drive_link 
                ...
            parses_JJ/ ## the Binary Parsing dependency analysing files
                parses ## (default)
                ...
            tags_NN/ ##  ## the Unary PoS(Part-of-Speech) Tagging files
                tags ## (default)
                ...
            vocab_idx_word/ ## the Vocabulary files
                my_vg_vocab_prototype_originated_from_pro_vocab_name_list_json_313KB_22_5_19_20_35.json ## (default)
                ...
            clip_t2i_i2t_jj.py ## the Knowledge Inference code
            logs/ ## test logs
                screen_Table_1_CLIP_f30k_i2t_top_k_5000_24_6_13_1.out ## the i2t test log file(latest ver.) [BETTER performance]
                screen_Table_1_CLIP_f30k_i2t_top_k_5000_23_2_28_1.out ## the i2t test log file(former ver.) [WORSE performance]
                screen_Table_1_CLIP_f30k_t2i_top_k_1000_24_6_13_1.out ## the t2i test log file(latest ver.) [BETTER performance]
                screen_Table_1_CLIP_f30k_t2i_top_k_1000_23_2_28_1.out ## the t2i test log file(former ver.) [WORSE performance]
            README.txt ## the Main README file (for testing MACK)
            HOW TO REPRODUCE MACK.txt ## the Supplementary README file (for constructing MACK)




【Terminal/Shell Command】
conda activate <your_env_name>
cd <your_path>/




【Python Evaluation Command(CPU ONLY)】
## We give the experiment with ALL DEFAULT hyper-params ONLY, which is related to the FIRST experiment in the paper(see Table 3 of the latest ver., or the Table 1/Table 2 of ealier NeurIPS/former TPAMI-SI ver.). 
## You can try various combinations of the following hyper-params to REPRODUCE the experimental results of MACK. 
## You can use your self-made/DIY files[see HOW TO REPRODUCE MACK.txt] to explore MORE possible hyper-params/your favorite or interested experiments(if you like). 
## Note: cosine metric for [CLIP, VSRN, SAEM, ALBEF(coarse-grained)]; softmax metric for [UNITER, OSCAR]
## ★ If you wonder how these hyper-params be made, please see the Supplementary README file [HOW TO REPRODUCE MACK.txt] (for constructing MACK) [Friendship Link]. 
## ★ If you want to know how to construct MACK, please see the Supplementary README file [HOW TO REPRODUCE MACK.txt] (for constructing MACK) [Friendship Link]. 


Table 1 - CLIP x f30k x pretrained x top_k=1000/5000
## In the ealier NeurIPS/former TPAMI-SI version paper, the experiments are showed in Table 1/Table 2; 
## BUT in latest version of TPAMI-SI paper, the experiments are showed in Table 3. ==> You can follow [HOW TO REPRODUCE MACK.txt] to REPRODUCE the construction process of the proposed MACK. 
## You can see that: the latest version obtain BETTER PERFORMANCE. That is because we use more formal and more normative tokenize principles to make a higher quality knowledge vocabulary/dictionary. 

[A] --top_k=1000/5000 \

[A1] f30k t2i
nohup python clip_t2i_i2t_jj.py \
--vocab='./vocab_idx_word/my_vg_vocab_prototype_originated_from_pro_vocab_name_list_json_313KB_22_5_19_20_35.json' \
--p_feas='./p_feas_vlkb_word_idx_region_feat/my_p_feas_prototype_26276_240613.npy' \
--tags='./tags_NN/tags' \
--parses='./parses_JJ/parses' \
--img_feats='./bu_precomp_feats/test_acc_precomp_features.npy' \
--npy_redundance=1 \
--sims='./base_sims/f30k_RN50x16 test embedding_sim.npy' \
--metric='cosine' \
--test_mode='t2i' \
--top_k=1000 \
--t2i_scale=0.15 \
> ./logs/screen_Table_1_CLIP_f30k_t2i_top_k_1000_24_6_13_1.out \
&

## (former ver.) screen_Table_1_CLIP_f30k_t2i_top_k_1000_23_2_28_1.out
23-2-28 11:21~14:22
r1, r5, r10, r1+r5+r10 ## evaluation metrics: R@1/5/10/Sum
65.4, 87.2, 91.7, 244.3 ## sim  ## ITM score of the Sim matrix from base model
10.84, 26.66, 35.84, 73.34 ## rank  ## through MACK VLKB knowledge from top_k result of the Sim matrix from base model
66.86, 88.44, 92.88, 248.18 ## sim + rerank  ## reranking performance of the Sim matrix improved by the proposed MACK VLKB knowledge

## (latest ver.) screen_Table_1_CLIP_f30k_t2i_top_k_1000_24_6_13_1.out
24-6-13 18:23~18:58
r1, r5, r10, r1+r5+r10 ## evaluation metrics: R@1/5/10/Sum
65.4, 87.2, 91.7, 244.3
11.14, 27.6, 36.66, 75.4 ## better than the former version
66.84, 88.3, 93.22, 248.35999999999999


[A2] f30k i2t
nohup python clip_t2i_i2t_jj.py \
--vocab='./vocab_idx_word/my_vg_vocab_prototype_originated_from_pro_vocab_name_list_json_313KB_22_5_19_20_35.json' \
--p_feas='./p_feas_vlkb_word_idx_region_feat/my_p_feas_prototype_26276_240613.npy' \
--tags='./tags_NN/tags' \
--parses='./parses_JJ/parses' \
--img_feats='./bu_precomp_feats/test_acc_precomp_features.npy' \
--npy_redundance=1 \
--sims='./base_sims/f30k_RN50x16 test embedding_sim.npy' \
--metric='cosine' \
--test_mode='i2t' \
--top_k=5000 \
--principle_of_pool 'i2t' \
--i2t_scale=0.15 \
> ./logs/screen_Table_1_CLIP_f30k_i2t_top_k_5000_24_6_13_1.out \
&

## (former ver.) screen_Table_1_CLIP_f30k_i2t_top_k_5000_23_2_28_1.out
23-2-28
r1, r5, r10, r1+r5+r10 ## evaluation metrics: R@1/5/10/Sum
85.4, 97.1, 98.7, 281.2 ## sim  ## ITM score of the Sim matrix from base model
10.4, 25.4, 34.3, 70.1 ## rank  ## through MACK VLKB knowledge from top_k result of the Sim matrix from base model
87.6, 96.9, 99.0, 283.5 ## sim + rerank  ## reranking performance of the Sim matrix improved by the proposed MACK VLKB knowledge

## (latest ver.) screen_Table_1_CLIP_f30k_i2t_top_k_5000_24_6_13_1.out
24-6-13 18:24~18:54
r1, r5, r10, r1+r5+r10 ## evaluation metrics: R@1/5/10/Sum
85.4, 97.1, 98.7, 281.2
9.9, 25.9, 35.0, 70.8 ## better than the former version
87.1, 96.9, 98.9, 282.9




【Main Hyper-params】
[VLKB dictionary]
    --vocab ## VLKB dictionary (idx <-> word)【see dir "vocab_idx_word"】
    --p_feas ## VLKB dictionary (word idx -> prototype region feature)【see dir "p_feas_vlkb_word_idx_region_feat"】
[word/region@dataset]
    --tags ## one word annotation ('NN' is n.) by StanfordPOSTagger【see dir "tags_NN"】
    --parses ## two words' relation annotation ('JJ' is adj.) by StanfordDependencyParser【see dir "parses_JJ"】
    --img_feats ## (precomp bu/bottom-up region feats) from SCAN[ECCV 18]/VSRN[ICCV 19]【see dir "bu_precomp_feats"】
    --npy_redundance ## the Image Redundancy Factor of the .npy image feats (f30k/coco may have 5 times image feats, because in f30k/coco, every image have 5 texts)
[base model@dataset]
    --sims ## test similarity matrix, cosine similarity or probability score【see dir "base_sims"】
    --metric ## cosine metric for [CLIP, VSRN, SAEM, ALBEF(coarse-grained)]; softmax metric for [UNITER, OSCAR]【"cosine"/"softmax"】
[test]
    --test_mode ## test mode (t2i/i2t/t2i+i2t) 
    --test_type ## test type ['', '1K', '5-fold-1K', '5K', ] 
    --top_k ## top k results by base model 
    --principle_of_pool ## the way of pooling VLKB 2D Matrix into a scalar (t2i/i2t)  ## the performance will be best if --test_mode matches --principle_of_pool 
[hyper-params]
    --NN_scale ## when *NN* + *JJ*, scale factor of NN 
    --JJ_scale ## when *NN* + *JJ*, scale factor of JJ 
    --t2i_scale ## scale factor of reranked t2i sim matrix 
    --i2t_scale ## scale factor of reranked t2i sim matrix 
    --T ## temperature of softmax 
    --max_mean ## pooling type 
    --emb_dim ## the dimensions of the features, see --p_feas and --img_feats (2048/1024/512/256/...) 
    --each_separate_word ## if True, MACK will calculate NN and JJ features independently/separately; otherwise, NN feat will add JJ feat together 
    --NN_JJ ## when each_separate_word is True, choose which types of token feats are ready to be calculated separately ['NN', 'NN+JJ', 'NN+JJ+VB', 'ALL_TYPE', ] 
    --save_argsort ## save 3 types (sim, rerank, sim+rerank) of evaluation results(the specific retrieval ranking results for each query), which is beneficial to Visualization 




【Introduction】
vocab_idx_word & p_feas_vlkb_word_idx_region_feat: 
    The MACK Vision-Language Knowledge-Base (MACK-VLKB) is based on the Visual Genome dataset[IJCV 17] v1.2. 
    The Visual Genome dataset has 3.8M(3,802,374) object-words annotations, with 26276 different words in total. 
    "vocab_idx_word" is a bi-directional table/dictionary that can map word to index, as well as index to word. 
        word2idx dict: str      -> int
        idx2word dict: str(int) -> str
        Length: 26276
    "p_feas_vlkb_word_idx_region_feat" is a table/dictionary that can map index of word from "vocab_idx_word" to prototype features 
    by OD/bottom-up attention region encoder(BUTD Faster R-CNN[CVPR 18]). 
        Shape: (26276, 2048)
tags_NN & parses_JJ: 
    Tag/relation annotations of the texts from Flickr30k(F30k)/MSCOCO(COCO) dataset test split. 
    "tags_NN" is the tag annotation of one word by StanfordPOSTagger. ('NN' is n.)
        e.g. in the description "orange hat", the word "hat" is a "NN" concept. 
    "parses_JJ" is the relation annotation between two words by StanfordDependencyParser. ('JJ' is adj.)
        e.g. in the description "orange hat", the word "orange" is a "JJ" concept related to the "NN" concept "hat". 
    NN & JJ are the two main concerns in MACK. ## see https://en.wikiversity.org/wiki/Template:DT_NN_NN_CC_NN to know the meaning of abbreviations
bu_precomp_feats: 
    OD/Bottom-up attention(Faster R-CNN R101 fixed 36) pre-computed features (from BUTD[CVPR 18] and SCAN[ECCV 18]) of the Flickr30k(F30k)/MSCOCO(COCO) dataset test split. 
        F30k Shape: (1000, 36, 2048) ## 1K test images with 36 bbox region/object features of 2048 dimensions. 
        COCO Shape: (5000, 36, 2048) ## 5K test images with 36 bbox region/object features of 2048 dimensions. 
base_sims: 
    Sim(Similarity) Matrices from base models of open-sourced Image-text Matching(ITM). 
        F30k Shape: (1000, 5000) ## 1K Test only 
        COCO Shape: (5000, 25000) ## 5K/5-fold-1K/1K Test 











