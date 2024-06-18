# -*- coding: utf-8 -*-
import numpy, os, json, h5py, pickle
import numpy as np
import argparse
import torch


def norm2(feats=None): ## (15, N, 2048) ----norm along dim=2(the 2048 dim)----> (15, N, 2048)
    norm = numpy.sqrt(numpy.sum(feats**2, axis=2))
    feats /= norm[:, :, None]
    return feats


def norm2_2d(feats=None): ## 22-4-3 (N, 2048) ----norm along dim=1(the 2048 dim)----> (N, 2048)
    norm = numpy.sqrt(numpy.sum(feats**2, axis=1))
    feats /= norm[:, None]
    return feats


def np_softmax_1d(scores, T=1.0): 
    score_T = scores/T
    return np.exp(score_T) / np.sum( np.exp(score_T) )


def show_score(sim_clip): 
    str1 = ''
    str2 = ''
    (r1, r5, r10, medr, meanr, ranks, top1) = i2t(sim_clip)
    (r1i, r5i, r10i, medri, meanri, ranks_t2i, top1i) = t2i(sim_clip)
    str1 = "\t Image to text: %.1f, %.1f, %.1f, %.1f, %.1f \t Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr, r1i, r5i, r10i, medri, meanri)
    score= r1+r5+r10+r1i+r5i+r10i
    str2 = "\t currscore: %.1f" % (score)
    
    print(str1)
    print(str2)
    
    return str1, str2, score, (r1, r5, r10, medr, meanr, ranks, top1), (r1i, r5i, r10i, medri, meanri, ranks_t2i, top1i)


def top_k_max_pool(sim_rank_i_t5, k=1):
    if k == 1: 
        return sim_rank_i_t5.max(-1).mean(-1)
    if k >= 2: 
        return numpy.sort(sim_rank_i_t5, axis=-1)[...,-k:].mean(-1).mean(-1)



def i2t(sim, return_ranks=True):
    npts = sim.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sim[index])[::-1]
        
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
        return (r1, r5, r10, medr, meanr, ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(sim, return_ranks=True):
    npts = sim.shape[0]
    
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    
    sim = sim.T
    
    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sim[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]
    
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr, ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def NN_add_JJ( NN_scale, o_fea , JJ_scale, jj_fea ): ## 22-6-30
    if type(jj_fea) == type(0): 
        assert jj_fea == 0
        jj_fea = [0]
    
    l = []
    num_NN = len(o_fea)
    num_JJ = len(jj_fea)
    for i in range(num_NN):
        for j in range(num_JJ): 
            new_fea = NN_scale*o_fea[i] + JJ_scale*jj_fea[j]
            l += [ new_fea ]
    return l


def p_feas_polysemy_add( headword_fea, modifier_fea ):
    ##assert headword_fea.shape[0] >= 2
    ##assert modifier_fea.shape[0] >= 2
    l = []
    num_headword = headword_fea.shape[0]
    if type(modifier_fea) == type(0) or type(modifier_fea) == type(0.):
        num_modifier = 1
        modifier_fea = [0.]
    else: 
        num_modifier = modifier_fea.shape[0]
    for i in range(num_headword):
        for j in range(num_modifier): 
            new_fea = headword_fea[i] + modifier_fea[j]
            l += [ new_fea ]
    
    return l, len(l)


def get_NN_with_JJ_feats(idx, tags, word2idx, p_feas, parses, NN_scale=1.0, JJ_scale=1.0, opt=None): 
    words = tags[idx] ## idx th sentence' all words with tags
    
    NNs = []
    feas = []
    p_feas_polysemy_len = []
    
    if 'each_separate_word' in opt.__dict__ and opt.each_separate_word == True:
        ## NN_JJ', type=str, default='NN', help='NN or NN+JJ or NN+JJ+VB or ALL_TYPE' 
        assert 'NN_JJ' in opt.__dict__
        if opt.NN_JJ == 'NN': 
            valid_type = ['NN', ]
        if opt.NN_JJ == 'NN+JJ': 
            valid_type = ['NN', 'JJ', ]
        if opt.NN_JJ == 'NN+JJ+VB': 
            valid_type = ['NN', 'JJ', 'VB', ]
        if opt.NN_JJ == 'ALL_TYPE': 
            valid_type = ['', ]
        
        for word in words: 
            if True in [type in word[1] for type in valid_type] and word[0] in word2idx.keys():
                if opt.show_reason_info: 
                    print('valid word: {}, {}'.format(type, word[0]))
                if opt.word2idxs == 'True':
                    if opt.word2idxs_VG_CHN2EN != '': 
                        assert 'word2idxs_VG_CHN2EN' in opt.__dict__ and opt.word2idxs_VG_CHN2EN != ''
                        assert 'each_separate_word' in opt.__dict__ and opt.each_separate_word == True 
                        word2idxs = word2idx
                        word_idxs = word2idxs[word[0]]
                        o_feas = p_feas[word_idxs]
                        
                        new_feas = o_feas
                        for new_fea in new_feas: 
                            feas.append(new_fea)
                    elif opt.word2idxs_IN21k_split != '':
                        assert 'word2idxs_IN21k_split' in opt.__dict__ and opt.word2idxs_IN21k_split != ''
                        assert 'each_separate_word' in opt.__dict__ and opt.each_separate_word == True 
                        word2idxs = word2idx
                        word_idxs = word2idxs[word[0]]
                        if opt.word2idxs_IN21k_split == 'train': 
                            word_idxs = [ train_word_idx       for train_word_idx in word_idxs if train_word_idx <  10450 ]
                        elif opt.word2idxs_IN21k_split == 'val': 
                            word_idxs = [   val_word_idx       for   val_word_idx in word_idxs if   val_word_idx <  10450 ]
                        elif opt.word2idxs_IN21k_split == 'small': 
                            word_idxs = [ small_word_idx-10450 for small_word_idx in word_idxs if small_word_idx >= 10450 ]
                        else : 
                            assert 1==2
                        o_feas = p_feas[word_idxs]
                        
                        new_feas = o_feas
                        for new_fea in new_feas: 
                            feas.append(new_fea)
                    else: 
                        assert 1==2
                else:
                    word_idx = word2idx[word[0]]
                    o_fea = p_feas[word_idx]
                    
                    new_fea = o_fea
                    feas.append(new_fea)
            
        
        if len(feas) == 0:
            if opt.show_reason_info: 
                print('No valid word checked')
            temp = numpy.zeros(opt.emb_dim)
            temp[0] = 1.
            feas.append(temp)
        
        
        
    else:
        for word in words:
            if 'NN' in word[1] and word[0] in word2idx.keys():
                if opt.show_reason_info: 
                    print('valid NN word: {}'.format(word[0]))
                NNs.append(word[0])
                word_idx = word2idx[word[0]]
                o_fea = p_feas[word_idx]
                
                jj_fea = 0
                for row in parses[idx]:
                    if word[0] == row[0][0] and 'amod' in row[1] and 'JJ' in row[2][1] and row[2][0] in word2idx.keys():
                        if opt.show_reason_info: 
                            print('valid JJ word: {}'.format(row[2][0]))
                        jj_idx = word2idx[row[2][0]] ## adj ----add in----> list
                        jj_fea = p_feas[jj_idx] ## retrieve adj's (AvgPooled/mean) BU feat 
                        break
                    else:
                        if opt.show_reason_info: 
                            print('No valid JJ word checked')
                        
                if type(o_fea) == type(np.ndarray(0)):
                    if opt.p_feas_polysemy == True:
                        new_fea, len_new_fea = p_feas_polysemy_add(o_fea, JJ_scale*jj_fea)
                        feas += new_fea
                        p_feas_polysemy_len += [ len_new_fea ]
                    else:
                        new_fea = o_fea + JJ_scale*jj_fea ## old version
                        feas.append(new_fea)
                    
                elif type(o_fea) == type([]):
                    new_fea = NN_add_JJ( NN_scale, o_fea , JJ_scale, jj_fea )
                    feas += new_fea
                
                else: 
                    assert 1==2
            
        if len(feas) == 0:
            temp = numpy.zeros(opt.emb_dim)
            temp[0] = 1.
            feas.append(temp) ## p_feas[0]'s shape == numpy.zeros(2048)  ## We choose one-hot vector instead for simplicity. 21-9-30 
    
    return feas, p_feas_polysemy_len


def eval(opt, test_mode, len_test, sim_clip, tags, word2idx, p_feas, parses, test_feas): 
    ranks_un = []
    ranks_clip = []
    ranks_comb = []
    k = opt.top_k ## rerank only cares top 15
    acc_inds_clip_t5 = []
    acc_inds_i_t5 = []
    acc_inds_comb_i_t5 = []
    
    flag_no_sim_plus_rank = opt.flag_no_sim_plus_rank
    
    for i in range(len_test):
        if test_mode == 't2i': 
            idx = i ## text index
            index = idx//5 ## image index
        if test_mode == 'i2t': 
            index = i
        
        if test_mode == 't2i': 
            sim_clip_i = sim_clip.T[idx] ## (1000,)  ## equal to: sim_clip[:,0]  ## Sim's idx th colomn: text idx th to all 1000 images's (cosine/probs) similarity
        if test_mode == 'i2t': 
            sim_clip_i = sim_clip[index] ## (5000,)  ## equal to: sim_clip[0]  ## Sim's index th line: image index th to all 5000 texts' (cosine/probs) similarity
        inds_clip_t5 = numpy.argsort(sim_clip_i)[::-1][0:k] ## descent order sort by similarity(real, -1~1), and return corresponding descent order sorted rank/index(integer, 0~999), and then choose top k(k==15) rank/index
        acc_inds_clip_t5 += [inds_clip_t5]
        sim_clip_i_t5 = sim_clip_i[inds_clip_t5] ## numpy's high-order index method: translating(mapping) descent order rank(index) to corresponding descent order sort(similarity)
        try:
            if test_mode == 't2i': 
                rank_clip_t5 = numpy.where(inds_clip_t5 == idx//5)[0][0] ## correct index / real rank (one integer), 0 is best(top 1), N means top N+1
            if test_mode == 'i2t': 
                rank_clip_t5 = numpy.where(inds_clip_t5//5 == index)[0][0] ## correct index / real rank (one integer), 0 is best(top 1), N means top N+1
        except:
            rank_clip_t5 = 100 ## if not in top 15, return 100(ignore, means do not use "vocab/knowledge base", so >15 is OK, because it will not influence the R@K results, do not change anything) ==> HY's code is correct!!! 21-9-30 
        ranks_clip.append(rank_clip_t5)
        
        if test_mode == 't2i': 
            feas, p_feas_polysemy_len = get_NN_with_JJ_feats(idx, tags, word2idx, p_feas, parses, NN_scale=opt.NN_scale, JJ_scale=opt.JJ_scale, opt=opt)
            feas = numpy.array(feas)[numpy.newaxis,:,:] ## (N, 2048) ----expand----> (1, N, 2048)
            feas_expand = numpy.repeat(feas, k, 0) ## (1, N, 2048) ----expand----> (15, N, 2048)
            if opt.l2norm_off: 
                sim_rank_i_t5 = numpy.matmul(feas_expand, test_feas[inds_clip_t5].transpose(0,2,1))
            else: 
                sim_rank_i_t5 = numpy.matmul(norm2(feats=feas_expand), norm2(feats=test_feas[inds_clip_t5]).transpose(0,2,1)) ## normalized!!! 21-9-30 
            
            if opt.p_feas_polysemy == True and len(p_feas_polysemy_len) >= 1:
                p_feas_polysemy_pooling_sim_rank_i_t5 = np.zeros( ( sim_rank_i_t5.shape[0], len(p_feas_polysemy_len), sim_rank_i_t5.shape[-1] ) ).astype(np.float32)
                
                acc_p_feas_polysemy_pooling_num = 0
                for it_pooling_idx, it_pooling_num in enumerate( p_feas_polysemy_len ): 
                    if opt.p_feas_polysemy_pooling == 'max': 
                        p_feas_polysemy_pooling_sim_rank_i_t5[ :, it_pooling_idx ,: ] = sim_rank_i_t5[ : , acc_p_feas_polysemy_pooling_num : acc_p_feas_polysemy_pooling_num + it_pooling_num , : ].max(-2)
                    elif opt.p_feas_polysemy_pooling == 'mean': 
                        p_feas_polysemy_pooling_sim_rank_i_t5[ :, it_pooling_idx ,: ] = sim_rank_i_t5[ : , acc_p_feas_polysemy_pooling_num : acc_p_feas_polysemy_pooling_num + it_pooling_num , : ].mean(-2)
                    else: 
                        assert 1==0
                    acc_p_feas_polysemy_pooling_num += it_pooling_num
                
                sim_rank_i_t5 = p_feas_polysemy_pooling_sim_rank_i_t5
                
            else:
                pass
            
            if 'principle_of_pool' in list(opt.__dict__):
                if opt.principle_of_pool == 't2i':
                    pass
                if opt.principle_of_pool == 'i2t':
                    sim_rank_i_t5 = sim_rank_i_t5.transpose(0,2,1)
            else: 
                pass
            
            if opt.max_mean=='max_mean' and 'k_of_kmaxpool' in list(opt.__dict__):
                kk = opt.k_of_kmaxpool
            else: 
                kk = 1
            
            if opt.max_mean=='mean_mean': 
                sim_rank_i_t5 = sim_rank_i_t5.mean(-1).mean(-1)
            if opt.max_mean=='max_mean': 
                ##sim_rank_i_t5 = sim_rank_i_t5.max(-1).mean(-1) ## (15, N, 36) ----max(-1)----> (15, N) ----mean(-1)----> (15,) 
                sim_rank_i_t5 = top_k_max_pool(sim_rank_i_t5, k=kk)
            if opt.max_mean=='max_max': 
                sim_rank_i_t5 = sim_rank_i_t5.max(-1).max(-1) ## (15, N, 36) ----max(-1)----> (15, N) ----max(-1)----> (15,) 
            if opt.max_mean=='mean_max': 
                sim_rank_i_t5 = sim_rank_i_t5.mean(-1).max(-1) ## (15, N, 36) ----mean(-1)----> (15, N) ----max(-1)----> (15,) 
        if test_mode == 'i2t': 
            acc_sim_rank_i_t5 = [] ## 22-4-3 
            for idx in inds_clip_t5: ## 22-4-3 
                feas, p_feas_polysemy_len = get_NN_with_JJ_feats(idx, tags, word2idx, p_feas, parses, NN_scale=opt.NN_scale, JJ_scale=opt.JJ_scale, opt=opt)
                feas = numpy.array(feas) ## (N, 2048)
                if opt.l2norm_off: 
                    sim_rank_i_t5 = numpy.matmul(feas, test_feas[index].transpose(1,0)) ## 22-8-17 
                else: 
                    sim_rank_i_t5 = numpy.matmul(norm2_2d(feats=feas), norm2_2d(feats=test_feas[index]).transpose(1,0)) ## normalized!!! 21-9-30 
                
                if opt.p_feas_polysemy == True and len(p_feas_polysemy_len) >= 1:
                    p_feas_polysemy_pooling_sim_rank_i_t5 = np.zeros( ( len(p_feas_polysemy_len), sim_rank_i_t5.shape[-1] ) ).astype(np.float32)
                    
                    acc_p_feas_polysemy_pooling_num = 0
                    for it_pooling_idx, it_pooling_num in enumerate( p_feas_polysemy_len ): 
                        if opt.p_feas_polysemy_pooling == 'max': 
                            p_feas_polysemy_pooling_sim_rank_i_t5[ it_pooling_idx ,: ] = sim_rank_i_t5[ acc_p_feas_polysemy_pooling_num : acc_p_feas_polysemy_pooling_num + it_pooling_num , : ].max(-2)
                        elif opt.p_feas_polysemy_pooling == 'mean': 
                            p_feas_polysemy_pooling_sim_rank_i_t5[ it_pooling_idx ,: ] = sim_rank_i_t5[ acc_p_feas_polysemy_pooling_num : acc_p_feas_polysemy_pooling_num + it_pooling_num , : ].mean(-2)
                        else: 
                            assert 1==0
                        acc_p_feas_polysemy_pooling_num += it_pooling_num
                    
                    sim_rank_i_t5 = p_feas_polysemy_pooling_sim_rank_i_t5
                    
                else:
                    pass
                
                if 'principle_of_pool' in list(opt.__dict__):
                    if opt.principle_of_pool == 't2i':
                        pass
                    if opt.principle_of_pool == 'i2t':
                        sim_rank_i_t5 = sim_rank_i_t5.T
                else: 
                    pass
                
                if opt.max_mean=='max_mean' and 'k_of_kmaxpool' in list(opt.__dict__):
                    kk = opt.k_of_kmaxpool
                else: 
                    kk = 1
                
                if opt.max_mean=='mean_mean': 
                    sim_rank_i_t5 = sim_rank_i_t5.mean(-1).mean(-1) ## (N, 36) ----mean(-1)----> (N, ) ----mean(-1)----> () 
                if opt.max_mean=='max_mean': 
                    ##sim_rank_i_t5 = sim_rank_i_t5.max(-1).mean(-1) ## (N, 36) ----max(-1)----> (N, ) ----mean(-1)----> () 
                    sim_rank_i_t5 = top_k_max_pool(sim_rank_i_t5, k=kk)
                if opt.max_mean=='max_max': 
                    sim_rank_i_t5 = sim_rank_i_t5.max(-1).max(-1) ## (N, 36) ----max(-1)----> (N, ) ----max(-1)----> () 
                if opt.max_mean=='mean_max': 
                    sim_rank_i_t5 = sim_rank_i_t5.mean(-1).max(-1) ## (N, 36) ----mean(-1)----> (N, ) ----max(-1)----> () 
                acc_sim_rank_i_t5.append(sim_rank_i_t5)
                
                
            
        
        if test_mode == 't2i': 
            inds_i_t5_temp = numpy.argsort(sim_rank_i_t5)[::-1] ## e.g. array([ 0,  4,  1,  2, 13, 11,  6,  8, 10,  7,  3, 12,  5,  9, 14])
            inds_i_t5 = inds_clip_t5[inds_i_t5_temp] ## e.g. array([  0, 550, 239, 209, 327, 328, 594, 710, 716, 156, 463,  19,  42,  79, 502])
            acc_inds_i_t5 += [ inds_i_t5 ]
            try:
                rank_i_t5 = numpy.where(inds_i_t5 == idx//5)[0][0] ## e.g. 0 
            except:
                rank_i_t5 = 100
            ranks_un.append(rank_i_t5)
        if test_mode == 'i2t': 
            acc_sim_rank_i_t5 = numpy.array(acc_sim_rank_i_t5) ## (k, ) cosine  ## 22-4-3 
            inds_i_t5_temp = numpy.argsort(acc_sim_rank_i_t5)[::-1] ## e.g. array([ 0,  4,  1,  2, 13, 11,  6,  8, 10,  7,  3, 12,  5,  9, 14])
            inds_i_t5 = inds_clip_t5[inds_i_t5_temp] ## e.g. array([  0, 550, 239, 209, 327, 328, 594, 710, 716, 156, 463,  19,  42,  79, 502])
            acc_inds_i_t5 += [ inds_i_t5 ]
            try:
                rank_i_t5 = numpy.where(inds_i_t5//5 == index)[0][0] ## e.g. 0 
            except:
                rank_i_t5 = 100
            ranks_un.append(rank_i_t5)

        
        if not flag_no_sim_plus_rank: ## 23-2-24 
            if opt.metric == 'softmax': 
                sim_clip_i_t5 = np_softmax_1d( sim_clip_i_t5 , opt.T)
                if test_mode == 't2i': 
                    sim_rank_i_t5 = np_softmax_1d( sim_rank_i_t5 , opt.T)
                if test_mode == 'i2t': 
                    acc_sim_rank_i_t5 = np_softmax_1d( acc_sim_rank_i_t5 , opt.T)
            
            if test_mode == 't2i': 
                sim_comb_i_t5 = sim_clip_i_t5 + opt.t2i_scale* sim_rank_i_t5 ## (15,) 
            if test_mode == 'i2t': 
                sim_comb_i_t5 = sim_clip_i_t5 + opt.i2t_scale* acc_sim_rank_i_t5 ## (15,) 
            inds_comb_i_t5_temp = numpy.argsort(sim_comb_i_t5)[::-1] ## array([ 0,  4,  1,  2, 11,  6, 13,  8, 10,  7,  3,  5, 12,  9, 14]) 
            inds_comb_i_t5 = inds_clip_t5[inds_comb_i_t5_temp] ## (15,) 
            acc_inds_comb_i_t5 += [ inds_comb_i_t5 ]
            try:
                if test_mode == 't2i': 
                    rank_comb_i_t5 = numpy.where(inds_comb_i_t5 == index)[0][0] ## 0 
                if test_mode == 'i2t': 
                    rank_comb_i_t5 = numpy.where(inds_comb_i_t5//5 == index)[0][0] ## 0 
            except:
                rank_comb_i_t5 = 100
            ranks_comb.append(rank_comb_i_t5)
            
            if rank_clip_t5 > rank_comb_i_t5:
                if test_mode == 't2i': 
                    print(idx, 'th: clip', rank_clip_t5, 'rank', rank_i_t5, 'comb', rank_comb_i_t5 )
                if test_mode == 'i2t': 
                    print(index, 'th: clip', rank_clip_t5, 'rank', rank_i_t5, 'comb', rank_comb_i_t5 )
    
    
    result = []
    ranks_clip = numpy.array(ranks_clip)
    r1 = 100.0 * len(numpy.where(ranks_clip < 1)[0]) / len(ranks_clip)
    r5 = 100.0 * len(numpy.where(ranks_clip < 5)[0]) / len(ranks_clip)
    r10 = 100.0 * len(numpy.where(ranks_clip < 10)[0]) / len(ranks_clip)
    str1 = '{}, {}, {}, {}'.format(r1, r5, r10, r1+r5+r10)
    print(str1)
    result += [ [ r1, r5, r10, r1+r5+r10 ] ]


    ranks_un = numpy.array(ranks_un)
    r1 = 100.0 * len(numpy.where(ranks_un < 1)[0]) / len(ranks_un)
    r5 = 100.0 * len(numpy.where(ranks_un < 5)[0]) / len(ranks_un)
    r10 = 100.0 * len(numpy.where(ranks_un < 10)[0]) / len(ranks_un)
    str2 = '{}, {}, {}, {}'.format(r1, r5, r10, r1+r5+r10)
    print(str2)
    result += [ [ r1, r5, r10, r1+r5+r10 ] ]


    if not flag_no_sim_plus_rank: ## 23-2-24 
        ranks_comb = numpy.array(ranks_comb)
        r1 = 100.0 * len(numpy.where(ranks_comb < 1)[0]) / len(ranks_comb)
        r5 = 100.0 * len(numpy.where(ranks_comb < 5)[0]) / len(ranks_comb)
        r10 = 100.0 * len(numpy.where(ranks_comb < 10)[0]) / len(ranks_comb)
        str3 = '{}, {}, {}, {}'.format(r1, r5, r10, r1+r5+r10)
        print(str3)
        result += [ [ r1, r5, r10, r1+r5+r10 ] ]
    else: 
        result += [ [ -1, -1, -1, -3 ] ]
    
    
    if 'save_argsort' in list(opt.__dict__) and opt.save_argsort == True:
        np.save('./[sim][{}]acc_inds_clip_t5.npy'.format(opt.test_mode), np.array(acc_inds_clip_t5))
        np.save('./[rank][{}]acc_inds_i_t5.npy'.format(opt.test_mode), np.array(acc_inds_i_t5))
        if not flag_no_sim_plus_rank: ## 23-2-24 
            np.save('./[sim+rerank][{}]acc_inds_comb_i_t5.npy'.format(opt.test_mode), np.array(acc_inds_comb_i_t5))
    
    return result ## [[1, 5, 10, 1510], [1, 5, 10, 1510], [1, 5, 10, 1510]]


def eval_once(opt, sim_clip, tags, word2idx, p_feas, parses, test_feas): 
    img_num = sim_clip.shape[0] ## f30k: 1000
    txt_num = sim_clip.shape[1] ## f30k: 5000
    
    if opt.test_mode == 't2i': 
        len_test = txt_num
        result = eval(opt, opt.test_mode, len_test, sim_clip, tags, word2idx, p_feas, parses, test_feas)
        return result ## [[1, 5, 10, 1510], [1, 5, 10, 1510], [1, 5, 10, 1510]]
    if opt.test_mode == 'i2t': 
        len_test = img_num
        result = eval(opt, opt.test_mode, len_test, sim_clip, tags, word2idx, p_feas, parses, test_feas)
        return result ## [[1, 5, 10, 1510], [1, 5, 10, 1510], [1, 5, 10, 1510]]
    if opt.test_mode == 't2i+i2t': 
        len_test = txt_num
        result1 = eval(opt, 't2i', len_test, sim_clip, tags, word2idx, p_feas, parses, test_feas)
        len_test = img_num
        result2 = eval(opt, 'i2t', len_test, sim_clip, tags, word2idx, p_feas, parses, test_feas)
        return result1+result2 ## [[1, 5, 10, 1510], [1, 5, 10, 1510], [1, 5, 10, 1510], [1, 5, 10, 1510], [1, 5, 10, 1510], [1, 5, 10, 1510]]




## ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   

if __name__ == '__main__':
    parser = argparse.ArgumentParser() ## (default) f30k
    ## VLKB Dicts
    parser.add_argument('--vocab', default='./vocab_idx_word/vg_vocab.json') ## word idx (idx <-> word)  ## VLKB dictionary (idx <-> word)【see dir "vocab_idx_word"】
        ## 23-2-20 
        ## when activate --word2idxs , 
        ## --vocab and --p_feas will change : 
        ## VLKB will become 3 types of split modes adapted to IN-21k . 
    parser.add_argument('--word2idxs', default='') ## 23-2-20 (''|'True') whether change word2idx(one-to-one) of vocab into word2idxs(one-to-many) (default: '' represents one-to-one) 
    parser.add_argument('--word2idxs_IN21k_split', default='') ## 23-2-20 IN-21k VLKB ONLY (''|'train'|'val'|'small'), choose a certain split of IN-21k to do fancy indexing p_feas 
    parser.add_argument('--word2idxs_VG_CHN2EN', default='') ## 24-5-30 VG(CHN->EN) VLKB ONLY (''|'True') 
    parser.add_argument('--p_feas', default='./p_feas_vlkb_word_idx_region_feat/p_feas.npy') ## word-bbox VLKB (word idx -> object/region feature)  ## VLKB dictionary (word idx -> prototype region feature)【see dir "p_feas_vlkb_word_idx_region_feat"】
        ## 23-2-20 
        ## when --word2idxs=='True' , 
        ## --vocab='./vocab_idx_word/my_vocab_IN_21k_prototype_idx2words_19167_word2idxs_22331_230210.json' is the ONLY choice of vocab 
        ## --p_feas='./p_feas_vlkb_word_idx_region_feat/prototype_fea_train|val|small.npy' are the ONLY 3 choices of p_feas 
        ## and MORE hyper-params needed: 
        ## --each_separate_word \ ## True is neccessary 
        ## --NN_JJ 'NN' \ ## 'NN or NN+JJ or NN+JJ+VB or ALL_TYPE' are the ONLY 4 choices of p_feas 
    parser.add_argument('--p_feas_polysemy', action='store_true') ## 23-10-9 whether p_feas is polysemous (whether p_feas has one-to-many word2feats) (False | True), (default: False)
    ## datasets/word analysis/object analysis
    parser.add_argument('--tags', default='./tags_NN/tags') ## (Unary) PoS(Part-of-Speech) Tagging file of certain test dataset  ## one word annotation ('NN' is n.) by StanfordPOSTagger【see dir "tags_NN"】
    parser.add_argument('--parses', default='./parses_JJ/parses') ## (Binary) Parsing dependency analysing file of certain test dataset  ## two words' relation annotation ('JJ' is adj.) by StanfordDependencyParser【see dir "parses_JJ"】
    parser.add_argument('--img_feats', default='./bu_precomp_feats/f30k_test_buctxbox.h5') ## OD/CNN based pre_computed image features of certain test dataset  ## (precomp bu/bottom-up region feats) from SCAN[ECCV 18]/VSRN[ICCV 19]【see dir "bu_precomp_feats"】
    parser.add_argument('--npy_redundance', default=5, type=int) ## the Image Redundancy Factor of the .npy image feats of certain test dataset(f30k/coco may have 5 times image feats, because in f30k/coco, every image have 5 texts)
    ## base models
    parser.add_argument('--sims', default='./base_sims/f30k_RN50x16 test embedding_sim.npy') ## Similarity Matrix by ITM base model of certain test dataset (usually test split Sim Matrix, and cosine similarity metric)  ## test similarity matrix, cosine similarity or probability score【see dir "base_sims"】
    parser.add_argument('--metric', default='cosine') ## the Interpretation of Meaning/Way of Using of the Sim Matrix when doing i2t/t2i testing (usually is cosine similarity[CLIP, VSRN, SAEM, ALBEF(coarse-grained)], and may also be softmax probability[UNITER, OSCAR]) ("cosine"/"softmax")  ## cosine metric for [CLIP, VSRN, SAEM, ALBEF(coarse-grained)]; softmax metric for [UNITER, OSCAR]【"cosine"/"softmax"】
    ## test ways/methods
    parser.add_argument('--test_mode', default='t2i') ## test mode (t2i/i2t/t2i+i2t)
    parser.add_argument('--test_type', default='') ## test type ['', '1K', '5-fold-1K', '5K', '100', ]
    parser.add_argument('--top_k', default=15, type=int) ## top k sim results before reranking  ## top k results by base model 
    parser.add_argument('--fc_ft', default='') ## 22-8-1  ## ['', 'fc_ft', ], using FC(B=Ax+b, one for img_enc and one for txt_enc) self-supervised way to FT, (default: '') 
    parser.add_argument('--fc_ft_model_file', default='') ## 22-8-1  ## fc_ft model file  ## using when '--fc_ft'='fc_ft' is activated 
    parser.add_argument('--l2norm_off', action='store_true') ## 22-8-17 l2norm off. This will influence testing score. Using l2norm may reduce testing score. 
    parser.add_argument('--eval_type', default='eval_once') ## 22-9-20  ## evaluation type (default: 'eval_once' )
    parser.add_argument('--show_reason_info', action='store_true') ## 22-9-21 open print function in get_NN_with_JJ_feats, to show evaluation/inference process and reason, in order to examine the cause of errors 
    parser.add_argument('--k_of_kmaxpool', default=1, type=int) ## the k of kmaxpool [1,...] (default: 1, which is equivalent to [max_mean])  ## used ONLY when opt.max_mean=='max_mean' is activated  ## top_k_max_pool is the extension of the original [max_mean] from k=1 to k=any positive integer 
    parser.add_argument('--principle_of_pool', default='t2i') ## the principle of pooling [t2i/i2t] (default: t2i)  ## the performance will be best if --test_mode matches --principle_of_pool 
    ## hyper-params
    parser.add_argument('--NN_scale', default=1.0, type=float) ## when *NN* + *JJ*, scale factor of NN 
    parser.add_argument('--JJ_scale', default=1.0, type=float) ## when *NN* + *JJ*, scale factor of JJ 
    parser.add_argument('--t2i_scale', default=0.1, type=float) ## scale factor of reranked t2i sim matrix 
    parser.add_argument('--i2t_scale', default=0.03, type=float) ## scale factor of reranked t2i sim matrix  ## 23-2-23 when --test_mode='i2t', the best performance is from [--principle_of_pool 'i2t' \][--i2t_scale=0.1 \] 
    parser.add_argument('--T', default=1.0, type=float) ## temperature of softmax 
    parser.add_argument('--p_feas_polysemy_pooling', default='max') ## 23-10-9 (used ONLY when p_feas_polysemy == True is activated)  ## p_feas_polysemy_pooling is used ONLY for polysemous p_feas 
    parser.add_argument('--max_mean', default='max_mean') ## pooling type (default: opt.max_mean=='max_mean'), and may be 'mean_mean', 'max_max', 'mean_max' 
    parser.add_argument('--emb_dim', default=2048, type=int) ## (default: 2048), and may be 1024/512/256(manual setting)  ## the dimensions of the features, see --p_feas and --img_feats (2048/1024/512/256/...) 
    parser.add_argument('--each_separate_word', action='store_true') ## 22-8-16 NN/JJ/VB token in the sentence will be added/calculated independently/separately  ## if True, MACK will calculate NN and JJ features independently/separately; otherwise, NN feat will add JJ feat together 
    parser.add_argument('--NN_JJ', type=str, default='NN',
                        help='NN or NN+JJ or NN+JJ+VB or ALL_TYPE') ## 22-8-16 add!  ##  when each_separate_word is True, choose which types of token feats are ready to be calculated separately ['NN', 'NN+JJ', 'NN+JJ+VB', 'ALL_TYPE', ] 
    parser.add_argument('--flag_no_sim_plus_rank', action='store_true') ## 23-2-24 when top_k = 1000/5000, it will speed up testing, the result of [sim] and [rank] will unchanged, but the result of [sim+rerank] will be ignored 
    ## else
    parser.add_argument('--save_argsort', action='store_true') ## 22-8-2 save specific rank of 3 types of test methods(sim, rerank, sim+rerank) for visualization  ## save 3 types (sim, rerank, sim+rerank) of evaluation results(the specific retrieval ranking results for each query), which is beneficial to Visualization 
    opt = parser.parse_args()
    
    
    with open(opt.vocab, mode='r') as f: ## bidirectional-vocab. 27801 words. 
        dicts = json.load(f)
        word2idx = dicts[0] ## dict: str      -> int
        idx2word = dicts[1] ## dict: str(int) -> str
        
    
    p_feas = numpy.load(opt.p_feas, allow_pickle=True) ## (27801, 2048)  ## 22-6-30 my_p_feas_prototype_N_2_26276_220514.npy needs allow_pickle=True to correctly open. 
    
    
    if opt.img_feats.endswith('.h5'): ## dtype('<f4') 
        file=h5py.File(opt.img_feats)
        test_feas = file['ctx'][:] ## (1000, 36, 2048)  ## f30k test dataset has 1000 test (i,t) pairs, with 36 region features (from BU, 2048 dim) per image. 
        file.close()
    
    
    if opt.img_feats.endswith('.npy'):
        test_feas=np.load(opt.img_feats)
        if 'npy_redundance' in opt.__dict__: 
            test_feas = test_feas[::opt.npy_redundance]
        else:
            test_feas = test_feas[::5]
    
    f1 = open(opt.tags, 'rb')
    tags = pickle.load(f1) ## list: 5000 sentences' words with tags  ## tags[0] == [('the', 'DT'), ('man', 'NN'), ('with', 'IN'), ('pierced', 'JJ'), ('ears', 'NNS'), ('is', 'VBZ'), ('wearing', 'VBG'), ('glasses', 'NNS'), ('and', 'CC'), ('an', 'DT'), ('orange', 'JJ'), ('hat', 'NN')]
    f1.close()
    
    f2 = open(opt.parses, 'rb')
    parses = pickle.load(f2) ## list: 5000 sentences' words with (word, word, relation)s  ## parses[0] == [(('wearing', 'VBG'), 'nsubj', ('man', 'NN')), (('man', 'NN'), 'det', ('the', 'DT')), (('man', 'NN'), 'nmod', ('ears', 'NNS')), (('ears', 'NNS'), 'case', ('with', 'IN')), (('ears', 'NNS'), 'amod', ('pierced', 'VBN')), (('wearing', 'VBG'), 'aux', ('is', 'VBZ')), (('wearing', 'VBG'), 'obj', ('glasses', 'NNS')), (('glasses', 'NNS'), 'conj', ('hat', 'NN')), (('hat', 'NN'), 'cc', ('and', 'CC')), (('hat', 'NN'), 'det', ('an', 'DT')), (('hat', 'NN'), 'amod', ('orange', 'JJ'))]
    f2.close()
    
    
    
    if 'fc_ft' in list(opt.__dict__) and opt.fc_ft == 'fc_ft':
        '''## who is A ?'''
        A_0 = test_feas
        A_1 = p_feas ## (27801, 2048)
        
        
        '''## who are W and b ?'''
        checkpoint = torch.load(opt.fc_ft_model_file)
        
        ## img_enc 
        W_0 = checkpoint['model'][0]['module.fc.weight'].cpu().numpy() ## (2048, 2048)
        b_0 = checkpoint['model'][0]['module.fc.bias'].cpu().numpy() ## (2048,)
        
        ## txt_enc 
        W_1 = checkpoint['model'][1]['module.fc.weight'].cpu().numpy() ## (2048, 2048)
        b_1 = checkpoint['model'][1]['module.fc.bias'].cpu().numpy() ## (2048,)
        
        
        '''## B=WA+b ?'''
        B_0 = np.matmul(A_0, W_0.T)+b_0 ## [1000*36, 2048] x [in_features=2048, out_features=2048] + [out_features=2048, ]  ## 19s
        B_1 = np.matmul(A_1, W_1.T)+b_1 ## [27801, 2048] x [in_features=2048, out_features=2048] + [out_features=2048, ]  ## 1s
        B_0.shape ## (1000, 36, 2048)
        B_1.shape ## (27801, 2048)
        
        
        '''## after A→B，who is B ?'''
        test_feas = B_0 ## (1000, 36, 2048)
        p_feas = B_1 ## (27801, 2048)
    
    
    
    if opt.eval_type == 'eval_once': 
        eval_func = eval_once
    
    sim_clip = numpy.load(opt.sims) ## f30k: (1000, 5000); coco: (5000, 25000)
    
    if opt.test_type == ['', '1K', '5-fold-1K', '5K', '100', ][1]: ## '1K'
        sim_clip = sim_clip[:1000,:5000]
        assert sim_clip.shape[0] == 1000
        _ = show_score(sim_clip)
        str1, str2, score, _2, _3 = _
        (r1, r5, r10, medr, meanr, ranks, top1) = _2
        (r1i, r5i, r10i, medri, meanri, ranks_t2i, top1i) = _3
        result = eval_func(opt, sim_clip, tags[:5000], word2idx, p_feas, parses[:5000], test_feas[:1000])
        print('r1, r5, r10, r1+r5+r10')
        for itm in result: 
            print('{}, {}, {}, {}'.format(itm[0], itm[1], itm[2], itm[3]))
    elif opt.test_type == ['', '1K', '5-fold-1K', '5K', '100', ][4]: ## '100'
        sim_clip = sim_clip[:100,:500]
        assert sim_clip.shape[0] == 100
        _ = show_score(sim_clip)
        str1, str2, score, _2, _3 = _
        (r1, r5, r10, medr, meanr, ranks, top1) = _2
        (r1i, r5i, r10i, medri, meanri, ranks_t2i, top1i) = _3
        result = eval_func(opt, sim_clip, tags[:500], word2idx, p_feas, parses[:500], test_feas[:100])
        print('r1, r5, r10, r1+r5+r10')
        for itm in result: 
            print('{}, {}, {}, {}'.format(itm[0], itm[1], itm[2], itm[3]))
    elif opt.test_type == ['', '1K', '5-fold-1K', '5K', '100', ][2]: ## '5-fold-1K'
        assert sim_clip.shape[0] == 5000
        strs1 = []
        strs2 = []
        rslts = []
        
        sim_avg_1 = []
        sim_avg_5 = []
        sim_avg10 = []
        sim_avg_r = []
        
        rank_avg_1 = []
        rank_avg_5 = []
        rank_avg10 = []
        rank_avg_r = []
        
        comb_avg_1 = []
        comb_avg_5 = []
        comb_avg10 = []
        comb_avg_r = []
        
        _sim_avg_1 = []
        _sim_avg_5 = []
        _sim_avg10 = []
        _sim_avg_r = []
        
        _rank_avg_1 = []
        _rank_avg_5 = []
        _rank_avg10 = []
        _rank_avg_r = []
        
        _comb_avg_1 = []
        _comb_avg_5 = []
        _comb_avg10 = []
        _comb_avg_r = []
        
        for i in range(5):
            _ = show_score(sim_clip[i*1000:(i+1)*1000,i*5000:(i+1)*5000])
            str1, str2, score, _2, _3 = _
            (r1, r5, r10, medr, meanr, ranks, top1) = _2
            (r1i, r5i, r10i, medri, meanri, ranks_t2i, top1i) = _3
            strs1 += [ str1 ]
            strs2 += [ str2 ]
            result = eval_func(opt, sim_clip[i*1000:(i+1)*1000,i*5000:(i+1)*5000], tags[i*5000:(i+1)*5000], word2idx, p_feas, parses[i*5000:(i+1)*5000], test_feas[i*1000:(i+1)*1000])
            rslts += [ result ]
            
            sim_avg_1 += [ result[0][0] ]
            sim_avg_5 += [ result[0][1] ]
            sim_avg10 += [ result[0][2] ]
            sim_avg_r += [ result[0][3] ]
            
            rank_avg_1 += [ result[1][0] ]
            rank_avg_5 += [ result[1][1] ]
            rank_avg10 += [ result[1][2] ]
            rank_avg_r += [ result[1][3] ]
            
            comb_avg_1 += [ result[2][0] ]
            comb_avg_5 += [ result[2][1] ]
            comb_avg10 += [ result[2][2] ]
            comb_avg_r += [ result[2][3] ]
            
            if len(result) == 6: ## t2i+i2t
                _sim_avg_1 += [ result[3][0] ]
                _sim_avg_5 += [ result[3][1] ]
                _sim_avg10 += [ result[3][2] ]
                _sim_avg_r += [ result[3][3] ]
                
                _rank_avg_1 += [ result[4][0] ]
                _rank_avg_5 += [ result[4][1] ]
                _rank_avg10 += [ result[4][2] ]
                _rank_avg_r += [ result[4][3] ]
                
                _comb_avg_1 += [ result[5][0] ]
                _comb_avg_5 += [ result[5][1] ]
                _comb_avg10 += [ result[5][2] ]
                _comb_avg_r += [ result[5][3] ]
        
        print(strs1)
        print(strs2)
        print('r1, r5, r10, r1+r5+r10')
        for i, result in enumerate(rslts): 
            print("{} th 1K".format(i+1))
            for itm in result: 
                print('{}, {}, {}, {}'.format(itm[0], itm[1], itm[2], itm[3]))
        
        print('5-fold-1K: (sim, rank, comb)')
        print('r1, r5, r10, r1+r5+r10')
        print('{}, {}, {}, {}'.format(np.mean(sim_avg_1), np.mean(sim_avg_5), np.mean(sim_avg10), np.mean(sim_avg_r)))
        print('{}, {}, {}, {}'.format(np.mean(rank_avg_1), np.mean(rank_avg_5), np.mean(rank_avg10), np.mean(rank_avg_r)))
        print('{}, {}, {}, {}'.format(np.mean(comb_avg_1), np.mean(comb_avg_5), np.mean(comb_avg10), np.mean(comb_avg_r)))
        if len(result) == 6: ## t2i+i2t
            print('{}, {}, {}, {}'.format(np.mean(_sim_avg_1), np.mean(_sim_avg_5), np.mean(_sim_avg10), np.mean(_sim_avg_r)))
            print('{}, {}, {}, {}'.format(np.mean(_rank_avg_1), np.mean(_rank_avg_5), np.mean(_rank_avg10), np.mean(_rank_avg_r)))
            print('{}, {}, {}, {}'.format(np.mean(_comb_avg_1), np.mean(_comb_avg_5), np.mean(_comb_avg10), np.mean(_comb_avg_r)))
        
    elif opt.test_type == ['', '1K', '5-fold-1K', '5K', ][3]: ## '5K'
        assert sim_clip.shape[0] == 5000
        _ = show_score(sim_clip)
        str1, str2, score, _2, _3 = _
        (r1, r5, r10, medr, meanr, ranks, top1) = _2
        (r1i, r5i, r10i, medri, meanri, ranks_t2i, top1i) = _3
        result = eval_func(opt, sim_clip, tags[:25000], word2idx, p_feas, parses[:25000], test_feas[:5000])
        print('r1, r5, r10, r1+r5+r10')
        for itm in result: 
            print('{}, {}, {}, {}'.format(itm[0], itm[1], itm[2], itm[3]))
    else: ## ''
        _ = show_score(sim_clip)
        str1, str2, score, _2, _3 = _
        (r1, r5, r10, medr, meanr, ranks, top1) = _2
        (r1i, r5i, r10i, medri, meanri, ranks_t2i, top1i) = _3
        result = eval_func(opt, sim_clip, tags, word2idx, p_feas, parses, test_feas)
        print('r1, r5, r10, r1+r5+r10')
        for itm in result: 
            print('{}, {}, {}, {}'.format(itm[0], itm[1], itm[2], itm[3]))
    



