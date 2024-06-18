# -*- coding: utf-8 -*-

"""
Dataset loading
"""
import numpy, os, nltk, json, h5py, pickle
#from vocab import build_dictionary
#from PIL import Image
from nltk.parse.stanford import StanfordDependencyParser
import os
from collections import OrderedDict
#java_path = "C:/Program Files (x86)\Java\jre1.8.0_321/bin/java.exe"
java_path = "C:/Program Files/Java/jre1.8.0_311/bin/java.exe"
os.environ['JAVAHOME'] = java_path



caps = []
with open('coco_test.txt', 'rb') as f:
    for line in f:
        caps.append(line.decode('utf-8').strip().lower())



eng_parser = StanfordDependencyParser("./stanford-parser-full-2020-11-17/stanford-parser.jar", 
                                      "./stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar")



parses = []

def f(idx):
    global parses
    
    if idx % 100 == 0:
        print('idx/25000 == {} %'.format(idx/25000*100.))
    
    cap_i = caps[idx]
    res = list(eng_parser.parse(cap_i.split()))
    
    dicts = []
    for row in res[0].triples():
        if idx % 100 == 0:
            print("idx={}, row={}".format(idx, row))
        dicts.append(row)
        
    
    parses.append( (idx, dicts) )


from multiprocessing.pool import ThreadPool
idxs = list(range(0, len(caps)))
results = ThreadPool(6).imap_unordered(f, idxs)
sorted_parses = sorted(parses, key=lambda x:x[0])



final_parses = []
for i in range( len( sorted_parses ) ):
    final_parses.append( sorted_parses[i][1] )

f= open('coco_parses', 'wb') 
pickle.dump(final_parses, f)
f.close()

















