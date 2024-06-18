# -*- coding: utf-8 -*-

"""
Dataset loading
"""
import numpy, os, nltk, json, h5py, pickle
from vocab import build_dictionary
from PIL import Image
from nltk.tag import StanfordPOSTagger
import os
java_path = "C:/Program Files (x86)\Java\jre1.8.0_321/bin/java.exe"
os.environ['JAVAHOME'] = java_path

  
    
    


caps = []
with open('coco_test.txt', 'rb') as f:
    for line in f:
        caps.append(line.decode('utf-8').strip().lower())
#caps = numpy.array(caps)[::5][:,numpy.newaxis]
caps_ = numpy.array(caps)[:,numpy.newaxis]


eng_tagger = StanfordPOSTagger(model_filename='.\stanford-postagger-full-2020-11-17\models\english-bidirectional-distsim.tagger', path_to_jar='.\stanford-postagger-full-2020-11-17\stanford-postagger.jar')

tags = []
for i in range(25000):
    idx = i
    print(i)
    index = idx//5
  
    
    cap_i = nltk.word_tokenize(caps[idx])
    words = eng_tagger.tag(cap_i)
    tags.append(words)
    
    list_log = []

f= open('coco_tags', 'wb') 
pickle.dump(tags, f)
f.close()

#f1 = open('test', 'rb')
#tags_new = pickle.load(f1)


a = 1
    
    
    

        

    

