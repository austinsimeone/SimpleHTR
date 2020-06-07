#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:45:44 2020

@author: austin
"""

from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import os

class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2


class Model:
        "Minimalistic TF model for HTR"
        
        #model constants
        batchSize = 50
        imgSize =(128,32)
        maxTextLen = 32
        
        def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore = False, dump = False):
            self.dump = dump
            self.charList = charList
            self.decoderType = decoderType
            self.mustRestore = mustRestore
            self.snapID = 0
            
            self.is_train = tf.keras.Input(dtype = tf.bool, name='is_train')
            
            
            self.inputImgs = tf.keras.input(dtype = tf.float32,
                                            shape=(None, 
                                                   Model.imgSize[0], 
                                                   Model.imgSize[1]
                                                   )
                                            )
            
            self.setupCNN()
            self.setupRNN()
            self.setupCTC()
            
            # setup optimizer to train NN
            self.batchesTrained = 0
            self.learningRate = tf.keras.input(dtype =  tf.float32, shape = [])
        
        def setupCNN(self):
            "create CNN layers and return output of these layers"
            cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)
            
            # list of parameters for the layers
            kernelVals = [5, 5, 3, 3, 3]
            featureVals = [1, 32, 64, 128, 128, 256]
            strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
            numLayers = len(strideVals)
            
            # create layers
            pool = cnnIn4d # input to first CNN layer
            for i in range(numLayers):
                kernel = tf.Variable(tf.random.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
                conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
                conv_norm = tf.layers.keras.Batch_Normalization(conv, training=self.is_train)
                relu = tf.nn.relu(conv_norm)
                pool = tf.nn.max_pool2d(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')
                
            self.cnnOut4d = pool
            
        def setupRNN(self):
            "create RNN layers and return output of these layers"
            rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])
  
            # basic cells which is used to build RNN
            numHidden = 256
            cells = [tf.keras.layers.LSTMCell(units=numHidden) for _ in range(2)] # 2 layers
  
            # stack basic cells
            stacked = tf.keras.layers.StackedRNNCells(cells)
            # bidirectional RNN
            # BxTxF -> BxTx2H
            ((fw, bw), _) = tf.keras.layers.Bidirectional(layer=stacked, backward_layer=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
  									
            # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
            concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
  									
            # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
            kernel = tf.Variable(tf.random.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
            self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
            
            
        def setupCTC(self):
            "create CTC loss and decoder and return them"
            # BxTxC -> TxBxC
            self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
            # ground truth text as sparse tensor
            self.gtTexts = tf.SparseTensor(tf.keras.Input(tf.int64, shape=[None, 2]) , tf.keras.Input(tf.int32, [None]), tf.keras.Input(tf.int64, [2]))
    
            # calc loss for batch
            self.seqLen = tf.keras.Input(tf.int32, [None])
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))
    
            # calc loss for each element to compute label probability
            self.savedCtcInput = tf.keras.Input(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
            self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)
    
            # decoder: either best path decoding or beam search decoding
            if self.decoderType == DecoderType.BestPath:
                self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
            elif self.decoderType == DecoderType.BeamSearch:
                 self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50) 
# =============================================================================
#             elif self.decoderType == DecoderType.WordBeamSearch:
#     			# import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
#                 word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')
#     
#     			# prepare information about language (dictionary, characters in dataset, characters forming words) 
#                 chars = str().join(self.charList)
#                 wordChars = open('../model/wordCharList.txt').read().splitlines()[0]
#                 corpus = open('../data/corpus.txt').read()
#     
#                 # decode using the "Words" mode of word beam search
#                 self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, axis=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))
# 
# =============================================================================



        def toSparse(self, texts):
        		"put ground truth texts into sparse tensor for ctc_loss"
        		indices = []
        		values = []
        		shape = [len(texts), 0] # last entry must be max(labelList[i])
        
        		# go over all texts
        		for (batchElement, text) in enumerate(texts):
        			# convert to string of label (i.e. class-ids)
        			labelStr = [self.charList.index(c) for c in text]
        			# sparse tensor must have size of max. label-string
        			if len(labelStr) > shape[1]:
        				shape[1] = len(labelStr)
        			# put each label into sparse tensor
        			for (i, label) in enumerate(labelStr):
        				indices.append([batchElement, i])
        				values.append(label)
        
        		return (indices, values, shape)
            
        def decoderOutputToText(self, ctcOutput, batchSize):
        		"extract texts from output of CTC decoder"
        		
        		# contains string of labels for each batch element
        		encodedLabelStrs = [[] for i in range(batchSize)]
        
        		# word beam search: label strings terminated by blank
        		if self.decoderType == DecoderType.WordBeamSearch:
        			blank=len(self.charList)
        			for b in range(batchSize):
        				for label in ctcOutput[b]:
        					if label==blank:
        						break
        					encodedLabelStrs[b].append(label)
        
        		# TF decoders: label strings are contained in sparse tensor
        		else:
        			# ctc returns tuple, first element is SparseTensor 
        			decoded=ctcOutput[0][0] 
        
        			# go over all indices and save mapping: batch -> values
        			idxDict = { b : [] for b in range(batchSize) }
        			for (idx, idx2d) in enumerate(decoded.indices):
        				label = decoded.values[idx]
        				batchElement = idx2d[0] # index according to [b,t]
        				encodedLabelStrs[batchElement].append(label)
        
        		# map labels to chars for all batch elements
        		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]
                   
