# ------------------------------------------
# TextDiffuser: Diffusion Models as Text Painters
# Paper Link: https://arxiv.org/abs/2305.10855
# Code Link: https://github.com/microsoft/unilm/tree/master/textdiffuser
# Copyright (c) Microsoft Corporation.
# This file provides the inference script.
# ------------------------------------------

import os
import re
import copy
import fastwer
import numpy as np

gts = {
    # 'ChineseDrawText': [],
    # 'DrawBenchText': [],
    # 'DrawTextCreative': [],
    'LAIONEval4000': [],
    # 'OpenLibraryEval500': [],
    'TMDBEval500': [],
}

results = {
    # 'controlnet_canny': {'cnt':0, 'p':0, 'r':0, 'f':0, 'acc':0},
    # 'controlnet_seg': {'cnt':0, 'p':0, 'r':0, 'f':0, 'acc':0},
    # 'controlnet_seg_glyph': {'cnt':0, 'p':0, 'r':0, 'f':0, 'acc':0},
    # 'controlnet_scribble': {'cnt':0, 'p':0, 'r':0, 'f':0, 'acc':0},

    'controlnet_canny': {'cer':[], 'wer':[]},
    # 'controlnet_seg': {'cer':[], 'wer':[]},
    # 'controlnet_seg_glyph': {'cer':[], 'wer':[]},
    # 'controlnet_scribble': {'cer':[], 'wer':[]},

}

def get_key_words(text: str):
    words = []
    text = text
    matches = re.findall(r"'(.*?)'", text) # find the keywords enclosed by ''
    if matches:
        for match in matches:
            words.extend(match.split())
   
    return words


# load gt
files = os.listdir('MARIOEval')
for file in files:
    print(file)
    if file == 'OpenLibraryEval500':
      continue
    lines = open(os.path.join('MARIOEval', file, f'{file}.txt')).readlines()
    for line in lines:
        line = line.strip().lower()
        gts[file].append(get_key_words(line))
# print(gts['TMDBEval500'][:10])

def get_ocr_metrics(pred,gt):
    pred_str = ' '.join(pred) 
    gt_str = ' '.join(gt)

    cer = (fastwer.score_sent(pred_str, gt_str, char_level=True) + 1e-8)/100
    wer = (fastwer.score_sent(pred_str, gt_str, char_level=False) + 1e-8)/100
    return cer, wer

def get_p_r_acc(method, pred, gt):

    pred = [p.strip().lower() for p in pred] 
    gt = [g.strip().lower() for g in gt]

    pred_orig = copy.deepcopy(pred)
    gt_orig = copy.deepcopy(gt)

    pred_length = len(pred)
    gt_length = len(gt)

    for p in pred:
        if p in gt_orig:
            pred_orig.remove(p) 
            gt_orig.remove(p)

    p = (pred_length - len(pred_orig)) / (pred_length + 1e-8)
    r = (gt_length - len(gt_orig)) / (gt_length + 1e-8)
   
    pred_sorted = sorted(pred)
    gt_sorted = sorted(gt)
    if ''.join(pred_sorted) == ''.join(gt_sorted):
        acc = 1
    else:
        acc = 0

    return p, r, acc


files = [f for f in os.listdir('generation/ocr_result/') if os.path.isfile(f'generation/ocr_result/{f}')]
print(len(files))

for file in files:
    method, dataset,image_index = file.strip().split('-')
    if (method== 'controlnet_canny' and dataset == 'TMDBEval500'):
      with open(os.path.join('generation/ocr_result/', file), 'r') as f:
        ocrs = f.readlines()
        for i in range(500):
          print(i, ocrs[i].strip().split(","),gts[dataset][i])
          # p, r, acc = get_p_r_acc(method, ocrs[i].strip().split(","), gts[dataset][i])
          # results[method]['cnt'] += 1
          # results[method]['p'] += p
          # results[method]['r'] += r
          # results[method]['acc'] += acc

          cer, wer = get_ocr_metrics(ocrs[i].strip().split(","), gts[dataset][i])
          results[method]['cer'].append(cer)
          results[method]['wer'].append(wer)

for method in results.keys():
    print(method)
    # results[method]['p'] /= results[method]['cnt']
    # results[method]['r'] /= results[method]['cnt']
    # results[method]['f'] = 2 * results[method]['p'] * results[method]['r'] / (results[method]['p'] + results[method]['r'] + 1e-8)
    # results[method]['acc'] /= results[method]['cnt']
    
    results[method]['cer']  = np.array(results[method]['cer'],dtype=np.float64)
    results[method]['wer']  = np.array(results[method]['wer'],dtype=np.float64)

    results[method]['cer'] = np.mean(results[method]['cer'][np.isfinite(results[method]['cer'])])
    results[method]['wer'] = np.mean(results[method]['wer'][np.isfinite(results[method]['wer'])])


print(results)

