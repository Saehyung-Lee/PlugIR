# [ACL 2024 main] Interactive Text-to-Image Retrieval with Large Language Models: A Plug-and-Play Approach

This repository is the official implementation of [Interactive Text-to-Image Retrieval with Large Language Models: A Plug-and-Play Approach](https://arxiv.org/abs/2405.00000), published as a main (long) paper at ACL 2024.
Our code is based on
1. [ChatIR (Levy et al., NeurIPS 2023)](https://github.com/levymsn/ChatIR)
2. [PyTorch image classification reference](https://github.com/pytorch/vision/tree/main/references/classification)
3. [CLIP](https://github.com/openai/CLIP)

## Prerequisites
### Packages
* [pytorch](https://pytorch.org/get-started/locally/)
* [clip](https://github.com/openai/CLIP)
* [transformers](https://github.com/huggingface/transformers)
* [openai](https://openai.com/index/openai-api/)
### datasets
* [COCO 2017 Unlabeled images](http://images.cocodataset.org/zips/unlabeled2017.zip)
* [VisDial](https://visualdialog.org/data)
```
    VisDial  
    ├── train                    
    │    ├── images          
    │    └── visdial_1.0_train.json                
    └── val
         ├── images  
	 └── visdial_1.0_val.json
```
## Context Reformulation and Context-aware Dialogue Generation
Our method, PlugIR, actively utilizes the general instruction-following capability of LLMs in two ways. First, by transforming the dialogue-form context into a caption-style query, we eliminate the need to fine-tune a retrieval model on existing visual dialogue data, thereby enabling the use of any arbitrary black-box model. Second, we construct the LLM questioner to generate non-redundant questions about the attributes of the target image, based on the information of retrieval candidate images in the current context. This approach mitigates the issues of noisiness and redundancy in the generated questions.
```PlugIR
python generate_dialog.py
```

## Evaluation
In addition to existing metrics (Hits@K and Recall@K), we introduce the Best log Rank Integral (BRI) metric. BRI is a novel metric aligned with human judgment, specifically designed to provide a comprehensive and quantifiable evaluation of interactive retrieval systems.
### Zero-shot baseline
```zero-shot
model=blip
python eval.py --retriever ${model} --cache-corpus cache/corpus_${model}.pth --data-dir <a directory path containing "unlabeled2017">
```
### Fine-tuning baseline
```fine-tuned
model=blip
python eval.py --retriever ${model} --cache-corpus cache/corpus_finetuned_${model}.pth --data-dir <a directory path containing "unlabeled2017"> --ft-model-path <fine-tuned-model.pth>
```
### PlugIR (ours)
```PlugIR
model=blip
python eval.py --retriever ${model} --cache-corpus cache/corpus_finetuned_${model}.pth --data-dir <a directory path containing "unlabeled2017"> --queries-path <our_queries.json> --split
```

## BLIP Text Encoder Fine-tuning
```FT
cd finetune
torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 train.py \
	--data-path <path to the VisDial directory> \
	--amp
```
