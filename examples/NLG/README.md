# Adapting GPT-2 using LoRA

This folder contains the implementation of LoRA in GPT-2 using the Python package `lora` and steps to replicate the results in our recent paper

**LoRA: Low-Rank Adaptation of Large Language Models** <br>
*Edward J. Hu\*, Yelong Shen\*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen* <br>
Paper: https://arxiv.org/abs/2106.09685 <br>

<p>
<img src="figures/LoRA_GPT2.PNG" width="800" >
</p>

This repo reproduces our experiments on GPT-2.

## Repository Overview

Our implementation is based on the fine-tuning code for GPT-2 in [Hugging Face](https://huggingface.co/).
There are several directories in this repo:
* [src/](src) contains the source code used for data processing, training, and decoding.
* [eval/](eval) contains the code for task-specific evaluation scripts.
* [data/](data) contains the raw data we used in our experiments.
* [vocab/](vocab) contains the GPT-2 vocabulary files.

## Getting Started

 1. You can start with the following docker image: `nvcr.io/nvidia/pytorch:20.03-py3` on a GPU-capable machine, but any generic PyTorch image should work.
 ```
 docker pull nvcr.io/nvidia/pytorch:25.03-py3
 // 这里安装最新的NVIDIA镜像即可，见https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
 // 下载的镜像是一个文件，会占用本地的空间；下载之后就可以在这个镜像上运行容器了
 docker run -it --gpus all nvcr.io/nvidia/pytorch:25.03-py3
 // 这个操作是运行容器；--gpus all可以让容器使用GPU
 // 这两行命令都在cmd中运行，然后就可以去docker desktop找对应的容器了
 ```

 2. Clone the repo and install dependencies in a virtual environment (remove sudo if running in docker container):
 ```
 // 这些命令可以在cmd中运行，也可以在docker desktop中对应容器的终端运行
 // 一般来说为了方便，我们在docker desktop中找到对应的容器，然后点击Exec，然后点击Open in external terminal
 // 这个样子就可以在外部终端中运行命令，同时使用docker desktop对下载的文件进行修改（容器中的Files选项可以修改）
 sudo apt-get update
 // 更新软件包
 sudo apt-get -y install git jq virtualenv
 // 安装必要的库：git用于版本控制，下载和管理代码仓库；jq是一个轻量级的 JSON 处理工具，常用于解析和操作 JSON 数据；virtualenv用于创建 Python 虚拟环境，隔离项目依赖
 git clone https://github.com/microsoft/LoRA.git; cd LoRA
 // 克隆仓库到容器的文件夹中；cd LoRA表示进入这个目录
 // 如果要运行本地的代码文件，那么就将这句命令换成docker对本地文件的挂载即可
 virtualenv -p $(which python3) ./venv --system-site-packages
 // 创建名为venv的虚拟环境；最后的--system-site-packages可以让这个虚拟环境访问全局环境的包（全局环境是NVIDIA镜像，所以包含Pytorch）
 // 注意我们下载的NVIDIA镜像已经包含Pytorch了，所以在下面安装requirement的包的时候就不用再安装torch了
 . ./venv/bin/activate
 // 激活虚拟环境
 cd ./examples/NLG
 pip install pillow
 pip install -r requirement.txt
 pip install loralib
 bash download_pretrained_checkpoints.sh
 bash create_datasets.sh
 cd ./eval
 bash download_evalscript.sh
 cd ..
 ```

#### Now we are ready to replicate the results in our paper.

## Replicating Our Result on E2E

1. Train GPT-2 Medium with LoRA (see our paper for hyperparameters for GPT-2 Medium)
首先我们要先对要运行的文件（gpt2_ft.py）进行修改。具体要进行的修改对照https://pytorch.org/docs/stable/elastic/run.html#module-torch.distributed.run即可。如果使用的是torch.distributed.launch，就按照左边的修改；如果使用的是torchrun，就按照右边的修改
注意下面的参数堆GPU的内存要求很高，本机是不可以的，我实验的时候将所有大参数全部改的很小，然后就可以跑了
```
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 1000 \
    --lora_dim 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --random_seed 110
```

2. Generate outputs from the trained model using beam search:
```
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/e2e/model.26289.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --output_file predict.26289.b10p08r4.jsonl
```

3. Decode outputs from step (2)
```
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/e2e/predict.26289.b10p08r4.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt
```

4. Run evaluation on E2E test set

```
python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p
```

## Replicating Our Result on WebNLG

1. Follow steps 1 and 2 from E2E pipeline by replacing references to E2E with webnlg (see our paper for hyperparameters)

2. Decode outputs from beam search (step 2 above)
```
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/webnlg/predict.20000.b10p08.jsonl \
    --input_file ./data/webnlg_challenge_2017/test_formatted.jsonl \
    --ref_type webnlg \
    --ref_num 6 \
    --output_ref_file eval/GenerationEval/data/references_webnlg \
    --output_pred_file eval/GenerationEval/data/hypothesis_webnlg \
    --tokenize --lower
```

3. Run evaluation on WebNLG test set
```
cd ./eval/GenerationEval/
python eval.py \
    -R data/references_webnlg/reference \
    -H data/hypothesis_webnlg \
    -nr 6 \
    -m bleu,meteor,ter 
cd ../..
```

## Replicating Our Result on DART

1. Follow steps 1 and 2 from E2E pipeline by replacing references to E2E with dart (see our paper for hyperparameters)

2. Decode outputs from beam search (step 2 above)
```
python src/gpt2_decode.py \
        --vocab ./vocab \
        --sample_file ./trained_models/GPT2_M/dart/predict.20000.b10p08.jsonl \
        --input_file ./data/dart/test_formatted.jsonl \
        --ref_type dart \
        --ref_num 6 \
        --output_ref_file eval/GenerationEval/data/references_dart \
        --output_pred_file eval/GenerationEval/data/hypothesis_dart \
        --tokenize --lower
```

3. Run evaluation on Dart test set
```
cd ./eval/GenerationEval/
python eval.py \
    -R data/references_dart/reference \
    -H data/hypothesis_dart \
    -nr 6 \
    -m bleu,meteor,ter 
cd ../..
```

## Citation
```
@misc{hu2021lora,
    title={LoRA: Low-Rank Adaptation of Large Language Models},
    author={Hu, Edward and Shen, Yelong and Wallis, Phil and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Lu and Chen, Weizhu},
    year={2021},
    eprint={2106.09685},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
