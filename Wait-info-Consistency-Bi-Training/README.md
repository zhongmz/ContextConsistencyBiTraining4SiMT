
# Wait-info with Context-Consistency-Bi Training

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

**Installing Fairseq**
```bash
cd Wait-info-Consistency-Bi-Training
pip install --editable .
```

# Training
Train wait-info using context-consistency-bi with the following command:
```bash
CUDA_VISIBLE_DEVICES=$gpu python $code_dir/train.py  --ddp-backend=no_c10d $data_bin -s $SRC -t $TGT \
    --left-pad-source False \
    --arch $arch \
    --save-dir $MODEL --fp16 \
    --seed 1 --no-epoch-checkpoints --no-progress-bar --log-interval 10  \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 --clip-norm 0.1 --dropout 0.3 \
    --max-tokens 4000 --update-freq 2 --max-update 2300 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --lr 5e-4 --min-lr '1e-9' \
    --min-lr '1e-9' \
    --share-decoder-input-output-embed \
    --criterion minimum_risk_training_loss_with_al \
    --reset-optimizer --reset-lr-scheduler  --reset-meters \
    --mrt-beam-size $beam_size \
    --mrt-seq-max-len-a 1.5 --mrt-seq-max-len-b 5 \
    --mrt-length-penalty $length_penalty  \
    --mrt-temperature $temperature \
    --mrt-greedy $greedy \
    --mrt-waitk $k \
    --mrt-alpha $alpha 
```


# Testing
```bash
CUDA_VISIBLE_DEVICES=$gpu python $code_dir/generate.py $data_bin \
    -s $SRC -t $TGT --gen-subset $subset \
    --path $MODEL/checkpoint_best.pt --test-wait-k $testk \
    --left-pad-source False  --fp16 \
    --no-progress-bar \
    --max-tokens 8000 --remove-bpe --beam 1 
```