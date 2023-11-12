
# Wait-k with Context-Consistency-Bi Training

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

**Installing Fairseq**
```bash
cd Waitk-Consistency-Bi-Training
pip install --editable .
```

# Training
Train wait-k model using context-consistency-bi with the following command:
```bash
CUDA_VISIBLE_DEVICES=$gpu python $code_dir/train.py $data_bin -s $SRC -t $TGT --left-pad-source False \
    --user-dir $code_dir/examples/waitk --arch $arch --save-dir $MODEL  \
    --ddp-backend=no_c10d --fp16 \
    --seed 1 --no-epoch-checkpoints --no-progress-bar --log-interval 10  \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 --clip-norm 0.1 --dropout 0.3 \
    --max-tokens 4000 --update-freq 2  --max-update 2300 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --lr 5e-4 --min-lr '1e-9' \
    --criterion minimum_risk_training_loss_with_al \
    --share-decoder-input-output-embed --waitk  $k  \
    --reset-optimizer --reset-lr-scheduler  --reset-meters \
    --mrt-beam-size $beam_size \
    --mrt-seq-max-len-a 1.5 --mrt-seq-max-len-b 5 \
    --mrt-temperature $temperature \
    --mrt-greedy false \
    --mrt-alpha $alpha 
```

# Testing
```bash
CUDA_VISIBLE_DEVICES=$gpu python $code_dir/generate.py $data_bin \
    -s $SRC -t $TGT --gen-subset $subset \
    --path $MODEL/checkpoint_best.pt --task waitk_translation --eval-waitk $testk \
    --model-overrides "{'max_source_positions': 1024, 'max_target_positions': 1024}" --left-pad-source False  \
    --user-dir $code_dir/examples/waitk --no-progress-bar \
    --max-tokens 8000 --remove-bpe --beam 1 
```