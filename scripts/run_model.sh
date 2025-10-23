#!/bin/bash

device=0
method=SLD-strong

# choose from:
# text-img text-cls text-match img-cls img-clip GuardT2I
# SD-NP SLD-strong SLD-max ESD SafeGen AdvUnlearn SAFREE DUO
# SD2 SD3 SafetyDPO

path=output/${method}
mkdir $path

python scripts/eval_model.py --iter 0 --device $device --path $path --method $method
for i in {1..9}
do
    python scripts/train_score.py --iter $i --device $device --path $path
    CUDA_VISIBLE_DEVICES=$device python scripts/dpo_llm.py --iter $i --path $path
    python scripts/eval_model.py --iter $i --device $device --path $path --method $method
done
python scripts/train_score.py --iter 10 --device $device --path $path
CUDA_VISIBLE_DEVICES=$device python scripts/dpo_llm.py --iter 10 --path $path

python scripts/eval_model.py --iter 10 --device $device --path $path --method $method --eval
