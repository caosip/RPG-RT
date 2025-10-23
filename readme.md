
<a href="https://arxiv.org/abs/2505.21074"><img src="https://img.shields.io/static/v1?label=Paper&message=2505.21074&color=red"></a>

# RPG-RT: Red-Teaming Text-to-Image Systems by Rule-based Preference Modeling

This repository contains the code for *Red-Teaming Text-to-Image Systems by Rule-based Preference Modeling*, accepted by NeurIPS 2025.

## Installation

We provide the `environment.yaml` to create a conda environment. You could create the environment with:
```
conda env create -f environment.yaml
```

## Fine-tuning for RPG-RT

You can choose your target T2I systems in `scripts/run_model.sh` and run RPG-RT with:
```
bash scripts/run_model.sh
```

## Evaluation

Evaluation code for RPG-RT could be found in the notebook `evaluation/evaluation.ipynb`, including the ASR, CLIP similarity (CS), FID and per​​p​​le​​x​​ity (PPL).

## Citation

Please feel free to email us at ```caoyichuan@amss.ac.cn```. If you find this useful, please consider citing our work!

```
@article{cao2025red,
  title={Red-Teaming Text-to-Image Systems by Rule-based Preference Modeling},
  author={Cao, Yichuan and Miao, Yibo and Gao, Xiao-Shan and Dong, Yinpeng},
  journal={arXiv preprint arXiv:2505.21074},
  year={2025}
}
```
