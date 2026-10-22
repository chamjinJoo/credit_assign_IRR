# Rethinking Credit Assignment in Cooperative MARL via Interventional Reward Response
This is the official implementaion for the paper [Rethinking Credit Assignment in Cooperative MARL via Interventional Reward Response](https://openreview.net/forum?id=FOmIe6PtvT&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2026%2FConference%2FAuthors%23your-submissions)) 
<p align="center"> by Chamjin Joo, Myoung Hoon Ha, Sang Wan Lee </p>

**(TL;DR)**: We address the **credit assginemnt** challenge in cooperative MARL **by formulating agent-level credit as an interventional reward response (IRR)**, using **Proximal Causal Inference (PCI)** to identify credit from observable proxies via an outcome bridge function. Building on this identification strategy, we design a practical credit-aligned update signal and integrate it into policy gradient methods.

## Our approach
We define an IRR, rather than a conditional expectation. Based on the theoretical framework of causal inference. The IRR desribes the causation between an individual agent and the global reward.
<p align="center"><img src="./assets/images/IRR.jpg" width="50%"></p>

To identify IRR, we analyze the Dec-POMDP causal graph to figure out the causal effect of an individual agent on the global reward. Here, another challenge, partial observability, hinders the identification of the causal effect. 

<p align="center"><img src="./assets/images/decpomdp.jpg" alt="IRR" width="80%"></p>

Therefore, we adopt PCI to comprehensivelty handle the two challenges and finally propose an credit-aligned policy-based online MARL framework, **PCI-IPPO/MAPPO**.
<p align="center"><img src="./assets/images/fig2_oracle_val.jpg" alt="Oracle Results" width="60%"></p>
<p align="center"><img src="./assets/images/fig4_perf_prof.jpg" alt="Experimental Results" width="60%"></p>

## What does this code do?

## Get started

Create a conda environment and install dependencies:

```bash
conda create -n env_name python=3.10.18 -y
conda activate env_name
bash install.sh
```

Note: 
- Python ver, Pytorch, Jax 등 필수 library version/version범위 명시.
- `requirements.txt` does not contain all necessary libraries. Please see the `install.sh` for more information. 

## Training scripts

- `scripts/staghunt/`: Torch Stag Hunt experiments.
- `scripts/smax/`: JAX SMAX experiments.

Each script exposes its main settings near the top of the file, such as
`config`, `model`, `seed` or `seeds`, `device`, and environment-specific
options.

Example commands:

```bash
bash scripts/staghunt/train_ippo.sh
bash scripts/smax/train_ippo.sh
```

Some scripts enable Weights & Biases logging by default. If external logging is
not needed, remove or comment out the corresponding `--use-wandb` arguments in
the script before running it.

## Code layout

- `models/torch/`: Torch algorithm implementations used for Stag Hunt.
- `models/jax/`: JAX algorithm implementations used for SMAX.
- `configs/`: YAML and dataclass-based experiment configuration.
- `envs/`: multi-agent environments and wrappers.
- `runner/torch/`, `runner/jax/`: training loops for Torch and JAX.
- `utils/`: shared utility code.

## Citation
```
@inproceedings{anonymous2026rethinking,
    title={Rethinking Credit Assignment in Cooperative {MARL} via Interventional Reward Response},
    author={Anonymous},
    booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
    year={2026},
    url={https://openreview.net/forum?id=FOmIe6PtvT}
}
```