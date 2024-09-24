# Improve Cross-domain Mixed Sampling with Guidance Training for Adaptive Segmentation

**by Wenlve Zhou, Zhiheng Zhou, Tianlei Wang, Delu Zeng**

**[[Arxiv]](https://arxiv.org/abs/2403.14995)**
**[[Paper]](https://arxiv.org/pdf/2403.14995)**

**Notice:**

* We apologize for the late release of the code. In fact, we have not yet finalized the design of the method and are still working on improvements to Guidance Training based on the DAFormer architecture.

* The current code corresponds to the version available on Arxiv, but it is not compatible with DAFormer, and using it directly with DAFormer will result in reduced performance.

* Fortunately, the final version is nearing completion, and its performance will surpass that of the previous iteration. The latest code will be released in the project soon, so please stay tuned if you're interested.

## Checkpoints
We have also provided the weight files and log files for "MIC + Guidance Training" on the GTA → Cityscapes task.

**[[log files]](checkpoints/20240128_123801.log)**
**[[weights]](checkpoints/best_mIoU_iter_36000.pth)**

## Training
1. Our code is implemented entirely based on MIC's code, so please follow MIC's steps for environment configuration and dataset preparation before running Guidance Training, 
please refer to [README_mic.md](README_mic.md).


2. For convenience, we provide an annotated config file of the final Guidance Training on GTA→Cityscapes. A training job can be launched using:
```shell
python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_dlv2_s0.py
```

```shell
python run_experiments.py --config configs/hrda/gtaHR2csHR_hrda_dlv2.py
```

```shell
python run_experiments.py --config configs/mic/gtaHR2csHR_mic_hrda_dlv2.py
```
## Acknowledgements

Guidance Training is based on the following open-source projects. We thank their
authors for making the source code publicly available.
* [MIC](https://github.com/lhoyer/MIC)
* [HRDA](https://github.com/lhoyer/HRDA)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
