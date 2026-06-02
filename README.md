<div align="center">

<img src="assets/robodream_logo.svg" alt="RoboDream Logo" height="90">

# RoboDream: Compositional World Models for Scalable Robot Data Synthesis

[![arXiv](https://img.shields.io/badge/arXiv-2606.02577-b31b1b.svg)](https://arxiv.org/abs/2606.02577)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](http://junjieye.com/RoboDream/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**[Junjie Ye](https://junjieye.com)<sup>1,2</sup>, Rong Xue<sup>1</sup>, Basile Van Hoorick<sup>2</sup>, Runhao Li<sup>1</sup>, Harshitha Belagavi Rajaprakash<sup>1</sup>, Pavel Tokmakov<sup>2</sup>, Muhammad Zubair Irshad<sup>2</sup>, Vitor Guizilini<sup>2,&dagger;</sup>, Yue Wang<sup>1,&dagger;</sup>**

<sup>1</sup>USC Physical Superintelligence (PSI) Lab &nbsp;&nbsp; <sup>2</sup>Toyota Research Institute &nbsp;&nbsp; <sup>&dagger;</sup>Equal advising

<img src="assets/teaser.gif" alt="RoboDream teaser" width="100%">

</div>

## Abstract

Scaling robot learning requires large-scale, diverse demonstrations, yet real-world data collection via teleoperation remains prohibitively expensive and time-consuming. While video diffusion models offer a promising avenue for data scaling, existing generative approaches are often limited to superficial visual augmentation, or suffer from embodiment hallucinations that yield physically infeasible motions.

We present **RoboDream**, a generalizable embodiment-centric world model that achieves scalable data generation by synthesizing photorealistic demonstrations with novel objects, in novel scenes, and from novel viewpoints. Our approach anchors generation to rendered robot motion while conditioning on explicit scene and object priors, effectively decoupling trajectory execution from environment synthesis. This formulation unlocks two powerful data scaling capabilities:

1. **Retrieval and rebirth** — repurposing existing trajectories into entirely new contexts without new motion data; and
2. **Prop-free teleoperation** — where operators manipulate empty air and the model hallucinates the target objects and scene afterwards.

We demonstrate with real-world experiments that our generated data consistently improves downstream policy performance and significantly reduces real-world data requirements across diverse manipulation tasks.

## Code Release

🚧 **Code coming soon.** We are cleaning up the codebase and will release training, data generation, and policy-learning pipelines here. Please ⭐ watch this repository for updates.

## Citation

If you find RoboDream useful in your research, please consider citing:

```bibtex
@article{ye2026robodream,
  title={RoboDream: Compositional World Models for Scalable Robot Data Synthesis},
  author={Ye, Junjie and Xue, Rong and Van Hoorick, Basile and Li, Runhao and Belagavi Rajaprakash, Harshitha and Tokmakov, Pavel and Irshad, Muhammad Zubair and Guizilini, Vitor and Wang, Yue},
  journal={arXiv preprint arXiv:2606.02577},
  year={2026}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE).
