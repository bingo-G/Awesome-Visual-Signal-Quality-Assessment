# Visual-Signal-Quality-Assessment

Quality assessment crowdsourcing tools: https://github.com/microsoft/P.910 

## paper for visual signal quality assessment
### Table of contents
<!-- - [Survey paper](#survey-paper)
- [Table of contents](#table-of-contents) -->
- [VQA](#VQA)
- [IQA](#IQA)
- [Explainable IQA](#Explainable_IQA)
   - [LMM based IQA](#LMM-based_IQA)
- [Openworld IQA](#Openworld_IQA)
   - [Unified Pretraining](#Unified_Pretraining)
   - [Continual Learning](#Continual_Learning)
   - [UDA IQA](#UDA_IQA)
- [Competition](#Competition)
- [Aesthetic](#Aesthetic)
- [Medical](#Medical)
- [Benchmark](#Benchmark)
  <!-- - [Recommended Datasets](#recommended-datasets)
  - [All Datasets](#all-datasets) -->

### VQA
| Models | Paper | First Author | Venue | Topic | Project |
| :--- | :---: | :---: | :---: | :---: | :---: |
| PTQE | [Blind Video Quality Prediction by Uncovering Human Video Perceptual Representation](https://ieeexplore.ieee.org/document/10667010) | Liang Liao | TIP2024 | UGC | []() |
| ExplainableVQA | [Towards Explainable In-the-Wild Video Quality Assessment: A Database and a Language-Prompted Approach](https://arxiv.org/abs/2305.12726) | Haoning Wu | ACM MM2023 | UGC | [![Stars](https://img.shields.io/github/stars/VQAssessment/ExplainableVQA.svg?style=social&label=Star)](https://github.com/VQAssessment/ExplainableVQA) |
| MD-VQA | [MD-VQA: Multi-Dimensional Quality Assessment for UGC Live Videos](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_MD-VQA_Multi-Dimensional_Quality_Assessment_for_UGC_Live_Videos_CVPR_2023_paper.pdf) | Zicheng Zhang | CVPR2023 | Live Video | [![Stars](https://img.shields.io/github/stars/zzc-1998/MD-VQA.svg?style=social&label=Star)](https://github.com/zzc-1998/MD-VQA) |
| DOVER | [Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Exploring_Video_Quality_Assessment_on_User_Generated_Contents_from_Aesthetic_ICCV_2023_paper.pdf) | Haoning Wu | ICCV2023 | UGC | [![Stars](https://img.shields.io/github/stars/VQAssessment/DOVER.svg?style=social&label=Star)](https://github.com/VQAssessment/DOVER) |
| FAST-VQA | [FAST-VQA: Efficient End-to-End Video Quality Assessment with Fragment Sampling](https://link.springer.com/chapter/10.1007/978-3-031-20068-7_31) | Haoning Wu | ECCV2022 | UGC | [![Stars](https://img.shields.io/github/stars/VQAssessment/FAST-VQA-and-FasterVQA.svg?style=social&label=Star)](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA) |
| BVQI | [Exploring Opinion-unaware Video Quality Assessment with Semantic Affinity Criterion](https://link.springer.com/chapter/10.1007/978-3-031-20068-7_31) | Haoning Wu | ICME2023 | zero-shot | [![Stars](https://img.shields.io/github/stars/vqassessment/BVQI.svg?style=social&label=Star)](https://github.com/vqassessment/BVQI) |
| DisCoVQA | [DisCoVQA: Temporal Distortion-Content Transformers for Video Quality Assessment](https://dl.acm.org/doi/10.1109/TCSVT.2023.3249741) | Haoning Wu | TCSVT2023 | UGC | [![Stars](https://img.shields.io/github/stars/VQAssessment/DisCoVQA.svg?style=social&label=Star)](https://github.com/VQAssessment/DisCoVQA) |
| SimpleVQA | [A Deep Learning based No-reference Quality Assessment Model for UGC Videos](https://dl.acm.org/doi/abs/10.1145/3503161.3548329) | Wei Sun | ACM MM2022 | UGC | [![Stars](https://img.shields.io/github/stars/sunwei925/SimpleVQA.svg?style=social&label=Star)](https://github.com/sunwei925/SimpleVQA) |
| VSFA | [Quality Assessment of In-the-Wild Videos](https://dl.acm.org/doi/abs/10.1145/3503161.3548329) | Dingquan Li | ACM MM2019 | UGC | [![Stars](https://img.shields.io/github/stars/lidq92/VSFA.svg?style=social&label=Star)](https://github.com/lidq92/VSFA) |

### IQA
| Paper | First Author | Training Way | Venue | Topic | Project |
| :--- | :---: | :--: | :--: |:--: | :--: |
| [MUSIQ: Multi-scale Image Quality Transformer (ICCV2021) google research](https://openaccess.thecvf.com/content/ICCV2021/papers/Ke_MUSIQ_Multi-Scale_Image_Quality_Transformer_ICCV_2021_paper.pdf) | Junjie Ke | | ICCV2021 | Transformer | | [code](https://github.com/google-research/google-research/tree/master/musiq) |
| [Image Quality Assessment: From Mean Opinion Score to Opinion Score Distribution](https://dl.acm.org/doi/abs/10.1145/3503161.3547872) | Yixuan Gao |  | ACM MM2022| | []() |
| [No-Reference Image Quality Assessment via Transformers, Relative Ranking, and Self-Consistency](https://openaccess.thecvf.com/content/WACV2022/papers/Golestaneh_No-Reference_Image_Quality_Assessment_via_Transformers_Relative_Ranking_and_Self-Consistency_WACV_2022_paper.pdf) | S. Alireza Golestaneh |  | WACV2022 | Transformer | [![Stars](https://img.shields.io/github/stars/isalirezag/TReS.svg?style=social&label=Star)](https://github.com/isalirezag/TReS) |
| [Local Distortion Aware Efficient Transformer Adaptation for Image Quality Assessment](https://arxiv.org/pdf/2308.12001) | Kangmin Xu |  |  | Transformer | | []() |
| [Data-Efficient Image Quality Assessment with Attention-Panel Decoder](https://ojs.aaai.org/index.php/AAAI/article/view/25302) | Guanyi Qin |  | AAAI2023 | Transformer | [![Stars](https://img.shields.io/github/stars/narthchin/DEIQT.svg?style=social&label=Star)](https://github.com/narthchin/DEIQT) |
| [DifFIQA: Face Image Quality Assessment Using Denoising Diffusion Probabilistic Models (axriv2023)](https://arxiv.org/pdf/2305.05768) | Žiga Babnik |  | | DDPM | [![Stars](https://img.shields.io/github/stars/LSIbabnikz/DifFIQA.svg?style=social&label=Star)](https://github.com/LSIbabnikz/DifFIQA) |
| [Image Quality Assessment: Unifying Structure and Texture Similarity  (TIPAMI2020)](https://ieeexplore.ieee.org/abstract/document/9298952) | Keyan Ding |  | TIPAMI2020 | VGG16 | []() |
| [Content-Variant Reference Image Quality Assessment via Knowledge Distillation(AAAI2022)](https://ojs.aaai.org/index.php/AAAI/article/view/20221) | Guanghao Yin |  | AAAI2022 | | [![Stars](https://img.shields.io/github/stars/guanghaoyin/CVRKD-IQA.svg?style=social&label=Star)](https://github.com/guanghaoyin/CVRKD-IQA) |
| [Learning Conditional Knowledge Distillation for Degraded-Reference Image Quality Assessment (ICCV2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Learning_Conditional_Knowledge_Distillation_for_Degraded-Reference_Image_Quality_Assessment_ICCV_2021_paper.pdf) | Heliang Zheng |  | ICCV2021 | | [![Stars](https://img.shields.io/github/stars/researchmm/CKDN.svg?style=social&label=Star)](https://github.com/researchmm/CKDN) |
| [Data-Driven Transform-Based Compressed Image Quality Assessment](https://ieeexplore.ieee.org/abstract/document/9274532) | Xinfeng Zhang |  |TCSVT2020 | Transformer | []() |
| [Spatio-Temporal Deformable Convolution for Compressed Video Quality Enhancement(for enhancement)](https://ojs.aaai.org/index.php/AAAI/article/view/6697) | Jianing Deng |  | AAAI2020 | | [![Stars](https://img.shields.io/github/stars/ryanxingql/stdf-pytorch.svg?style=social&label=Star)](https://github.com/ryanxingql/stdf-pytorch) |
| [Adaptive Image Quality Assessment via Teaching Large Multimodal Model to Compare](https://arxiv.org/pdf/2405.19298) | Hanwei Zhu |  | NeurIPS2024 |  | [![Stars](https://img.shields.io/github/stars/Q-Future/Compare2Score.svg?style=social&label=Star)](https://github.com/Q-Future/Compare2Score) |
### Explainable IQA
#### LMM-based IQA
| Model| Paper |  Venue | Topic | Project |
| :--- | :--: | :--: |:--: | :--: |
|| [Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective ]() | CVPR2023 | CLIP | [code](https://github.com/zwx8981/LIQE) |
|| [Exploring CLIP for Assessing the Look and Feel of Images ]() |  AAAI2023| CLIP | [code](https://github.com/IceClear/CLIP-IQA) |
|| [Towards Robust Text-Prompted Semantic Criterion for In-the-Wild Video Quality Assessment]() |axriv2023 | | [code]() |
|| [VILA: Learning Image Aesthetics from User Comments with Vision-Language Pretraining]() | CVPR2023 | VILA | [code]() |
|| [Advancing Zero-Shot Digital Human Quality Assessment through Text-Prompted Evaluation]() | axriv2023 | CLIP| [code](https://github.com/zzc-1998/SJTU-H3D) |
|| [Q-Ground: Image Quality Grounding with Large Multi-modality Models]() | ACMMM2024 | PixeLM | [code]() |
|| [Towards open-ended visual quality comparison]() | ECCV2024 | | [code](https://github.com/Q-Future/Co-Instruct) |
|| [A-Bench: Are LMMs Masters at Evaluating AI-generated Images? ]| Axriv2024 |  | [code](https://github.com/Q-Future/A-Bench) |
|| [Q-BENCH: A Benchmark for Multi-modal Foundation Models on Low-level Vision from Single Images to Pairs]() | TIPAMI2024 |  | [code](https://huggingface.co/datasets/q-future/Q-Bench2-HF) |
|| [Q-Instruct: Improving low-level visual abilities for multi-modality foundation models ]() | CVPR2024 | MLLM | [code](https://github.com/Q-Future/Q-Instruct) |
|| [Descriptive Image Quality Assessment in the Wild ]() | Axriv2024 | | [code](https://depictqa.github.io/depictqa-wild/) |
|| [Depicting beyond scores: Advancing image quality assessment through multi-modal language models]() | ECCV2024 |  | [code](https://github.com/XPixelGroup/DepictQA) |


### Openworld IQA

#### Unified Pretraining
| Model | Paper | First Author | Training Way | Venue | Topic | Project |
| :--- | :--- | :---: | :--: | :--: |:--: | :--: |
|| [Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild (CVPR2023)]() |  |  | | | []() |
|| [Image Quality Assessment using Contrastive Learning (TIP2022)]() |  |  | | | []() |
|| [Adaptive Image Quality Assessment via Teaching Large Multimodal Model to Compare(Axriv2024)]() |  |  | | | []() |
|| [Q-Align: Teaching lmms for visual scoring via discrete text-defined levels (ICML2024)]() |  |  | | | []() |
|| [PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts (ECCV2023)]() |  |  | | | []() |
|| []() |  |  | | | []() |

#### Continual Learning
|Model| Paper | First Author | Training Way | Venue | Topic | Project |
| :--- | :---: | :---: | :--: | :--: |:--: | :--: |
|| [Continual Learning for Blind Image Quality Assessment (TIPAMI2021)]() |  |  | | | []() |
|| []() |  |  | | | []() |
|| []() |  |  | | | []() |
|| []() |  |  | | | []() |
#### UDA IQA
|Model|Paper | First Author | Training Way | Venue | Topic | Project |
| :--- | :---: | :---: | :--: | :--: |:--: | :--: |
|| [No-Reference Point Cloud Quality Assessment via Domain Adaptatio (CVPR2022)]() |  |  | | | []() |
|| [TTL-IQA: Transitive Transfer Learning based No-reference Image Quality Assessment (TMM)]() |  |  | | | []() |
|| []() |  |  | | | []() |
|| []() |  |  | | | []() |
|| []() |  |  | | | []() |
|| []() |  |  | | | []() |
|| []() |  |  | | | []() |


## Competition
|Model| Paper | First Author | Venue | Topic | Project |
| :--- | :---: | :---: | :--: | :--: |:--: |
|| [NTIRE 2022 Challenge on Perceptual Image Quality Assessment](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Gu_NTIRE_2022_Challenge_on_Perceptual_Image_Quality_Assessment_CVPRW_2022_paper.pdf) |  |  | | | []() |


## Aesthetic
|Model| Paper | First Author | Venue | Topic | Project |
| :--- | :---: | :---: | :--: | :--: |:--: |
|| [Towards Artistic Image Aesthetics Assessment: A Large-Scale Dataset and a New Method](https://github.com/Dreemurr-T/BAID) |  |  | | | []() |


## Medical
|Model| Paper | First Author | Venue | Topic | Project |
| :--- | :---: | :---: | :--: | :--: |:--: |
|| [Md-iqa: Learning Multi-scale Distributed Image Quality Assessment with Semi-supervised Learning for Low Dose CT](https://ieeexplore.ieee.org/abstract/document/10635481/) |  |  | | | []() |


### Benchmark
|Dataset|Task|Usage|Year|
|:----:|:----:|:----:|:----:|
|LLVisionQA,LLDescribe| [Q-Bench: A Benchmark for General-Purpose Foundation Models on Low-level Vision(ICLR2024)](https://github.com/Q-Future/Q-Bench) | MLLM benchmark | 2023.9 | | | []() |
|AIGIQA-20K| [Aigiqa-20k: A large database for ai-generated image quality assessment (CVPRW2024)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Li_AIGIQA-20K_A_Large_Database_for_AI-Generated_Image_Quality_Assessment_CVPRW_2024_paper.pdf) | AIGC image | 2024.4 | | | []() |
|T2VQA-DB| [Subjective-Aligned Dateset and Metric for Text-to-Video Quality Assessment(Axriv2024)](https://github.com/QMME/T2VQA) | Text-to-Video  | 2024.3 | | | []() |

