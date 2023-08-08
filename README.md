# Diffusion Reading Group at EleutherAI

This is an ongoing study group occuring the EleutherAI Discord server. You can join the server over [here](https://discord.gg/zBGx3azzUn), then head to the "Diffusion Reading Group" thread under the #reading-groups channel.

[Here is a playlist](https://www.youtube.com/playlist?list=PLXqc0KMM8ZtKVEh8fIWEUaIU43SmWnfdM) of the previous session recordings.

<!--ts-->
* [Diffusion Reading Group at EleutherAI](#diffusion-reading-group-at-eleutherai)
   * [Week 1: DDPM paper](#week-1-ddpm-paper)
   * [Week 2: Score-based generative modeling](#week-2-score-based-generative-modeling)
   * [Week 3: NCSNv2 and Score SDEs](#week-3-ncsnv2-and-score-sdes)
   * [Week 4: Score SDEs and DDIM](#week-4-score-sdes-and-ddim)
   * [Week 5: IDDPM, ADM, and Classifier-Free Guidance](#week-5-iddpm-adm-and-classifier-free-guidance)
   * [Week 6: Review](#week-6-review)
   * [Week 7: Classifier-Free Guidance, VDMs, and Denoising Diffusion GANs](#week-7-classifier-free-guidance-vdms-and-denoising-diffusion-gans)
   * [Week 8: Perception-Prioritized Training, Elucidating Design Spaces](#week-8-perception-prioritized-training-elucidating-design-spaces)
   * [Week 9: DDPM paper, EDM paper code walk-thrus](#week-9-ddpm-paper-edm-paper-code-walk-thrus)
   * [Week 10: Paella](#week-10-paella)
   * [Week 11: Progressive Distillation, Distillation of Guided Models](#week-11-progressive-distillation-distillation-of-guided-models)
   * [Week 12: SDEDit](#week-12-sdedit)
   * [Week 13: Latent Diffusion and Stable Diffusion](#week-13-latent-diffusion-and-stable-diffusion)
   * [Week 14: Q&amp;A with Robin Rombach](#week-14-qa-with-robin-rombach)
   * [Week 15: Soft Diffusion](#week-15-soft-diffusion)
   * [Week 16 &amp; 17: Flow Matching](#week-16--17-flow-matching)
   * [Week 18: Consistency Models](#week-18-consistency-models)
   * [List of papers to cover:](#list-of-papers-to-cover)
<!--te-->

## Week 1: DDPM paper
* Readings:
  * [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
  * [AssemblyAI blog post](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)
* [Recording](https://www.youtube.com/watch?v=B5gfJF8mOPo)
* [Slides](%231%20DDPM%20paper.pdf)

## Week 2: Score-based generative modeling
* Readings:
  * [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)
  * [Yang Song's blog post](https://yang-song.net/blog/2021/score/)
* [Recording](https://youtu.be/iv6K7yo5KgQ)
* [Slides](%232%20Score-based%20generative%20modeling.pdf)

## Week 3: NCSNv2 and Score SDEs
* Readings:
  * [Improved Techniques for Training Score-Based Generative Models](https://arxiv.org/abs/2006.09011)
  * [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
  * [Yang Song's blog post](https://yang-song.net/blog/2021/score/)
* [Recording](https://www.youtube.com/watch?v=NwfkNEGjNus)
* [Slides](%233%20NCSNv2%20and%20Score%20SDE.pdf)

## Week 4: Score SDEs and DDIM
* Readings:
  * [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
  * [Yang Song's blog post](https://yang-song.net/blog/2021/score/)
  * [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
  * [Lilian Weng's blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#speed-up-diffusion-model-sampling)
* [Recording](https://youtu.be/o4dr7tUQryQ)
* [Slides](%234%20Score%20SDEs%20and%20DDIM.pdf)

## Week 5: IDDPM, ADM, and Classifier-Free Guidance
* Readings:
  * [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
  * [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
  * [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
  * [Sander Dieleman's blog post on guidance](https://benanne.github.io/2022/05/26/guidance.html)
* [Recording](https://youtu.be/L4PHJn1VHbY)
* [Slides](%235%20IDDPM%20and%20ADM.pdf)

## Week 6: Review
* Readings:
  * All prior readings
* [Recording](https://www.youtube.com/watch?v=S01qKbC6wdA)
* [Slides](%236%20Review.pdf)

## Week 7: Classifier-Free Guidance, VDMs, and Denoising Diffusion GANs
* Readings: 
  * [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
  * [Sander Dieleman's blog post on guidance](https://benanne.github.io/2022/05/26/guidance.html)
  * [Variational Diffusion Models](https://arxiv.org/abs/2107.00630)
  * [Variational Diffusion Model Colab Notebook](https://colab.research.google.com/github/google-research/vdm/blob/main/colab/SimpleDiffusionColab.ipynb)
  * [Calvin Luo's blog post](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html#variational-diffusion-models)
  * [Tackling the Generative Learning Trilemma with Denoising Diffusion GANs](https://arxiv.org/abs/2112.07804)
  * [Vahdat and Kreis's blog post](https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-2/)
  * [Arash Vahdat's DLCT talk slides](https://rosanneliu.com/dlctfs/dlct_220204.pdf)
* [Recording](https://youtu.be/EuK9BeuOSPs)
* [Slides](%237%20CFG%2C%20VDM%20and%20DDGAN.pdf)

## Week 8: Perception-Prioritized Training, Elucidating Design Spaces
* Readings:
  * [Perception Prioritized Training of Diffusion Models](https://arxiv.org/abs/2204.00227)
    * [Official Codebase](https://github.com/jychoi118/P2-weighting)
  * [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
    * [Official Codebase](https://github.com/NVlabs/edm) 
    * [Katherine's codebase](https://github.com/crowsonkb/k-diffusion/)
  * [Recording](https://youtu.be/ffq6RAmfGzk)
  * [Notebook - P2 paper](https://marii-moe.github.io/quatro-blog/posts/perception-prioritized-training-of-diffusion-models/Perception%20Prioritized%20Training%20of%20Diffusion%20Models.html) (presented by Molly Beavers)
  * [Slides - EDM paper](%238%20EDM.pdf)

## Week 9: DDPM paper, EDM paper code walk-thrus
* Readings:
  * [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
  * [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
    * [Katherine's codebase](https://github.com/crowsonkb/k-diffusion/) (walk-thru by Jeremy Howard)
* [Recording](https://youtu.be/Ke5Wqdj3O5Q)
* [DDPM Colab notebook](https://colab.research.google.com/github/tmabraham/diffusion_reading_group/blob/main/Diffusion%20Reading%20Study%20Group%20%239%20-%20DDPM%20Code%20Walk-Thru.ipynb)

## Week 10: Paella
* Readings: 
  * [Fast Text-Conditional Discrete Denoising on Vector-Quantized Latent Spaces](https://arxiv.org/abs/2211.07292) 
* [Recording](https://youtu.be/MY01FGCyaaA)


## Week 11: Progressive Distillation, Distillation of Guided Models
* Readings:
  * [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) 
  * [On Distillation of Guided Diffusion Models](https://arxiv.org/abs/2210.03142)
* [Recording](https://youtu.be/ssAed4eNhuY)
* [Slides - Progressive Distillation](%2311B%20On%20Distillation%20of%20Guided%20Diffusion%20Models.pdf) - Presented by marii
* [Slides - Distillation of Guided Models](%2311B%20On%20Distillation%20of%20Guided%20Diffusion%20Models.pdf) - Presented by Griffin Floto

## Week 12: SDEDit
* Readings:
  * [SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073)
* [Recording](https://www.youtube.com/watch?v=ZMjVKWlmQbM)
* [Slides](%2312%20SDEdit.pdf)

## Week 13: Latent Diffusion and Stable Diffusion
* Readings:
  * [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
  * [Annotated Paper Implementation of Latent/Stable Diffusion](https://nn.labml.ai/diffusion/stable_diffusion/latent_diffusion.html)
  * [Lilian Weng's blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#speed-up-diffusion-model-sampling)
  * [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
  * [HuggingFace blog post on Stable Diffusion](https://huggingface.co/blog/stable_diffusion#how-does-stable-diffusion-work)
* [Recording](https://youtu.be/EZdB63fGum4)
* [Slides](%2313%20Latent%20Diffusion%20and%20Stable%20Diffusion.pdf)

## Week 14: Q&A with Robin Rombach
* Readings:
    * Same as Week 13
* [Recording](https://youtu.be/GZZvgxIm6WU)

## Week 15: Soft Diffusion
* Readings:
  * [Soft Diffusion: Score Matching for General Corruptions](https://arxiv.org/abs/2209.05442)
* [Recording](https://www.youtube.com/watch?v=Ped_I1uPL8Q)

## Week 16 & 17: Flow Matching
* Readings:
  * [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
* [Recording](https://youtu.be/DM8Ft15QmzI)
* [Recording](https://youtu.be/ZeDpfxIVXto)
* [Slides](%2316%20Flow%20Matching.pdf)
  
 ## Week 18: Consistency Models
 * Readings:
   * [Consistency Models](https://arxiv.org/abs/2303.01469)
 * [Recording](https://youtu.be/M1-jaVuZC4I)
 * [Slides](%2318%20-%20Consistency%20Models.pdf)

## Week 19: Conditional Flow Matching
* Readings:
  * [Conditional Flow Matching: Simulation-Free Dynamic Optimal Transport](https://arxiv.org/abs/2302.00482)
* [Recording](https://youtu.be/d5ljj_pEkUg)

## Week 20: Inverse Heat Dissipation
* Readings:
  * [Generative Modelling With Inverse Heat Dissipation](https://arxiv.org/abs/2206.13397)
* [Recording](https://youtu.be/q_ozv1VVdAA)

## Week 21: Poisson Flow Generative Models
* Readings:
  * [Poisson Flow Generative Models](https://arxiv.org/abs/2209.11178)
  * [AssemblyAI blog post](https://www.assemblyai.com/blog/an-introduction-to-poisson-flow-generative-models/)
* [Recording](https://youtu.be/l28qf2Jn7ZA)

## Week 22: Min-SNR Weighting Strategy
* Readings:
  * [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556)
* [Recording](https://youtu.be/SX_IKmil0ao)

## Week 23: PFGM++
* Readings:
  * [PFGM++: Unlocking the Potential of Physics-Inspired Generative Models](https://arxiv.org/abs/2302.04265)

## Week 24: Blurring Diffusion Models
* Readings:
  * [Blurring Diffusion Models](https://arxiv.org/abs/2209.05557)
* [Recording](https://youtu.be/-zOtwta31LU)

## Week 25: ControlNet
* Readings:
  * [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)
* [Recording](https://youtu.be/S_HCnM8BFk0)

## Week 26: DDPO (RLHF Diffusion)
* Readings:
  * [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301)
* [Recording](https://youtu.be/xzaZ19z8tOQ)

## Week 27: Diffusion Transformers
* Readings:
  * [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
* [Recording](https://youtu.be/3UBayMBsuqk)

## Week 28: simple diffusion
* Readings:
  * [simple diffusion: End-to-end diffusion for high resolution images](https://arxiv.org/abs/2301.11093)
* [Recording](https://youtu.be/KIO_amGmqlA)

## Week 29: Wuerstchen
* Readings:
  * [Wuerstchen: Efficient Pretraining of Text-to-Image Models](https://arxiv.org/abs/2306.00637)
* [Recording](https://youtu.be/YFJDATJcqhg)

## Week 30: Palette
* Readings:
  * [Palette: Image-to-Image Diffusion Models](https://arxiv.org/abs/2111.05826)
* [Recording](https://youtu.be/7IEue0u2XRE)

  
## List of papers to cover:
1. ~~Denoising Diffusion Probabilistic Models~~
2. ~~Generative Modeling by Estimating Gradients of the Data Distribution~~
3. ~~Improved techniques for training score-based generative models~~
4. ~~Score-Based Generative Modeling through Stochastic Differential Equations~~
5. ~~Denoising Diffusion Implicit Models~~
6. ~~Improved Denoising Diffusion Probabilistic Models~~
7. ~~Diffusion Models Beat GANs on Image Synthesis~~
8. ~~Classifier-Free Diffusion Guidance~~
9. ~~Variational Diffusion Models~~
10. ~~Tackling the Generative Learning Trilemma with Denoising Diffusion GANs~~
11. ~~Perception Prioritized Training of Diffusion Models~~
12. ~~Elucidating the Design Space of Diffusion-Based Generative Models~~
13. ~~Fast Text-Conditional Discrete Denoising on Vector-Quantized Latent Spaces~~
14. ~~Progressive Distillation for Fast Sampling of Diffusion Models~~
15. ~~On Distillation of Guided Diffusion Models~~
16. ~~SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations~~
17. ~~High-Resolution Image Synthesis with Latent Diffusion Models~~
18. ~~Stable Diffusion~~
19. ~~Soft Diffusion: Score Matching for General Corruptions~~
20. ~~Flow Matching for Generative Modeling~~
21. ~~Consistency Models~~
22. ~~Conditional Flow Matching: Simulation-Free Dynamic Optimal Transport~~
23. ~~Generative Modelling With Inverse Heat Dissipation~~
24. ~~Poisson Flow Generative Models~~
25. ~~Efficient Diffusion Training via Min-SNR Weighting Strategy~~
26. ~~PFGM++: Unlocking the Potential of Physics-Inspired Generative Models~~
27. ~~Blurring Diffusion Models~~
28. ~~Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)~~
29. ~~Training Diffusion Models with Reinforcement Learning~~
30. ~~Scalable Diffusion Models with Transformers~~
31. ~~simple diffusion: End-to-end diffusion for high resolution images~~
32. ~~Wuerstchen: Efficient Pretraining of Text-to-Image Models~~
33. ~~Palette: Image-to-image diffusion models~~
34. Reflected Diffusion Models
35. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
36. Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise
37. Pyramidal Denoising Diffusion Probabilistic Models
38. Cascaded Diffusion Models for High Fidelity Image Generation
39. Image super-resolution via iterative refinement
40. Denoising Diffusion Probabilistic Models for Robust Image Super-Resolution in the Wild
41. I$^2$SB: Image-to-Image Schr\"odinger Bridge
42. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models
43. Hierarchical Text-Conditional Image Generation with CLIP Latents
44. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding
45. eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers
46. Semi-Parametric Neural Image Synthesis
47. On the Importance of Noise Scheduling for Diffusion Models
48. Pseudo Numerical Methods for Diffusion Models on Manifolds
49. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Step
50. DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models
51. GENIE: Higher-Order Denoising Diffusion Solvers
52. Journey to the BAOAB-limit: ﬁnding effective MCMC samplers for score-based models
53. Riemannian Score-Based Generative Modeling
54. DiffWave: A Versatile Diffusion Model for Audio Synthesis
55. Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech
56. Video diffusion models
57. MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation
58. Imagen Video: High Definition Video Generation with Diffusion Models
59. Make-A-Video: Text-to-Video Generation without Text-Video Data
60. DreamFusion: Text-to-3D using 2D Diffusion
61. Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation
62. Magic3D: High-Resolution Text-to-3D Content Creation
63. Diffusion-LM Improves Controllable Text Generation.
64. Autoregressive Diffusion Models
65. Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning
66. Continuous diffusion for categorical data
67. DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models
68. Vector quantized diffusion model for text-to-image synthesis
69. Improved Vector Quantized Diffusion Models
70. DiVAE: Photorealistic Images Synthesis with Denoising Diffusion Decoder
71. Diffusion Models already have a Semantic Latent Space
72. Understanding ddpm latent codes through optimal transport
73. Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling
74. Dual Diffusion Implicit Bridges for Image-to-Image Translation
75. Unifying Diffusion Models' Latent Space, with Applications to CycleDiffusion and Guidance
76. Zero-shot Image-to-Image Translation
