# pytorch-kaldi-neural-speaker-embeddings
A light weight neural speaker embeddings extraction based on Kaldi and PyTorch. \
The repository serves as a starting point for users to reproduce and experiment several recent advances in speaker recognition literature. 
Kaldi is used for pre-processing and post-processing and PyTorch is used for training the neural speaker embeddings.

This repository contains PyTorch+Kaldi pipeline to reproduce the core results for: 
* [Exploring the Encoding Layer and Loss Function in End-to-End Speaker and Language Recognition System](https://arxiv.org/pdf/1804.05160.pdf)
* [A Novel Learnable Dictionary Encoding Layer for End-to-End Language Identification](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8462025)

With some modifications, you can easily adapt the pipeline for:
* []()
* []()
* []()

Please cite our paper(s) if you find this repository useful. Cite both if you are kind enough!
```
@article{villalba2019state,
  title={State-of-the-art speaker recognition with neural network embeddings in nist sre18 and speakers in the wild evaluations},
  author={Villalba, Jes{\'u}s and Chen, Nanxin and Snyder, David and Garcia-Romero, Daniel and McCree, Alan and Sell, Gregory and Borgstrom, Jonas and Garc{\'\i}a-Perera, Leibny Paola and Richardson, Fred and Dehak, R{\'e}da and others},
  journal={Computer Speech \& Language},
  pages={101026},
  year={2019},
  publisher={Elsevier}
}
```
```
@article{cooper2019zero,
  title={Zero-Shot Multi-Speaker Text-To-Speech with State-of-the-art Neural Speaker Embeddings},
  author={Cooper, Erica and Lai, Cheng-I and Yasuda, Yusuke and Fang, Fuming and Wang, Xin and Chen, Nanxin and Yamagishi, Junichi},
  journal={arXiv preprint arXiv:1910.10838},
  year={2019}
}
```

# Overview
neural speaker embeddings: encoder --> pooling --> classification 
put LDE plot 

# Requirements
put requirements.txt
PyTorch 
Python 
Kaldi

# Getting Started 
`./pipeline.sh`

# Datasets 
VoxCeleb I+II

# Pre-Trained Models 


# Benchmarking Speaker Verification EERs
c.f. x-vectors 

# Speaker Adaptation for Tacotron2
embedding space visualizaiton 

# Benchmarking TTS MOS scores
c.f. x-vectors 

# Credits
Base code written by [Nanxin Chen](https://github.com/bobchennan), Johns Hopkins University \
Experiments done by [Cheng-I Lai](http://people.csail.mit.edu/clai24/), MIT
