# w2v-cif-bert

Code for paper: [Efficiently Fusing Pretrained Acoustic and Linguistic Encoders for Low-resource Speech Recognition](https://arxiv.org/abs/2101.06699)

We only provide key files of our model, `w2v-cif-bert`, which can be reimplement based on [fairseq](https://github.com/pytorch/fairseq/tree/master/fairseq/models/wav2vec).
If you have any questions on the reimplementation, please consult yicheng2016@ia.ac.cn.

## update
- 2021.5.14
Following others' requirement of the baselines used in our paper, we reveal the implementation of `w2v-seq2seq` and `w2v-nar` (relative scripts are in `baselines/*`).
NOTE: These codes are based on the out-of-date commit（23d8502bdde88a3e58e0910e2ee49834f8478b39 upstream/master）of Fairseq without testing in the new one.

Please cite as:
``` bibtex
@article{yi2021efciently,
  title={Efciently Fusing Pretrained Acoustic and Linguistic Encoders for Low-resource Speech Recognition},
  author={Yi, Cheng and Zhou, Shiyu and Xu, Bo},
  journal={IEEE Signal Processing Letters},
  year={2021},
  volume={28},
  pages={788-792},
  publisher={IEEE}
}
```
