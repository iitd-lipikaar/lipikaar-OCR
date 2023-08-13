# Lipikaar-OCR: Urdu

**Based on the paper: *"UTRNet: High-Resolution Urdu Text Recognition In Printed Documents"***
This is inference code for printed Urdu text recognition based on the UTRNet paper. For details, you may refer to the paper, the project website and the official repository below:

[![UTRNet](https://img.shields.io/badge/UTRNet:%20High--Resolution%20Urdu%20Text%20Recognition-blueviolet?logo=github&style=flat-square)](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition)
[![Website](https://img.shields.io/badge/Website-Visit%20Here-brightgreen?style=flat-square)](https://abdur75648.github.io/UTRNet/)
[![arXiv](https://img.shields.io/badge/arXiv-2306.15782-darkred.svg)](https://arxiv.org/abs/2306.15782)

## Using This Repository
### Environment
* Python 3.7
* Pytorch 1.9.1+cu111
* Torchvision 0.10.1+cu111
* CUDA 11.4

### Running the code

Run the following command for inference on a single image:

```
CUDA_VISIBLE_DEVICES=0 python3 read.py --image_path path/to/image.png --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC  --saved_model path/to/model.pth
```

## Trained Model
[UTRNet-Large](https://csciitd-my.sharepoint.com/:u:/g/personal/ch7190150_iitd_ac_in/EeUZUQsvd3BIsPfqFYvPFcUBnxq9pDl-LZrNryIxtyE6Hw?e=MLccZi)


# Citation
If you use the code/dataset, please cite the following paper:

```BibTeX
@article{rahman2023utrnet,
      title={UTRNet: High-Resolution Urdu Text Recognition In Printed Documents}, 
      author={Abdur Rahman and Arjun Ghosh and Chetan Arora},
      journal={arXiv preprint arXiv:2306.15782},
      year={2023},
      eprint={2306.15782},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      doi = {https://doi.org/10.48550/arXiv.2306.15782},
      url = {https://arxiv.org/abs/2306.15782}
}
```

# License
[![Creative Commons License](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/). This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/) for Noncommercial (academic & research) purposes only and must not be used for any other purpose without the author's explicit permission.
