# Lipikaar-OCR

All models uploaded currently support **Printed Text** as well as **Scene Text** data. [run.py](run.py) can be modified to execute on multiple instances.

## Models that are currently available:
Model checkpoints can be downloaded from this [link](https://csciitd-my.sharepoint.com/:f:/g/personal/agarai_cstaff_iitd_ac_in/EpB6Cr98expDnB78qsdb4WEBTF1-MzKDqZvc-ARkaug9Wg?e=ADBnhb).</br>
We support the following languages:</br>
1. **English**
2. **Kannada**</br>
3. **Tamil**</br>
4. **Hindi**</br>
5. **Assamese**</br>
6. **Gujarati**</br>
7. **Punjabi**</br>
8. **Telugu**</br>
9. **Malayalam**</br>
10. **Bengali**</br>
11. **Oriya**</br>
12. **Marathi**</br>
13. **Manipuri**</br>
14. **Urdu**</br>

      
## Convert mdb files to images
[check_lmdb_to_jpg.py](check_lmdb_to_jpg.py) file can be used to convert mdb files to images. (Path to mdb folder consists of data.mdb and lock.mdb files)

## Test the model
Python version 3.9.0 is recommended
```
python run.py --model_path /path/to/model.ckpt --image_path /path/to/image.jpg
```

## Citation
This repository contains code related to the paper **UTRNet**. [![arXiv](https://img.shields.io/badge/arXiv-2306.15782-darkred.svg)](https://arxiv.org/abs/2306.15782)</br>
If the code related to Urdu is helpful, please cite the paper:
```
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

