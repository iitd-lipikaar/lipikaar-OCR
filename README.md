# Lipikaar-OCR

All models uploaded currently support **Printed Text** datasets. [run.py](run.py) can be modified to execute on multiple instances.

## Models that are currently available:
Model checkpoints can be downloaded from this [link](https://csciitd-my.sharepoint.com/:f:/g/personal/agarai_cstaff_iitd_ac_in/EpB6Cr98expDnB78qsdb4WEBTF1-MzKDqZvc-ARkaug9Wg?e=ADBnhb).</br>
1. **English OCR**
2. **Kannada OCR**</br>
3. **Tamil OCR**</br>
4. **Hindi OCR**</br>
5. **Assamese OCR**</br>
6. **Urdu OCR***</br>

**Based on a different architecture than others, for higher accuracy.*
      
## Convert mdb files to images
[check_lmdb_to_jpg.py](check_lmdb_to_jpg.py) file can be used to convert mdb files to images. (Path to mdb folder consists of data.mdb and lock.mdb files)

## Test the model
Python version 3.9.0 is recommended
```
python run.py --model_path /path/to/model.ckpt --image_path /path/to/image.jpg
```

