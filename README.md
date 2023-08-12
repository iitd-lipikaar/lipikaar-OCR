# Lipikaar-OCR

Model checkpoints can be downloaded from [here](https://csciitd-my.sharepoint.com/:f:/g/personal/agarai_cstaff_iitd_ac_in/EpB6Cr98expDnB78qsdb4WEBTF1-MzKDqZvc-ARkaug9Wg?e=ADBnhb).

All models uploaded currently support **Printed Text** datasets. run.py can be modified to execute on multiple instances.

## Models that are currently available:
1. **English OCR**
2. **Kannada OCR**</br>
3. **Tamil OCR**</br>
4. **Hindi OCR**</br>
5. **Assamese OCR**</br>
6. **Urdu OCR** (For Urdu we have a seperate GitHub repo: [UTRNet](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition))</br>


      
## Convert mdb files to images
check_lmdb_to_jpg.py file can be used to convert mdb files to images. (Path to mdb folder consists of data.mdb and lock.mdb files)

## Test the model
Python version 3.9.0 is recommended
```
python run.py --model_path /path/to/model.ckpt --image_path /path/to/image.jpg
```

