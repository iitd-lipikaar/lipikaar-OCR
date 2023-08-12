import os
import cv2
import numpy as np
import lmdb
from strhub.data.utils import CharsetAdapter
import unicodedata


def check_label_charset(label, charset):
    for char in label:
        if char not in charset:
            return False
    return True

def unpack_mdb(data_dir, output_dir):
    """Unpack data from .mdb files and save images with labels"""

    # Open the .mdb environment
    env = lmdb.open(data_dir, readonly=True)

    # Open the transaction for data retrieval
    txn = env.begin()

    # from other code
    #charset_adapter = CharsetAdapter(charset)
    labels_list = []
    img_name = 0

    # Retrieve the data and labels
    with open(output_dir+'/'+'gt.txt', 'w+', encoding='utf-8') as f:
        with txn.cursor() as cursor:
            for key, value in cursor:
                # Convert the image data to numpy array
                image_data = np.frombuffer(value, dtype=np.uint8)

                # Decode and save the image
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                #image_name = key.decode()
                image_name = img_name
                image_path = os.path.join(output_dir, f'{image_name}.jpg')
                # cv2.imwrite(image_path, image)


                # Extract and save the label
                label_key = key.decode()
                label_key = f'label-{label_key.split("-")[1]}'.encode()
                print(label_key)
                label = txn.get(label_key).decode()

                label_path = os.path.join(output_dir, f'{image_name}.txt')

                lbl = str(image_name) + ".jpg" + " " + str(label)

                check = True

                if check == True:

                    labels_list.append(lbl)

                    img_name += 1

                    cv2.imwrite(image_path, image)
                    f.write(lbl+"\n")

    env.close()

    print('Unpacking completed successfully.')


# Set the path to the directory containing the .mdb files and the output directory
data_dir = '/path/to/mdb/files/'
output_dir = '/path/to/store/decoded/images/'

# Unpack the data and labels from the .mdb files
unpack_mdb(data_dir, output_dir)
