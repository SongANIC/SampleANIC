"""
Created on 2021-03-08 10:57:00

@Author: Kai Wang，xinghuiSong
"""
import argparse
import numpy as np
import h5py
import os
import json
from collections import Counter
from scipy.misc import imread, imresize
from random import seed, choice, sample
from tqdm import tqdm

def create_input_files(dataset1, dataset2, json_path1, json_path2, image_folder1,
image_folder2, captions_per_image1,captions_per_image2,min_word_freq, output_folder, max_len=512):
    '''
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'fashion'
    :param json_path: path of JSON file with splits and captions
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold
                          are binned as <unk>s
    :param output_path: path to save files
    :param max_len: don't sample captions longer than this length
    '''


    # read JSON
    with open(json_path1, 'r') as j:
        data1 = json.load(j)
        
    with open(json_path2, 'r') as j:
        data2 = json.load(j)

    # Read image paths and captions for each image
    train_image_paths1 = []
    train_image_paths2 = []
    train_image_captions1 = []
    train_image_captions2 = []
    val_image_paths1 = []
    val_image_paths2 = []
    val_image_captions1 = []
    val_image_captions2 = []
    word_freq = Counter()

    for i, img in enumerate(data1['images']):
        captions1 = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions1.append(c['tokens'])
                
        
        if len(captions1) == 0:
            continue
        
        path1 = os.path.join(image_folder1, img['filename'])  

        if img['split'] in {'train'}:
            train_image_paths1.append(path1)
            train_image_captions1.append(captions1)
        elif img['split'] in {'val'}:
            val_image_paths1.append(path1)
            val_image_captions1.append(captions1)
    for i, img in enumerate(data2['images']):
        captions2 = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions2.append(c['tokens'])
                
        
        if len(captions2) == 0:
            continue
        
        path2 = os.path.join(image_folder2, img['filename'])  

        if img['split'] in {'train'}:
            train_image_paths2.append(path2)
            train_image_captions2.append(captions2)
        elif img['split'] in {'val'}:
            val_image_paths2.append(path2)
            val_image_captions2.append(captions2)       

    # Sanity check
    assert len(train_image_captions1) == len(train_image_paths1)
    assert len(val_image_captions1) == len(val_image_paths1)
    assert len(train_image_captions2) == len(train_image_paths2)
    assert len(val_image_captions2) == len(val_image_paths2)
    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v+1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0
    print(word_map)

    # Create a base/root name for all output files
    base_filename1 = dataset1 + str(min_word_freq) + '_min_word_freq'
    base_filename2 = dataset2 + str(min_word_freq) + '_min_word_freq'

    # print(base_filename)

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP.json'), 'w') as j:
        json.dump(word_map, j)
    
    # Sample captions for each image, save images to HDF5 file, and captions
    # and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [
            (train_image_paths1, train_image_captions1,'TRAIN'),
            (val_image_paths1, val_image_captions1,'VAL'),
    ]:
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' \
            + base_filename1 + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image1
            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256),
                                      dtype='uint8')
            print("\nReading %s images and captions, storing to file...\n" % split)
            
            enc_captions1 = []
            caplens1 = []

            for i, path in enumerate(tqdm(impaths)):
                # Sample captions
                if len(imcaps[i]) < captions_per_image1:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image1 - len(imcaps[i]))]
                else:
                    #captions = sample(imcaps[i], k=captions_per_image)
                    captions = imcaps[i][:captions_per_image1]
                # Sanity check
                assert len(captions) == captions_per_image1

                # Read images
                # print(impaths[i])
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # print('c:  {}'.format(c))
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions1.append(enc_c)
                    caplens1.append(c_len)


            # print(split, ':', images.shape[0], captions_per_image, len(enc_captions), len(caplens))
            # Sanity check
            assert images.shape[0] * captions_per_image1 == len(enc_captions1) == len(caplens1)
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename1 + '.json'), 'w') as j:
                json.dump(enc_captions1, j)
            
            # print("caplens: {}".format(caplens))
            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename1 + '.json'), 'w') as j:
                json.dump(caplens1, j)
                
    for impaths, imcaps,split in [
        (train_image_paths2, train_image_captions2, 'TRAIN'),
        (val_image_paths2, val_image_captions2, 'VAL'),
    ]:
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' \
                                                   + base_filename2 + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image2
            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256),
                                      dtype='uint8')
            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions2 = []
            caplens2 = []

            for i, path in enumerate(tqdm(impaths)):
                # Sample captions
                if len(imcaps[i]) < captions_per_image2:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in
                                            range(captions_per_image2 - len(imcaps[i]))]
                else:
                    # captions = sample(imcaps[i], k=captions_per_image)
                    captions = imcaps[i][:captions_per_image2]
                # Sanity check
                assert len(captions) == captions_per_image2

                # Read images
                # print(impaths[i])
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # print('c:  {}'.format(c))
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions2.append(enc_c)
                    caplens2.append(c_len)

            # print(split, ':', images.shape[0], captions_per_image, len(enc_captions), len(caplens))
            # Sanity check
            assert images.shape[0] * captions_per_image2 == len(enc_captions2) == len(caplens2)

            # Save encoded captions and their lengths to JSON files
            # print("enc_captions: {}".format(len(enc_captions)))
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename2 + '.json'), 'w') as j:
                json.dump(enc_captions2, j)
            
            # print("caplens: {}".format(caplens))
            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename2 + '.json'), 'w') as j:
                json.dump(caplens2, j)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create input files')
    parser.add_argument('--dataset1', type=str, default='flickr')
    parser.add_argument('--dataset2', type=str, default='PCCD')
    
    parser.add_argument('--json_path1', type=str, 
                        default='../../data/flick8k/flickrresult.json')
    parser.add_argument('--json_path2', type=str, 
                        default='../../data/PCCD/PCCDresult.json')
    parser.add_argument('--image_folder1', type=str, 
                        default='../../data/flick8k/images/')
    parser.add_argument('--image_folder2', type=str, 
                        default='../../data/PCCD/images/')
    parser.add_argument('--captions_per_image1', type=int, default=5,
                        help='number of captions to sample per image')
    parser.add_argument('--captions_per_image2', type=int, default=8,
                        help='number of captions to sample per image')
    parser.add_argument('--min_word_freq', type=int, default=5, 
                        help='words occuring less frequently than this \
                        threshold are binned as <unk>s')
    parser.add_argument('--output_folder', type=str, default='../../data/OutputDataset/')
    parser.add_argument('--max_len', type=int, default=512, 
                        help="don't sample captions longer than this length")
    args = parser.parse_args()

    create_input_files(args.dataset1,args.dataset2, args.json_path1,args.json_path2, 
    args.image_folder1, args.image_folder2,args.captions_per_image1, args.captions_per_image2,args.min_word_freq,args.output_folder, args.max_len)

'''
python3 create_input_files.py --dataset='fashion' --json-path='/data/fashion/annotations/data_fashion.json' --image-path='/data/fashion/images' --captions-per-image=3 --output-path='/data/fashion/'
'''
