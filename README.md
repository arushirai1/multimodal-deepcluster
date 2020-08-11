# COIN_HowTo
## Pretraining steps
1. Change 'coin_howto_overlap_captions.pickle' in ``` generate_freq_words_script.py ``` to where the captions are stored
2. Run ``` python generate_freq_words_script.py ``` in the same directory as xdc_main.py and video_main.py
3. For training change the files listed in the next section
4. Then run ```python xdc_main.py --exp {path experiment results} --cliplen 64 --batch 5 --workers 16 --modal joint --k 100 --verbose --contrastive --alpha 0.5``` where adding contrastive flag runs the contrastive approach. Alpha is the weight of the cross entropy loss. 1- alpha is the weight of the contrastive loss during training.
5. For single modal approaches run ```video-main.py --nmb_cluster 70 --epochs 100 --batch 5 --sobel --workers 16 --modal video_only --exp ./video_only_64 --lr 0.0001 --verbose --cliplen 64``` switch to --modal text_only for text only
## Run **Downstream Task** on COIN data
1. Change paths in build_paths in Utils.py
2. COIN_Dataset.py is the Dataset object
3. How to load data:
```
    from COIN_Dataset import COIN
    from Utils import build_paths
    root, dictionary_pickle, metadata_path = build_paths()

    train_dataset = COIN(root, dictionary_pickle, metadata_path, train=True, do_crop=False)

    val_dataset = COIN(root, dictionary_pickle, metadata_path, train=False, do_crop=False)
```
