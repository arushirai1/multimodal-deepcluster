# COIN_HowTo

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
