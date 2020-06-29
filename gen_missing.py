path_to='/home/mschiappa/data/COIN/'

import os
import pickle
with open('coin_howto_overlap_captions.pickle','rb') as f:
    captions_df = pickle.load(f)
missing_paths=[]
keys=captions_df.keys()
total_videos=len(keys)
present_videos = 0
for key in keys:
    key+=".mp4"
    if not os.path.exists(os.path.join(path_to, 'train', key)):
        if not os.path.exists(os.path.join(path_to, 'val', key)):
            missing_paths.append(key)
        else:
            present_videos+=1
    else:
        present_videos+=1

print(missing_paths)
with open('missing_paths.pickle','wb') as f:
    pickle.dump(missing_paths,f)

print("Percent present:", present_videos/total_videos)
print("Present", present_videos)
