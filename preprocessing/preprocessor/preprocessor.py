#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
import json
import ast
import os

# In[58]:


from tqdm.notebook import tqdm

# In[59]:


tqdm.pandas()

# In[60]:


args = pd.Series({
    "root_dir": "/mnt/disks/data/",
    "dataset_path": "/mnt/disks/data/fma/fma_large",
    "embeddings": "music_style",
    "train_id": "hierarchical_single",
    'sample_size': 1
})

# In[61]:


job_path = "/mnt/disks/data/fma/trains"

train_path = os.path.join(job_path, args.train_id)

# In[16]:

base_path = os.path.join(args.root_dir, "fma")

# In[17]:

models_path = os.path.join(args.root_dir, "models")

metadata_path_fma = os.path.join(base_path, "fma_metadata")

# In[18]:

metadata_file = os.path.join(train_path, "metadata.json")

categories_labels_path = os.path.join(train_path, "labels.json")


def __load_json__(path):
    with open(path, 'r') as f:
        tmp = json.loads(f.read())

    return tmp


def create_dir(path):
    # checking if the directory demo_folder2
    # exist or not.
    if not os.path.isdir(path):
        # if the demo_folder2 directory is
        # not present then create it.
        os.makedirs(path)
    return True


create_dir(train_path)



genres_df = pd.read_csv(os.path.join(metadata_path_fma, 'genres.csv'))
# Cria um dicionário que associa o ID de cada música aos IDs de seus gêneros musicais
tracks_df = pd.read_csv(os.path.join(metadata_path_fma, 'tracks_valid.csv'))

tracks_df = tracks_df.sample(frac=args.sample_size)

## Get complete genre structure
def get_all_structure(estrutura, df_genres):
    ## Get structure from df_genres
    def get_all_structure_from_df(estrutura, df_genres, structure=[]):
        if estrutura == 0:
            return structure
        else:
            structure.append(int(estrutura))
            get_all_structure_from_df(df_genres[df_genres["genre_id"] == int(estrutura)].parent.values[0], df_genres,
                                      structure)
            return structure

    return get_all_structure_from_df(estrutura, df_genres, structure=[])

# tracks_df['valid_genre'] = tracks_df.track_genres.apply(lambda x: x.strip('][').split(', ') if x != '[]' else None)
tracks_df['valid_genre'] = tracks_df.valid_genre.apply(lambda x: ast.literal_eval(x))
tracks_df['last_genre_id'] = tracks_df.valid_genre.apply(lambda x: x[-1] if x != None else None)

tracks_df.dropna(inplace=True)
tracks_df['full_genre_id'] = tracks_df.last_genre_id.progress_apply(lambda x: get_all_structure(x, genres_df)[::-1])

# In[80]:


tracks_df.full_genre_id.value_counts()
tracks_df = tracks_df[['track_id', 'full_genre_id']]
labels_size = tracks_df.full_genre_id.apply(lambda x: len(x))
labels_size = labels_size.max()


# ### Parse of label to structure

### Function for parse label to sctructure of hierarhical scheme

def parse_label(label, label_size=5):
    # label = label.split('-')
    # preencher com 0 no caso de haver menos de 5 níveis
    labels = np.zeros(label_size, dtype=int)
    for i, label in enumerate(label):
        if i == 5:
            break
        # Aqui você pode fazer a conversão do label em um índice inteiro usando um dicionário ou outro método
        # Neste exemplo, estou apenas usando a posição da label na lista como índice
        labels[i] = label
    return labels


parsed_labels = tracks_df.full_genre_id.apply(lambda x: parse_label(x))

def convert_label_to_string(x, level=2):
    return '-'.join([str(value) for value in x[:level]])



tracks_df['labels_1'] = parsed_labels.progress_apply(lambda x: str(x[:1][0]))
tracks_df['labels_2'] = parsed_labels.progress_apply(lambda x: convert_label_to_string(x, level=2))
tracks_df['labels_3'] = parsed_labels.progress_apply(lambda x: convert_label_to_string(x, level=3))
tracks_df['labels_4'] = parsed_labels.progress_apply(lambda x: convert_label_to_string(x, level=4))
tracks_df['labels_5'] = parsed_labels.progress_apply(lambda x: convert_label_to_string(x, level=5))


tracks_df.to_csv(os.path.join(train_path, "tracks.csv"), index=False)


tracks_df = pd.read_csv(os.path.join(train_path, "tracks.csv"))



categories_df = pd.DataFrame({'level5': tracks_df.labels_5.unique()})



categories_df['level1'] = categories_df.level5.progress_apply(lambda x: '-'.join(x.split('-')[:1]))
categories_df['level2'] = categories_df.level5.progress_apply(lambda x: '-'.join(x.split('-')[:2]))
categories_df['level3'] = categories_df.level5.progress_apply(lambda x: '-'.join(x.split('-')[:3]))
categories_df['level4'] = categories_df.level5.progress_apply(lambda x: '-'.join(x.split('-')[:4]))



def get_labels_name(x, genres_df):
    levels = 5
    full_name = []
    last_level = 0
    genre_root = ""
    for genre in x.split('-'):
        genre_df = genres_df[genres_df['genre_id'] == int(genre)]
        if genre_df.empty:
            genre_name = genre_root
        else:
            genre_name = genre_df.title.values.tolist()[0]
            genre_root = genre_name

        full_name.append(genre_name)
    full_name = '>'.join(full_name)

    return full_name
    # return genres_df[genres_df['genre_id'] == int(x)].title.values.tolist()[0]

categories_df['level5_name'] = categories_df.level5.apply(lambda x: get_labels_name(x, genres_df))


def __create_labels__(categories_df):
    data = {
        "label1": {},
        "label2": {},
        "label3": {},
        "label4": {},
        "label5": {},
        "label1_inverse": [],
        "label2_inverse": [],
        "label3_inverse": [],
        "label4_inverse": [],
        "label5_inverse": [],
        "label1_name": {},
        "label2_name": {},
        "label3_name": {},
        "label4_name": {},
        "label5_name": {},
    }

    idx = 0

    for id_x, cat in enumerate(set(categories_df.level1.values.tolist())):
        data['label1'][cat] = idx
        data['label1_inverse'].append(cat)
        data['label1_count'] = idx + 1
        idx += 1

    for id_x, cat in enumerate(set(categories_df.level2.values.tolist())):
        data['label2'][cat] = idx
        data['label2_inverse'].append(cat)
        data['label2_count'] = idx + 1
        idx += 1

    for id_x, cat in enumerate(set(categories_df.level3.values.tolist())):
        data['label3'][cat] = idx
        data['label3_inverse'].append(cat)
        data['label3_count'] = idx + 1
        idx += 1

    for id_x, cat in enumerate(set(categories_df.level4.values.tolist())):
        data['label4'][cat] = idx
        data['label4_inverse'].append(cat)
        data['label4_count'] = idx + 1
        idx += 1

    for idx, cat in enumerate(set(categories_df.level5.values.tolist())):
        data['label5'][cat] = idx
        data['label5_inverse'].append(cat)
        data['label5_count'] = idx + 1
        idx += 1

    for cat5, cat1, cat2, cat3, cat4, name5 in categories_df.values:
        name1 = '>'.join(name5.split('>')[:1])
        name2 = '>'.join(name5.split('>')[:2])
        name3 = '>'.join(name5.split('>')[:3])
        name4 = '>'.join(name5.split('>')[:4])

        data['label1_name'][cat1] = name1
        data['label2_name'][cat2] = name2
        data['label3_name'][cat3] = name3
        data['label4_name'][cat4] = name4
        data['label5_name'][cat5] = name5

    return data

with open(categories_labels_path, 'w+') as f:
    f.write(json.dumps(__create_labels__(categories_df)))


