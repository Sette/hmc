#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import os
from essentia.standard import MonoLoader


# In[2]:


from tqdm.notebook import tqdm


# In[3]:


tqdm.pandas()


# In[4]:



args = pd.Series({
    "root_dir":"/mnt/disks/data/",
    "dataset_path":"/mnt/disks/data/fma/fma_large",
    "embeddings":"music_style",
    "train_id": "hierarchical_test",
    'sample_size':1
})


# In[5]:





job_path = "/mnt/disks/data/fma/trains"


# In[15]:


train_path = os.path.join(job_path,args.train_id)


# In[16]:


base_path = os.path.join(args.root_dir,"fma")


# In[17]:


models_path = os.path.join(args.root_dir,"models")


metadata_path_fma = os.path.join(base_path,"fma_metadata")


# In[18]:


metadata_file = os.path.join(train_path,"metadata.json")


categories_labels_path = os.path.join(train_path,"labels.json")


# In[6]:




def __load_json__(path):
    with open(path, 'r') as f:
        tmp = json.loads(f.read())

    return tmp



# In[7]:



def create_dir(path):
    # checking if the directory demo_folder2 
    # exist or not.
    if not os.path.isdir(path):

        # if the demo_folder2 directory is 
        # not present then create it.
        os.makedirs(path)
    return True


# In[8]:


create_dir(train_path)


# ## Load genres file. Contains relationships beetwen genres

# In[9]:


genres_df = pd.read_csv(os.path.join(metadata_path_fma,'genres.csv'))


# In[10]:


genres_df


# In[11]:


genres_df[genres_df['genre_id'] == 76]


# In[12]:



# Filtra as colunas relevantes
genres = genres_df[['genre_id', 'parent']]

# Cria um dicionário que mapeia o ID de cada gênero musical aos IDs de seus subgêneros
genre_dict = {}
for i, row in genres.iterrows():
    genre_id = row['genre_id']
    parent_id = row['parent']
    if pd.notna(parent_id):
        if parent_id not in genre_dict:
            genre_dict[parent_id] = []
        genre_dict[parent_id].append(genre_id)


# In[13]:


genre_dict


# In[14]:


# Cria um dicionário que associa o ID de cada música aos IDs de seus gêneros musicais
tracks_df = pd.read_csv(os.path.join(metadata_path_fma,'tracks.csv'), header=[0,1], index_col=0)


# In[15]:


tracks_df


# In[19]:


tracks_df.track.genres.value_counts()


# In[21]:


tracks_df['valid_genre'] = tracks_df.track.genres.apply(lambda x: x.strip('][').split(', ') if x != '[]' else None)


# In[22]:


tracks_df.valid_genre.dropna()


# In[23]:


tracks_df.track.genre_top


# In[24]:


tracks_df.track


# In[25]:


genre_tracks_dict = {}
for track_id, track_genres in tracks_df.valid_genre.dropna().items():
    # track_genres = row[('track', 'genres')].split('|')[0]
    # track_genres = track_genres.strip('][').split(', ')
    # print(track_genres)
    track_genre_ids = []
    genre = track_genres[0]
    if int(genre) in genre_dict:
        track_genre_ids.extend(genre_dict[int(genre)])
    track_genre_ids.append(int(genre))
    genre_tracks_dict[track_id] = list(reversed(track_genre_ids))


# In[26]:


genre_tracks_dict


# In[27]:


## Get complete genre structure
def get_all_structure(estrutura,df_genres):
    ## Get structure from df_genres
    
    def get_all_structure_from_df(estrutura,df_genres):
        if estrutura == 0:
            return 
        else:
            return f'{estrutura}-{get_all_structure_from_df(df_genres[df_genres["genre_id"]==int(estrutura)].parent.values[0],df_genres)}'
        
    
    return get_all_structure_from_df(estrutura,df_genres)
    


# In[28]:


tracks_df['first_genre_id'] = tracks_df.valid_genre.apply(lambda x:x[0] if x != None else None)


# In[29]:


tracks_df = tracks_df[['valid_genre','first_genre_id']]


# In[30]:


tracks_df


# In[31]:


tracks_df.dropna(inplace=True)


# In[32]:


tracks_df


# In[33]:


tracks_df["full_genre_id"] = tracks_df.first_genre_id.apply(lambda x: get_all_structure(x,genres_df))


# In[34]:


tracks_df.reset_index(inplace=True)


# In[35]:


def find_path(track_id,dataset_path):
    track_id = track_id.zfill(6)
    folder_id = track_id[0:3]
    file_path = os.path.join(dataset_path,folder_id,track_id+'.mp3')
    return file_path
    


# In[36]:


tracks_df['file_path'] = tracks_df.track_id.apply(lambda x: find_path(str(x),args.dataset_path))


# In[37]:


tracks_df.iloc[0].file_path


# In[38]:


def valid_music(file_path):
    try:
        # we start by instantiating the audio loader:
        loader = MonoLoader(filename=file_path)

        return True
    except:
        return False


# In[ ]:


valid_music('/mnt/disks/data/fma/fma_large/000/000002.mp3')


# In[39]:


teste = tracks_df.iloc[:100].file_path.astype(str).progress_apply(lambda x: valid_music(x))


# In[41]:


teste


# In[ ]:


tracks_df[['track_id','file_path']].shape


# In[ ]:


tracks_df.shape


# In[ ]:


# tracks_df.set_index('track_id',inplace=True)


# ### Delete all crashed files

# In[36]:


tracks_df = tracks_df[tracks_df['Valid'] == True]


# In[ ]:


tracks_df.to_csv(os.path.join(train_path,"tracks_genres_id_full_valid.csv"),index=False)


# In[14]:


df = pd.read_csv(os.path.join(train_path,"tracks_genres_id_full_valid.csv"))


# ### Parse of label to structure

# In[15]:


### Function for parse label to sctructure of hierarhical scheme

def parse_label(label):
    label = label.split('-')
    # preencher com 0 no caso de haver menos de 5 níveis
    labels = np.zeros(5,dtype=int)
    for i, label in enumerate(label):
        if i == 5:
            break
        # Aqui você pode fazer a conversão do label em um índice inteiro usando um dicionário ou outro método
        # Neste exemplo, estou apenas usando a posição da label na lista como índice
        labels[i] = label
    return labels


# In[16]:


df.full_genre_id.value_counts()


# In[17]:


df['labels'] = df.full_genre_id.apply(lambda x: parse_label(x))


# In[18]:


def convert_label_to_string(x,level=2):
    return '-'.join([str(value) for value in x[:level]])


# In[19]:


df['labels_1'] = df.labels.progress_apply(lambda x: str(x[:1][0]))
df['labels_2'] = df.labels.progress_apply(lambda x: convert_label_to_string(x,level=2))
df['labels_3'] = df.labels.progress_apply(lambda x: convert_label_to_string(x,level=3))
df['labels_4'] = df.labels.progress_apply(lambda x: convert_label_to_string(x,level=4))
df['labels_5'] = df.labels.progress_apply(lambda x: convert_label_to_string(x,level=5))


# In[20]:


df.labels_2.unique()


# In[21]:


labels_level_1 = df.labels_1.unique()
labels_level_2 = df.labels_2.unique()
labels_level_3 = df.labels_3.unique()
labels_level_4 = df.labels_4.unique()
labels_level_5 = df.labels_5.unique()


# In[22]:


categories_df = pd.DataFrame(
    {'level5':labels_level_5,
    'level4':labels_level_4,
    'level3':labels_level_3,
    'level2':labels_level_2,
    'level1':labels_level_1})


# In[23]:


def get_labels_name(x,genres_df):
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


# In[24]:


categories_df['level5_name'] = categories_df.level5.apply(lambda x: get_labels_name(x,genres_df))


# In[25]:


categories_df['level5_name']


# In[27]:


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

    for idx, cat in enumerate(set(categories_df.level1.values.tolist())):
        data['label1'][cat] = idx
        data['label1_inverse'].append(cat)
        data['label1_count'] = idx + 1

    for idx, cat in enumerate(set(categories_df.level2.values.tolist())):
        data['label2'][cat] = idx
        data['label2_inverse'].append(cat)
        data['label2_count'] = idx + 1
        
    for idx, cat in enumerate(set(categories_df.level3.values.tolist())):
        data['label3'][cat] = idx
        data['label3_inverse'].append(cat)
        data['label3_count'] = idx + 1

    for idx, cat in enumerate(set(categories_df.level4.values.tolist())):
        data['label4'][cat] = idx
        data['label4_inverse'].append(cat)
        data['label4_count'] = idx + 1
        
    for idx, cat in enumerate(set(categories_df.level5.values.tolist())):
        data['label5'][cat] = idx
        data['label5_inverse'].append(cat)
        data['label5_count'] = idx + 1
        
    for cat5,cat4,cat3,cat2,cat1,name5 in categories_df.values:
        
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


# In[28]:


with open(categories_labels_path, 'w+') as f:
    f.write(json.dumps(__create_labels__(categories_df)))


# In[ ]:



