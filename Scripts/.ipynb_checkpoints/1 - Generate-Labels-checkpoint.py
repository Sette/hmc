#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import os
from essentia.standard import MonoLoader


# In[2]:


from tqdm import tqdm


# In[3]:


tqdm.pandas()


# In[4]:



args = pd.Series({
    "root_dir":"/mnt/disks/data/",
    "dataset_path":"/mnt/disks/data/fma/fma_large",
    "embeddings":"music_style",
    "train_id": "hierarchical_mini_test",
    'sample_size':0.01
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


labels_file = os.path.join(train_path,"labels.json")


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
genres_df = genres_df[['genre_id', 'parent']]

# Cria um dicionário que mapeia o ID de cada gênero musical aos IDs de seus subgêneros
genre_dict = {}
for i, row in genres_df.iterrows():
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


# In[18]:


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


tracks_df['Valid'] = tracks_df.file_path.astype(str).progress_apply(lambda x: valid_music(x))


# In[41]

# In[ ]:


tracks_df[['track_id','file_path']].shape


# In[ ]:


tracks_df.shape


# In[ ]:


# tracks_df.set_index('track_id',inplace=True)


# In[ ]:


tracks_df.to_csv(os.path.join(train_path,"tracks_genres_id_full_valid.csv"),index=False)


# In[ ]:


tracks = pd.read_csv(os.path.join(train_path,"tracks_genres_id_full_valid.csv"))


# In[ ]:


tracks.dropna()


# In[ ]:



