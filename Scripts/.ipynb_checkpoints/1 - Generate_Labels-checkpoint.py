#!/usr/bin/env python
# coding: utf-8

# ## Análise do genres.csv

# In[52]:


import pandas as pd
import os
import json


from essentia.standard import MonoLoader, TensorflowPredictMusiCNN


# In[3]:


base_path = "/mnt/disks/data/fma"


# In[4]:


metadata_path = os.path.join(base_path,"fma_metadata")


# In[5]:


dataset_path =  os.path.join(base_path,"fma_large")


# ## Análise do raw_genres.csv

# In[6]:


raw_genres = os.path.join(metadata_path,"raw_genres.csv")


# In[7]:


df_raw_genres = pd.read_csv(raw_genres)


# ## Análise do genres.csv

# In[11]:


genres = os.path.join(metadata_path,"genres.csv")


# In[12]:


df_genres = pd.read_csv(genres)



# ## Análise do raw_tracks.csv

# In[14]:


raw_tracks = os.path.join(metadata_path,"raw_tracks.csv")


# In[15]:


df_raw_tracks = pd.read_csv(raw_tracks)



# ## Análise do tracks.csv

# In[17]:


tracks = os.path.join(metadata_path,"tracks.csv")


# In[18]:


df_tracks = pd.read_csv(tracks,header=[1])


# In[19]:


df_tracks.rename(columns={"Unnamed: 0":"track_id"},inplace=True)
df_tracks.drop(index=[0],inplace=True)


# In[20]:


df_tracks.reset_index(inplace=True)
df_tracks.drop(columns=['index'],inplace=True)


# In[21]:


def get_structure(df,values):
    if values[2] == 0:
        return f'{values[0]}'
    else:
        return f'{values[0]}-{retorna_estrutura(df,df[df["genre_id"]==values[2]].values[0])}'


# In[22]:


import re


# In[23]:


df_tracks.genres.dropna(inplace=True)


# In[24]:


def convert_to_int(lista):
    lista.reverse()
    value = sum([(10**i)*number for i,number in enumerate(lista)])
    return value


# In[25]:


def extract_id_from_string(sentence):
    s = []
    for t in re.sub("[[]]","", sentence.split()[0]):
        try:
            s.append(int(t))
        except ValueError:
            pass
    return convert_to_int(s)


# In[26]:


df_tracks["first_genre_id"] = df_tracks.genres.apply(lambda x : extract_id_from_string(x))


# In[27]:


df_tracks.first_genre_id.unique()


# ## Gerando labels

# In[ ]:





# In[28]:


def create_labels(arr):
    labels = sorted(list(set(arr)))
    return dict([(label, i) for i, label in enumerate(labels)])   


# In[29]:


df_tracks


# In[30]:


labels = {
        'global': create_labels(df_tracks.first_genre_id)
}


# In[31]:


labels


# In[32]:


df_tracks['first_genre_id_label'] = df_tracks.first_genre_id.apply(lambda x:  labels['global'][x])


# In[33]:


df_tracks


# In[34]:


labels_path = os.path.join(metadata_path,"labels.json")


# In[35]:


with open(labels_path, "w+") as f:
    f.write(json.dumps(labels))


# ## Join com tabela de generos

# ### Geração da hierarquia a partir das tracks
# 

# In[36]:


## Get complete genre structure
def get_all_structure(estrutura):
    ## Get structure from df_genres
    genres = os.path.join(metadata_path,"genres.csv")
    df_genres = pd.read_csv(genres)
    
    
    def get_all_structure_from_df(df_genres,estrutura):

        if estrutura == 0:
            return f'{estrutura}'
        else:
            return f'{estrutura}-{get_all_structure_from_df(df_genres,df_genres[df_genres["genre_id"]==estrutura].parent.values[0])}'
        
    
    return get_all_structure_from_df(df_genres,estrutura)
    


# In[37]:


print(get_all_structure(df_tracks.iloc[158].first_genre_id))


# In[38]:


df_tracks.iloc[158]


# In[40]:


from tqdm.notebook import tqdm


# In[42]:


tqdm.pandas()


# In[43]:


df_tracks["full_genre_id"] = df_tracks.first_genre_id.progress_apply(lambda x: get_all_structure(x))


# In[44]:


df_tracks


# In[45]:


dataset_path


# In[46]:


def find_path(track_id,dataset_path):
    track_id = track_id.zfill(6)
    folder_id = track_id[0:3]
    file_path = os.path.join(dataset_path,folder_id,track_id+'.mp3')
    return file_path
    
    


# In[47]:


df_tracks['file_path'] = df_tracks.track_id.apply(lambda x: find_path(str(x),dataset_path))


# In[48]:


df_tracks.iloc[0].file_path


# In[53]:


def valid_music(file_path):
    try:
        # we start by instantiating the audio loader:
        loader = MonoLoader(filename=file_path)

        # and then we actually perform the loading:
        audio = loader()
        
        return True
    except:
        return False
    
    


# In[ ]:


df_tracks['valid'] = df_tracks.file_path.progress_apply(lambda x: valid_music(x))


# In[ ]:


df_tracks.to_csv(os.path.join(metadata_path,"tracks_genres_id_full.csv"),index=False)


# In[74]:






