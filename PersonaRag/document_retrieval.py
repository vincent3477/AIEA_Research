import os
from datasets import load_dataset
import numpy as np
import OpenMatch as om
import re
import faiss #1.11.0
from sentence_transformers import SentenceTransformer
import progressbar
import duckdb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import snapshot_download





class k_doc_retriever:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/msmarco-roberta-base-v2')
        self.dataset_path = snapshot_download(repo_id = "vsiu2/my_persona_rag", repo_type = "dataset")
        self.wiki_db = duckdb.connect("wiki_chunked.duckdb")            
        

    def get_model(self):
        return self.model
    
    def get_dataset_path(self):
        return self.dataset_path
    
    def map_and_merge(self, iter1, iter2, curr_dict):
        """    
        for i in range(len(iter1)):
            if iter1[i] in curr_dict:
                curr_dict[iter1[i]].append(iter2[i])
            else:
                curr_dict[iter1[i]] = [iter2[i]]

        return curr_dict
        """

        #print("passed into map and merge", iter1, iter2)

        key_values = zip(iter1, iter2) # distances, indexes

        curr_dict.extend(key_values)

        return curr_dict


    
    def embed_query(self, query, num_results):
        
        """
        Return indexes of semantically relevant articles.

        Args:
            query: Search terms
            num_results: Number of results to fetch (default to 5)
        """


        num_entries = int(num_results / 12)
        if num_entries == 0:
            num_entries = 1

        input_embedding = self.model.encode(query)

        
        embeds_kv_pair = [] # a list of tuples that has distances and indexes.
        index13 = faiss.read_index(os.path.join(self.dataset_path,"embeddings_768d_1.bin"))
        d1, i1 = index13.search(np.array([input_embedding]), k = num_entries * 3)
        embeds_kv_pair = self.map_and_merge(d1[0], i1[0], embeds_kv_pair)
        del index13

        index34 = faiss.read_index(os.path.join(self.dataset_path,"embeddings_768d_2.bin"))
        d2, i2 = index34.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d2[0], i2[0], embeds_kv_pair)
        del index34

        index45 = faiss.read_index(os.path.join(self.dataset_path,"embeddings_768d_3.bin"))
        d3, i3 = index45.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d3[0], i3[0], embeds_kv_pair)
        del index45

        index56 = faiss.read_index(os.path.join(self.dataset_path,"embeddings_768d_4.bin"))
        d4, i4 = index56.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d4[0], i4[0], embeds_kv_pair)
        del index56

        index67 = faiss.read_index(os.path.join(self.dataset_path,"embeddings_768d_5.bin"))
        d5, i5 = index67.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d5[0], i5[0], embeds_kv_pair) 
        del index67  

        index78 = faiss.read_index(os.path.join(self.dataset_path,"embeddings_768d_6.bin"))
        d6, i6 = index78.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d6[0], i6[0], embeds_kv_pair)
        del index78

        index89 = faiss.read_index(os.path.join(self.dataset_path,"embeddings_768d_7.bin"))
        d7, i7 = index89.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d7[0], i7[0], embeds_kv_pair)  

        index910 = faiss.read_index(os.path.join(self.dataset_path,"embeddings_768d_8.bin"))
        d8, i8 = index910.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d8[0], i8[0], embeds_kv_pair)  
        del index910

        index1011 = faiss.read_index(os.path.join(self.dataset_path,"embeddings_768d_9.bin"))
        d9, i9 = index1011.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d9[0], i9[0], embeds_kv_pair)
        del index1011

        index1112 = faiss.read_index(os.path.join(self.dataset_path,"embeddings_768d_10.bin"))
        d10, i10 = index1112.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d10[0], i10[0], embeds_kv_pair)
        del index1112  

        index1213 = faiss.read_index(os.path.join(self.dataset_path,"embeddings_768d_11.bin"))
        d11, i11 = index1213.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d11[0], i11[0], embeds_kv_pair)  
        del index1213

        index13plus = faiss.read_index(os.path.join(self.dataset_path,"embeddings_768d_12.bin"))
        d12, i12 = index13plus.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d12[0], i12[0], embeds_kv_pair)  
        del index13plus

        embeds_kv_pair = sorted(embeds_kv_pair, key= lambda item: item[0])

        # then we find the similarities between the query and the vectorized articles, either with the use of cosine similarities.


        indexes_for_query = []
        
        for i in range(len(embeds_kv_pair)):
            indexes_for_query.append(int(embeds_kv_pair[i][1]))
            if i == num_results - 1:
                break
        
        #text_list = {}
        #text_list_raw = []    

        return indexes_for_query
        