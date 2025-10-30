from datasets import load_dataset
import numpy as np
import OpenMatch as om
import re
import faiss
from sentence_transformers import SentenceTransformer
import progressbar
import duckdb
from langchain_text_splitters import RecursiveCharacterTextSplitter





class k_doc_retriever:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/msmarco-roberta-base-v2')
        self.index13 = faiss.read_index("embeddings_768d_1.bin") #Chunks 1-3 (in millions)
        self.index34 = faiss.read_index("embeddings_768d_2.bin") #Chunks 3-4
        self.index45 = faiss.read_index("embeddings_768d_3.bin") #Chunks 4-5
        self.index56 = faiss.read_index("embeddings_768d_4.bin") #Chunks 5-6
        self.index67 = faiss.read_index("embeddings_768d_5.bin") #Chunks 6-7
        self.index78 = faiss.read_index("embeddings_768d_6.bin") #Chunks 7-8
        self.index89 = faiss.read_index("embeddings_768d_7.bin") #Chunks 8-9
        self.index910 = faiss.read_index("embeddings_768d_8.bin") #Chunks 9-10
        self.index1011 = faiss.read_index("embeddings_768d_9.bin") #Chunks 10-11
        self.index1112 = faiss.read_index("embeddings_768d_10.bin") #Chunks 11-12
        self.index1213 = faiss.read_index("embeddings_768d_11.bin") #Chunks 12-13
        self.index13plus = faiss.read_index("embeddings_768d_12.bin") #Chunks 13-13.4
        self.wiki_db = duckdb.connect("wiki_chunked.duckdb")
                    
        

    def get_model(self):
        return self.model
    
    def map_and_merge(self, iter1, iter2, curr_dict):
        """    
        for i in range(len(iter1)):
            if iter1[i] in curr_dict:
                curr_dict[iter1[i]].append(iter2[i])
            else:
                curr_dict[iter1[i]] = [iter2[i]]

        return curr_dict
        """

        key_values = zip(iter1, iter2) # distances, indexes

        curr_dict.extend(key_values)

        return curr_dict


    
    def embed_query(self, query, num_results):
        
        #If k is more than 12, then do two for each, use % to determine how many to get from each then merge.

        num_entries = int(num_results / 12)
        if num_entries == 0:
            num_entries = 1

        input_embedding = self.model.encode(query)

        
        embeds_kv_pair = [] # a list of tuples that has distances and indexes.
        
        d1, i1 = self.index13.search(np.array([input_embedding]), k = num_entries * 3)
        embeds_kv_pair = self.map_and_merge(d1[0], i1[0], embeds_kv_pair)

        print(embeds_kv_pair)

        d2, i2 = self.index34.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d2[0], i2[0], embeds_kv_pair)

        print(embeds_kv_pair)
        
        d3, i3 = self.index45.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d3[0], i3[0], embeds_kv_pair)

        print(embeds_kv_pair)

        d4, i4 = self.index56.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d4[0], i4[0], embeds_kv_pair)

        print(embeds_kv_pair)

        d5, i5 = self.index67.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d5[0], i5[0], embeds_kv_pair)   

        print(embeds_kv_pair)

        d6, i6 = self.index78.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d6[0], i6[0], embeds_kv_pair)

        print(embeds_kv_pair)

        d7, i7 = self.index89.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d7[0], i7[0], embeds_kv_pair)  

        print(embeds_kv_pair)

        d8, i8 = self.index910.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d8[0], i8[0], embeds_kv_pair)  

        print(embeds_kv_pair)

        d9, i9 = self.index1011.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d9[0], i9[0], embeds_kv_pair)   

        print(embeds_kv_pair) 

        d10, i10 = self.index1112.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d10[0], i10[0], embeds_kv_pair)  

        print(embeds_kv_pair)  

        d11, i11 = self.index1213.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d11[0], i11[0], embeds_kv_pair)  

        print(embeds_kv_pair)

        d12, i12 = self.index13plus.search(np.array([input_embedding]), k = num_entries)
        embeds_kv_pair = self.map_and_merge(d12[0], i12[0], embeds_kv_pair)  

        print(embeds_kv_pair) 

        embeds_kv_pair = sorted(embeds_kv_pair, key= lambda item: item[0])

        #print("we have to the top k index")
        # then we find the similarities between the query and the vectorized articles, either with the use of cosine similarities.

        print(embeds_kv_pair) 


        
        
        """
        articles = {}
        for indiv_file in self.filenames:
            with open(indiv_file) as f:
                line = f.readline()
                index = 0
                articles_found = 0
                
                while(line != "" and articles_found < num_results):
                    if index in I[0]:
                        articles_found += 1
                        articles[index] = line
                    line = f.readline()
                    index += 1
        """

        indexes_for_query = []
        
        for i in range(len(embeds_kv_pair)):
            indexes_for_query.append(int(embeds_kv_pair[i][1]))
            if i == num_results - 1:
                break
        
        text_list = {}
        text_list_raw = []    


        query = "SELECT * FROM wiki_chunked WHERE column0 IN (SELECT UNNEST(?))"
        df = self.wiki_db.execute(query, [indexes_for_query]).fetchall()


        print(df)
        text_list = dict(df)

        for i in df:
            text_list_raw.append(i[1])



        return text_list, text_list_raw
    

            