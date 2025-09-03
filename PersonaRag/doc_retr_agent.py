from datasets import load_dataset
import numpy as np
import OpenMatch as om
import re
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import progressbar
import duckdb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch




class k_doc_retriever:
    def __init__(self, index = None, index_file = None, use_index_file = False, filenames = None):
        self.index_file = index_file
        self.model = AutoModel.from_pretrained("OpenMatch/cocodr-base-msmarco")
        self.tokenizer = AutoTokenizer.from_pretrained("OpenMatch/cocodr-base-msmarco")
        self.index = index
        self.initialized = False
        if not isinstance(filenames, list) and filenames != None:
            raise TypeError("Filename is not of type list.") 
        self.use_index_file = use_index_file
            
    def embed_documents(self):
        if not self.use_index_file:
            summary_list = []
            summary_id = []
            
            wiki = load_dataset("wikimedia/wikipedia", "20251101.en")
            
            for i, row in enumerate(wiki["train"]):
                text = row["text"].replace("\n", " ").replace("\n", " ")
                summary_list.append(text)
                summary_id.append(row["id"])
                if len(summary_list) > 512:
                    embeddings = self.model.encode(summary_list)
                    ids_array = np.array(summary_id, dtype=np.int64)
                    self.index.add_with_ids(embeddings, ids_array)
                    summary_list = []
                    summary_id = []
                print(i)
            if summary_list:
                embeddings = self.model.encode(summary_list)
                ids_array = np.array(summary_id, dtype=np.int64)
                self.index.add_with_ids(embeddings, ids_array)
            faiss.write_index(self.index, "wiki_embed.index")
            self.initialized = True
        else:
            self.initialized = True
            return self.index_file

        

                    
        

    def get_model(self):
        if not self.initialized:
            return None
        return self.model

    def get_index(self):
        if not self.initialized:
            return None
        return self.index
    
    def embed_query(self, query, num_results):
        #print("the query passed in was", query)
        #if not self.initialized:
        #    raise ModuleNotFoundError("Unable to embed query. Model must be initialized.") 
        print("embed query")
        print(self.model)
        device = torch.device("cpu")
        self.model = self.model.to(device)
        
        #input_embedding = self.model.encode(query)
        inputs = self.tokenizer(query, return_tensors="pt", padding = True, truncation = True)
        #for key, tensor in inputs.items():
        #    print(f"{key}: {tensor.shape}")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        print(type(inputs))


        
        print("make a representation of input")
        output  = self.model(**inputs)
        print("get the output")
        #embeddings = output.last_hidden_state.mean(dim=1)

        print("make it past the part we embed the query")

        if self.index == None:
            self.index = faiss.read_index(self.index_file)
        D, I = self.index.search(embeddings.deteach().numpy(), k = num_results)
        #print("we have to the top k index")
        # then we find the similarities between the query and the vectorized articles, either with the use of cosine similarities.

        assert len(I[0]) == num_results
        

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap=0)
        text_list = {}
        con = duckdb.connect("wiki.duckdb")
        for i in I[0]:
            
            df = con.execute(f"SELECT * FROM wiki WHERE column0={i}").fetchone()
            if df != None:
                text_list[df[0]] = [df[1], text_splitter.split_text(df[2])]
                print(f"article id {df[0]}, title: {df[1]}")
                #print("this is the retrieved item",df[2])
                #print(df[1])
                #print(text_splitter.split_text(df[2]))
            else:
                print(f"{i} has none")

        print(f"Top-{num_results} distances:", D)
        print(f"Top-{num_results} indices:", I)


        return text_list
    

        

