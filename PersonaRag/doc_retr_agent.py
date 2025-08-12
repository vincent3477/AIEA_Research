from datasets import load_dataset
import numpy as np
import OpenMatch as om
import re
import faiss
from sentence_transformers import SentenceTransformer



class k_doc_retriever:
    def __init__(self, model, index):
        self.model = model
        self.index = index
        self.initialized = False

    def embed_documents(self, filename):

        #self.index = faiss.IndexFlat(384)
        #self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


        summary_list = []
        batch_id = 0
        with open(filename) as f:
            exp = re.compile("(\d+)\t(.+)\t(.+)")
            line = f.readline()
            print(line)
            while line != "":
                text = exp.findall(line)[0][2]
                print("appending to a list")
                summary_list.append(text)
                if len(summary_list) % 10 == 0:
                    batch_id += 1
                    print(batch_id)
                    # encode the summary list in batches of 10
                    embedding = self.model.encode(summary_list)
                    self.index.add(embedding)
                    summary_list = []
                line = f.readline()
        

        embeddings = self.model.encode(summary_list)
        self.index.add(embeddings)

        self.initialized = True

    
    def get_model(self):
        if not self.initialized:
            return None
        return self.model

    def get_index(self):
        if not self.initialized:
            return None
        return self.index
    
    def embed_query(self, query, num_results):
        if not self.initialized:
            raise ModuleNotFoundError("Unable to embed query. Model must be initialized.") 
        input_embedding = self.model.encode(query)
        D, I = self.index.search(np.array([input_embedding]), k = num_results)
        # then we find the similarities between the query and the vectorized articles, either with the use of cosine similarities.


        articles = []

        with open("corpus.tsv") as f:
            line = f.readline()
            index = 0
            while(line != ""):
                if index in I:
                    articles.append(line)
                



        print(f"Top-{num_results} distances:", D)
        print(f"Top-{num_results} indices:", I)

        
        


        return articles
            