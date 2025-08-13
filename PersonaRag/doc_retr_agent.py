from datasets import load_dataset
import numpy as np
import OpenMatch as om
import re
import faiss
from sentence_transformers import SentenceTransformer



class k_doc_retriever:
    def __init__(self, model, index, filename):
        self.model = model
        self.index = index
        self.initialized = False
        self.filename = filename

    def embed_documents(self):

        #self.index = faiss.IndexFlat(384)
        #self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


        summary_list = []
        batch_id = 0
        with open(self.filename) as f:
            exp = re.compile("(\d+)\t(.+)\t(.+)")
            line = f.readline()
            while line != "":
                text = exp.findall(line)[0][2]
                #print("Appending to a list")
                summary_list.append(text)
                if len(summary_list) % 10 == 0:
                    batch_id += 1
                    #print(batch_id)
                    # encode the summary list in batches of 10
                    embedding = self.model.encode(summary_list)
                    self.index.add(embedding)
                    summary_list = []
                line = f.readline()
        

        embeddings = self.model.encode(summary_list)
        self.index.add(embeddings)

        self.initialized = True

        print("Done Initializing.")

    
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

        assert len(I[0]) == num_results
        

        articles = {}

        with open(self.filename) as f:
            line = f.readline()
            index = 0
            articles_found = 0
            
            while(line != "" and articles_found < num_results):
                if index in I[0]:
                    articles_found += 1
                    articles[index] = line
                line = f.readline()
                index += 1
                
        print(f"Top-{num_results} distances:", D)
        print(f"Top-{num_results} indices:", I)


        return articles
            