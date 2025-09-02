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
    def __init__(self, model, index = None, index_file = None, use_index_file = False, filenames = None):
        self.index_file = index_file
        self.model = model
        self.index = index
        self.initialized = False
        if not isinstance(filenames, list) and filenames != None:
            raise TypeError("Filename is not of type list.") 
        self.use_index_file = use_index_file
            

    """def embed_documents(self):

        #self.index = faiss.IndexFlat(384)
        #self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        for indiv_file in self.filenames:
            print(indiv_file)
            summary_list = []
            batch_id = 0
            with open(indiv_file) as f:
                
                exp = re.compile("(\d+)\t(.+)\t(.+)")
                line = f.readline()
                it = 0
                while line != "":
                    print(it)
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
                    it += 1

            

            embeddings = self.model.encode(summary_list)
            self.index.add(embeddings)
            faiss.write(self.index, "wiki_articles.index")

        self.initialized = True

        print("Done Initializing.")

    """
    
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
        #print("embed query")
        input_embedding = self.model.encode(query)
        if self.index == None:
            self.index = faiss.read_index(self.index_file)
        D, I = self.index.search(np.array([input_embedding]), k = num_results)
        #print("we have to the top k index")
        # then we find the similarities between the query and the vectorized articles, either with the use of cosine similarities.

        assert len(I[0]) == num_results
        
        
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
    

            