"""from openai import OpenAI
client = OpenAI()

query = input("type in your query here")

# scrape for documents here

memory = {}

user_profile = client.chat.completions.create(
    model = "gpt-4o-muni",
    messages=[
        {"role": "system", "content": f"From the memory pool get a gist of what his personality may be like including previous queries, main interests, and immediate needs that are most common." },
        {"role": "user", "content": f"Based on this knowledge base, describe this person's main interests filter his main interest based on the queries made. This is the person's knowledge base {memory}"}
    ]
)
"""

# User profile agent: should capture how the user interacts with search results includi

#from agents import Agent, Runner

from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import re
import wikipedia
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

memory = {} # use this as a form of memory for now
passages = {}


def get_wikipedia_summary(topic, sentences=5):
    try:
        return wikipedia.summary(topic, sentences=sentences)
    except wikipedia.DisambiguationError as e:
        # Use the first suggested option
        print(wikipedia.summary(e.options[0], sentences=sentences))

        #for i in range(0, len(e.options)):
        #    print(wikipedia.summary(e.options[i], sentences=sentences))
        #    return wikipedia.summary(e.options[i], sentences=sentences)
        
    except wikipedia.PageError:
        return None
    



def get_top_k_articles(test, num_articles):

    topics = ["4th & King", "Balboa Park", "Embarcadero", "Civic Center", "Ferry Building", "Powell", "West Portal", "Pier 41"]

    summary_list = []
    
    for i in range(0,len(topics)):
        summary = get_wikipedia_summary(topics[i])
        if isinstance(summary, str):
            summary_list.append(summary)
            print(f"Topic {topics[i]}: {summary}")
            print("\n\n\n")


    index = faiss.IndexFlat(384)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(summary_list)
    index.add(embeddings)
    input_embedding = model.encode(test)


    D, I = index.search(np.array([input_embedding]), k = num_articles)


    print("Top-5 distances:", D)
    print("Top-5 indices:", I)

    # get the query
    # vectorize the query
    # then we find the similarities between the query and the vectorized articles, either with the use of cosine similarities.
    
    best_ranked_articles = []

    print("The best matching results. First is highest match and last is the lowest match.")
    for i in I[0]:
        print(summary_list[i])
        print("\n\n\n")
        best_ranked_articles.append(summary_list[i])
    return best_ranked_articles



# get the user input
test = input("what is your query")
articles = get_top_k_articles(test, 5)

client = OpenAI()

user_profile = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"You are a search technology export guidiing the contextual retrieval agent to deliver context-aware document retrieval." },
        {"role": "user", "content": f""}
    ]
)

reply = user_profile.choices[0].message.content
print(reply)










