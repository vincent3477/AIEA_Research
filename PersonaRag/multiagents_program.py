from agents import Agent, Runner
from openai import OpenAI
import wikipedia
from doc_retr_agent import k_doc_retriever
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import faiss

#index = None
#model = None
"""
def get_wikipedia_summary(topic, sentences=5):
    try:
        return wikipedia.summary(topic, sentences=sentences)
    except wikipedia.DisambiguationError as e:
        # Use the first suggested option
        return wikipedia.summary(e.options[0], sentences=sentences)

        #for i in range(0, len(e.options)):
        #    print(wikipedia.summary(e.options[i], sentences=sentences))
        #    return wikipedia.summary(e.options[i], sentences=sentences)
        
    except wikipedia.PageError:
        return None
    

def get_top_k_articles(test, num_articles):

    

    topics = ["4th & King", "Balboa Park", "Embarcadero", "Civic Center", "Ferry Building", "Powell", "West Portal", "Pier 41, Yerba Buena/Moscone", "Castro", "Van Ness", "Forest Hill", "Chinatown"]

    summary_list = []
    
    for i in range(0,len(topics)):
        summary = get_wikipedia_summary(topics[i])
        if isinstance(summary, str):
            summary_list.append(summary)
            #print(f"Topic {topics[i]}: {summary}")
            #print("\n\n\n")


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
        #print(summary_list[i])
        #print("\n\n\n")
        best_ranked_articles.append(summary_list[i])
    return best_ranked_articles

"""

doc_retriever = k_doc_retriever(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'), faiss.IndexFlat(384))
doc_retriever.embed_documents("corpus.txt")


gmp_format = "User Profile Agent: {user_profile_answer}, Contextual Retrieval Agent: {contextual_answer}, Live Session Agent: {live_session_answer}, Document Ranking Agent: {document_ranking_answer}"



# initialize all the agents

client = OpenAI()

# Serves as a hub for interagent communication
glob_message_pool = Agent(name = "Global Message Pool", instructions=f"You are responsible for maintaining and enriching the Global Message Pool, serving as a central hub for inter-agent communication. Using the responses from individual agents \
                        and the existing global memory, consolidate key insights into a shared repository. Your goal is to organize a comprehensive message pool that includes agent-specific findings, historical user preferences, session-specific behaviors, search \
                        queries, and user feedback. This structure should provide all agents with meaningful data points and strategic recommendations, reducing redundant communication and improving the system's overall efficiency. Your response must only include\
                          the message pool in the form of a python dictionary in the following manner: (agent specific findings: {gmp_format}, query: (current query), passages: (passages as retrieved)).\
                          Do not include anything else other than the message pool. If there are no findings yet for any of the agent specific findings fields in the message pool, put 'Nothing' in the field.")

# Responsible for retrieving the top relevant documents based on user needs and interests
cont_retr_agent = Agent(name="Contextual Retrieval Agent", instructions = "You are a search technology expert guiding the Contextual Retrieval Agent to deliver context-aware document retrieval. Using the global memory pool and the retrieved passages, \
                        identify strategies to refine document retrieval. Highlight how user preferences, immediate needs, and global insights can be leveraged to adjust search queries and prioritize results that align with the user's interests. Ensure the Contextual \
                        Retrieval Agent uses this shared information to deliver more relevant and valuable results.")

# Responsible for adjusting search results real time to best suite what the user is looking for right now.
live_sess_agent = Agent(name = "Live Session Agent", instructions = "Your expertise in session analysis is required to assist the Live Session Agent in dynamically adjusting results. Examine the retrieved passages and information in the global memory pool. \
                        Determine how the Live Session Agent can use this data to refine its understanding of the user's immediate needs. Suggest ways to dynamically adjust search results or recommend new queries in real-time, ensuring that session adjustments align with user \
                        preferences and goals.")

# Responsible for priortizing documents to match the context.
doc_rank_agent = Agent(name = "Document Ranking Agent", instructions="Your task is to help the prioritize documents for better ranking. Analyze the retrieved passages and global memory pool to identify ways to rank documents effectively. Focus on combining \
                       historical user preferences, immediate needs, and session behavior to refine ranking algorithms. Your insights should ensure that documents presented by the Document Ranking Agent are prioritized to match user interests and search context.")


user_prof_agent = Agent(name = "User Profile Agent", instructions="Your task is to help the User Profile Agent improve its understanding of user preferences based on ranked document lists and the shared global memory pool. From the provided passages and global \
                        memory pool, analyze clues about the user's search preferences. Look for themes, types of documents, and navigation behaviors that reveal user interest. Use these insights to recommend how the User Profile Agent can refine and expand the \
                        user profile to deliver better-personalized results. If the global memory is empty, meaning a new profile was just made, output 'Nothing'.")


cog_agent = Agent(name = "Cognitive Agent", instructions="Your responsbility is to help the cognitiive agent to enahance its understanding of user insights to continuous improve the system's response")

chain_of_thought = Agent(name = "Chain Of Thought Agent", instructions="To solve the problem, please think and reason step by step, then answer.")



# First we have the user profile agent that retrives that current user profile



while True:
    embed_articles()

    query = input("what is your query? \n")
    articles = get_top_k(query, 20)
    inputs = f"(Question: {query}, Passages: {articles}, Global Memory: {gmp_format})"


    # Update the user profile
    profile_agent_output = Runner.run_sync(user_prof_agent, inputs).final_output

    # then pipe it into the the global message pool
    new_gmp = Runner.run_sync(glob_message_pool, f"Update the global message pool with new information from the user profile agent {profile_agent_output}").final_output
    print("FIRST TIME UPDATING THE GMP", new_gmp)

    # Retrieve the best of the top k-articles
    article_output = Runner.run_sync(cont_retr_agent, new_gmp).final_output

    new_gmp = Runner.run_sync(glob_message_pool, f"Update the global message pool with new information from the context retrieval agent {article_output}").final_output

    # Re-rank the articles.
    ranked_article_output  = Runner.run_sync(doc_rank_agent, inputs).final_output
    new_gmp = Runner.run_sync(glob_message_pool, f"Update the global message pool with new information from the document ranking agent {ranked_article_output}").final_output


    cot_prompt = f"Question: {query}, Passages: {ranked_article_output}, Read the given question and passages to gather relevant information. 2. Write reading notes summarizing the key points from these passages. 3. Discuss the relevance of the given question and passages. 4. If some passages are relevant to the given question, provide a brief answer based on the passages. 5. If no passage is relevant, directly provide the answer without considering the passages."

    cot_answer = Runner.run_sync(chain_of_thought, cot_prompt).final_output

    cot_gmp = f"(Question: {query}, Passages: {articles}, Global Memory: {new_gmp}, Initial Answer: {cot_answer})"


    print(cot_gmp)

    user_insights = Runner.run_sync(cog_agent, cot_gmp).final_output





    print(user_insights)




