from agents import Agent, Runner
from openai import OpenAI
import wikipedia
from doc_retr_agent import k_doc_retriever
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import faiss
from ast import literal_eval

#index = None
#model = None


def get_articles_by_index(article_dict, list_articles):

    exp = re.compile("(\d+)")
    indexed_list = exp.findall(list_articles)

    return_vals = {}
    for l in indexed_list:
        return_vals[int(l)] = article_dict[int(l)]
    return return_vals

# initialize the vector db of articles.
doc_retriever = k_doc_retriever(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'), faiss.IndexFlat(384), 'corpus.tsv')
doc_retriever.embed_documents()


gmp_format = "User Profile Agent: {user_profile_answer}, Live Session Agent: {live_session_answer}, Document Ranking Agent: {document_ranking_answer}"



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

    query = input("what is your query? \n")
    articles = doc_retriever.embed_query(query, 20)
    global_message_pool = {}
    global_message_pool["Query"] = query
    global_message_pool["Passages"] = articles
    global_message_pool["Global Memory"] = gmp_format
    str_glob_mess_pool = str(global_message_pool)


    # Update the user profile
    profile_agent_output = Runner.run_sync(user_prof_agent, f"Update the user profile based on the query given in the message pool{global_message_pool}").final_output

    # then pipe into the global message pool updating it
    glob_memory_state = Runner.run_sync(glob_message_pool, f"Update the global message pool with new information from the user profile agent {profile_agent_output}").final_output
    # Update the global message Pool
    global_message_pool["Global Memory"] = glob_memory_state
    print("GMP update #1", new_gmp)


    # With the articles given priortize the most relevant articles
    cont_retr_output = Runner.run_sync(cont_retr_agent, f"As said in the instructions above, priortize the most relevant articles given in the \"Passages\" field in here: {str(global_message_pool)}. Just list the passage IDs that are best relevant in form of a Python list.").final_output
    print("agent's article output", cont_retr_output)
    # tell the agent to give the passage IDs only so it doesnt need to output the entire wikipedia text.


    cont_retr_articles = get_articles_by_index(articles, cont_retr_output)
    global_message_pool["Passages"] = cont_retr_articles
 

    #new_gmp = Runner.run_sync(glob_message_pool, f"Update the global message pool (here {new_gmp}) by modifying the \"passages\"  field with new information from the context retrieval agent {cont_retr_output}").final_output


    lve_ses_suggestions = Runner.run_sync(live_sess_agent, f"Using the context given, suggest queries or adjusting search results based on the retrieved passages and queries based on the current findings {global_message_pool}").final_output

    glob_memory_state = Runner.run_sync(glob_message_pool, f"Update the gloval message pool with new information from the live session agent {lve_ses_suggestions}").final_output


    # Re-rank the articles.
    ranked_article_output  = Runner.run_sync(doc_rank_agent, f"Rank the documents in the current field \"passages\" {new_gmp}").final_output
    new_gmp = Runner.run_sync(glob_message_pool, f"Update the global message pool with new information from the document ranking agent {ranked_article_output}").final_output
    print("reranked articles", new_gmp)


    cot_prompt = f"Question: {query}, Passages: {ranked_article_output}, Read the given question and passages to gather relevant information. 2. Write reading notes summarizing the key points from these passages. 3. Discuss the relevance of the given question and passages. 4. If some passages are relevant to the given question, provide a brief answer based on the passages. 5. If no passage is relevant, directly provide the answer without considering the passages."

    cot_answer = Runner.run_sync(chain_of_thought, cot_prompt).final_output
    

    cot_gmp = f"(Question: {query}, Passages: {articles}, Global Memory: {new_gmp}, Initial Answer: {cot_answer})"
    print("chain of thought output", cot_gmp)

    user_insights = Runner.run_sync(cog_agent, cot_gmp).final_output
    print("udpated user insights", user_insights)





    print(user_insights)




