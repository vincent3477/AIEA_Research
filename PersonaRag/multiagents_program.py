from agents import Agent, Runner
from openai import OpenAI
import wikipedia
from doc_retr_agent import k_doc_retriever
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import numpy as np
from datasets import load_dataset
import re
import faiss
from ast import literal_eval
from transformers import pipeline
from datasets import load_dataset



    

#index = None
#model = None

class persona_rag:
    def __init__(self):
         # initialize all the agents

        client = OpenAI()

        # initialize the vector db of articles.
        self.doc_retriever = k_doc_retriever(index_file = "wiki_articles_1.index", use_index_file=True)
        self.doc_retriever.embed_documents()

        #self.doc_retriever.embed_documents()



        self.gmp_format = "User Profile Agent: {user_profile_answer}, Live Session Agent: {live_session_answer}, Document Ranking Agent: {document_ranking_answer}"


        self.global_message_pool = {}
        self.global_message_pool["Global Memory"] = self.gmp_format
            

        # Serves as a hub for interagent communication
        self.glob_message_pool = Agent(name = "Global Message Pool", instructions=f"You are responsible for maintaining and enriching the Global Message Pool, serving as a central hub for inter-agent communication. Using the responses from individual agents \
                                and the existing global memory, consolidate key insights into a shared repository. Your goal is to organize a comprehensive message pool that includes agent-specific findings, historical user preferences, session-specific behaviors, search \
                                queries, and user feedback. This structure should provide all agents with meaningful data points and strategic recommendations, reducing redundant communication and improving the system's overall efficiency. Your response must only include\
                                the message pool in the form of a python dictionary in the following manner: (agent specific findings: {self.gmp_format}, query: (current query), passages: (passages as retrieved)).\
                                Do not include anything else other than the message pool. If there are no findings yet for any of the agent specific findings fields in the message pool, put 'Nothing' in the field.")

        # Responsible for retrieving the top relevant documents based on user needs and interests
        self.cont_retr_agent = Agent(name="Contextual Retrieval Agent", instructions = "You are a search technology expert guiding the Contextual Retrieval Agent to deliver context-aware document retrieval. Using the global memory pool and the retrieved passages, \
                                identify strategies to refine document retrieval. Highlight how user preferences, immediate needs, and global insights can be leveraged to adjust search queries and prioritize results that align with the user's interests. Ensure the Contextual \
                                Retrieval Agent uses this shared information to deliver more relevant and valuable results.")

        # Responsible for adjusting search results real time to best suite what the user is looking for right now.
        self.live_sess_agent = Agent(name = "Live Session Agent", instructions = "Your expertise in session analysis is required to assist the Live Session Agent in dynamically adjusting results. Examine the retrieved passages and information in the global memory pool. \
                                Determine how the Live Session Agent can use this data to refine its understanding of the user's immediate needs. Suggest ways to dynamically adjust search results or recommend new queries in real-time, ensuring that session adjustments align with user \
                                preferences and goals.")

        # Responsible for priortizing documents to match the context.
        self.doc_rank_agent = Agent(name = "Document Ranking Agent", instructions="Your task is to help the prioritize documents for better ranking. Analyze the retrieved passages and global memory pool to identify ways to rank documents effectively. Focus on combining \
                            historical user preferences, immediate needs, and session behavior to refine ranking algorithms. Your insights should ensure that documents presented by the Document Ranking Agent are prioritized to match user interests and search context.")


        self.user_prof_agent = Agent(name = "User Profile Agent", instructions="Your task is to help the User Profile Agent improve its understanding of user preferences based on ranked document lists and the shared global memory pool. From the provided passages and global \
                                memory pool, analyze clues about the user's search preferences. Look for themes, types of documents, and navigation behaviors that reveal user interest. Use these insights to recommend how the User Profile Agent can refine and expand the \
                                user profile to deliver better-personalized results. If the global memory is empty, meaning a new profile was just made, output 'Nothing'.")


        self.cog_agent = Agent(name = "Cognitive Agent", instructions="Your responsbility is to help the cognitive agent to enahance its understanding of user insights to continuous improve the system's response")

        self.chain_of_thought = Agent(name = "Chain Of Thought Agent", instructions="To solve the problem, please think and reason step by step, then answer.")

        
    
    def get_articles_by_index(self, article_dict, list_articles):

        exp = re.compile("(\d+)")
        indexed_list = exp.findall(list_articles)

        print(list_articles)

        return_vals = {}
        print(indexed_list)
        print(article_dict.keys())
        for l in indexed_list:
            try:
                return_vals[int(l)] = article_dict[int(l)]
            except KeyError:
                print(f"ID {l} was NOT found. This can can lead to potential information loss.")
        return return_vals

    def update_glob_mem_state(self, query):
        glob_memory_state = Runner.run_sync(self.glob_message_pool, query).final_output
        # Update the global message Pool
        #self.global_message_pool["Global Memory"] = glob_memory_state
        return glob_memory_state

    
    def get_cont_retr_docs(self, query):
        doc_output_list = Runner.run_sync(self.doc_rank_agent, query).final_output
        return doc_output_list
    
    def get_live_sess_sugg(self, query):
        suggestions = Runner.run_sync(self.live_sess_agent, query).final_output
        return suggestions
    
    def update_user_profile(self, query):
        doc_output_list = Runner.run_sync(self.user_prof_agent, query).final_output
        return doc_output_list
    
    def cot(self, query):
        cot_output = Runner.run_sync(self.chain_of_thought, query).final_output
        return cot_output
    
    def rank_docs(self, query):
        rank_doc_output = Runner.run_sync(self.doc_rank_agent, query).final_output
        return rank_doc_output


    def final_cog_output(self, query):
        cog_output = Runner.run_sync(self.cog_agent, query).final_output
        return cog_output






        


    def ask_question(self, query):
        

        
        #query = input("what is your query? \n")
        articles = self.doc_retriever.embed_query(query, 5)
        self.global_message_pool["Query"] = query
        self.global_message_pool["Passages"] = articles
            
        #str_glob_mess_pool = str(global_message_pool)


        # Update the user profile
        profile_agent_output = self.update_user_profile(f"Update the user profile based on the query \"{query}\" given in the message pool{self.global_message_pool}")

        # then pipe into the global message pool updating it
        glob_memory_state = self.update_glob_mem_state(f"Update the global message pool with new information from the user profile agent {profile_agent_output}")
        # Update the global message Pool
        self.global_message_pool["Global Memory"] = glob_memory_state
        print("GMP update #1", glob_memory_state)


        # With the articles given priortize the most relevant articles
        
        cont_retr_output = self.get_cont_retr_docs(f"As said in the instructions above, priortize the most relevant articles given in the \"Passages\" field in here: {str(self.global_message_pool)}. Just list the passage IDs that are best relevant in form of a Python list.")
        print("agent's article output", cont_retr_output)
        # tell the agent to give the passage IDs only so it doesnt need to output the entire wikipedia text.


        cont_retr_articles = self.get_articles_by_index(articles, cont_retr_output)
        self.global_message_pool["Passages"] = cont_retr_articles
        #print(cont_retr_articles)


        #new_gmp = Runner.run_sync(glob_message_pool, f"Update the global message pool (here {new_gmp}) by modifying the \"passages\"  field with new information from the context retrieval agent {cont_retr_output}").final_output


        lve_ses_suggestions = self.get_live_sess_sugg(f"Using the context given, suggest queries or adjusting search results based on the retrieved passages and queries based on the current findings {str(self.global_message_pool)}")

        glob_memory_state = self.update_glob_mem_state(f"Update the current global memory \"{glob_memory_state}\" with new information from the live session agent {lve_ses_suggestions}")
        print(glob_memory_state)
        self.global_message_pool["Global Memory"] = glob_memory_state


        # Re-rank the articles.
        ranked_article_output  = self.rank_docs(f"Rerank the documents in the current field \"passages\. Simply give out the IDs of the passages. {self.global_message_pool}")
                                                
        retr_ranked_articles = self.get_articles_by_index(articles, ranked_article_output)
        del self.global_message_pool["Passages"]
        self.global_message_pool["Passages ranked most relevant to least relevant"] = retr_ranked_articles
        print(self.global_message_pool.keys())

        #new_gmp = Runner.run_sync(glob_message_pool, f"Update the global message pool with new information from the document ranking agent {ranked_article_output}").final_output
        


        cot_prompt = f"Question: {query}, Passages: {retr_ranked_articles}, Read the given question and passages to gather relevant information. 2. Write reading notes summarizing the key points from these passages. 3. Discuss the relevance of the given question and passages. 4. If some passages are relevant to the given question, provide a brief answer based on the passages. 5. If no passage is relevant, directly provide the answer without considering the passages."
        cot_answer = self.cot(cot_prompt)
        print(cot_answer)

        #cot_gmp = f"({global_message_pool}, Initial Answer: {cot_answer})"
        #print("chain of thought output", cot_gmp)

        cognitive_agent_prompt = f"Verify the reasoning process in the initial response shown here \"{cot_answer}\" for errors or misalignments. Use insights from user interaction analysis shown here \"{self.global_message_pool}\" to refine this response, correcting any inaccuracies and enhancing the query answers based on user profile. Ensure that your refined response aligns more closely with the user's immediate needs and incorporates foundational or advanced knowledge from other sources. Do not restate intructions."

        final_answer = self.final_cog_output(cognitive_agent_prompt)

        profile_agent_output = self.update_user_profile(f"Update the user profile based on this past query{self.global_message_pool}")


        glob_memory_state = self.update_glob_mem_state(f"Update the global memory with new information from the user profile agent {profile_agent_output}")
        self.global_message_pool["Global Memory"] = glob_memory_state


        print("udpated user insights", profile_agent_output)

        print(final_answer)

        return final_answer

"""

def test_agent(n = 20):


    


    agent = persona_rag()

    dataset = load_dataset("trivia_qa", "rc")
    #pdds = dataset.to_pandas()
    print(type(dataset))
    print(len(dataset))


    correct = 0
        
    for i, row in enumerate(dataset["validation"]):
        q = row["question"]
        gold = row["answer"]
        #print(gold)
        #print(type(gold))
        #print(q)
        #print("type for q", type(q))
        
        #print("This was asked to the agent", q)
        pred = agent.ask_question(q)
        
        s = set(gold["aliases"])
        s1 = s.union(set(gold["normalized_aliases"]))

        for al in s1:
            if al in pred.lower():
                print("response is correct!")
                correct += 1
                break

        #for alias in gold:
        #    print(f"{gold[alias]} AND {gold}")
        #    if gold[alias] in pred.lower():
        #        print("response is correct!")
        #        correct += 1
        if i > n:
            break

        
        print(f"Q {i} of {n}: {q}")
        print(f"Pred: {pred}")
        print(f"Gold: {gold}")
        print(f"Accuracy: {correct}/{n}")
        print("---")

    print(f"Accuracy: {correct}/{n}")

test_agent()



"""
agent = persona_rag()

while True:
    query = input("ask something")
    agent.ask_question(query)
    