from agents import Agent, Runner
from openai import OpenAI
import wikipedia
from document_retrieval import k_doc_retriever
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import numpy as np
from datasets import load_dataset
from datasets import Dataset
import re
import faiss
from ast import literal_eval
from transformers import pipeline
from datasets import load_dataset
import json
from datasets import load_dataset
from langchain_openai.chat_models import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from langchain.schema import SystemMessage, HumanMessage
from ragas.metrics import (answer_relevancy, faithfulness, context_recall, answer_correctness, context_precision)
from ragas import evaluate
from datasets import load_dataset, Dataset
import duckdb
from json import loads
import json


#index = None
#model = None

class persona_rag:
    def __init__(self):
         # initialize all the agents

        client = OpenAI()

        self.wiki_text_db = duckdb.connect("wiki_chunked.duckdb")

        # initialize the vector db of articles.
        self.doc_retriever = k_doc_retriever()
       


        self.gmp_format = "User Profile Agent: {user_profile_answer}, Live Session Agent: {live_session_answer}, Document Ranking Agent: {document_ranking_answer}"


        self.global_message_pool = {}
        self.global_message_pool["Global Memory"] = self.gmp_format
        self.global_message_pool["Past 10 Passages"] = [] # a queue of dicts tracks the past number passages. Apended by the document ranking agent.
        # structure: [{P1: a brief summary, P2: a brief summary}]
        # if the live session agent is taken into consideration, then there would more profound information of knowing which documents are ACTUALLY relevant
            

        # Serves as a hub for interagent communication
        self.glob_message_pool = Agent(name = "Global Message Pool", instructions=f"You are responsible for maintaining and enriching the Global Message Pool, serving as a central hub for inter-agent communication. Using the responses from individual agents \
                                and the existing global memory, consolidate key insights into a shared repository. Your goal is to organize a comprehensive message pool that includes agent-specific findings, historical user preferences, session-specific behaviors, search \
                                queries, and user feedback. This structure should provide all agents with meaningful data points and strategic recommendations, reducing redundant communication and improving the system's overall efficiency. Your response must only include\
                                the message pool in the form of a python dictionary in the following manner: (agent specific findings: {self.gmp_format}, query: (current query), passages: (passages as retrieved)).\
                                Do not include anything else other than the message pool. If there are no findings yet for any of the agent specific findings fields in the message pool, put 'Nothing' in the field.")

        # Responsible for retrieving the top relevant documents based on user needs and interests
        self.cont_retr_agent = Agent(name="Contextual Retrieval Agent", instructions = "You are a search technology expert guiding the Contextual Retrieval Agent to deliver context-aware document retrieval. Using the global memory pool and the retrieved passages, \
                                identify strategies to refine document retrieval. Highlight how user preferences, immediate needs, and global insights can be leveraged to adjust search queries and prioritize results that align with the user's interests. Ensure the Contextual \
                                Retrieval Agent uses this shared information to deliver more relevant and valuable results.  In your responses, just show the json without the json declaration and without any extra text. ")

        # Responsible for adjusting search results real time to best suite what the user is looking for right now.
        self.live_sess_agent = Agent(name = "Live Session Agent", instructions = "Your expertise in session analysis is required to assist the Live Session Agent in dynamically adjusting results. Examine the retrieved passages and information in the global memory pool. \
                                Determine how the Live Session Agent can use this data to refine its understanding of the user's immediate needs. Suggest ways to dynamically adjust search results or recommend new queries in real-time, ensuring that session adjustments align with user \
                                preferences and goals.")

        # Responsible for priortizing documents to match the context.
        self.doc_rank_agent = Agent(name = "Document Ranking Agent", instructions="Your task is to help the prioritize documents for better ranking. Analyze the retrieved passages and global memory pool to identify ways to rank documents effectively. Focus on combining \
                            historical user preferences, immediate needs, and session behavior to refine ranking algorithms. Your insights should ensure that documents presented by the Document Ranking Agent are prioritized to match user interests and search context. In your responses, just show the json without the json declaration and without any extra text. ")


        self.user_prof_agent = Agent(name = "User Profile Agent", instructions="Your task is to help the User Profile Agent improve its understanding of user preferences based on ranked document lists and the shared global memory pool. From the provided passages and global \
                                memory pool, analyze clues about the user's search preferences. Look for themes, types of documents, and navigation behaviors that reveal user interest. Use these insights to recommend how the User Profile Agent can refine and expand the \
                                user profile to deliver better-personalized results. If the global memory is empty, meaning a new profile was just made, output 'Nothing'.")


        self.feedback_agent = Agent(name = "Feedback Agent", instructions="You are an expert in feedback collection and analyis, guiding the Feedback Agent to gather and utilize insights.")

        self.cog_agent = Agent(name = "Cognitive Agent", instructions="Your responsbility is to help the cognitive agent to enahance its understanding of user insights to continuous improve the system's response")

        self.chain_of_thought = Agent(name = "Chain Of Thought Agent", instructions="To solve the problem, please think and reason step by step, then answer.")

        
    
    def retrieve_articles(self, indexes):

        print("indexes passed", indexes)

        if indexes is not None:

            for i in range(len(indexes)):
                if not isinstance(indexes[i], int):
                    indexes[i] = int(indexes[i])
                    

            query = "SELECT * FROM wiki_chunked WHERE column0 IN (SELECT UNNEST(?))"
            df = self.wiki_text_db.execute(query, [indexes]).fetchall()     

            text_list_raw = []

            print("THE DATAFRAME WE JUST GOT", df)
            text_list = dict(df)

            for i in df:
                text_list_raw.append(i[1])
            return text_list, text_list_raw
        else:
            print("Invalid indexes passed in. Returning empty fields.")
            return {}, []



           
    
    def json_to_list(self, json_string):
        try:
            print("json string passed in ", json_string)
            index_list = []
            json_format = json.loads(json_string)
            for i in json_format:
                index_list.append(int(i.get("Article_ID")))
            return index_list

        except Exception as e:
            print(f"Could not properly parse articles, error {e}")
            return None

    def get_articles_by_index(self, article_dict, list_articles):



        exp = re.compile("(\d+)")
        indexed_list = exp.findall(list_articles)

        return_vals = {}
        raw_articles = []
        for l in indexed_list:
            try:
                raw_articles.append(article_dict[int(l)])
                return_vals[int(l)] = article_dict[int(l)]
            except KeyError:
                try:
                    wiki_db = duckdb.connect("wiki_chunked.duckdb")
                    res = wiki_db.execute(f"SELECT column1 FROM wiki_chunked WHERE column0 = {int(l)}").fetchone()
                    raw_articles.append(res[0])
                    return_vals[int(l)] = res[0] 
                except Exception as e:
                    print(f"Could not retrieve document {l}. Error {e}")



                print(f"ID {l} was NOT found.")
        return return_vals, raw_articles

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
    
    def get_feedback(self, query):
        feedback_output = Runner.run_sync(self.feedback_agent_agent, query).final_output
        return feedback_output


    def ask_question(self, query):
        

        
        #query = input("what is your query? \n")
        indexes_for_retrieval = self.doc_retriever.embed_query(query, 3)

        articles, raw_articles = self.retrieve_articles(indexes_for_retrieval)
        print("ARTICLES WITH ID", articles)
        print("\n\n RAW ARTICLES", raw_articles)
        #articles with ID | just the articles in the form of a list.

        self.global_message_pool["Query"] = query
        self.global_message_pool["Passages"] = articles
            
        #str_glob_mess_pool = str(global_message_pool)


        # Update the user profile
        profile_agent_output = self.update_user_profile(f"Update the user profile based on the query \"{query}\" given in the message pool \"{str(self.global_message_pool)}\". If possible build on the information that is already supplied, otherwise do not lose the supplied info")

        # then pipe into the global message pool updating it
        glob_memory_state = self.update_glob_mem_state(f"Update the global message pool as shown here \"{str(self.global_message_pool)}\" with new information from the user profile agent \"{profile_agent_output}\". If possible build on the information that is already supplied, otherwise do not lose the supplied info")
        # Update the global message Pool
        self.global_message_pool["Global Memory"] = glob_memory_state
        #print(" #### glob memory state after updating the user insights ####\n", glob_memory_state)

        # With the articles given priortize the most relevant articles
        print("CURRENT STATE OF GLOBAL MESSAGE POOL ", str(self.glob_message_pool))
        cont_retr_output = self.get_cont_retr_docs(f"As said in the instructions above, prioritize the most relevant articles given in the \"Passages\" field in here: {str(self.global_message_pool)}. In your responses, just show the json without the json declaration and do NOT show include any extra keys, explanations or text outside the json. Start the entire message with \"[\".[{{\"Article_ID\": \"<id here>\", \"Brief_Summary\": \"<summary here>\"}}]")
        #print("agent's article output", cont_retr_output)
        # tell the agent to give the passage IDs only so it doesnt need to output the entire wikipedia text.
        print(cont_retr_output)
        cont_retr_article_list = self.json_to_list(cont_retr_output)


        cont_retr_articles, raw_articles = self.retrieve_articles(cont_retr_article_list)
        self.global_message_pool["Passages"] = cont_retr_articles
        #print(cont_retr_articles)


        #new_gmp = Runner.run_sync(glob_message_pool, f"Update the global message pool (here {new_gmp}) by modifying the \"passages\"  field with new information from the context retrieval agent {cont_retr_output}").final_output


        lve_ses_suggestions = self.get_live_sess_sugg(f"Using the context given, suggest queries or adjusting search results based on the retrieved passages and queries based on the current findings {str(self.global_message_pool)}. ")


        update_user_suggested_topics = self.update_user_profile(f"From the live session agent integrate new insights of the scope of the query that include suggested topics as given by the live session agent. This is the current memory \"{str(self.global_message_pool)}\".  These are the suggestions from the live session agent \"{lve_ses_suggestions}\"")


        glob_memory_state = self.update_glob_mem_state(f"Update the current global memory \"{glob_memory_state}\" with profile-based contextual suggestions from {update_user_suggested_topics}. If possible build on the information that is already supplied, otherwise do not lose the supplied info")
        #print(" #### glob memory state after live suggestions ####\n", glob_memory_state)

        self.global_message_pool["Global Memory"] = glob_memory_state
        # update the user profile from the live sess agent

        past_10_query_articles = str(self.global_message_pool["Past 10 Passages"])

        # Re-rank the articles. Include the past articles if they are relevant.
        ranked_article_output  = self.rank_docs(f"Rerank the documents in the current field \"passages\". Included are articles that were ranked from the past 10 queries. Past articles from last 10 queries: {past_10_query_articles}\". If any entries from article corpus from the past 10 queries are highly relevant, \
                                                include them in the ranking. {self.global_message_pool}. In your responses, just show the json without the json declaration and do NOT show include any extra keys, explanations or text outside the json. Start the entire message with \"[\". [{{\"Article_ID\": \"<id here>\", \"Brief_Summary\": \"<summary here>\"}}].") #need to include the correct form of output.
        print("RERANKED ARTICLE RESULTS", ranked_article_output)   

        ranked_index_list = self.json_to_list(ranked_article_output)

        retr_ranked_articles, raw_articles = self.retrieve_articles(ranked_index_list)
        del self.global_message_pool["Passages"]
        self.global_message_pool["Passages ranked most relevant to least relevant"] = retr_ranked_articles
        
        #new_gmp = Runner.run_sync(glob_message_pool, f"Update the global message pool with new information from the document ranking agent {ranked_article_output}").final_output

        #past_article_output = self.get_articles_by_index(articles, self.glob_message_pool["Past 10 Passages"])
        
        cot_prompt = f"Question: {query}, Passages: {retr_ranked_articles}, 1. Read the given question and passages to gather relevant information. 2. Write reading notes summarizing the key points from these passages. 3. Discuss the relevance of the given question and passages. 4. If some passages are relevant to the given question, provide a brief answer based on the passages. 5. If and only if no passage is relevant at all, you can simply state the answer to the question without looking at the passages."
        cot_answer = self.cot(cot_prompt)
        #print(cot_answer)

        #cot_gmp = f"({global_message_pool}, Initial Answer: {cot_answer})"
        #print("chain of thought output", cot_gmp)

        cognitive_agent_prompt = f"Verify the reasoning process in the initial response shown here \"{cot_answer}\" for errors or misalignments to the query as shown here \"{query}\". Use insights from user interaction analysis shown here \"{str(self.global_message_pool)}\" to refine this response, correcting any inaccuracies and enhancing the query answers based on user profile. Ensure that your refined response aligns more closely with the user's immediate needs and incorporates foundational or advanced knowledge from other sources. Be sure to mirror the way they ask the query reflecting the same tone the user gives (based on the query). Do not restate intructions."

        final_answer = self.final_cog_output(cognitive_agent_prompt)

        profile_agent_output = self.update_user_profile(f"Update the user profile based on this past query \"{str(self.global_message_pool)}\"")


        glob_memory_state = self.update_glob_mem_state(f"Update the global memory as shown here \"{str(self.global_message_pool)}\" with new information from the user profile agent {profile_agent_output}")
        self.global_message_pool["Global Memory"] = glob_memory_state

        self.global_message_pool["Past 10 Passages"].insert(0, retr_ranked_articles) #this should contain the most relevant articles
        if len(self.global_message_pool["Past 10 Passages"]) == 10:
            self.global_message_pool["Past 10 Passages"].pop(-1)
        print("THE LAST 10 PASSAGES", self.global_message_pool["Past 10 Passages"])


        

        #print("#### updated user insights #### \n\n", profile_agent_output)


        #print(" #### last glob mem update ####\n", glob_memory_state)

        print("#### final answer #### \n\n")
        print(final_answer)


        return final_answer, raw_articles
    

def main():
    agent = persona_rag()
    while(1):
        p = input("Ask something here\n")
        agent.ask_question(p)

if __name__ == "__main__":
    main()
    


"""

def test_agent():


    llm_judge = ChatOpenAI(model="gpt-4o")

    correctness_measurement = []

    agent = persona_rag()
    
    dataset = load_dataset("web_questions")

    qa_set = []

    limit = 24
    i = 0


    


    print(len(dataset["test"]))

    
    for entry in dataset["test"]:
        query = entry["question"]
        reference = entry["answers"]
        answer, context_articles = agent.ask_question(query)
        #question -> string
        #contexts -> list of strings
        #answer -> string
        #reference -> string

        

        for a in range(len(context_articles)):
            nt = "".join(c for c in context_articles[a] if c.isprintable())
            context_articles[a] = nt
        

        system_instructions = "You are an expert at evaluating whether or not an answer is correct."

        eval_prompt = f"Given the following \n Question: \"{query}\" \n Retrieved Articles: \"{context_articles}\" \n Model answer: \"{answer}\" \n Reference answer(s): \"{reference}\", evaluate is the model answer correct given the question and the reference. Correct answers in the reference are separated by \";\". When  the Reference Answers field has multiple answers, you can deduct points if and only if the the context of the query require some or all of of the multiple responses. Only output a json formatted exactly as shown in the following {{Evaluation: <put evaluation here>, Score: 0.89}}"

        messages = [SystemMessage(content = system_instructions), HumanMessage(content = eval_prompt)]

        accuracy_results = llm_judge.invoke(messages).content

        correctness_measurement.append(accuracy_results)

        qa = {"question": query, "contexts": context_articles, "answer": answer, "reference": "; ".join(reference)}


        qa_set.append(qa)

        i += 1
        print("Progress i", i)
        if i >= limit:
            break

    eval_dataset = Dataset.from_list(qa_set)

    result = evaluate(eval_dataset, metrics = [context_precision, context_recall, faithfulness, answer_relevancy])

    print("#### correctness results #####")
    for i in correctness_measurement:
        print(i)
        print("\n")

    results_df = result.to_pandas()

    results_df.to_csv('results1.csv', index = False)

    print(results_df.head())

    results_df

test_agent()


Agent textual evaluation guidelines
- Metrics to check for:
    - Hallucination: do responses contain facts or claims not present in the provided context
    - Faithfulness: do responses accuracy represent the context.
    - Content Similarity: are the responses consistent even if the query is qworded slightly differently
    - Completeness
    - Answer Similarity: How well do the responses address the query

- Understanding the conbtext (evlauating how well the agent is using the context)
    - context position: where does the context appear in the responses (should be normally at the top)
    - context precision: are the chunks grouped logically and does the response contain the original meaning
    - Context relevancy: is the respoinse using the most appropriate pieces of context
    - context recall: does the llm response recall the context provided.
"""