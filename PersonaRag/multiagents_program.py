"""
References:
https://chatgpt.com/share/6981982c-0a78-8001-8170-4be7344e35f7
https://chatgpt.com/share/69819b4d-c9cc-8001-a52f-4297c0056234
https://chatgpt.com/share/69819d6f-2054-8001-b4e8-37888b5c7177
https://chatgpt.com/share/69819dfa-b3f4-8001-a726-9017540b51a1
https://youtu.be/dOKHuw52YTA?si=bWg7AvQK18tUVn23
https://docs.ragas.io/en/stable/
Sam Bhagwhat - Principles of Building AI Agents
https://arxiv.org/abs/2407.09394
"""
from agents import Agent, Runner
from openai import OpenAI
import openai
from document_retrieval import k_doc_retriever
import numpy as np # 2.1.3
from datasets import load_dataset
from datasets import Dataset
import re
import json
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from ragas.metrics import (answer_relevancy, faithfulness, context_recall, answer_correctness, context_precision)
from ragas import evaluate
import duckdb
from json import loads
import json
import pandas as pd
from prompts.live_session_prompts import (
    build_live_session_suggestions_prompt,
    build_update_user_from_live_session_prompt,
)

from prompts.retrieval_prompts import build_rank_articles_prompt

from prompts.cot_prompt import (
    build_cot_prompt,
    build_cognitive_agent_prompt,
)

from prompts.update_mem import (
    build_update_global_memory_from_live_prompt,
    build_update_user_profile_prompt,
    build_update_global_memory_from_profile_prompt,
    build_final_gmp_update_prompt,
)


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
        self.glob_message_pool = Agent(
            name="Global Message Pool",
            instructions=f"""
        You are responsible for maintaining and enriching the Global Message Pool,
        serving as a central hub for inter-agent communication.

        Using the responses from individual agents and the existing global memory,
        consolidate key insights into a shared repository.

        Your goal is to organize a comprehensive message pool that includes:
        - agent-specific findings
        - historical user preferences
        - session-specific behaviors
        - search queries
        - user feedback

        This structure should provide all agents with meaningful data points and
        strategic recommendations, reducing redundant communication and improving
        the system's overall efficiency.

        Your response must only include the message pool in the form of a python
        dictionary in the following manner:
        (agent specific findings: {self.gmp_format},
        query: (current query),
        passages: (passages as retrieved)).

        Do not include anything else other than the message pool.

        If there are no findings yet for any of the agent specific findings fields
        in the message pool, put 'Nothing' in the field.
        """
        )

        # Contextual Retrieval Agent
        self.cont_retr_agent = Agent(
            name="Contextual Retrieval Agent",
            instructions="""
        You are a search technology expert guiding the Contextual Retrieval Agent
        to deliver context-aware document retrieval.

        Using the global memory pool and the retrieved passages, identify strategies
        to refine document retrieval.

        Highlight how user preferences, immediate needs, and global insights can be
        leveraged to adjust search queries and prioritize results that align with
        the user's interests.

        Ensure the Contextual Retrieval Agent uses this shared information to
        deliver more relevant and valuable results.

        In your responses, just show the json without the json declaration and
        without any extra text.
        """
        )

        # Live Session Agent
        self.live_sess_agent = Agent(
            name="Live Session Agent",
            instructions="""
        Your expertise in session analysis is required to assist the Live Session
        Agent in dynamically adjusting results.

        Examine the retrieved passages and information in the global memory pool.

        Determine how the Live Session Agent can use this data to refine its
        understanding of the user's immediate needs.

        Suggest ways to dynamically adjust search results or recommend new queries
        in real-time, ensuring that session adjustments align with user preferences
        and goals.
        """
        )

        # Document Ranking Agent (renamed but same structure)
        self.doc_rank_agent = Agent(
            name="Contextual Retrieval Agent",
            instructions="""
        You are a search technology expert guiding the Contextual Retrieval Agent
        that performs context-aware document ranking.

        Using the global memory pool and the retrieved passages, identify strategies
        to refine document retrieval.

        Highlight how user preferences, immediate needs, and global insights can be
        leveraged to adjust search queries and prioritize results that align with
        the user's interests (both historical and current).

        Ensure the Contextual Retrieval Agent uses this shared information to
        deliver more relevant and valuable results.

        In your responses, just show the json without the json declaration and
        without any extra text.
        """
        )

        # User Profile Agent
        self.user_prof_agent = Agent(
            name="User Profile Agent",
            instructions="""
        Your task is to help the User Profile Agent improve its understanding of
        user preferences based on ranked document lists and the shared global
        memory pool.

        From the provided passages and global memory pool, analyze clues about the
        user's search preferences.

        Look for themes, types of documents, and navigation behaviors that reveal
        user interest.

        Use these insights to recommend how the User Profile Agent can refine and
        expand the user profile to deliver better-personalized results.

        If the global memory is empty, meaning a new profile was just made,
        output 'Nothing'.
        """
        )

        # Feedback Agent
        self.feedback_agent = Agent(
            name="Feedback Agent",
            instructions="""
        You are an expert in feedback collection and analysis, guiding the Feedback
        Agent to gather and utilize insights.
        """
        )

        # Cognitive Agent
        self.cog_agent = Agent(
            name="Cognitive Agent",
            instructions="""
        Your responsibility is to help the cognitive agent to enhance its
        understanding of user insights to continuously improve the system's
        response.
        """
        )

        # Chain of Thought Agent
        self.chain_of_thought = Agent(
            name="Chain Of Thought Agent",
            instructions="""
        To solve the problem, please think and reason step by step, then answer.
        """
        )

        # JSON Formatting Agent
        self.string_format_agent = Agent(
            name="Json Formatting Agent",
            instructions="""
        Because LLM outputs can be inconsistent, your role is to adjust the output
        such that json.loads() can parse LLM outputs.
        """
        )

        # Global Message Pool Analyst
        self.gmp_analyst = Agent(
            name="Global Message Pool Analyst",
            instructions=f"""
        Your job is to analyze the change in the overall content in the global
        message pool.

        Does the new global message pool reflect the user's new interests?

        Does the message pool still keep track of the user's previous interests
        and queries while evolving with the context from the new query?

        Suggest improvements in maintaining a cohesive memory structure.
        """
        )

        self.gmp_analyst_instructions = ""

    def retrieve_articles(self, indexes):

        #text_splitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap=0)

        print("indexes passed", indexes)

        if indexes is not None:

            for i in range(len(indexes)):
                if not isinstance(indexes[i], int):
                    indexes[i] = int(indexes[i])
                    

            query = "SELECT * FROM wiki_chunked WHERE column0 IN (SELECT UNNEST(?))"
            df = self.wiki_text_db.execute(query, [indexes]).fetchall()     

            text_list_raw = []

            print("THE DATAFRAME WE JUST GOT", df)
            #text_list = dict(df)
            text_list = []
            for i in df:
                doc_test = i[1][0:3000]
                text_list.append({i[0]: doc_test})
                text_list_raw.append(doc_test)
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
            print(f"Could not properly parse articles, error {e}. Trying to correct the format...")
            # try removing the declaration headers, GPT normally generates

            reformatted_output = Runner.run_sync(self.string_format_agent, f"Fix the following json here - {json_string}. Your response should just be a json. Do **not** include any extra keys, explanations or text outside the JSON. The format must be [{{\"Article_ID\": \"<id here>\", \"Brief_Summary\": \"<summary here>\"}}]. Start the entire message with \"[\".").final_output

            try:
                stripped_headers_beg = re.sub("```json", "", reformatted_output)
                stripped_headers_end = re.sub("```","",stripped_headers_beg)
                json_format = json.loads(stripped_headers_end)
                for i in json_format:
                    index_list.append(int(i.get("Article_ID")))
                return index_list
            except Exception as e:
                print(f"Could not parse JSON output after implementing attempted fixes. {e}")

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
                    ds_path = self.doc_retriever.get_dataset_path()
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
    

    def analyze_gmp(self, gm1, gm2, prev_instructions):
        global_memory_analyzation = Runner.run_sync(self.gmp_analyst, f"Analyze the change between the old global message pool {gm1} and the new global message pool {gm2}. Look for any clues where the users's interests evolve as stated in the system instructions and suggests changes in maintaining coherency. If the analyzation was run previously, analyze how the global message pool was previously improved given these instructions {prev_instructions} and suggest changes to ensure consistency.").final_output
        return global_memory_analyzation


    def ask_question(self, query):
        
        beginning_gmp = self.global_message_pool
        
        #query = input("what is your query? \n")
        indexes_for_retrieval = self.doc_retriever.embed_query(query, 3)

        articles, raw_articles = self.retrieve_articles(indexes_for_retrieval)
        print("ARTICLES WITH ID", articles)
        print("\n\n RAW ARTICLES", raw_articles)
        #articles with ID | just the articles in the form of a list.

        self.global_message_pool["Query"] = query
        #self.global_message_pool["Passages"] = articles
            
        #str_glob_mess_pool = str(global_message_pool)


        # Update the user profile (New proposal: For this agent to give suggestions.) -- No Passages
        profile_agent_output = self.update_user_profile(build_update_user_profile_prompt(self.global_message_pool["Global Memory"])) #need query here


        # then pipe into the global message pool updating it. -- No Passages
        glob_memory_state = self.update_glob_mem_state(build_update_global_memory_from_profile_prompt(self.global_message_pool["Global Memory"], profile_agent_output, self.gmp_analyst_instructions))
        
        # Update the global message Pool
        self.global_message_pool["Global Memory"] = glob_memory_state
        #print(" #### glob memory state after updating the user insights ####\n", glob_memory_state)

        # With the articles given priortize the most relevant articles
        
        # -- Proposal to remove this agent call to cut down api calls and reduce response time. With only n=3, this becomes redundant.
        #cont_retr_output = self.get_cont_retr_docs(f"As said in the instructions above, prioritize the most relevant articles given in the \"Passages\" field in here: {str(self.global_message_pool)}. In your responses, just show the json without the json declaration and do NOT show include any extra keys, explanations or text outside the json. Start the entire message with \"[\".[{{\"Article_ID\": \"<id here>\", \"Brief_Summary\": \"<summary here>\"}}]")
        #print("agent's article output", cont_retr_output)
        # tell the agent to give the passage IDs only so it doesnt need to output the entire wikipedia text.
        #print(cont_retr_output)
        #cont_retr_article_list = self.json_to_list(cont_retr_output)


        #cont_retr_articles, raw_articles = self.retrieve_articles(cont_retr_article_list)
        #self.global_message_pool["Passages"] = cont_retr_articles
        #print(cont_retr_articles)

        


        #new_gmp = Runner.run_sync(glob_message_pool, f"Update the global message pool (here {new_gmp}) by modifying the \"passages\"  field with new information from the context retrieval agent {cont_retr_output}").final_output


        lve_ses_suggestions = self.get_live_sess_sugg(
            build_live_session_suggestions_prompt(
                self.global_message_pool,
                articles
            )
        )
        print(lve_ses_suggestions)


        update_user_suggested_topics = self.update_user_profile(
            build_update_user_from_live_session_prompt(
                self.global_message_pool,
                lve_ses_suggestions
            )
        )
        print(update_user_suggested_topics)
        

        glob_memory_state = self.update_glob_mem_state(
            build_update_global_memory_from_live_prompt(
                self.global_message_pool,
                update_user_suggested_topics,
                lve_ses_suggestions,
                self.gmp_analyst_instructions
            )
        )
        #print(" #### glob memory state after live suggestions ####\n", glob_memory_state)
        print(glob_memory_state)
        self.global_message_pool["Global Memory"] = glob_memory_state
        # update the user profile from the live sess agent

        

        past_10_query_articles = str(self.global_message_pool["Past 10 Passages"])

        # Re-rank the articles. Include the past articles if they are relevant.
        ranked_article_output = self.get_cont_retr_docs(
            build_rank_articles_prompt(
                query,
                articles,
                past_10_query_articles,
                self.global_message_pool
            )
        )
        print("RERANKED ARTICLE RESULTS", ranked_article_output)   
        # merge this into one agent. Maybe include passages for agents calls if it is absolutely needed, otherwise its not.
        

        ranked_index_list = self.json_to_list(ranked_article_output)

        

        retr_ranked_articles, raw_articles = self.retrieve_articles(ranked_index_list)
        
        #new_gmp = Runner.run_sync(glob_message_pool, f"Update the global message pool with new information from the document ranking agent {ranked_article_output}").final_output

        #past_article_output = self.get_articles_by_index(articles, self.glob_message_pool["Past 10 Passages"])
        
        cot_prompt = build_cot_prompt(query, retr_ranked_articles)
        cot_answer = self.cot(cot_prompt)
        #print(cot_answer)

        

        #cot_gmp = f"({global_message_pool}, Initial Answer: {cot_answer})"
        #print("chain of thought output", cot_gmp)

        cognitive_agent_prompt = build_cognitive_agent_prompt(
            query,
            cot_answer,
            self.global_message_pool
        )

        final_answer = self.final_cog_output(cognitive_agent_prompt)

        

        profile_agent_output = self.update_user_profile(
            build_update_user_profile_prompt(self.global_message_pool)
        )
                


        glob_memory_state = self.update_glob_mem_state(
            build_update_global_memory_from_profile_prompt(
                self.global_message_pool,
                profile_agent_output,
                self.gmp_analyst_instructions
            )
        )

        self.global_message_pool["Global Memory"] = glob_memory_state

        self.global_message_pool["Past 10 Passages"].insert(0, retr_ranked_articles) #this should contain the most relevant articles
        print(self.global_message_pool["Global Memory"])
        if len(self.global_message_pool["Past 10 Passages"]) == 10:
            self.global_message_pool["Past 10 Passages"].pop(-1)
        print("THE LAST 10 PASSAGES", self.global_message_pool["Past 10 Passages"])


        

        #print("#### updated user insights #### \n\n", profile_agent_output)


        #print(" #### last glob mem update ####\n", glob_memory_state)

        analyzation = self.analyze_gmp(beginning_gmp, self.global_message_pool, self.gmp_analyst_instructions)
        print("GMP ANALYZATION", analyzation)


        glob_memory_state = self.update_glob_mem_state(
            build_final_gmp_update_prompt(analyzation)
        )
        self.gmp_analyst_instructions = analyzation

        self.global_message_pool["Global Memory"] = glob_memory_state
        print("#### final answer #### \n\n")
        print(final_answer)

        print(raw_articles)
        return final_answer, raw_articles
    

def main():

    llm_judge = ChatOpenAI(model="gpt-4o")

    correctness_measurement = []

    agent = persona_rag()
    
    dataset = load_dataset("web_questions")

    

    qa_set = []

    limit = 545
    i = 0

    
    dataset_1 = pd.read_csv("custom_test.csv", sep = ",")
    


    #print(len(dataset["test"]))

    
    for entry in dataset_1.iterrows():
        #query = entry["question"]
        #reference = entry["answers"]

        # This is for the custom transit-related dataset
        query = (entry[1]["query"])
        reference = entry[1]["correct_answers"]
      


        try:
            answer, context_articles = agent.ask_question(query)
        except openai.RateLimitError as e:
            print(f"error trying to query {e}, skipping to next")
            i += 1
            continue

        
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

        #qa = {"question": query, "contexts": context_articles, "answer": answer, "reference": "; ".join(reference)}
        qa = {"question": query, "contexts": context_articles, "answer": answer, "reference": reference}


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

    results_df.to_csv('results4.csv', index = False)

    print(results_df.head())

    results_df

if __name__ == "__main__":
    main()
    


"""


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