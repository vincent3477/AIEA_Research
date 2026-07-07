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
from agents import Agent, Runner, function_tool
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
from prompts.retrieval_prompts import build_rank_articles_prompt, build_cont_retr_prompt
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


class persona_rag:
    def __init__(self, verbose = False):

        # initialize agent tools
        self.tools = self.get_agent_tools()

        self.wiki_text_db = duckdb.connect("wiki_chunked.duckdb")

        # initialize the vector db of articles.
        self.doc_retriever = k_doc_retriever()
       

        self.verbose = verbose
        self.gmp_format = "User Profile Agent: {user_profile_answer}, Live Session Agent: {live_session_answer}, Document Ranking Agent: {document_ranking_answer}"
        self.contextual_retriever_memory = {}
        self.global_message_pool = {}
        self.global_message_pool["Global Memory"] = self.gmp_format
        self.global_message_pool["Past 10 Passages"] = [] # a queue of dicts tracks the past number passages. Apended by the document ranking agent.
        # structure: [{P1: a brief summary, P2: a brief summary}]
        # if the live session agent is taken into consideration, then there would more profound information of knowing which documents are ACTUALLY relevant
        self.global_message_pool["Past 5 Queries"] = []


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
        You are a Search Technology Expert.

        You use the user's query, global memory pool, and retrieved passages to create a refined search query.

        You have access to:
        get_articles_by_query(query, k)

        Workflow:
        1. Understand the user's immediate information need.
        2. Use relevant memory/passage context to refine the search query.
        3. Ignore unrelated memory or irrelevant passages.
        4. Call get_articles_by_query(query, k) by passing in a sentence or query that reflects the intent the user is asking. Please do not change up meaning.
        5. Rank articles that are defined in the tool and that were passed in.
        6. If none of the articles are relevant, then call get_articles_by_query(query, k) again with a differently worded query. Then rank articles again repeating step 5.
        7. Return only valid JSON in the required article-ranking format.

        Tool rules:
        - Call get_articles_by_query unless the user query is empty or impossible to interpret.
        - If none of the articles from the initial call are relevant, do a second call on get_articles_by_query while changing up the query.
        - Use k = 5 unless another value is provided.
        - When the user query is in form of a question, keep your query in form of a question. When the user query is in in form of a sentence, then keep in form of a sentence. 
        - Do not invent article IDs.
        - Only use article IDs from returned tool results.
        - If no relevant articles are found after you did multiple call to get_articles_by_query, return [].

        Final output format:
        [
        {
            "Article_ID": "<id here>",
            "Brief_Summary": "<summary here>"
        }
        ]
        """,
            tools = self.tools
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
            name="Document Ranking Agent",
            instructions="""
        Your task is to help the prioritize documents for better ranking. Analyze the retrieved 
        passages and global memory pool to identify ways to rank documents effectively. Focus on 
        combining historical user preferences, immediate needs, and session behavior to refine ranking 
        algorithms. Your insights should ensure that documents presented by the Document Ranking Agent are 
        prioritized to match user interests and search context. In your responses, just show the json without 
        the json declaration and without any extra text. 
        
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

    def retrieve_articles(self, indexes: list[int]):

        self.print_debug("Article Indexes Passed", indexes)

        if indexes is not None:

            for i in range(len(indexes)):
                if not isinstance(indexes[i], int):
                    indexes[i] = int(indexes[i])
                    

            query = "SELECT * FROM wiki_chunked WHERE column0 IN (SELECT UNNEST(?))"
            df = self.wiki_text_db.execute(query, [indexes]).fetchall()     

            text_list_raw = []

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

        if json_string == "" or json_string == None or json_string == "[]":
            return None
        
        try:
            index_list = []
            json_format = json.loads(json_string)
            for i in json_format:
                index_list.append(int(i.get("Article_ID")))
            return index_list

        except Exception as e:
            print(f"Could not properly parse articles, error {e}. Trying to correct the format...")
            # try removing the declaration headers, GPT normally generates

            reformatted_output = Runner.run_sync(self.string_format_agent, f"""Fix the following json here - {json_string}. Your response should just be a json. 
                                                 Do **not** include any extra keys, explanations or text outside the JSON. The format must be [{{\"Article_ID\": \"<id here>\", \"Brief_Summary\": \"<summary here>\"}}]. 
                                                 If there are no article IDs found, just give []. Please DO NOT make up article numbers if none are existent.
                                                 Start the entire message with \"[\".""").final_output

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
    
    def get_articles_by_query(self, query, k):

        """
        Returns a k-length dict containing article IDs and articles based on query.

        Args:
            query: Search terms (string list)
            num_results: Number of results to fetch (default to 5)
        """
        self.print_debug("Article Query from Contextual Retriever Agent", query)

        indexes = self.doc_retriever.embed_query(query, k)

        self.print_debug("Indexes retrieved from tool call", indexes)

        article_id_map, _ =  self.retrieve_articles(indexes)

        return article_id_map
    
    
    def get_agent_tools(self):
        @function_tool("search articles")
        def get_articles_by_query_tool(query: str, k: int):
            """
            Returns a k-length dict containing article IDs and articles based on query.

            Args:
                query: Search terms (string list)
                num_results: Number of results to fetch (default to 5)
            """
            return self.get_articles_by_query(query, k) 
        return [get_articles_by_query_tool]


    def update_glob_mem_state(self, query):
        glob_memory_state = Runner.run_sync(self.glob_message_pool, query).final_output
        # Update the global message Pool
        # self.global_message_pool["Global Memory"] = glob_memory_state
        return glob_memory_state

    
    def get_cont_retr_docs(self, query):
        doc_output_list = Runner.run_sync(self.cont_retr_agent, query).final_output
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
    
    def print_debug(self, title, content):
        if self.verbose:
            print("---------------")
            print("title")
            print(title)
            print("content")
            print(content)
            print("---------------")

    

    def analyze_gmp(self, gm1, gm2, prev_instructions):
        global_memory_analyzation = Runner.run_sync(self.gmp_analyst, f"""Analyze the change between the old global message pool {gm1} 
                                                    nd the new global message pool {gm2}. Look for any clues where the users's interests 
                                                    evolve as stated in the system instructions and suggests changes in maintaining coherency. 
                                                    If the analyzation was run previously, analyze how the global message pool was previously 
                                                    improved given these instructions {prev_instructions} and suggest changes to ensure consistency.""").final_output
        return global_memory_analyzation


    def ask_question(self, query):
        self.print_debug("query", query)
        
        beginning_gmp = self.global_message_pool
        
        indexes_for_retrieval = self.doc_retriever.embed_query(query, 3)

        articles, raw_articles = self.retrieve_articles(indexes_for_retrieval)


        self.global_message_pool["Query"] = query
        self.global_message_pool["Past 5 Queries"].append(query)
        if len(self.global_message_pool["Past 5 Queries"]) > 5:
            self.global_message_pool["Past 5 Queries"].pop(0)
        self.global_message_pool["Passages"] = articles
            

        # Update the user profile (New proposal: For this agent to give suggestions.) -- No Passages
        profile_agent_output = self.update_user_profile(build_update_user_profile_prompt(self.global_message_pool["Global Memory"])) #need query here
        self.print_debug("first user profile update", profile_agent_output)


        # then pipe into the global message pool updating it. -- No Passages
        glob_memory_state = self.update_glob_mem_state(build_update_global_memory_from_profile_prompt(self.global_message_pool["Global Memory"], profile_agent_output, self.gmp_analyst_instructions))
        
        # Update the global message Pool
        self.global_message_pool["Global Memory"] = glob_memory_state
        self.print_debug("First Update to Global Message Pool", glob_memory_state)
        
        cont_retr_prompt = build_cont_retr_prompt(query, self.global_message_pool)
        cont_retr_output = self.get_cont_retr_docs(cont_retr_prompt)
        self.print_debug("context retrieval output", cont_retr_output)
        # tell the agent to give the passage IDs only so it doesnt need to output the entire wikipedia text. It seems to be relating back to the past 10 passages in which the most recently retrieved have NOTHING to do with the query,
        cont_retr_article_list = self.json_to_list(cont_retr_output)

        # self. retrieve articles is what the agent can use instead, so nothing is deterministic.
        cont_retr_articles, raw_articles = self.retrieve_articles(cont_retr_article_list)
        self.global_message_pool["Passages"] = cont_retr_articles

        


        #new_gmp = Runner.run_sync(glob_message_pool, f"Update the global message pool (here {new_gmp}) by modifying the \"passages\"  field with new information from the context retrieval agent {cont_retr_output}").final_output


        lve_ses_suggestions = self.get_live_sess_sugg(
            build_live_session_suggestions_prompt(
                self.global_message_pool,
                articles
            )
        )
        self.print_debug("live session agent suggestions", lve_ses_suggestions)


        update_user_suggested_topics = self.update_user_profile(
            build_update_user_from_live_session_prompt(
                self.global_message_pool,
                lve_ses_suggestions
            )
        )
        self.print_debug("updated user suggested topics", update_user_suggested_topics)
        

        glob_memory_state = self.update_glob_mem_state(
            build_update_global_memory_from_live_prompt(
                self.global_message_pool,
                update_user_suggested_topics,
                lve_ses_suggestions,
                self.gmp_analyst_instructions
            )
        )
        #print(" #### glob memory state after live suggestions ####\n", glob_memory_state)
        self.global_message_pool["Global Memory"] = glob_memory_state
        # update the user profile from the live sess agent

        

        past_10_query_articles = str(self.global_message_pool["Past 10 Passages"])
        self.print_debug("the past 10 articles", past_10_query_articles)

        # Re-rank the articles. Include the past articles if they are relevant.
        ranked_article_output = self.rank_docs(
            build_rank_articles_prompt(
                query,
                articles,
                past_10_query_articles,
                self.global_message_pool
            )
        )
        self.print_debug("ranked article result from ranking agent", ranked_article_output)
        
        ranked_index_list = self.json_to_list(ranked_article_output)

        retr_ranked_articles, raw_articles = self.retrieve_articles(ranked_index_list)

        cot_prompt = build_cot_prompt(query, retr_ranked_articles)
        cot_answer = self.cot(cot_prompt)
        self.print_debug("chain of thought answer", cot_answer)


        cognitive_agent_prompt = build_cognitive_agent_prompt(
            query,
            cot_answer,
            self.global_message_pool
        )

        final_answer = self.final_cog_output(cognitive_agent_prompt)
        self.print_debug("cognitive agent answer", final_answer)

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

        self.print_debug("Final Global Message Pool", glob_memory_state)

        if len(self.global_message_pool["Past 10 Passages"]) == 10:
            self.global_message_pool["Past 10 Passages"].pop(-1)


        analyzation = self.analyze_gmp(beginning_gmp, self.global_message_pool, self.gmp_analyst_instructions)
       
       
        self.print_debug("Global Memory Analysis", analyzation)


        glob_memory_state = self.update_glob_mem_state(
            build_final_gmp_update_prompt(analyzation)
        )
        self.gmp_analyst_instructions = analyzation

        self.global_message_pool["Global Memory"] = glob_memory_state

        self.print_debug("Articles that support the answer", raw_articles)
      
        return final_answer, raw_articles
    


