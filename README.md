# AIEA_Research

## Overview: 
This repository includes the main research and project I have done with AIEA at UC Santa Cruz. I contributed in developing a Persona-based RAG pipeline that semantically extracts wikipedia articles (2023 snapshot), in which LLM agents can use to answer the query. 

## Purpose and Relevance: 
The main purpose of this project is to enable anyone to evaluate different RAG and ProSlm approaches.

## How to use:
1. Make a virtual environment and activate it.
2. Install all libraries as listed in requirements.txt. 
3. You can either evaluate the performance of the pipeline or use it as a chatbot.\n
    \t a. To run as chatbot: `Python3 query_model.py --verbose`
    \t b. To evaluate: `Python3 evaluate_agent.py --limit 3 --output_file_name output_name --dataset dataset_name`


## How this pipeline was evaluated
Custom datasets were developed in order to evaluate how faithful the answers are, answer correctness, context recall, context precision. Several queries have obscured references to test, whether the pipeline remembers what is being discussed.
