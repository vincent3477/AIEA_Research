from multiagents_program import persona_rag
from datasets import load_dataset
from langchain_openai.chat_models import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (answer_relevancy, faithfulness, context_recall, context_precision, answer_correctness)
from ragas import evaluate
from datasets import load_dataset, Dataset
from langchain.schema import SystemMessage, HumanMessage
import openai
import pandas as pd
import argparse
from prompts.eval_prompt import build_eval_prompt

"""
Goal: Evaluate the agent performance against a particular dataset

Usage:
--limit                 : (Optional) Max number of questions pass into the pipeline.
--dataset               : Name of the input dataset. Must be in CSV format with a comma "," separating query and reference answers
--output_file_name      : Name of file to display test results.
--verbose               : (Optional) Display detailed printed output.

"""

def test_agent(verbose, limit, dataset_name, output_file_name):

    dataset_name += ".csv"
    output_file_name += ".csv"

    llm_judge = ChatOpenAI(model="gpt-4o")
    correctness_measurement = []
    agent = persona_rag(verbose)
    qa_set = []
    dataset = pd.read_csv(dataset_name, sep = ",")
    i = 0
 
    for entry in dataset.iterrows():
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
        eval_prompt = build_eval_prompt(query, context_articles, answer, reference)
        messages = [SystemMessage(content = system_instructions), HumanMessage(content = eval_prompt)]

        accuracy_results = llm_judge.invoke(messages).content
        correctness_measurement.append(accuracy_results)
        qa = {"question": query, "contexts": context_articles, "answer": answer, "reference": reference}
        qa_set.append(qa)

        i += 1
        if i >= limit:
            break

    eval_dataset = Dataset.from_list(qa_set)
    result = evaluate(eval_dataset, metrics = [context_precision, context_recall, faithfulness, answer_relevancy])
    results_df = result.to_pandas()
    results_df.to_csv(output_file_name, index = False)

    

def main():
    set_verbose = False
    set_limit = 3
    dataset_name = None
    output_file_name = None

    parser = argparse.ArgumentParser(description="To test PersonaRag against a dataset of questions.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed logging")

    parser.add_argument("-l", "--limit", type=int, help="Max number of questions")

    parser.add_argument("-d", "--dataset", type=str, help="A CSV file that will be fed to the pipeline")

    parser.add_argument("-o", "--output_file_name", type=str, help="Output file name")

    args = parser.parse_args()

    if args.verbose:
        set_verbose = True

    if args.limit:
        set_limit = args.limit

    if args.dataset:
        dataset_name = args.dataset
    else:
        raise ValueError("A dataset name is required in order to test the pipeline")
    
    if args.output_file_name:
        output_file_name = args.output_file_name
    else:
        raise ValueError("An output file name to display results is required in order to test the pipeline")

    test_agent(set_verbose, set_limit, dataset_name, output_file_name)

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