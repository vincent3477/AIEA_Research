from multiagents_program import persona_rag
from datasets import load_dataset
from langchain_openai.chat_models import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (answer_relevancy, faithfulness, context_recall, context_precision, answer_correctness)
from ragas import evaluate
from datasets import load_dataset, Dataset
from langchain.schema import SystemMessage, HumanMessage

## a small program for testing the personarag pipeline.

def test_agent():


    llm_judge = ChatOpenAI(model="gpt-4o")

    correctness_measurement = []

    agent = persona_rag()
    
    dataset = load_dataset("web_questions")

    qa_set = []

    limit = 11
    i = 0


    


    print(len(dataset["test"]))

    
    for entry in dataset["test"]:
        query = entry["question"]
        reference = entry["answers"]

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

    results_df.to_csv('results4.csv', index = False)

    print(results_df.head())

    results_df
    

def main():
    test_agent()


if __name__ == "__main__":
    main()
