from multiagents_program import persona_rag
from datasets import load_dataset
from langchain_openai.chat_models import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (answer_relevancy, faithfulness, context_recall, context_precision, LLMContextPrecisionWithoutReference)
from ragas import evaluate
from datasets import load_dataset, Dataset


def test_agent(n = 20):

    agent = persona_rag()


    
    dataset = load_dataset("web_questions")

    qa_set = []

    limit = 24
    i = 0


    

    llm_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    evaluator_llm = LangchainLLMWrapper(llm_model)  

    print(len(dataset["test"]))

    context_precision = LLMContextPrecisionWithoutReference(llm = evaluator_llm)
    
    for entry in dataset["test"]:
        query = entry["question"]
        answers = entry["answers"]
        answer, context_articles = agent.ask_question(query)
        #question -> string
        #contexts -> list of strings
        #answer -> string
        #reference -> string
        qa = {"question": query, "contexts": context_articles, "answer": answer, "reference": ";".join(answers)}

        for a in range(len(context_articles)):
            nt = "".join(c for c in context_articles[a] if c.isprintable())
            context_articles[a] = nt

        qa_set.append(qa)

        i += 1
        print("Progress i", i)
        if i >= limit:
            break

    eval_dataset = Dataset.from_list(qa_set)

    result = evaluate(eval_dataset, metrics = [context_precision, context_recall, faithfulness, answer_relevancy])

    

    results_df = result.to_pandas()

    results_df.to_csv('results.csv', index = False)
    print(results_df.head())

    results_df
    

def main():
    test_agent()


if __name__ == "__main__":
    main()
