from transformers import pipeline
from datasets import load_dataset

qa_pipeline = pipeline("question_answering", model = "distilbert-base-based-distilled-squad")

triviaqa = load_dataset("trivia_qa", "rc.nocontext", split = "validation[:100]")

def test_agent(agent, dataset, n = 20):
    for i in range(n):
        q = dataset[i]["question"]
        gold = dataset[i]["aliases"]

    
"""



    def test_agent(n = 20):

        qa_pipeline = pipeline("document-question-answering", model = "distilbert-base-cased-distilled-squad")

        dataset = load_dataset("trivia_qa", "rc.nocontext", split = "validation[:100]")
        for i in range(n):
            q = dataset[i]["question"]
            gold = dataset[i]["aliases"]
            correct = 0
            pred = run_agent(q)

            for alias in gold:
                if alias.lower() in pred.lower():
                    correct += 1


        print(f"Q: {q}")
        print(f"Pred: {pred}")
        print(f"Gold: {gold}")
        print("---")

        print(f"Accuracy: {correct}/{n}")
"""