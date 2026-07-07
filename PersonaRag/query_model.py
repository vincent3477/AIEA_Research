from multiagents_program import persona_rag
import argparse


"""
Goal: Interact with PersonaRag like a chatbot

Input: String
Output: String
"""

def main():
    set_verbose = False

    parser = argparse.ArgumentParser(description="To test PersonaRag against a dataset of questions.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed logging")

    args = parser.parse_args()

    if args.verbose:
        set_verbose = True
    
    agent = persona_rag(set_verbose)

    while(1):
        query = input("Type your question here: \n")
        answer, _ = agent.ask_question(query)
        print(answer)


if __name__ == "__main__":
    main()