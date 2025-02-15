from openai import OpenAI
import janus_swi as janus
client = OpenAI()

input_prompt = input("Type in your prompt here: ")

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that can answer in about anything."},
        {"role": "user", "content": input_prompt}
    ],
    logprobs=True
)


nl_output = completion.choices[0].message.content
print(f"Here is the following response from gpt-40-mini:\n {nl_output}")
print("\n\n\n---------------------------------------------------------------------\n\n\n")

prologCompletionMessage = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Your job is to translate into code."},
        {"role": "user", "content": f"Translate the following input into Prolog code. Give code only. Here is the following input to be translated: {nl_output}" }
    ],
    logprobs=True
)

pl_output = prologCompletionMessage.choices[0].message.content

print(f"Here is the following CODE response from gpt-40-mini:\n {pl_output}")
print("\n\n\n---------------------------------------------------------------------\n\n\n")
print(f"Checking if the generated prolog code will run....")

fileName = "generated_pl_code4.pl"

with open(fileName, "a") as file:
    file.write(pl_output[9:len(pl_output)-3]) #strip the headers.

janus.consult(fileName)

print("If you DO NOT see any errors, that means the prolog file was able to compile successfully. Program finished.")









