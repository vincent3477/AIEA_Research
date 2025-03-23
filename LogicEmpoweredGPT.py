from openai import OpenAI
import janus_swi as janus
client = OpenAI()

input_prompt = input("Type in your problem here: ")


completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that can help with anything."},
        #{"role": "user", "content": input_prompt},
        {"role": "user", "content": f"Translate the following input into Prolog code. The last line should be in the form of 'rule(object)' and nothing else. Ensure the proofs are thorough going through step by step before arriving to the conclusion. Give code only. Here is the following input to be translated: {input_prompt}"}
    ],
    logprobs=True
)


pl_output = completion.choices[0].message.content

fileName = "generated_pl_code7.pl"

with open(fileName, "a") as file:
    file.write(pl_output[9:len(pl_output)-3]) #strip the headers.

janus.consult(fileName)

last_line = ""
with open(fileName, "r") as file:
    lines = file.readlines()
    if lines != "":
        last_line = lines[-1].rstrip('\n')

print(last_line)
 
result = janus.query_once(last_line)
print(result)

