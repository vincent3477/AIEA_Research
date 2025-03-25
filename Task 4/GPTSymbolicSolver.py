from openai import OpenAI
import janus_swi as janus
client = OpenAI()

#Ask user for input problem and where to save the generated .pl file.
input_prompt = input("Type in your problem here: ")
file_prompt  = input("Where do you want to save the file?")
fileName = file_prompt + ".pl"

#Create a GPT completion instance with the OpenAI API, where GPT must translate the input_prompt to prolog code.
#The last line of the generated code will the conclusion that is to be drawn.
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that can help with anything."},
        #{"role": "user", "content": input_prompt},
        {"role": "user", "content": f"Translate the following input into Prolog code. The last line should be in the form of 'rule(object)' and nothing else. Ensure the proofs are thorough going through step by step before arriving to the conclusion. Give code only. Here is the following input to be translated: {input_prompt}"}
    ],
    logprobs=True
)

#store the output.
pl_output = completion.choices[0].message.content

#write all of the lines (except the headers generated) into the said file.
with open(fileName, "a") as file:
    file.write(pl_output[9:len(pl_output)-3]) #strip the headers.

#Use janus to connsult the filename ensuring the .pl file generated is valid.
janus.consult(fileName)

#Get the last line of the file, which will be the conclusion made.
last_line = ""
with open(fileName, "r") as file:
    lines = file.readlines()
    if lines != "":
        last_line = lines[-1].rstrip('\n')

print(last_line)

#query that line (the conclusion), and return whether or not it's valid (true or false). 
result = janus.query_once(last_line)
print(result)

