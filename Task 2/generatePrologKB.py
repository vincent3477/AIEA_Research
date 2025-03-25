from openai import OpenAI
import janus_swi as janus
client = OpenAI()

input_prompt = input("Type in your problem here: ")
fileName = input("Where do you want to save the file?")
fileName = fileName + ".pl"


completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that can help with anything."},
        #{"role": "user", "content": input_prompt},
        {"role": "user", "content": f"Translate the following input into Prolog code. Give code only. Here is the following input to be translated: {input_prompt}"}
    ],
    logprobs=True
)


pl_output = completion.choices[0].message.content

with open(fileName, "a") as file:
    file.write(pl_output[9:len(pl_output)-3]) #strip the headers.

janus.consult(fileName)

print("If you DO NOT see any errors, that means the prolog file was able to compile successfully. Program finished.")
