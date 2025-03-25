from openai import OpenAI
import janus_swi as janus
client = OpenAI()

#Ask user for input problem and where to save the generated .pl file.
input_prompt = input("Type in your problem here: ")
fileName = input("Where do you want to save the file?")
fileName = fileName + ".pl"

#Create a GPT completion instance with the OpenAI API, where GPT must translate the input_prompt to prolog code.
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that can help with anything."},
        #{"role": "user", "content": input_prompt},
        {"role": "user", "content": f"Translate the following input into Prolog code. Give code only. Here is the following input to be translated: {input_prompt}"}
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

print("Program finished.")
