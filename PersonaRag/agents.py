from openai import OpenAI
client = OpenAI()

query = input("type in your query here")

# scrape for documents here

memory = {}

user_profile = client.chat.completions.create(
    model = "gpt-4o-muni",
    messages=[
        {"role": "system", "content": f"From the memory pool get a gist of what his personality may be like including previous queries, main interests, and immediate needs that are most common." },
        {"role": "user", "content": f"Based on this knowledge base, describe this person's main interests filter his main interest based on the queries made. This is the person's knowledge base {memory}"}
    ]
)

