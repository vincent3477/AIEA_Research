def build_cot_prompt(query, retrieved_ranked_articles):
    return f"""
Question:
{query}

Passages:
{retrieved_ranked_articles}

1. Read the given question and passages to gather relevant information.
2. Write reading notes summarizing the key points from these passages.
3. Discuss the relevance of the given question and passages.
4. If some passages are relevant, provide a brief answer based on them.
5. If and only if no passage is relevant at all, state:
   "no answer found"
"""


def build_cognitive_agent_prompt(query, cot_answer, global_message_pool):
    return f"""
Verify the reasoning process in the initial response:
"{cot_answer}"

Check for errors or misalignments to the query:
"{query}"

If the reasoning indicates no response, simply say:
"No Response"

Use insights from user interaction analysis:
"{str(global_message_pool)}"

Refine the response by:
- correcting inaccuracies
- enhancing answers based on the user profile
- aligning with immediate user needs
- incorporating foundational or advanced knowledge

Be sure to:
- mirror the user's tone based on the query
- NOT restate instructions
"""