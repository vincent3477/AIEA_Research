def build_eval_prompt(
    query: str,
    context_articles: str,
    answer: str,
    reference: str,
) -> str:
    return f"""
You are evaluating whether a model's answer is correct.

Question:
{query}

Retrieved Articles:
{context_articles}

Model Answer:
{answer}

Reference Answer(s):
{reference}

Evaluation Instructions:
1. Determine whether the model answer correctly answers the question based on:
   - The reference answer(s)
   - The information available in the retrieved articles

2. Multiple reference answers are separated by semicolons (";").

3. When multiple reference answers are provided, deduct points only when the
   question requires the model to include multiple answers and the model omits
   one or more required answers.

4. Do not penalize the model for omitting reference answers that are not
   necessary to satisfy the question.

5. Return a score between 0.0 and 1.0:
   - 1.0 means fully correct
   - 0.0 means fully incorrect

6. Output only valid JSON in exactly this structure:
{{
  "Evaluation": "<brief explanation>",
  "Score": 0.89
}}
""".strip()