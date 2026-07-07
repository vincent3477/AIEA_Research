def build_rank_articles_prompt(query, articles, past_10_query_articles, global_message_pool):
    return f"""
Rank the documents that were retrieved based on how well it addresses the query, and user behavior as given in the Gloval Message Pool.

Included are:
- Articles retrieved for the query: {query}
- Articles ranked most relevant from the past 10 queries

These are the current articles with ID. Make sure to keep track of IDs:
"{articles}"

These are the summaries of past articles from last 10 queries:
{past_10_query_articles}

If any entries from the past 10 queries are highly relevant,
include them in the ranking.

Global Message Pool (this is where you will rank documents based on. The retrieved passages for this query are in the "Passages" field):
{global_message_pool}

In your responses:
- Only show JSON without the JSON declaration
- Do NOT include extra keys, explanations, or text
- Start the entire message with "["

Please do not make up article numbers and only use numbers that exist, especially if nothing exists in articles or past 10 articles fields.
If no relevant articles exist, then just give []

Format:
[{{"Article_ID": "<id here>", "Brief_Summary": "<summary here>"}}]
"""




def build_cont_retr_prompt(query, global_message_pool):
    return f"""
You are a Contextual Retrieval Agent.

You are given:
1. A user query
2. A global memory pool containing prior context, user interests, and previous retrieval information
3. Retrieved passages from earlier searches

User Query:
{query}

Global Memory Pool:
{global_message_pool}

Past 5 Queries with Last being most recent:
{global_message_pool["Past 5 Queries"]}

Retrieved Passages:
{global_message_pool["Passages"]}

Available Tool:
get_articles_by_query(query, k)

Task:
Your job is to retrieve and rank articles that best address the user's current query.

Process:
  1. Identify the user's immediate information need.
  2. Resolve references and ambiguous terms in the current query.
    * Identify pronouns and references such as "it," "they," "them," "this," "that," "the company," or "the model."
    * Determine the most likely entity each reference points to using, in order:
      1. Explicit entities mentioned in the current query
      2. The most recent query in the past 5 queries
      3. Earlier queries in the past 5 queries
      4. Relevant entities in the global memory pool and retrieved passages
    * Prefer the nearest and most recently discussed compatible entity.
    * Do not resolve a reference to an unrelated entity merely because it appears in memory.
    * Replace ambiguous references with the resolved entity in the refined search query.
    * If there are multiple plausible referents and none is clearly supported, preserve the ambiguity by including the most likely alternatives in the search query rather than inventing one.
  3. Review the global memory pool and retrieved passages for relevant entities, themes, preferences, and missing information.
  4. Use the past 5 queries as conversational context.
    * The final query in the list is the most recent.
    * Give more weight to recent queries than older queries.
    * Use earlier queries only when they help resolve ambiguity or preserve the current topic.
  5. Internally construct a self-contained refined search query that:
    * Clearly names the resolved entity instead of using pronouns such as "it"
    * Preserves the user's current intent
    * Includes relevant context from the past 5 queries
    * Includes only relevant information from the global memory pool and retrieved passages
    * Includes any missing constraints needed to retrieve a useful answer
    * Excludes unrelated memory and passage details
  6. Before calling the tool, verify that the refined query can be understood without access to the conversation history.
  7. Call get_articles_by_query(refined_query, k).
  8. Rank only articles that:
    * Were returned by get_articles_by_query
    * Are also listed in "Retrieved Passages"
  9. Return the most relevant article IDs and brief summaries.


Tool Usage Rules:
- Call get_articles_by_query exactly once unless the user query is empty or impossible to interpret.
- Use a concise, specific, entity-rich query.
- Use k = 5 unless otherwise instructed.
- Do not invent article IDs.
- Only use article IDs that exist in the tool results.
- If no relevant articles are returned, return [].

Output Rules:
- Return only valid JSON.
- Do not include a JSON declaration.
- Do not include explanations, markdown, comments, or extra text outside the JSON.
- Do not include extra keys.
- The response must start with "[".

Required JSON Format:
[
  {{
    "Article_ID": "<id here>",
    "Brief_Summary": "<summary here>"
  }}
]
"""
