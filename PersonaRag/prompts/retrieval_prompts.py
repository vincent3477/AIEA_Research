def build_rank_articles_prompt(query, articles, past_10_query_articles, global_message_pool):
    return f"""
Rank the documents that were retrieved.

Included are:
- Articles retrieved for the query: {query}
- Articles ranked most relevant from the past 10 queries

These are the current articles with ID. Make sure to keep track of IDs:
"{articles}"

These are the summaries of past articles from last 10 queries:
{past_10_query_articles}

If any entries from the past 10 queries are highly relevant,
include them in the ranking.

Rank the documents based on:
{global_message_pool}

In your responses:
- Only show JSON without the JSON declaration
- Do NOT include extra keys, explanations, or text
- Start the entire message with "["

Format:
[{{"Article_ID": "<id here>", "Brief_Summary": "<summary here>"}}]
"""