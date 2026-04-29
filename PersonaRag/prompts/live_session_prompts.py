def build_live_session_suggestions_prompt(global_message_pool, articles):
    return f"""
Using the context given, suggest queries or adjust search results
from passages that have relevance to the query and historical
user interactions:

{str(global_message_pool)}

If no passages are found to be relevant at all, just say:
"no suggestions".

These are the contexts:

"{articles}"
"""


def build_update_user_from_live_session_prompt(global_message_pool, live_session_suggestions):
    return f"""
From the live session agent, integrate new insights of the scope
of the query that include suggested topics.

This is the current memory:
"{str(global_message_pool)}"

These are the suggestions from the live session agent:
"{live_session_suggestions}"
"""