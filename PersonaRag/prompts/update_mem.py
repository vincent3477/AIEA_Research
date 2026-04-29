def build_update_global_memory_from_live_prompt(
    global_message_pool,
    update_user_suggested_topics,
    live_session_suggestions,
    gmp_analyst_instructions,
):
    return f"""
Update the current global memory:
"{global_message_pool}"

with profile-based contextual suggestions:
"{update_user_suggested_topics}"

while integrating insights from the live session agent:
"{live_session_suggestions}"

If possible, build on the information that is already supplied,
otherwise do not lose the supplied info.

To ensure a consistent global message pool, follow the
instructions here:
{gmp_analyst_instructions}
"""


def build_update_user_profile_prompt(global_message_pool):
    return f"""
Update the user profile based on this past query:
"{str(global_message_pool)}"
"""




def build_update_global_memory_from_profile_prompt(
    global_message_pool,
    profile_agent_output,
    gmp_analyst_instructions,
):
    return f"""
Update the global memory:
"{str(global_message_pool)}"

with new information from the user profile agent:
{profile_agent_output}

To ensure a consistent global message pool, follow:
{gmp_analyst_instructions}
"""


def build_final_gmp_update_prompt(analyzation):
    return f"""
This is the updated global memory pool after the query has been
completed and user interests have been updated.

With suggestions from the global message pool analyst, reformat
it as needed to maintain structural coherence while preserving
information about the user's interests.

Suggestions and analysis:
{analyzation}
"""