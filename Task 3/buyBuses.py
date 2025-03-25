import janus_swi as janus

# Load Prolog grammar file
janus.consult("busKnowledgeBase.pl")

# Input sentence (dynamically from user or script)
sentence = input("Low Emissions or Zero Emissions?") 
if sentence.lower() == "low emissions":
    prolog_query = "print_zero_emissions_options(StringList)."
elif sentence.lower() == "zero emissions":
    prolog_query = "print_low_emissions_options(StringList)."
else:
    raise Exception("You did not choose one of the two choices.")

# Construct Prolog query dynamically

# Send query to Prolog and retrieve parse tree
try:
    result = next(janus.query(prolog_query))
    print(result)
    

except StopIteration:
    print("Invalid sentence structure!")
