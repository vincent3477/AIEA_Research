% a knowledge base about what type of bus you should buy


% the two main bus manufacturers
new_flyer_made(xn40).
new_flyer_made(xde40).
new_flyer_made(xe40).
new_flyer_made(xhe40).
new_flyer_made(xd40).

gillig_made(low_Floor_PLUS_CNG).
gillig_made(low_Floor_PLUS_Electric).
gillig_made(low_Floor_PLUS_Hybrid).
gillig_made(low_Floor_PLUS_Diesel).


% the models that are made by each manufacturer, assume everything is a 40 foot bus.
cng_fueled(xn40).
cng_fueled(low_Floor_PLUS_CNG).

hybrid_fueled(xde40).
hybrid_fueled(low_Floor_PLUS_Hybrid).

battery_powered(xe40).
battery_powered(low_Floor_PLUS_Electric).

hydrogen_fueled(xhe40).

diesel_fueled(xd40).
diesel_fueled(low_Floor_PLUS_Diesel).


isLowEmissions(X) :- battery_powered(X) | cng_fueled(X) | hybrid_fueled(X).
isZeroEmissions(X) :- battery_powered(X) | hydrogen_fueled(X).


print_zero_emissions_options(StringList) :-
    findall(RelationString, (
        isZeroEmissions(Vehicle),
        term_string(isZeroEmissions(Vehicle), RelationString)
    ), StringList).

print_low_emissions_options(StringList) :-
    findall(RelationString, (
        isLowEmissions(Vehicle),
        term_string(isLowEmissions(Vehicle), RelationString)
    ), StringList).










