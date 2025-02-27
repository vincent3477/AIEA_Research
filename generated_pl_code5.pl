
% Facts
made_by(xe40, new_flyer).
made_by(xhe40, new_flyer).
made_by(xde40, new_flyer).
made_by(low_floor_plus_CNG, gillig).
made_by(low_floor_plus_hybrid, gillig).
made_by(low_floor_plus_diesel, gillig).
made_by(low_floor_plus_electric, gillig).

% Rules
battery_electric(Bus) :- Bus = xe40.
battery_electric(Bus) :- Bus = low_floor_plus_electric.

% Query
% Is xd40 battery electric?
% ?- battery_electric(xd40).
