
wet(rain).
wet(snow).

% Nice weather equivalence 
nice_weather(nice).
nice_weather(clear).

% Temperature classification rules
temp_category(Temp, very_cold) :- Temp < 20.
temp_category(Temp, cold) :- Temp >= 20, Temp =< 40.
temp_category(Temp, cool) :- Temp > 40, Temp =< 60.
temp_category(Temp, warm) :- Temp > 60, Temp =< 80.
temp_category(Temp, hot) :- Temp > 80.

% Shoes selection rules
shoes(_, formal, dressshoes).
% Boots for wet weather with casual/semiformal appointments
shoes(Weather, Appointment, boots) :-
    wet(Weather),
    (Appointment = casual ; Appointment = semiformal).
% Sneakers for nice weather with casual/semiformal appointments
shoes(Weather, Appointment, sneakers) :-
    nice_weather(Weather),
    (Appointment = casual ; Appointment = semiformal).

% Tie selection rules
tie(formal, 'four-in-hand').
tie(casual, 'no-tie').
tie(semiformal, 'no-tie').

% Pants selection rules
pants(Temp, shorts) :- Temp > 80.
pants(Temp, insulatedpants) :- Temp < 20.
pants(Temp, jeans) :- Temp >= 20, Temp =< 40.
pants(Temp, jeans) :- Temp > 40, Temp =< 80.
pants(Temp, khakis) :- Temp > 40, Temp =< 80.

% Outerwear selection rules

outerwear(_, Temp, parka) :- Temp < 20.
outerwear(_, Temp, coat) :- Temp >= 20, Temp =< 40.
outerwear(_, Temp, jacket) :- Temp > 40, Temp =< 80.
outerwear(_, Temp, none) :- Temp > 80.
% Weather-based outerwear 
outerwear(Weather, _, raincoat) :- wet(Weather).

% Shirt selection rules 
% Formal always allows dress shirt
shirt(_, _, formal, dressshirt).
% Very cold always allows flannel shirt (regardless of formality)
shirt(_, Temp, _, flannelshirt) :- Temp < 20.
% Casual cool/cold (20-60) allows flannel or sweatshirt
shirt(_, Temp, casual, flannelshirt) :- Temp >= 20, Temp =< 60.
shirt(_, Temp, casual, sweatshirt) :- Temp >= 20, Temp =< 60.
% Casual not-cold (>= 40) allows t-shirt
shirt(_, Temp, casual, 't-shirt') :- Temp >= 40.
% Semiformal always allows dress shirt (note: dresshirt not dressshirt)
shirt(_, _, semiformal, dresshirt).
% Semiformal warm/cool (40-80) allows polo
shirt(_, Temp, semiformal, polo) :- Temp > 40, Temp =< 80.

% Main outfit predicate
% Uses findall to collect all valid options for each category
% list_to_set removes duplicates while preserving order
outfit(Weather, Temp, Appointment, [Shirts, Ties, Pants, Shoes, Outerwear]) :-
    findall(S, shirt(Weather, Temp, Appointment, S), ShirtList),
    list_to_set(ShirtList, Shirts),
    findall(T, tie(Appointment, T), TieList),
    list_to_set(TieList, Ties),
    findall(P, pants(Temp, P), PantList),
    list_to_set(PantList, Pants),
    findall(Sh, shoes(Weather, Appointment, Sh), ShoeList),
    list_to_set(ShoeList, Shoes),
    findall(O, outerwear(Weather, Temp, O), OuterList),
    list_to_set(OuterList, Outerwear).

% Test cases
go :-
    nl, write("F1: "), outfit(snow, 15, casual, F1), write(F1),
    nl, write("F2: "), outfit(snow, 15, formal, F2), write(F2),
    nl, write("F3: "), outfit(snow, 30, casual, F3), write(F3),
    nl, write("F4: "), outfit(snow, 30, semiformal, F4), write(F4),
    nl, write("F5: "), outfit(rain, 35, casual, F5), write(F5),
    nl, write("F6: "), outfit(clear, 50, casual, F6), write(F6),
    nl, write("F7: "), outfit(rain, 50, semiformal, F7), write(F7),
    nl, write("F8: "), outfit(clear, 50, formal, F8), write(F8),
    nl, write("F9: "), outfit(nice, 70, casual, F9), write(F9),
    nl, write("Fa: "), outfit(clear, 85, semiformal, Fa), write(Fa),
    nl, write("Fb: "), outfit(rain, 98, casual, Fb), write(Fb),
    nl, write("Fc: "), outfit(clear, 98, formal, Fc), write(Fc).
