import epydeck

# Read from a file with `epydeck.load`
with open("./test_2d/input.deck") as f:
    deck = epydeck.load(f)

#print(deck)

print(deck.keys())
# dict_keys(['control', 'boundaries', 'constant', 'species', 'laser', 'output_global', 'output', 'dist_fn'])

# Modify the deck as a usual python dict:
deck["species"]["Positron"]["charge"] = 1.0

# Write to file
with open("./output.deck", "w") as f:
    epydeck.dump(deck, f)

print(epydeck.dumps(deck))
# ...
# begin:species
#   name = proton
#   charge = 2.0
#   mass = 1836.2
#   fraction = 0.5
#   number_density = if((r gt ri) and (r lt ro), den_cone, 0.0)
#   number_density = if((x gt xi) and (x lt xo) and (r lt ri), den_cone, number_density(proton))
#   number_density = if(x gt xo, 0.0, number_density(proton))
# end:species
# ...