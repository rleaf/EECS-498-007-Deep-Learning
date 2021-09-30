toad = {
   "toad1": "tom",
   "toad2": "jerry",
   'toad3': 'ra',
   'toad4': 4,
   4: 5
}

print(toad['toad1'])
print(toad[4])

toad.setdefault('toad5', []).append([1, 2, 3])
print(toad)


tom = [2, 5, 6, 8]

for k in tom:
   print(k)
