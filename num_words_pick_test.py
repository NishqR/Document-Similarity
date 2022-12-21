import random

num_words_rel_irrel = [(60,50), (55,45), (70,55), (65,50), (65,55),(55,50)]

config_picked = num_words_rel_irrel[random.randint(0, len(num_words_rel_irrel) - 1)]

num_words_relevant = config_picked[0]
num_words_irrelevant = config_picked[1]

print(num_words_relevant)
print(num_words_irrelevant)



'''
for i in range(100000):
	num_generated = random.randint(0, len(num_words_rel_irrel) - 1) 
	if num_generated >= (len(num_words_rel_irrel) - 1):
		print("err0r")

'''