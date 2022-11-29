remove_words = [word.strip("\n") for word in open("remove_words.txt", "r")]

print(remove_words)
print(type(remove_words))
print(remove_words[1])