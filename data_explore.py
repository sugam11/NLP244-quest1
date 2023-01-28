import os
path = "./data/wikitext-2"

word_fre = {}
with open(os.path.join(path, "wiki.train.tokens"), "r", encoding="utf8") as f:
    for line in f:
        words = line.split()
        for word in words:
            if word not in word_fre:
                word_fre[word] = 0
            word_fre[word] += 1

print(
    f" Frequency of <unk> token is: {word_fre['<unk>']}\n"
    f"Total word frequency is: {sum(word_fre.values())}\n"
    f"%age of <unk> token is: {word_fre['<unk>']/sum(word_fre.values())*100}\n"
)



