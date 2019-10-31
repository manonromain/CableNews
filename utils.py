


def get_lexicon():
    # Load words
    words = []
    with open("words.lex", "rt") as lex_file:
        for line in lex_file.readlines():
            word = line.split("\t")[-1].lower()
            words.append(word)
    print("Lexicon loaded")
    return words
