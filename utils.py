


def get_lexicon():
    # Load words
    words = []
    with open("words.lex", "rt") as lex_file:
        for line in lex_file.readlines():
            word = line.split("\t")[-1].lower().rstrip()
            words.append(word)
    print("Lexicon loaded")
    return words


def get_channels():
    # Load channels
    channels = {}
    with open("docs.list", "rt") as doc_file:
        for i, line in enumerate(doc_file.readlines()):
            channel = line.split("\t")[-1].split("_")[0]
            channels[i] = channel
    print("Channels loaded")
    return channels


