# physformer/tokenizer.py
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {ch:i for i,ch in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        self.inv_vocab = {i:ch for i,ch in self.vocab.items()}

    def encode(self, text, max_len=16):
        tokens = [self.vocab.get(ch, 0) for ch in text.lower()]
        if len(tokens) < max_len:
            tokens += [0]*(max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return tokens

    def decode(self, tokens):
        return "".join([self.inv_vocab.get(t, "?") for t in tokens])
