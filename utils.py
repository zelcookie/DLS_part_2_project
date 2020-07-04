
def flatten(l):
    return [item for sublist in l for item in sublist]

def remove_tech_tokens(mystr, tokens_to_remove=['<eos>', '<sos>', '<unk>', '<pad>']):
    return [x for x in mystr if x not in tokens_to_remove]


def get_text(x, TRG_vocab):
    tokens = []
    for ind in x:
        if TRG_vocab.itos[ind] == '<eos>':
            break
        tokens.append(TRG_vocab.itos[ind])
    text = remove_tech_tokens(tokens)
    if len(text) < 1:
        text = []
    return text


def generate_translation(src, trg, model, TRG_vocab):
    model.eval()

    output = model(src, trg, 0)[1:] #turn off teacher forcing
    output = output.argmax(dim=-1).cpu().numpy()

    original = [TRG_vocab.itos[x] for x in list(trg[:,0].cpu().numpy())]
    generated = []
    for ind in list(output[:, 0]):
        if TRG_vocab.itos[ind] == '<eos>':
            break
        generated.append(TRG_vocab.itos[ind])
    original = remove_tech_tokens(original)
    generated = remove_tech_tokens(generated)
    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()
