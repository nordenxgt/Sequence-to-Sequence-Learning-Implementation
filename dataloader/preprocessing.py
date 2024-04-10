import spacy


# python -m spacy download en_core_news_sm
# python -m spacy download fr_core_news_sm

en_nlp = spacy.load("en_core_web_sm")
fr_nlp = spacy.load("fr_core_news_sm")

sos_token = "<sos>"
eos_token = "<eos>"
max_length = 1_000

def tokenize_ds(data, en_nlp, fr_nlp, max_length, sos_token, eos_token):
    return {
        "en_tokens": [sos_token] + [token.text.lower() for token in en_nlp.tokenizer(data["en"])][:max_length] + [eos_token], 
        "fr_tokens": [sos_token] + [token.text.lower() for token in fr_nlp.tokenizer(data["fr"])][:max_length] + [eos_token]
    }

fn_kwargs = {
    "en_nlp": en_nlp,
    "fr_nlp": fr_nlp,
    "max_length": max_length,
    "sos_token": sos_token,
    "eos_token": eos_token
}

train_data = train_data.map(tokenize_ds, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(tokenize_ds, fn_kwargs=fn_kwargs)
test_data = test_data.map(tokenize_ds, fn_kwargs=fn_kwargs)