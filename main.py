import os

import torch
from flask import Flask, request, jsonify
from torchtext.data.utils import get_tokenizer, ngrams_iterator

from classifier.model import TextSentiment

VOCAB_SIZE = 4303416
EMBED_DIM = 32
NUM_CLASS = 2
ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, 'model')


LABEL = {
    0: "real",
    1: "fake",
}

app = Flask(__name__)

device = torch.device('cpu')
weights_file = os.path.join(MODEL_PATH, 'model.pth')
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)
model.load_state_dict(torch.load(weights_file, map_location=device))
model.eval()


def predict(text, model, vocab, n_grams):
    tokenizer = get_tokenizer("basic_english")
    text = torch.tensor([vocab[token]
                         for token in ngrams_iterator(tokenizer(text), n_grams)])
    output = model(text, torch.tensor([0]))
    print(output)
    return output.argmax(1).item()


@app.route('/get_result', methods=['GET', 'POST'])
def get_result():
    vocab = torch.load(os.path.join(MODEL_PATH, 'vocab.pth'))
    content = request.get_json()
    key = content['id']
    text = content['text']
    print(text, type(text))
    pred = predict(text, model, vocab, 2)

    return jsonify({
        'id': key,
        'label': LABEL[pred],
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
