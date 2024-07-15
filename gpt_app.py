from flask import Flask, request, jsonify, render_template
import torch
from reversing_model import ReverseGPT
from config import ReverseGPTConfig
import os


app = Flask(__name__)
port = int(os.environ.get("PORT", 8000))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_input = data['initial_string']

    if len(user_input) > 256:
        return jsonify({'error': 'Input exceeds the maximum limit of 256 characters.'}), 400
    # neither w nor k are in the vocabulary so remove them if used
    char_in_vocab = [char in config.vocabulary for char in user_input]
    user_input = "".join([char for char, in_vocab in zip(user_input, char_in_vocab) if in_vocab])
    if len(user_input) == 0:
        user_input = "\n"
    context = torch.tensor(config.str2int(user_input), device=config.device).view(1, -1)
    response = model.generate(context, device=config.device, max_new_tokens=10+1,
                              min_new_tokens=2, context_size=config.context_size,
                              idx_to_char=config.int_to_str)
    response = response.replace("\n", "<br>")
    return jsonify({'response': response + (" (Correct: {})".format(''.join(sorted(user_input[1:]))))})


if __name__ == '__main__':
    # Load the configuration
    config = ReverseGPTConfig()
    config.device = 'cpu'
    config.seq_len = 10
    config.context_size = 2*config.seq_len + 1
    alphabet = "abcdefghijklmnopqrstuvwxyz:."
    vocabulary = sorted(list(set(alphabet)))
    vocab_size = len(vocabulary)
    config.vocabulary = vocabulary
    config.vocabulary_size = vocab_size

    # Options for weights
    path = "models/next_token_200_10000_new.pth"
    # Load model
    model = ReverseGPT(config)
    model.load_state_dict(torch.load(path, map_location=config.device))
    model.eval()  # Set the model to evaluation mode, not training

    # Launch the app
    app.run(host='0.0.0.0', port=port, debug=False)