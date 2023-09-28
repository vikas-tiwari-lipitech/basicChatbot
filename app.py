# server/app.py
from flask import Flask, request, jsonify
from chatbot.chatbot_script import evaluate

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')
    if user_input.lower() == 'quit':
        response = "Goodbye!"
    else:
        response = evaluate(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
