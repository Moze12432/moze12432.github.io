from flask import Flask, request, jsonify
from model import generate_reply

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.json["message"]

    reply = generate_reply(user_message)

    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(port=5000)
