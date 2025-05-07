from flask import Flask, render_template, request, jsonify
import joblib
import torch
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

# Load the trained sentiment model
clf = joblib.load("sentiment_model.pkl")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./saved_tokenizer")  # Path to your local tokenizer



# Load the sentence embedding model
embedding_model = AutoModel.from_pretrained("distilbert-base-uncased")
embedding_model.load_state_dict(torch.load("embedding_model.pth"))
embedding_model.eval()

# Function to encode text into embeddings
def encode_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Extract CLS token embedding

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["text"]
        embedding = encode_text(user_input)
        prediction = clf.predict(embedding)[0]
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        return jsonify({"sentiment": sentiment_map[prediction]})
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
