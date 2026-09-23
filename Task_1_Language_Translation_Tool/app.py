from flask import Flask, request, render_template_string
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


# Download required NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# FAQ Dataset
FAQS = [
    {
        "question": "What is Python?",
        "answer": "Python is a popular programming language used for Artificial Intelligence, Machine Learning, web development, data science, and automation."
    },
    {
        "question": "What is Artificial Intelligence?",
        "answer": "Artificial Intelligence (AI) is technology that enables computers to perform tasks that normally require human intelligence."
    },
    {
        "question": "What is Machine Learning?",
        "answer": "Machine Learning is a branch of Artificial Intelligence that allows computers to learn patterns from data and make predictions or decisions."
    },
    {
        "question": "What is CodeAlpha?",
        "answer": "CodeAlpha provides internship opportunities where students can gain practical experience by working on real-world projects."
    },
    {
        "question": "What is this internship?",
        "answer": "This is an Artificial Intelligence internship where students can develop practical AI projects and improve their technical skills."
    },
    {
        "question": "What is Flask?",
        "answer": "Flask is a lightweight Python web framework used to build web applications and APIs."
    },
    {
        "question": "What is HTML?",
        "answer": "HTML stands for HyperText Markup Language. It is used to create the structure of web pages."
    },
    {
        "question": "What is CSS?",
        "answer": "CSS stands for Cascading Style Sheets. It is used to design and style web pages."
    },
    {
        "question": "What is GitHub?",
        "answer": "GitHub is a platform used to store, manage, and share source code using Git."
    },
    {
        "question": "What is NLP?",
        "answer": "NLP stands for Natural Language Processing. It is a field of Artificial Intelligence that helps computers understand and process human language."
    },
    {
        "question": "What is NLTK?",
        "answer": "NLTK stands for Natural Language Toolkit. It is a Python library used for Natural Language Processing and text analysis."
    },
    {
        "question": "What is an AI chatbot?",
        "answer": "An AI chatbot is a software application that communicates with users and provides responses using Artificial Intelligence or Natural Language Processing."
    },
    {
        "question": "How can I submit my internship task?",
        "answer": "You can submit your completed internship tasks according to the submission instructions provided by CodeAlpha."
    },
    {
        "question": "Thank you",
        "answer": "You're welcome! 😊"
    },
    {
        "question": "Goodbye",
        "answer": "Goodbye! 👋 All the best for your internship!"
    }
]


# -----------------------------
# NLP PREPROCESSING
# -----------------------------

def preprocess_text(text):

    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# Prepare FAQ questions
faq_questions = [
    preprocess_text(item["question"])
    for item in FAQS
]


# -----------------------------
# TF-IDF MODEL
# -----------------------------

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(faq_questions)


# -----------------------------
# FIND BEST ANSWER
# -----------------------------

def get_answer(user_question):

    processed_question = preprocess_text(user_question)

    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarities = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match_index = similarities.argmax()

    best_score = similarities[0][best_match_index]

    print("Question:", user_question)
    print("Best similarity:", best_score)

    if best_score < 0.15:

        return (
            "Sorry 😅 I couldn't find a suitable answer. "
            "Try asking about Python, AI, Machine Learning, "
            "CodeAlpha, Flask, HTML, CSS, GitHub, NLP, or NLTK."
        )

    return FAQS[best_match_index]["answer"]


# -----------------------------
# HTML INTERFACE
# -----------------------------

HTML = """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>AI FAQ Chatbot</title>


<style>

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    font-family: Arial, sans-serif;

    background: linear-gradient(
        135deg,
        #667eea,
        #764ba2
    );

    min-height: 100vh;

    display: flex;

    justify-content: center;

    align-items: center;

    padding: 20px;
}


.chat-container {

    width: 100%;

    max-width: 700px;

    background: white;

    border-radius: 20px;

    overflow: hidden;

    box-shadow:
        0 15px 40px rgba(0,0,0,0.25);
}


.header {

    background: #667eea;

    color: white;

    padding: 25px;

    text-align: center;
}


.header h1 {

    margin: 0;

    font-size: 28px;
}


.header p {

    margin: 8px 0 0;

    opacity: 0.9;
}


.chat-area {

    padding: 25px;

    min-height: 350px;

    max-height: 500px;

    overflow-y: auto;
}


.bot-message {

    background: #f1f3f8;

    padding: 15px;

    border-radius: 15px;

    margin-bottom: 15px;

    color: #333;

    line-height: 1.6;
}


.user-message {

    background: #667eea;

    color: white;

    padding: 15px;

    border-radius: 15px;

    margin-bottom: 15px;

    text-align: right;

    line-height: 1.6;
}


.input-area {

    padding: 20px;

    border-top: 1px solid #ddd;

    display: flex;

    gap: 10px;
}


input {

    flex: 1;

    padding: 14px;

    border: 2px solid #ddd;

    border-radius: 10px;

    font-size: 16px;

    outline: none;
}


input:focus {

    border-color: #667eea;
}


.send-button {

    border: none;

    background: #667eea;

    color: white;

    padding: 14px 20px;

    border-radius: 10px;

    cursor: pointer;

    font-size: 15px;
}


.send-button:hover {

    background: #5568d8;
}


.suggestions {

    padding: 0 20px 15px;

    text-align: center;
}


.suggestions p {

    color: #777;

    font-size: 13px;
}


.suggestion {

    border: none;

    background: #f1f3f8;

    color: #444;

    padding: 8px 12px;

    margin: 4px;

    border-radius: 20px;

    cursor: pointer;

}


.suggestion:hover {

    background: #667eea;

    color: white;
}


.footer {

    text-align: center;

    padding: 15px;

    color: #888;

    font-size: 13px;
}


@media (max-width: 600px) {

    .input-area {

        flex-direction: column;
    }

    .send-button {

        width: 100%;
    }

}

</style>

</head>


<body>


<div class="chat-container">


<div class="header">

<h1>🤖 AI FAQ Chatbot</h1>

<p>
Ask me a question and I'll find the best answer!
</p>

</div>


<div class="chat-area">


<div class="bot-message">

🤖 <strong>Bot:</strong>

<br><br>

Hello! 👋 Welcome to the AI FAQ Chatbot.

<br><br>

You can ask me about:

Python, Artificial Intelligence,
Machine Learning, CodeAlpha,
Flask, HTML, CSS, GitHub,
NLP, or NLTK.

</div>


{% if question %}


<div class="user-message">

👤 <strong>You:</strong>

<br><br>

{{ question }}

</div>


<div class="bot-message">

🤖 <strong>Bot:</strong>

<br><br>

{{ answer }}

</div>


{% endif %}


</div>


<div class="suggestions">

<p>💡 Try asking:</p>


<button
class="suggestion"
onclick="askQuestion('What is Python?')">

Python

</button>


<button
class="suggestion"
onclick="askQuestion('What is Artificial Intelligence?')">

AI

</button>


<button
class="suggestion"
onclick="askQuestion('What is Machine Learning?')">

Machine Learning

</button>


<button
class="suggestion"
onclick="askQuestion('What is CodeAlpha?')">

CodeAlpha

</button>


<button
class="suggestion"
onclick="askQuestion('What is NLP?')">

NLP

</button>


</div>


<form
method="POST"
class="input-area">


<input
type="text"
name="question"
id="question"
placeholder="Type your question here..."
required
autocomplete="off">


<button
type="submit"
class="send-button">

Send 🚀

</button>


</form>


<div class="footer">

AI FAQ Chatbot | CodeAlpha Internship

</div>


</div>


<script>

function askQuestion(question) {

    document.getElementById("question").value =
        question;

}

</script>


</body>

</html>

"""


# -----------------------------
# FLASK ROUTE
# -----------------------------

@app.route("/", methods=["GET", "POST"])

def home():

    question = ""

    answer = ""


    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        )


        if question.strip():

            answer = get_answer(question)


    return render_template_string(

        HTML,

        question=question,

        answer=answer

    )


# -----------------------------
# START APPLICATION
# -----------------------------

if __name__ == "__main__":

    app.run(debug=True)