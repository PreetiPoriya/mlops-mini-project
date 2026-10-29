FROM python:3.10

WORKDIR /app

COPY flask_app/ /app/

# app.py looks for ../models/vectorizer.pkl, which is /models/ inside the container
COPY models/vectorizer.pkl /models/vectorizer.pkl

RUN pip install --no-cache-dir -r requirements.txt

# NLTK data used by preprocessing_utility.py
RUN python -m nltk.downloader stopwords wordnet omw-1.4

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]