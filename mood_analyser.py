import sqlite3
import datetime
import logging
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def handle_analysis_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper


class MoodTracker:
    def __init__(self, db_path='/Users/debabratapanda/PycharmProjects/projects/mood_analyser/mood.db'):
        logger.info("Initializing Mood Tracker...")
        self.db_path = db_path
        self._init_db()
        self.emotion_classifier = pipeline(
            "text-classification",
            model="joeddav/distilbert-base-uncased-go-emotions-student",
            top_k=None,
            device="cpu"
        )
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        logger.info("Mood Tracker initialized successfully")

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        entry TEXT,
                        sentiment_score REAL,
                        sentiment_label TEXT,
                        dominant_emotion TEXT,
                        emotions TEXT
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    @handle_analysis_error
    def analyze_emotions(self, text):
        logger.info("Starting emotion analysis")
        emotions = self.emotion_classifier(text)[0]
        emotion_dict = {item['label']: item['score'] for item in emotions}

        sorted_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
        dominant_emotion = sorted_emotions[0]['label']

        if dominant_emotion == 'neutral' and len(sorted_emotions) > 1:
            if sorted_emotions[0]['score'] - sorted_emotions[1]['score'] < 0.1:
                dominant_emotion = sorted_emotions[1]['label']

        logger.info(f"Emotion analysis completed. Dominant emotion: {dominant_emotion}")
        return emotion_dict, dominant_emotion

    @handle_analysis_error
    def analyze_sentiment(self, text):
        logger.info("Starting sentiment analysis")
        scores = self.sentiment_analyzer.polarity_scores(text)
        compound_score = scores['compound']

        if compound_score >= 0.05:
            sentiment_label = "Positive"
        elif compound_score <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        logger.info(f"Sentiment analysis completed. Label: {sentiment_label}")
        return compound_score, sentiment_label

    @handle_analysis_error
    def add_entry(self, text):
        logger.info("Adding new journal entry")
        sentiment_score, sentiment_label = self.analyze_sentiment(text)
        emotions, dominant_emotion = self.analyze_emotions(text)

        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO entries (date, entry, sentiment_score, sentiment_label, dominant_emotion, emotions)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (date, text, sentiment_score, sentiment_label, dominant_emotion, str(emotions)))
                conn.commit()
            return sentiment_score, sentiment_label, emotions, dominant_emotion
        except sqlite3.Error as e:
            logger.error(f"Error in add_entry: {str(e)}")
            raise

    @handle_analysis_error
    def get_entries(self, limit=10):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM entries ORDER BY date DESC LIMIT ?', (limit,))
                return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error in get_entries: {str(e)}")
            raise