import pytest
import re

class TestEmotionDetection:
    def test_crisis_keywords_flagged(self):
        crisis = {"suicide","hopeless","worthless","end it","self harm"}
        text = "I feel completely hopeless and worthless"
        tokens = set(text.lower().split())
        assert len(tokens & crisis) > 0

    def test_empathy_response_not_empty(self):
        def respond(emotion):
            responses = {"sadness":"I hear you.", "joy":"Wonderful!", "anger":"I understand."}
            return responses.get(emotion, "I'm here to listen.")
        assert len(respond("sadness")) > 0
        assert len(respond("unknown")) > 0

    def test_text_cleaned(self):
        raw = "  Hello World!!  "
        clean = re.sub(r"[^a-zA-Z\s]", "", raw).strip().lower()
        assert "!" not in clean

    def test_emotion_keywords_mapped(self):
        emap = {"sad":"sadness","happy":"joy","angry":"anger","anxious":"anxiety"}
        text = "I feel sad and anxious"
        found = [emap[w] for w in text.split() if w in emap]
        assert "sadness" in found and "anxiety" in found

    def test_empty_input_handled(self):
        def process(t):
            return "Please share your thoughts." if not t.strip() else t
        assert process("") != ""

class TestContextualEmpathy:
    def test_multi_turn_context(self):
        history = ["I feel sad", "Why do you feel sad?", "I lost my job"]
        assert len(history) == 3
        assert "sad" in history[0]

    def test_sentiment_score_range(self):
        scores = [0.8, -0.3, 0.1, -0.9]
        assert all(-1.0 <= s <= 1.0 for s in scores)

    def test_sentence_boundary(self):
        text = "I am sad. I need help. Please listen."
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        assert len(sentences) == 3
