import numpy as np
import tensorflow as tf
from gtts import gTTS
import pygame

# Initialize pygame mixer for audio playback
try:
    pygame.mixer.init()
except Exception:
    pass


class RealTimeSignDetector:
    def __init__(
        self, 
        model_path="models/transformer_best.keras", 
        threshold=0.8, 
        sequence_length=30
    ):
        """
        Initializes the real-time gesture recognition detector.
        """
        print(f"[INFO] Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold
        self.sequence_length = sequence_length
        
        # State tracking variables for real-time inference
        self.sequence = []
        self.predictions = []
        self.sentence = []

    def process_keypoints(self, keypoints_1d, actions=None):
        """
        Processes a single-frame landmark vector, applies sliding window,
        threshold filtering, and temporal smoothing.
        """
        # Return empty result if no keypoints detected in the current frame
        if keypoints_1d is None:
            return {"label": "", "confidence": 0.0, "sentence": self.sentence}

        # 1. Update sliding window buffer and keep only the latest 30 frames
        self.sequence.append(keypoints_1d)
        self.sequence = self.sequence[-self.sequence_length:]

        current_label = ""
        confidence = 0.0

        # 2. Perform prediction once buffer reaches the required sequence length
        if len(self.sequence) == self.sequence_length:
            input_tensor = np.expand_dims(self.sequence, axis=0)
            res = self.model.predict(input_tensor, verbose=0)[0]
            
            best_class = int(np.argmax(res))
            confidence = float(res[best_class])
            self.predictions.append(best_class)

            # 3. Temporal smoothing (consensus over last 10 frames) and confidence threshold check
            recent_preds = self.predictions[-10:]
            if len(recent_preds) >= 10 and np.unique(recent_preds)[0] == best_class:
                if confidence > self.threshold:
                    label_name = actions[best_class] if actions else f"Class_{best_class}"
                    current_label = label_name

                    # 4. Append to sentence and trigger Text-to-Speech if label changes
                    if len(self.sentence) == 0 or label_name != self.sentence[-1]:
                        self.sentence.append(label_name)
                        self.speak_text(label_name)

            # Keep only the last 5 words for display
            if len(self.sentence) > 5:
                self.sentence = self.sentence[-5:]

        return {
            "label": current_label,
            "confidence": confidence,
            "sentence": self.sentence
        }

    def speak_text(self, text):
        try:
            temp_audio = "temp_tts.mp3"
            tts = gTTS(text=text, lang="vi", slow=False)
            tts.save(temp_audio)
            pygame.mixer.music.load(temp_audio)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"[TTS Warning] {e}")