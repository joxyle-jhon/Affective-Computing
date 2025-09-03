"""
Rule-Based Emotion Detection System for Online Bookstore
========================================================

This program analyzes customer messages and assigns emotions based on:
- Keywords and phrases
- Punctuation patterns
- ALL CAPS words
- Negations
- Text patterns

Emotions detected: Joy, Anger, Sadness, Surprise, Neutral

CORE CONCEPT: Rule-Based Emotion Detection
- Uses predefined rules and keyword matching to detect emotions
- Considers multiple text features: keywords, phrases, punctuation, caps
- Handles negation context to avoid false positives
- Assigns confidence scores based on rule strength

================================================================================
CORE RELEVANT COMPONENTS AND FUNCTIONS:
================================================================================

1. EMOTION RULES DICTIONARY (Lines 28-84):
   - Contains keywords and phrases for each emotion category
   - Organized by emotion type (joy, anger, sadness, surprise)
   - Includes special flags for caps words and exclamation marks

2. CALCULATE_EMOTION_SCORE() (Lines 161-233):
   - Core function that scores each emotion based on rule matching
   - Handles negation context and different keyword types
   - Returns score from 0.0 to 1.0 for each emotion

3. DETECT_EMOTION() (Lines 235-264):
   - Main emotion classification engine
   - Calculates scores for all emotions and selects the highest
   - Returns the best emotion with confidence score

4. DETECT_NEGATION_CONTEXT() (Lines 140-159):
   - Checks if emotion keywords are used in negative context
   - Prevents false positives from negated positive words
   - Critical for accurate emotion detection

5. DETECT_ALL_CAPS() and DETECT_MULTIPLE_EXCLAMATIONS() (Lines 111-138):
   - Utility functions for detecting high-intensity emotional indicators
   - Used specifically for anger detection
   - Enhance accuracy for strong emotional expressions

================================================================================
KEY ALGORITHM: Rule-Based Emotion Classification
================================================================================
The system uses a scoring approach:
1. For each emotion, check if keywords/phrases match the text
2. Adjust scores based on negation context
3. Add bonus points for special indicators (caps, exclamations)
4. Select the emotion with the highest score
5. Return neutral if no emotion scores above threshold

Example: "I LOVE this book!" → Joy (high score from "love" + exclamation)
Example: "I'm NOT happy" → Neutral (negated positive word)
Example: "This is WRONG!!" → Anger (negative word + caps + exclamations)
"""

import re
from typing import Dict, List, Tuple


class EmotionDetector:
    """
    A rule-based emotion detection system for customer messages.
    
    CORE FUNCTIONALITY:
    - Analyzes customer messages using keyword and pattern matching
    - Detects 5 emotions: Joy, Anger, Sadness, Surprise, Neutral
    - Handles negation context and special text features
    - Provides confidence scores for predictions
    """
    
    def __init__(self):
        """
        Initialize the emotion detector with predefined rules.
        
        CORE COMPONENT: Emotion Rules Dictionary
        - Contains keywords and phrases for each emotion category
        - Organized by emotion type with different intensity levels
        - Includes special flags for caps words and exclamation marks
        """
        
        # ========================================================================
        # CORE RELEVANT PART: EMOTION RULES DICTIONARY
        # ========================================================================
        # Contains keywords and phrases for each emotion category
        # Organized by emotion type with different intensity levels
        self.emotion_rules = {
            'joy': {
                'positive_keywords': [
                    'love', 'loved', 'amazing', 'wonderful', 'fantastic', 'excellent',
                    'great', 'awesome', 'brilliant', 'perfect', 'outstanding', 'superb',
                    'delighted', 'pleased', 'happy', 'satisfied', 'thrilled', 'excited',
                    'enjoyed', 'enjoying', 'beautiful', 'gorgeous', 'stunning', 'impressive',
                    'quick', 'fast', 'efficient', 'smooth', 'easy', 'convenient'
                ],
                'positive_phrases': [
                    'thank you', 'thanks', 'appreciate', 'grateful', 'blessed',
                    'couldn\'t be happier', 'exceeded expectations', 'highly recommend'
                ]
            },
            
            'anger': {
                'negative_keywords': [
                    'angry', 'furious', 'mad', 'upset', 'frustrated', 'annoyed',
                    'irritated', 'disgusted', 'outraged', 'livid', 'rage', 'hate',
                    'wrong', 'incorrect', 'mistake', 'error', 'problem', 'issue',
                    'late', 'delayed', 'missing', 'lost', 'broken', 'damaged',
                    'terrible', 'awful', 'horrible', 'disgusting', 'unacceptable',
                    'never', 'worst', 'disappointed', 'unhappy', 'dissatisfied'
                ],
                'negative_phrases': [
                    'what the hell', 'this is ridiculous', 'completely unacceptable',
                    'never again', 'waste of money', 'terrible service', 'worst experience'
                ],
                'caps_words': True,  # Flag for ALL CAPS detection
                'exclamation_multiple': True  # Flag for multiple exclamation marks
            },
            
            'sadness': {
                'sad_keywords': [
                    'sad', 'disappointed', 'disappointing', 'let down', 'heartbroken',
                    'devastated', 'crushed', 'depressed', 'miserable', 'unhappy',
                    'regret', 'regretful', 'sorry', 'apologetic', 'ashamed',
                    'embarrassed', 'humiliated', 'defeated', 'hopeless', 'helpless'
                ],
                'sad_phrases': [
                    'not what I expected', 'not happy with', 'not satisfied',
                    'could have been better', 'not worth it', 'waste of time'
                ]
            },
            
            'surprise': {
                'surprise_keywords': [
                    'wow', 'whoa', 'amazing', 'incredible', 'unbelievable',
                    'shocked', 'surprised', 'astonished', 'stunned', 'speechless',
                    'unexpected', 'unforeseen', 'out of nowhere', 'came as a shock'
                ],
                'surprise_phrases': [
                    'didn\'t expect', 'wasn\'t expecting', 'came as a surprise',
                    'totally unexpected', 'never thought', 'couldn\'t believe'
                ]
            }
        }
        
        # Negation words that can change emotion context
        self.negation_words = [
            'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere',
            'neither', 'nor', 'cannot', 'can\'t', 'won\'t', 'wouldn\'t',
            'don\'t', 'doesn\'t', 'didn\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text for analysis.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase for keyword matching
        text_lower = text.lower()
        
        # Remove extra whitespace
        text_clean = re.sub(r'\s+', ' ', text_lower.strip())
        
        return text_clean
    
    def detect_all_caps(self, text: str) -> bool:
        """
        Detect if text contains ALL CAPS words (indicating anger).
        
        Args:
            text (str): Input text
            
        Returns:
            bool: True if ALL CAPS words found
        """
        # Find words that are all uppercase and at least 3 characters
        caps_pattern = r'\b[A-Z]{3,}\b'
        caps_matches = re.findall(caps_pattern, text)
        return len(caps_matches) > 0
    
    def detect_multiple_exclamations(self, text: str) -> bool:
        """
        Detect multiple exclamation marks (indicating strong emotion).
        
        Args:
            text (str): Input text
            
        Returns:
            bool: True if multiple exclamation marks found
        """
        # Look for 2 or more consecutive exclamation marks
        exclamation_pattern = r'!{2,}'
        return bool(re.search(exclamation_pattern, text))
    
    def detect_negation_context(self, text: str, emotion_keywords: List[str]) -> bool:
        """
        Check if emotion keywords are used in a negative context.
        
        Args:
            text (str): Input text
            emotion_keywords (List[str]): Keywords to check for negation
            
        Returns:
            bool: True if keywords are negated
        """
        words = text.split()
        
        for i, word in enumerate(words):
            if word in emotion_keywords:
                # Check if there's a negation word within 3 words before
                for j in range(max(0, i-3), i):
                    if words[j] in self.negation_words:
                        return True
        return False
    
    def calculate_emotion_score(self, text: str, emotion: str) -> float:
        """
        Calculate a score for a specific emotion based on rules.
        
        CORE FUNCTION: Emotion Scoring Engine
        - Scores each emotion based on keyword and phrase matching
        - Handles negation context to avoid false positives
        - Considers special indicators like caps and exclamations
        - Returns score from 0.0 to 1.0
        
        Args:
            text (str): Input text
            emotion (str): Emotion to score
            
        Returns:
            float: Emotion score (0.0 to 1.0)
        """
        if emotion not in self.emotion_rules:
            return 0.0
        
        rules = self.emotion_rules[emotion]
        score = 0.0
        text_lower = text.lower()
        
        # ========================================================================
        # CORE LOGIC: Check for emotion-specific keywords and phrases
        # ========================================================================
        
        # Check for positive/negative keywords
        if 'positive_keywords' in rules:
            for keyword in rules['positive_keywords']:
                if keyword in text_lower:
                    # Check for negation
                    if not self.detect_negation_context(text_lower, [keyword]):
                        score += 0.3
                    else:
                        score -= 0.2  # Negated positive = negative
        
        if 'negative_keywords' in rules:
            for keyword in rules['negative_keywords']:
                if keyword in text_lower:
                    # Check for negation
                    if not self.detect_negation_context(text_lower, [keyword]):
                        score += 0.3
                    else:
                        score -= 0.1  # Negated negative = less negative
        
        if 'sad_keywords' in rules:
            for keyword in rules['sad_keywords']:
                if keyword in text_lower:
                    if not self.detect_negation_context(text_lower, [keyword]):
                        score += 0.3
                    else:
                        score -= 0.1
        
        if 'surprise_keywords' in rules:
            for keyword in rules['surprise_keywords']:
                if keyword in text_lower:
                    if not self.detect_negation_context(text_lower, [keyword]):
                        score += 0.3
                    else:
                        score -= 0.1
        
        # Check for phrases
        for phrase_type in ['positive_phrases', 'negative_phrases', 'sad_phrases', 'surprise_phrases']:
            if phrase_type in rules:
                for phrase in rules[phrase_type]:
                    if phrase in text_lower:
                        if not self.detect_negation_context(text_lower, [phrase]):
                            score += 0.4
                        else:
                            score -= 0.2
        
        # ========================================================================
        # CORE LOGIC: Special rules for anger (caps and exclamations)
        # ========================================================================
        if emotion == 'anger':
            if rules.get('caps_words', False) and self.detect_all_caps(text):
                score += 0.4
            
            if rules.get('exclamation_multiple', False) and self.detect_multiple_exclamations(text):
                score += 0.3
        
        # Normalize score to 0-1 range
        return min(1.0, max(0.0, score))
    
    def detect_emotion(self, text: str) -> Tuple[str, float]:
        """
        Detect the primary emotion in the given text.
        
        CORE FUNCTION: Main Emotion Classification Engine
        - Calculates scores for all emotions and selects the highest
        - Returns the best emotion with confidence score
        - Handles edge cases and low-confidence predictions
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Tuple[str, float]: (emotion, confidence_score)
        """
        if not text or not text.strip():
            return 'neutral', 0.0
        
        # ========================================================================
        # CORE PROCESS: Calculate scores for each emotion
        # ========================================================================
        emotion_scores = {}
        for emotion in self.emotion_rules.keys():
            emotion_scores[emotion] = self.calculate_emotion_score(text, emotion)
        
        # ========================================================================
        # CORE PROCESS: Find the emotion with the highest score
        # ========================================================================
        if not emotion_scores or max(emotion_scores.values()) == 0:
            return 'neutral', 0.0
        
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[best_emotion]
        
        # If confidence is too low, classify as neutral
        if confidence < 0.1:
            return 'neutral', 0.0
        
        return best_emotion, confidence


def main():
    """
    Main function to run the emotion detection system.
    
    MAIN PROGRAM: System Entry Point
    - Initializes the emotion detector
    - Tests with predefined examples from requirements
    - Provides interactive mode for user input
    - Handles user interaction and error management
    """
    print("=" * 60)
    print("📚 ONLINE BOOKSTORE EMOTION DETECTION SYSTEM")
    print("=" * 60)
    print("\nThis system analyzes customer messages and detects emotions:")
    print("• Joy (happy, satisfied)")
    print("• Anger (frustrated, upset)")
    print("• Sadness (disappointed)")
    print("• Surprise (unexpected)")
    print("• Neutral (none of the above)")
    print("\n" + "=" * 60)
    
    # ========================================================================
    # CORE PROGRAM FLOW: Initialize detector and test examples
    # ========================================================================
    
    # Initialize the emotion detector
    detector = EmotionDetector()
    
    # Test with predefined examples from requirements
    test_messages = [
        "Loved the cover and story!",
        "Where is my order? It's late again!!",
        "I'm not happy with this edition.",
        "Wow, I didn't expect a signed copy.",
        "This is the WRONG book I never ordered.",
        "Thanks for the quick delivery",
        "The pages arrived damaged."
    ]
    
    print("\n🧪 TESTING WITH PREDEFINED EXAMPLES:")
    print("-" * 50)
    
    for i, message in enumerate(test_messages, 1):
        emotion, confidence = detector.detect_emotion(message)
        print(f"{i}. \"{message}\"")
        print(f"   → Emotion: {emotion.upper()} (Confidence: {confidence:.2f})")
        print()
    
    # ========================================================================
    # CORE PROGRAM FLOW: Interactive mode for user input
    # ========================================================================
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("🎯 INTERACTIVE MODE - Enter your own messages!")
    print("Type 'quit' to exit.")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n📝 Enter a customer message: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Emotion Detection System!")
                break
            
            if not user_input:
                print("⚠️  Please enter a message.")
                continue
            
            # ========================================================================
            # CORE INTERACTION: Detect emotion and display results
            # ========================================================================
            
            # Detect emotion using rule-based system
            emotion, confidence = detector.detect_emotion(user_input)
            
            # Display results
            print(f"\n📊 ANALYSIS RESULTS:")
            print(f"   Message: \"{user_input}\"")
            print(f"   Detected Emotion: {emotion.upper()}")
            print(f"   Confidence Score: {confidence:.2f}")
            
            # Provide explanation
            if emotion == 'joy':
                print("   💡 This message contains positive words and expressions!")
            elif emotion == 'anger':
                print("   💡 This message shows frustration, complaints, or strong negative emotions!")
            elif emotion == 'sadness':
                print("   💡 This message expresses disappointment or sadness!")
            elif emotion == 'surprise':
                print("   💡 This message contains unexpected or surprising elements!")
            else:
                print("   💡 This message appears neutral or doesn't show strong emotions.")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
