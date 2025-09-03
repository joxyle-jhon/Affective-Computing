"""
Dimensional Model Emotion Detection System for Mobile Wellness App
================================================================

This program analyzes user diary entries using the dimensional model of affect:
- Valence: How positive (pleasant) or negative (unpleasant) the experience is
- Arousal: How calm (low energy) or excited (high energy) the experience is

The system combines these dimensions to classify emotions into four quadrants:
- High Valence + High Arousal → Excited / Joyful
- High Valence + Low Arousal → Calm / Content  
- Low Valence + High Arousal → Angry / Anxious
- Low Valence + Low Arousal → Sad / Tired

CORE CONCEPT: Dimensional Model of Affect
- Emotions are represented as points in a 2D space (Valence × Arousal)
- Valence ranges from negative (unpleasant) to positive (pleasant)
- Arousal ranges from low (calm) to high (excited)
- Four quadrants represent different emotional states

================================================================================
CORE RELEVANT COMPONENTS AND FUNCTIONS:
================================================================================

1. VALENCE RULES DICTIONARY (Lines 29-67):
   - Contains keywords and phrases for positive/negative valence detection
   - Organized by intensity (high/moderate) and type (keywords/phrases)
   - Used to calculate valence score (-1.0 to 1.0)

2. AROUSAL RULES DICTIONARY (Lines 70-118):
   - Contains keywords and phrases for high/low arousal detection
   - Includes excitement, anger, anxiety, calm, tired, and sad indicators
   - Used to calculate arousal score (0.0 to 1.0)

3. CALCULATE_VALENCE_SCORE() (Lines 195-263):
   - Core function that determines valence based on positive/negative indicators
   - Handles negation context and different intensity levels
   - Returns score from -1.0 (very negative) to 1.0 (very positive)

4. CALCULATE_AROUSAL_SCORE() (Lines 265-356):
   - Core function that determines arousal based on energy/calm indicators
   - Considers excitement, anger, anxiety, calm, tired, and sad keywords
   - Returns score from 0.0 (very calm) to 1.0 (very excited)

5. CLASSIFY_EMOTION_QUADRANT() (Lines 358-380):
   - Maps valence and arousal scores to emotional quadrants
   - Uses thresholds to determine which quadrant the emotion falls into
   - Returns quadrant name and emotion label

6. DETECT_EMOTION() (Lines 382-419):
   - Main analysis pipeline that orchestrates the dimensional model
   - Calculates valence and arousal, then classifies into quadrant
   - Returns comprehensive results with confidence scores

================================================================================
KEY ALGORITHM: Dimensional Emotion Classification
================================================================================
The system uses a two-step process:
1. Calculate valence score (-1.0 to 1.0) based on positive/negative keywords
2. Calculate arousal score (0.0 to 1.0) based on energy/calm keywords
3. Map the (valence, arousal) point to one of four quadrants
4. Calculate confidence based on distance from neutral points

Example: High valence (0.8) + High arousal (0.9) → Excited/Joyful
Example: Low valence (-0.7) + Low arousal (0.2) → Sad/Tired
"""

import re
from typing import Dict, List, Tuple


class DimensionalEmotionDetector:
    """
    A dimensional model emotion detection system for wellness diary entries.
    
    CORE FUNCTIONALITY:
    - Analyzes text using valence and arousal dimensions
    - Maps emotional states to four quadrants
    - Provides confidence scores and interpretations
    """
    
    def __init__(self):
        """
        Initialize the dimensional emotion detector with valence and arousal rules.
        
        CORE COMPONENT: Valence and Arousal Rules
        - Defines keywords and phrases for positive/negative valence
        - Defines keywords and phrases for high/low arousal
        - Based on psychological research on dimensional emotion models
        """
        
        # ========================================================================
        # CORE RELEVANT PART: VALENCE RULES DICTIONARY
        # ========================================================================
        # Contains keywords and phrases for detecting positive/negative valence
        # Organized by intensity levels and types for accurate scoring
        self.valence_rules = {
            'positive': {
                'high_positive_keywords': [
                    'amazing', 'wonderful', 'fantastic', 'excellent', 'perfect', 'outstanding',
                    'brilliant', 'superb', 'incredible', 'unbelievable', 'magnificent',
                    'delighted', 'thrilled', 'ecstatic', 'overjoyed', 'blissful', 'euphoric',
                    'love', 'loved', 'adore', 'cherish', 'treasure', 'blessed', 'grateful'
                ],
                'moderate_positive_keywords': [
                    'good', 'great', 'nice', 'pleasant', 'enjoyable', 'satisfying',
                    'happy', 'pleased', 'content', 'comfortable', 'peaceful', 'calm',
                    'relaxed', 'serene', 'tranquil', 'satisfied', 'fulfilled', 'optimistic'
                ],
                'positive_phrases': [
                    'feeling great', 'having fun', 'enjoying myself', 'life is good',
                    'everything is fine', 'couldn\'t be better', 'on top of the world',
                    'feeling blessed', 'grateful for', 'thankful for', 'loving life'
                ]
            },
            
            'negative': {
                'high_negative_keywords': [
                    'terrible', 'awful', 'horrible', 'disgusting', 'disgusted', 'hate',
                    'loathe', 'despise', 'furious', 'livid', 'enraged', 'outraged',
                    'devastated', 'crushed', 'heartbroken', 'miserable', 'wretched',
                    'hopeless', 'desperate', 'despair', 'anguish', 'torment', 'agony'
                ],
                'moderate_negative_keywords': [
                    'bad', 'sad', 'disappointed', 'upset', 'frustrated', 'annoyed',
                    'worried', 'concerned', 'anxious', 'stressed', 'tired', 'exhausted',
                    'bored', 'lonely', 'empty', 'lost', 'confused', 'overwhelmed'
                ],
                'negative_phrases': [
                    'feeling down', 'having a bad day', 'everything is wrong',
                    'can\'t take it anymore', 'at my wit\'s end', 'feeling hopeless',
                    'nothing is working', 'completely lost', 'falling apart'
                ]
            }
        }
        
        # ========================================================================
        # CORE RELEVANT PART: AROUSAL RULES DICTIONARY
        # ========================================================================
        # Contains keywords and phrases for detecting high/low arousal (energy levels)
        # Includes excitement, anger, anxiety, calm, tired, and sad indicators
        self.arousal_rules = {
            'high_arousal': {
                'excitement_keywords': [
                    'excited', 'thrilled', 'pumped', 'energized', 'hyped', 'ecstatic',
                    'euphoric', 'elated', 'overjoyed', 'delirious', 'frenzied', 'wild',
                    'intense', 'passionate', 'fiery', 'explosive', 'dynamic', 'vibrant'
                ],
                'anger_keywords': [
                    'angry', 'furious', 'livid', 'enraged', 'outraged', 'irate',
                    'incensed', 'infuriated', 'raging', 'boiling', 'seething', 'fuming',
                    'explosive', 'volatile', 'aggressive', 'hostile', 'combative'
                ],
                'anxiety_keywords': [
                    'anxious', 'nervous', 'worried', 'stressed', 'panicked', 'frantic',
                    'agitated', 'restless', 'jittery', 'on edge', 'tense', 'uptight',
                    'hyperactive', 'manic', 'frenetic', 'overwhelmed', 'overstimulated'
                ],
                'high_arousal_phrases': [
                    'can\'t sit still', 'full of energy', 'ready to explode',
                    'heart racing', 'adrenaline pumping', 'on fire', 'flying high',
                    'bouncing off the walls', 'can\'t contain myself'
                ],
                'exclamation_multiple': True,  # Multiple exclamation marks
                'caps_words': True  # ALL CAPS words
            },
            
            'low_arousal': {
                'calm_keywords': [
                    'calm', 'peaceful', 'serene', 'tranquil', 'quiet', 'still',
                    'relaxed', 'chilled', 'mellow', 'laid-back', 'easygoing',
                    'composed', 'collected', 'centered', 'grounded', 'stable'
                ],
                'tired_keywords': [
                    'tired', 'exhausted', 'drained', 'weary', 'fatigued', 'spent',
                    'worn out', 'burned out', 'depleted', 'lethargic', 'sluggish',
                    'listless', 'apathetic', 'numb', 'empty', 'hollow', 'lifeless'
                ],
                'sad_keywords': [
                    'sad', 'depressed', 'down', 'blue', 'melancholy', 'gloomy',
                    'somber', 'mournful', 'dejected', 'disheartened', 'discouraged',
                    'defeated', 'resigned', 'hopeless', 'helpless', 'powerless'
                ],
                'low_arousal_phrases': [
                    'feeling drained', 'no energy left', 'completely exhausted',
                    'just want to rest', 'feeling empty', 'going through the motions',
                    'barely keeping up', 'running on empty', 'at the end of my rope'
                ]
            }
        }
        
        # Negation words that can change the meaning
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
        Detect if text contains ALL CAPS words (indicating high arousal).
        
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
        Detect multiple exclamation marks (indicating high arousal).
        
        Args:
            text (str): Input text
            
        Returns:
            bool: True if multiple exclamation marks found
        """
        # Look for 2 or more consecutive exclamation marks
        exclamation_pattern = r'!{2,}'
        return bool(re.search(exclamation_pattern, text))
    
    def detect_negation_context(self, text: str, keywords: List[str]) -> bool:
        """
        Check if keywords are used in a negative context.
        
        Args:
            text (str): Input text
            keywords (List[str]): Keywords to check for negation
            
        Returns:
            bool: True if keywords are negated
        """
        words = text.split()
        
        for i, word in enumerate(words):
            if word in keywords:
                # Check if there's a negation word within 3 words before
                for j in range(max(0, i-3), i):
                    if words[j] in self.negation_words:
                        return True
        return False
    
    def calculate_valence_score(self, text: str) -> float:
        """
        Calculate valence score (-1.0 to 1.0) based on positive/negative indicators.
        
        CORE FUNCTION: Valence Calculation Engine
        - Analyzes text for positive and negative emotional indicators
        - Handles negation context to avoid false positives
        - Returns score from -1.0 (very negative) to 1.0 (very positive)
        
        Args:
            text (str): Input text
            
        Returns:
            float: Valence score (-1.0 = very negative, 1.0 = very positive)
        """
        text_lower = text.lower()
        score = 0.0
        
        # ========================================================================
        # CORE LOGIC: Check positive indicators
        # ========================================================================
        positive_rules = self.valence_rules['positive']
        
        # High positive keywords (strong positive valence)
        for keyword in positive_rules['high_positive_keywords']:
            if keyword in text_lower:
                if not self.detect_negation_context(text_lower, [keyword]):
                    score += 0.4
                else:
                    score -= 0.3  # Negated positive = negative
        
        # Moderate positive keywords
        for keyword in positive_rules['moderate_positive_keywords']:
            if keyword in text_lower:
                if not self.detect_negation_context(text_lower, [keyword]):
                    score += 0.2
                else:
                    score -= 0.2
        
        # Positive phrases
        for phrase in positive_rules['positive_phrases']:
            if phrase in text_lower:
                if not self.detect_negation_context(text_lower, [phrase]):
                    score += 0.3
                else:
                    score -= 0.2
        
        # ========================================================================
        # CORE LOGIC: Check negative indicators
        # ========================================================================
        negative_rules = self.valence_rules['negative']
        
        # High negative keywords (strong negative valence)
        for keyword in negative_rules['high_negative_keywords']:
            if keyword in text_lower:
                if not self.detect_negation_context(text_lower, [keyword]):
                    score -= 0.4
                else:
                    score += 0.2  # Negated negative = less negative
        
        # Moderate negative keywords
        for keyword in negative_rules['moderate_negative_keywords']:
            if keyword in text_lower:
                if not self.detect_negation_context(text_lower, [keyword]):
                    score -= 0.2
                else:
                    score += 0.1
        
        # Negative phrases
        for phrase in negative_rules['negative_phrases']:
            if phrase in text_lower:
                if not self.detect_negation_context(text_lower, [phrase]):
                    score -= 0.3
                else:
                    score += 0.2
        
        # Normalize to -1.0 to 1.0 range
        return max(-1.0, min(1.0, score))
    
    def calculate_arousal_score(self, text: str) -> float:
        """
        Calculate arousal score (0.0 to 1.0) based on energy/calm indicators.
        
        CORE FUNCTION: Arousal Calculation Engine
        - Analyzes text for high/low energy emotional indicators
        - Considers excitement, anger, anxiety, calm, tired, and sad keywords
        - Returns score from 0.0 (very calm) to 1.0 (very excited)
        
        Args:
            text (str): Input text
            
        Returns:
            float: Arousal score (0.0 = very calm, 1.0 = very excited)
        """
        text_lower = text.lower()
        score = 0.0
        
        # ========================================================================
        # CORE LOGIC: Check high arousal indicators
        # ========================================================================
        high_arousal_rules = self.arousal_rules['high_arousal']
        
        # Excitement keywords
        for keyword in high_arousal_rules['excitement_keywords']:
            if keyword in text_lower:
                if not self.detect_negation_context(text_lower, [keyword]):
                    score += 0.3
                else:
                    score -= 0.2
        
        # Anger keywords
        for keyword in high_arousal_rules['anger_keywords']:
            if keyword in text_lower:
                if not self.detect_negation_context(text_lower, [keyword]):
                    score += 0.3
                else:
                    score -= 0.1
        
        # Anxiety keywords
        for keyword in high_arousal_rules['anxiety_keywords']:
            if keyword in text_lower:
                if not self.detect_negation_context(text_lower, [keyword]):
                    score += 0.3
                else:
                    score -= 0.1
        
        # High arousal phrases
        for phrase in high_arousal_rules['high_arousal_phrases']:
            if phrase in text_lower:
                if not self.detect_negation_context(text_lower, [phrase]):
                    score += 0.4
                else:
                    score -= 0.2
        
        # Special rules for high arousal
        if high_arousal_rules.get('exclamation_multiple', False) and self.detect_multiple_exclamations(text):
            score += 0.3
        
        if high_arousal_rules.get('caps_words', False) and self.detect_all_caps(text):
            score += 0.3
        
        # ========================================================================
        # CORE LOGIC: Check low arousal indicators
        # ========================================================================
        low_arousal_rules = self.arousal_rules['low_arousal']
        
        # Calm keywords
        for keyword in low_arousal_rules['calm_keywords']:
            if keyword in text_lower:
                if not self.detect_negation_context(text_lower, [keyword]):
                    score -= 0.2
                else:
                    score += 0.1
        
        # Tired keywords
        for keyword in low_arousal_rules['tired_keywords']:
            if keyword in text_lower:
                if not self.detect_negation_context(text_lower, [keyword]):
                    score -= 0.3
                else:
                    score += 0.1
        
        # Sad keywords
        for keyword in low_arousal_rules['sad_keywords']:
            if keyword in text_lower:
                if not self.detect_negation_context(text_lower, [keyword]):
                    score -= 0.2
                else:
                    score += 0.1
        
        # Low arousal phrases
        for phrase in low_arousal_rules['low_arousal_phrases']:
            if phrase in text_lower:
                if not self.detect_negation_context(text_lower, [phrase]):
                    score -= 0.3
                else:
                    score += 0.1
        
        # Normalize to 0.0 to 1.0 range
        return max(0.0, min(1.0, score))
    
    def classify_emotion_quadrant(self, valence: float, arousal: float) -> Tuple[str, str]:
        """
        Classify emotion into one of four quadrants based on valence and arousal.
        
        CORE FUNCTION: Quadrant Classification
        - Maps valence and arousal scores to emotional quadrants
        - Uses thresholds to determine which quadrant the emotion falls into
        - Returns quadrant name and emotion label
        
        Args:
            valence (float): Valence score (-1.0 to 1.0)
            arousal (float): Arousal score (0.0 to 1.0)
            
        Returns:
            Tuple[str, str]: (quadrant_name, emotion_label)
        """
        # ========================================================================
        # CORE LOGIC: Define thresholds for quadrant classification
        # ========================================================================
        valence_threshold = 0.0  # Positive vs negative
        arousal_threshold = 0.5  # High vs low arousal
        
        if valence >= valence_threshold and arousal >= arousal_threshold:
            return "Quadrant I", "Excited / Joyful"
        elif valence >= valence_threshold and arousal < arousal_threshold:
            return "Quadrant II", "Calm / Content"
        elif valence < valence_threshold and arousal >= arousal_threshold:
            return "Quadrant III", "Angry / Anxious"
        else:  # valence < 0 and arousal < 0.5
            return "Quadrant IV", "Sad / Tired"
    
    def detect_emotion(self, text: str) -> Dict[str, any]:
        """
        Detect emotion using the dimensional model.
        
        CORE FUNCTION: Main Analysis Pipeline
        - Orchestrates the dimensional emotion detection process
        - Calculates valence and arousal scores, then classifies into quadrant
        - Returns comprehensive results with confidence scores
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, any]: Analysis results including valence, arousal, and quadrant
        """
        if not text or not text.strip():
            return {
                'valence': 0.0,
                'arousal': 0.5,
                'quadrant': 'Quadrant II',
                'emotion': 'Calm / Content',
                'confidence': 0.0
            }
        
        # ========================================================================
        # CORE PROCESS: Calculate valence and arousal scores
        # ========================================================================
        valence = self.calculate_valence_score(text)
        arousal = self.calculate_arousal_score(text)
        
        # ========================================================================
        # CORE PROCESS: Classify into quadrant
        # ========================================================================
        quadrant, emotion = self.classify_emotion_quadrant(valence, arousal)
        
        # ========================================================================
        # CORE PROCESS: Calculate confidence based on distance from neutral
        # ========================================================================
        valence_confidence = abs(valence)
        arousal_confidence = abs(arousal - 0.5) * 2  # Distance from middle (0.5)
        overall_confidence = (valence_confidence + arousal_confidence) / 2
        
        return {
            'valence': valence,
            'arousal': arousal,
            'quadrant': quadrant,
            'emotion': emotion,
            'confidence': overall_confidence
        }


def main():
    """
    Main function to run the dimensional emotion detection system.
    """
    print("=" * 70)
    print("📱 MOBILE WELLNESS APP - DIMENSIONAL EMOTION DETECTION")
    print("=" * 70)
    print("\nThis system analyzes diary entries using the dimensional model:")
    print("• Valence: How positive (pleasant) or negative (unpleasant)")
    print("• Arousal: How calm (low energy) or excited (high energy)")
    print("\nFour Emotional Quadrants:")
    print("• High Valence + High Arousal → Excited / Joyful")
    print("• High Valence + Low Arousal → Calm / Content")
    print("• Low Valence + High Arousal → Angry / Anxious")
    print("• Low Valence + Low Arousal → Sad / Tired")
    print("\n" + "=" * 70)
    
    # Initialize the emotion detector
    detector = DimensionalEmotionDetector()
    
    # Test with predefined examples
    test_messages = [
        "I just got the job of my dreams!",
        "The beach was so peaceful today.",
        "This delay is making me furious!!",
        "I feel exhausted and hopeless."
    ]
    
    print("\n🧪 TESTING WITH PREDEFINED EXAMPLES:")
    print("-" * 60)
    
    for i, message in enumerate(test_messages, 1):
        result = detector.detect_emotion(message)
        print(f"{i}. \"{message}\"")
        print(f"   → Valence: {result['valence']:+.2f} | Arousal: {result['arousal']:.2f}")
        print(f"   → {result['quadrant']}: {result['emotion']}")
        print(f"   → Confidence: {result['confidence']:.2f}")
        print()
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("🎯 INTERACTIVE MODE - Enter your own diary entries!")
    print("Type 'quit' to exit.")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("\n📝 Enter a diary entry: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Dimensional Emotion Detection System!")
                break
            
            if not user_input:
                print("⚠️  Please enter a diary entry.")
                continue
            
            # Detect emotion
            result = detector.detect_emotion(user_input)
            
            # Display results
            print(f"\n📊 ANALYSIS RESULTS:")
            print(f"   Entry: \"{user_input}\"")
            print(f"   Valence Score: {result['valence']:+.2f} (Negative ← → Positive)")
            print(f"   Arousal Score: {result['arousal']:.2f} (Calm ← → Excited)")
            print(f"   Classification: {result['quadrant']}")
            print(f"   Emotional State: {result['emotion']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            
            # Provide interpretation
            valence_desc = "positive" if result['valence'] > 0 else "negative" if result['valence'] < 0 else "neutral"
            arousal_desc = "high energy" if result['arousal'] > 0.5 else "low energy" if result['arousal'] < 0.5 else "moderate energy"
            
            print(f"\n💡 INTERPRETATION:")
            print(f"   Your entry shows {valence_desc} valence and {arousal_desc}.")
            
            if result['confidence'] > 0.7:
                print("   The emotional state is clearly detected.")
            elif result['confidence'] > 0.4:
                print("   The emotional state is moderately detected.")
            else:
                print("   The emotional state is weakly detected - may be neutral.")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
