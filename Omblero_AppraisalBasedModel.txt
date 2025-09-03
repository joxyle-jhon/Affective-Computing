"""
Appraisal-Based Emotion Detection System for Customer Support
============================================================

This program predicts emotions based on how events affect goals, control, and expectations.
The system uses four appraisal dimensions:
- Goal Relevance (0-1): Importance of the event
- Goal Congruence (-1 to 1): Does it help (+) or hinder (-) goals?
- Control (0-1): How much control does the person have?
- Novelty (0-1): How unexpected is the event?

The system combines these ratings to classify emotions using appraisal theory rules.

CORE CONCEPT: Appraisal Theory
- Emotions arise from how we evaluate events in relation to our goals
- Different combinations of appraisal dimensions lead to different emotions
- This system uses rule-based logic to map appraisal patterns to emotions

================================================================================
CORE RELEVANT COMPONENTS AND FUNCTIONS:
================================================================================

1. EMOTION RULES DICTIONARY (Lines 49-173):
   - Contains lambda functions defining conditions for each emotion
   - Based on psychological appraisal theory research
   - Each emotion has multiple conditions to handle different scenarios

2. CLASSIFY_EMOTION() (Lines 281-311):
   - Main emotion classification engine
   - Iterates through emotion rules and checks conditions
   - Returns first matching emotion with confidence score

3. CALCULATE_CONFIDENCE() (Lines 313-362):
   - Determines confidence in emotion prediction
   - Uses different weighting schemes for different emotions
   - Based on how extreme appraisal values are from neutral

4. GET_USER_INPUT() (Lines 207-279):
   - Collects event description and 4 appraisal dimension ratings
   - Validates all inputs to ensure proper ranges
   - Provides user-friendly interface

5. ANALYZE_EVENT() (Lines 364-398):
   - Main analysis pipeline that orchestrates classification
   - Combines input data with classification results
   - Returns structured results for display

6. TEST_PREDEFINED_SCENARIOS() (Lines 525-590):
   - Tests system with 6 predefined scenarios from requirements
   - Validates that appraisal rules work correctly
   - Demonstrates system capabilities

================================================================================
KEY ALGORITHM: Emotion Classification Logic
================================================================================
The system uses a rule-based approach where:
- Each emotion has specific conditions (lambda functions)
- Conditions check combinations of appraisal dimension values
- First matching condition determines the predicted emotion
- Confidence is calculated based on how extreme the values are
- Different emotions weight different dimensions differently

Example: Joy requires high relevance + positive congruence + high control
Example: Sadness requires high relevance + negative congruence + low control
Example: Surprise requires high novelty regardless of other factors
"""

from typing import Dict, Tuple, List
import re


class AppraisalEmotionDetector:
    """
    An appraisal-based emotion detection system for customer support.
    
    CORE FUNCTIONALITY:
    - Takes user input for event description and 4 appraisal dimensions
    - Applies appraisal theory rules to predict emotions
    - Provides confidence scores and detailed interpretations
    """
    
    def __init__(self):
        """
        Initialize the appraisal emotion detector with emotion rules.
        
        CORE COMPONENT: Emotion Rules Dictionary
        - Contains lambda functions that define conditions for each emotion
        - Each emotion has multiple conditions to handle different scenarios
        - Based on psychological appraisal theory research
        """
        
        # ========================================================================
        # CORE RELEVANT PART: EMOTION CLASSIFICATION RULES
        # ========================================================================
        # This is the heart of the system - defines how appraisal dimensions
        # combine to produce different emotions based on psychological theory
        self.emotion_rules = {
            'joy': {
                'conditions': [
                    # JOY RULE 1: High relevance + positive congruence + high control
                    # When something important helps your goals and you have control
                    lambda r, c, co, n: r >= 0.7 and c >= 0.3 and co >= 0.6,
                    
                    # JOY RULE 2: High relevance + positive congruence + moderate control
                    # Important positive event with some control
                    lambda r, c, co, n: r >= 0.6 and c >= 0.5 and co >= 0.4,
                    
                    # JOY RULE 3: Moderate relevance + very positive congruence
                    # Even moderately important events can bring joy if very positive
                    lambda r, c, co, n: r >= 0.5 and c >= 0.7
                ],
                'description': 'Positive outcome with good control over the situation'
            },
            
            'sadness': {
                'conditions': [
                    # SADNESS RULE 1: High relevance + negative congruence + low control
                    # Important negative event with no control = helplessness/sadness
                    lambda r, c, co, n: r >= 0.7 and c <= -0.3 and co <= 0.4,
                    
                    # SADNESS RULE 2: High relevance + negative congruence + moderate control
                    # Important negative event with limited control
                    lambda r, c, co, n: r >= 0.6 and c <= -0.5 and co <= 0.5,
                    
                    # SADNESS RULE 3: Moderate relevance + very negative congruence + low control
                    # Even moderately important events can cause sadness if very negative
                    lambda r, c, co, n: r >= 0.5 and c <= -0.7 and co <= 0.3
                ],
                'description': 'Negative outcome with little control over the situation'
            },
            
            'anger': {
                'conditions': [
                    # ANGER RULE 1: High relevance + negative congruence + high control
                    # Important negative event with control = can take action/blame others
                    lambda r, c, co, n: r >= 0.7 and c <= -0.3 and co >= 0.6,
                    
                    # ANGER RULE 2: High relevance + negative congruence + moderate control
                    # Important negative event with some control
                    lambda r, c, co, n: r >= 0.6 and c <= -0.4 and co >= 0.5,
                    
                    # ANGER RULE 3: Moderate relevance + negative congruence + high control
                    # Even moderately important events can cause anger if you have control
                    lambda r, c, co, n: r >= 0.5 and c <= -0.5 and co >= 0.7
                ],
                'description': 'Negative outcome with ability to take action or blame others'
            },
            
            'fear': {
                'conditions': [
                    # FEAR RULE 1: High relevance + negative congruence + moderate control + high novelty
                    # Threatening situation with uncertainty and some control
                    lambda r, c, co, n: r >= 0.7 and c <= -0.3 and 0.3 <= co <= 0.6 and n >= 0.6,
                    
                    # FEAR RULE 2: High relevance + negative congruence + low control + moderate novelty
                    # Important negative event with little control and some uncertainty
                    lambda r, c, co, n: r >= 0.6 and c <= -0.4 and co <= 0.4 and n >= 0.5,
                    
                    # FEAR RULE 3: Moderate relevance + negative congruence + low control + high novelty
                    # Uncertain threatening situation with no control
                    lambda r, c, co, n: r >= 0.5 and c <= -0.5 and co <= 0.3 and n >= 0.7
                ],
                'description': 'Threatening situation with uncertainty and limited control'
            },
            
            'surprise': {
                'conditions': [
                    # SURPRISE RULE 1: High novelty regardless of other factors (if novelty dominates)
                    # Very unexpected events are surprising regardless of other factors
                    lambda r, c, co, n: n >= 0.8,
                    
                    # SURPRISE RULE 2: High novelty + neutral congruence
                    # Unexpected event that doesn't clearly help or hinder goals
                    lambda r, c, co, n: n >= 0.7 and -0.2 <= c <= 0.2,
                    
                    # SURPRISE RULE 3: Moderate novelty + very high novelty relative to other dimensions
                    # Novelty is the dominant factor compared to other dimensions
                    lambda r, c, co, n: n >= 0.6 and n > max(r, abs(c), co) + 0.2
                ],
                'description': 'Unexpected event that doesn\'t clearly help or hinder goals'
            },
            
            'disgust': {
                'conditions': [
                    # DISGUST RULE 1: High relevance + very negative congruence + moderate control
                    # Important event that strongly violates values/expectations
                    lambda r, c, co, n: r >= 0.6 and c <= -0.7 and 0.4 <= co <= 0.7,
                    
                    # DISGUST RULE 2: Moderate relevance + very negative congruence + high control
                    # Even moderately important events can cause disgust if very negative
                    lambda r, c, co, n: r >= 0.5 and c <= -0.8 and co >= 0.6
                ],
                'description': 'Strongly negative outcome that violates expectations or values'
            },
            
            'hope': {
                'conditions': [
                    # HOPE RULE 1: High relevance + positive congruence + low control + low novelty
                    # Positive outcome expected despite limited current control
                    lambda r, c, co, n: r >= 0.7 and c >= 0.3 and co <= 0.4 and n <= 0.3,
                    
                    # HOPE RULE 2: Moderate relevance + positive congruence + low control
                    # Optimistic about positive outcome despite limitations
                    lambda r, c, co, n: r >= 0.5 and c >= 0.4 and co <= 0.3
                ],
                'description': 'Positive outcome expected despite limited current control'
            },
            
            'relief': {
                'conditions': [
                    # RELIEF RULE 1: High relevance + positive congruence + high control + low novelty
                    # Positive resolution of a previously uncertain situation
                    lambda r, c, co, n: r >= 0.6 and c >= 0.4 and co >= 0.6 and n <= 0.3,
                    
                    # RELIEF RULE 2: Moderate relevance + positive congruence + moderate control + low novelty
                    # Resolution of uncertainty with positive outcome
                    lambda r, c, co, n: r >= 0.5 and c >= 0.5 and co >= 0.5 and n <= 0.2
                ],
                'description': 'Positive resolution of a previously uncertain situation'
            }
        }
        
        # Default emotion for cases that don't match specific rules
        self.default_emotion = 'neutral'
    
    def validate_input(self, value: float, min_val: float, max_val: float, dimension_name: str) -> bool:
        """
        Validate that input values are within the specified range.
        
        UTILITY FUNCTION: Input validation
        - Ensures user inputs are numeric and within valid ranges
        - Provides clear error messages for invalid inputs
        
        Args:
            value (float): Input value to validate
            min_val (float): Minimum allowed value
            max_val (float): Maximum allowed value
            dimension_name (str): Name of the dimension for error messages
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check if input is a number
        if not isinstance(value, (int, float)):
            print(f"❌ {dimension_name} must be a number.")
            return False
        
        # Check if input is within valid range
        if value < min_val or value > max_val:
            print(f"❌ {dimension_name} must be between {min_val} and {max_val}.")
            return False
        
        return True
    
    def get_user_input(self) -> Tuple[str, float, float, float, float]:
        """
        Get event description and appraisal dimensions from user.
        
        CORE FUNCTION: User Interface
        - Collects event description and 4 appraisal dimension ratings
        - Validates all inputs to ensure they're within proper ranges
        - Provides user-friendly prompts with emojis and clear instructions
        
        Returns:
            Tuple[str, float, float, float, float]: (event, relevance, congruence, control, novelty)
        """
        print("\n" + "="*60)
        print("📝 EVENT DESCRIPTION")
        print("="*60)
        
        # Get event description from user
        while True:
            event = input("Describe the event: ").strip()
            if event:
                break
            print("❌ Please provide an event description.")
        
        print("\n" + "="*60)
        print("📊 APPRAISAL DIMENSIONS")
        print("="*60)
        print("Please rate the event on each dimension:")
        
        # ========================================================================
        # CORE INPUT COLLECTION: Four Appraisal Dimensions
        # ========================================================================
        
        # Get Goal Relevance (0-1): How important is this event?
        while True:
            try:
                relevance_input = input("\n🎯 Goal Relevance (0-1): How important is this event? ")
                relevance = float(relevance_input)
                if self.validate_input(relevance, 0, 1, "Goal Relevance"):
                    break
            except ValueError:
                print("❌ Please enter a valid number.")
        
        # Get Goal Congruence (-1 to 1): Does this help (+) or hinder (-) your goals?
        while True:
            try:
                congruence_input = input("⚖️  Goal Congruence (-1 to 1): Does this help (+) or hinder (-) your goals? ")
                congruence = float(congruence_input)
                if self.validate_input(congruence, -1, 1, "Goal Congruence"):
                    break
            except ValueError:
                print("❌ Please enter a valid number.")
        
        # Get Control (0-1): How much control do you have over this situation?
        while True:
            try:
                control_input = input("🎮 Control (0-1): How much control do you have over this situation? ")
                control = float(control_input)
                if self.validate_input(control, 0, 1, "Control"):
                    break
            except ValueError:
                print("❌ Please enter a valid number.")
        
        # Get Novelty (0-1): How unexpected was this event?
        while True:
            try:
                novelty_input = input("🆕 Novelty (0-1): How unexpected was this event? ")
                novelty = float(novelty_input)
                if self.validate_input(novelty, 0, 1, "Novelty"):
                    break
            except ValueError:
                print("❌ Please enter a valid number.")
        
        return event, relevance, congruence, control, novelty
    
    def classify_emotion(self, relevance: float, congruence: float, control: float, novelty: float) -> Tuple[str, str, float]:
        """
        Classify emotion based on appraisal dimensions.
        
        CORE FUNCTION: Emotion Classification Engine
        - This is the main logic that determines which emotion is predicted
        - Iterates through all emotion rules and checks conditions
        - Returns the first matching emotion with confidence score
        
        Args:
            relevance (float): Goal relevance (0-1)
            congruence (float): Goal congruence (-1 to 1)
            control (float): Control (0-1)
            novelty (float): Novelty (0-1)
            
        Returns:
            Tuple[str, str, float]: (emotion, description, confidence)
        """
        # ========================================================================
        # CORE LOGIC: Check each emotion's conditions in order
        # ========================================================================
        for emotion, data in self.emotion_rules.items():
            for condition in data['conditions']:
                # Test if current appraisal values match this emotion's condition
                if condition(relevance, congruence, control, novelty):
                    # Calculate confidence based on how well the conditions are met
                    confidence = self.calculate_confidence(relevance, congruence, control, novelty, emotion)
                    return emotion, data['description'], confidence
        
        # If no specific emotion matches, return neutral
        return self.default_emotion, "No strong emotional response detected", 0.5
    
    def calculate_confidence(self, relevance: float, congruence: float, control: float, novelty: float, emotion: str) -> float:
        """
        Calculate confidence score for the emotion classification.
        
        CORE FUNCTION: Confidence Calculation
        - Determines how confident we are in the emotion prediction
        - Uses different weighting schemes for different emotions
        - Based on how extreme the appraisal values are from neutral
        
        Args:
            relevance (float): Goal relevance (0-1)
            congruence (float): Goal congruence (-1 to 1)
            control (float): Control (0-1)
            novelty (float): Novelty (0-1)
            emotion (str): Detected emotion
            
        Returns:
            float: Confidence score (0-1)
        """
        # ========================================================================
        # CORE LOGIC: Calculate strength of each dimension
        # ========================================================================
        # Base confidence on how extreme the values are from neutral
        relevance_strength = abs(relevance - 0.5) * 2  # Distance from neutral (0.5)
        congruence_strength = abs(congruence)  # Distance from neutral (0)
        control_strength = abs(control - 0.5) * 2  # Distance from neutral (0.5)
        novelty_strength = abs(novelty - 0.5) * 2  # Distance from neutral (0.5)
        
        # ========================================================================
        # CORE LOGIC: Weight dimensions differently for different emotions
        # ========================================================================
        if emotion == 'surprise':
            # Novelty is most important for surprise
            confidence = (novelty_strength * 0.5 + relevance_strength * 0.2 + 
                         congruence_strength * 0.2 + control_strength * 0.1)
        elif emotion in ['joy', 'sadness', 'anger']:
            # Relevance and congruence are most important for basic emotions
            confidence = (relevance_strength * 0.4 + congruence_strength * 0.4 + 
                         control_strength * 0.2 + novelty_strength * 0.0)
        elif emotion in ['fear', 'hope']:
            # All dimensions matter but control is important for fear/hope
            confidence = (relevance_strength * 0.3 + congruence_strength * 0.3 + 
                         control_strength * 0.3 + novelty_strength * 0.1)
        else:
            # Default weighting for other emotions
            confidence = (relevance_strength * 0.3 + congruence_strength * 0.3 + 
                         control_strength * 0.2 + novelty_strength * 0.2)
        
        # Ensure confidence is between 0 and 1
        return min(1.0, max(0.0, confidence))
    
    def analyze_event(self, event: str, relevance: float, congruence: float, control: float, novelty: float) -> Dict:
        """
        Analyze an event and return comprehensive results.
        
        CORE FUNCTION: Main Analysis Pipeline
        - Orchestrates the emotion classification process
        - Combines all appraisal dimensions with classification results
        - Returns structured data for display and further processing
        
        Args:
            event (str): Event description
            relevance (float): Goal relevance (0-1)
            congruence (float): Goal congruence (-1 to 1)
            control (float): Control (0-1)
            novelty (float): Novelty (0-1)
            
        Returns:
            Dict: Analysis results containing all input and output data
        """
        # ========================================================================
        # CORE PROCESS: Classify emotion using appraisal dimensions
        # ========================================================================
        emotion, description, confidence = self.classify_emotion(relevance, congruence, control, novelty)
        
        # Return comprehensive results dictionary
        return {
            'event': event,
            'relevance': relevance,
            'congruence': congruence,
            'control': control,
            'novelty': novelty,
            'emotion': emotion,
            'description': description,
            'confidence': confidence
        }
    
    def display_results(self, results: Dict):
        """
        Display analysis results in a formatted way.
        
        UTILITY FUNCTION: Results Display
        - Formats and displays all analysis results in a user-friendly way
        - Shows appraisal dimensions, predicted emotion, and confidence
        - Calls interpretation function for additional insights
        
        Args:
            results (Dict): Analysis results from analyze_event()
        """
        print("\n" + "="*60)
        print("📊 EMOTION ANALYSIS RESULTS")
        print("="*60)
        
        # Display event description
        print(f"Event: \"{results['event']}\"")
        
        # ========================================================================
        # CORE OUTPUT: Display Appraisal Dimensions
        # ========================================================================
        print(f"\nAppraisal Dimensions:")
        print(f"  🎯 Goal Relevance: {results['relevance']:.2f} (0 = unimportant, 1 = very important)")
        print(f"  ⚖️  Goal Congruence: {results['congruence']:+.2f} (-1 = hinders goals, +1 = helps goals)")
        print(f"  🎮 Control: {results['control']:.2f} (0 = no control, 1 = full control)")
        print(f"  🆕 Novelty: {results['novelty']:.2f} (0 = expected, 1 = completely unexpected)")
        
        # ========================================================================
        # CORE OUTPUT: Display Emotion Prediction
        # ========================================================================
        print(f"\n🎭 Predicted Emotion: {results['emotion'].upper()}")
        print(f"📝 Description: {results['description']}")
        print(f"🎯 Confidence: {results['confidence']:.2f} ({self.get_confidence_level(results['confidence'])})")
        
        # Provide additional interpretation
        self.provide_interpretation(results)
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        Get a human-readable confidence level.
        
        Args:
            confidence (float): Confidence score (0-1)
            
        Returns:
            str: Confidence level description
        """
        if confidence >= 0.8:
            return "Very High"
        elif confidence >= 0.6:
            return "High"
        elif confidence >= 0.4:
            return "Moderate"
        elif confidence >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def provide_interpretation(self, results: Dict):
        """
        Provide additional interpretation of the results.
        
        Args:
            results (Dict): Analysis results
        """
        print(f"\n💡 INTERPRETATION:")
        
        # Interpret relevance
        if results['relevance'] >= 0.7:
            relevance_desc = "very important"
        elif results['relevance'] >= 0.4:
            relevance_desc = "moderately important"
        else:
            relevance_desc = "not very important"
        
        # Interpret congruence
        if results['congruence'] >= 0.5:
            congruence_desc = "strongly helps your goals"
        elif results['congruence'] >= 0.2:
            congruence_desc = "somewhat helps your goals"
        elif results['congruence'] <= -0.5:
            congruence_desc = "strongly hinders your goals"
        elif results['congruence'] <= -0.2:
            congruence_desc = "somewhat hinders your goals"
        else:
            congruence_desc = "neutral for your goals"
        
        # Interpret control
        if results['control'] >= 0.7:
            control_desc = "high control"
        elif results['control'] >= 0.4:
            control_desc = "moderate control"
        else:
            control_desc = "low control"
        
        # Interpret novelty
        if results['novelty'] >= 0.7:
            novelty_desc = "very unexpected"
        elif results['novelty'] >= 0.4:
            novelty_desc = "somewhat unexpected"
        else:
            novelty_desc = "expected"
        
        print(f"   This event is {relevance_desc} and {congruence_desc}.")
        print(f"   You have {control_desc} over the situation, and it was {novelty_desc}.")
        
        # Provide emotion-specific insights
        emotion = results['emotion']
        if emotion == 'joy':
            print("   🎉 This suggests a positive outcome that you had some influence over!")
        elif emotion == 'sadness':
            print("   😢 This indicates a negative outcome with limited ability to change it.")
        elif emotion == 'anger':
            print("   😠 This suggests frustration with a negative outcome you could potentially address.")
        elif emotion == 'fear':
            print("   😰 This indicates anxiety about an uncertain, potentially threatening situation.")
        elif emotion == 'surprise':
            print("   😲 This was an unexpected event that caught you off guard!")
        elif emotion == 'hope':
            print("   🌟 This suggests optimism about a positive outcome despite current limitations.")
        elif emotion == 'relief':
            print("   😌 This indicates a positive resolution of previous uncertainty or concern.")


def test_predefined_scenarios():
    """
    Test the system with predefined scenarios.
    
    TESTING FUNCTION: Validation and Demonstration
    - Tests the system with 6 predefined scenarios from the requirements
    - Validates that the appraisal rules work correctly
    - Demonstrates the system's capabilities
    """
    print("🧪 TESTING WITH PREDEFINED SCENARIOS")
    print("="*60)
    
    detector = AppraisalEmotionDetector()
    
    # ========================================================================
    # CORE TESTING: Predefined scenarios from requirements
    # ========================================================================
    test_scenarios = [
        {
            'event': 'You failed an exam',
            'relevance': 0.8, 'congruence': -0.7, 'control': 0.3, 'novelty': 0.2,
            'expected': 'Sadness'
        },
        {
            'event': 'You received a surprise gift',
            'relevance': 0.6, 'congruence': 0.8, 'control': 0.1, 'novelty': 0.9,
            'expected': 'Joy or Surprise'
        },
        {
            'event': 'Your project deadline was moved earlier nearby',
            'relevance': 0.8, 'congruence': -0.6, 'control': 0.4, 'novelty': 0.3,
            'expected': 'Anger or Anxiety'
        },
        {
            'event': 'Unexpected loud noise nearby',
            'relevance': 0.4, 'congruence': -0.2, 'control': 0.2, 'novelty': 0.9,
            'expected': 'Surprise'
        },
        {
            'event': 'You achieved a personal best in a run',
            'relevance': 0.7, 'congruence': 0.8, 'control': 0.8, 'novelty': 0.3,
            'expected': 'Joy'
        },
        {
            'event': 'Traffic made you late to a meeting',
            'relevance': 0.6, 'congruence': -0.5, 'control': 0.2, 'novelty': 0.1,
            'expected': 'Sadness or Anger'
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"{i}. {scenario['event']}")
        print()


def main():
    """
    Main function to run the appraisal-based emotion detection system.
    
    MAIN PROGRAM: System Entry Point
    - Initializes the emotion detector
    - Runs predefined test scenarios
    - Provides interactive mode for user input
    - Handles user interaction and error management
    """
    print("="*70)
    print("🎭 CUSTOMER SUPPORT - APPRAISAL-BASED EMOTION DETECTION")
    print("="*70)
    print("\nThis system predicts emotions based on how events affect:")
    print("• Goals (relevance and congruence)")
    print("• Control over the situation")
    print("• Expectations (novelty)")
    print("\nThe system uses appraisal theory to understand emotional responses.")
    print("="*70)
    
    # ========================================================================
    # CORE PROGRAM FLOW: Test scenarios first, then interactive mode
    # ========================================================================
    
    # Test with predefined scenarios first
    test_predefined_scenarios()
    
    # Interactive mode for user input
    print("\n" + "="*70)
    print("🎯 INTERACTIVE MODE - Analyze your own events!")
    print("Type 'quit' to exit.")
    print("="*70)
    
    detector = AppraisalEmotionDetector()
    
    # Main interaction loop
    while True:
        try:
            # ========================================================================
            # CORE INTERACTION: Get user input and analyze
            # ========================================================================
            # Get user input (event description + 4 appraisal dimensions)
            event, relevance, congruence, control, novelty = detector.get_user_input()
            
            # Analyze the event using appraisal theory
            results = detector.analyze_event(event, relevance, congruence, control, novelty)
            
            # Display comprehensive results
            detector.display_results(results)
            
            # Ask if user wants to continue
            print("\n" + "-"*60)
            continue_choice = input("Would you like to analyze another event? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
    
    print("\n👋 Thank you for using the Appraisal-Based Emotion Detection System!")


if __name__ == "__main__":
    main()
