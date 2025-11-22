import os
import json
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification


class EmergencyClassifierDemo:
    """Interactive demo for emergency classification"""
    
    def __init__(self):
        self.model_dir = 'outputs/trained_model'
        
        print("🔄 Loading model and tokenizer...")
        
        # Load label mapping
        with open(f"{self.model_dir}/label_mapping.json", 'r') as f:
            label_config = json.load(f)
        self.id_to_label = {int(k): v for k, v in label_config['id_to_label'].items()}
        self.label_names = list(self.id_to_label.values())
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
        
        # Load model (lazy loading - only when needed)
        self.model = None
        
        print("✅ Ready! Enter your emergency text below.\n")
    
    def load_model(self):
        """Load model on first use"""
        if self.model is None:
            print("📥 Loading TensorFlow model (first time only)...")
            self.model = TFDistilBertForSequenceClassification.from_pretrained(self.model_dir)
            print("✅ Model loaded!\n")
    
    def predict(self, text):
        """Predict emergency category from text"""
        self.load_model()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='tf'
        )
        
        # Predict
        outputs = self.model(encoding)
        logits = outputs.logits
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        
        # Get predictions
        predicted_idx = np.argmax(probs)
        predicted_label = self.id_to_label[predicted_idx]
        confidence = float(probs[predicted_idx])
        
        # Get all probabilities
        all_probs = {self.id_to_label[i]: float(probs[i]) for i in range(len(self.id_to_label))}
        
        return {
            'text': text,
            'predicted_category': predicted_label,
            'confidence': confidence,
            'all_probabilities': all_probs
        }
    
    def display_result(self, result):
        """Display prediction results nicely"""
        print("\n" + "="*80)
        print("📝 INPUT TEXT:")
        print(f"   {result['text']}")
        print("\n" + "="*80)
        print("🎯 PREDICTION:")
        print(f"   Category: {result['predicted_category'].upper()}")
        print(f"   Confidence: {result['confidence']*100:.2f}%")
        print("\n" + "="*80)
        print("📊 ALL CATEGORIES (Confidence Scores):")
        for label in self.label_names:
            prob = result['all_probabilities'][label]
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"   {label:20s} {prob*100:6.2f}% |{bar}|")
        print("="*80 + "\n")


def main():
    """Interactive demo"""
    print("🚨 Emergency NLP Classifier - Interactive Demo")
    print("="*80)
    print("Categories: police, fire, ambulance, women_helpline, disaster")
    print("Type 'quit' or 'exit' to stop\n")
    
    demo = EmergencyClassifierDemo()
    
    # Example sentences
    examples = [
        "Help someone is attacking me with knife",
        "Fire in the building smoke everywhere",
        "Medical emergency heart attack symptoms",
        "Man following me feeling unsafe",
        "Landslide blocked road people trapped"
    ]
    
    print("💡 Example sentences you can try:")
    for i, ex in enumerate(examples, 1):
        print(f"   {i}. {ex}")
    print()
    
    while True:
        try:
            # Get user input
            text = input("Enter emergency text (or 'quit' to exit): ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not text:
                print("⚠️  Please enter some text.\n")
                continue
            
            # Predict
            result = demo.predict(text)
            
            # Display results
            demo.display_result(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()

