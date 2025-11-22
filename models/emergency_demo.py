import os
import json
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification


class EmergencyResponseDemo:
    """Interactive demo for emergency classification with resource recommendations"""
    
    def __init__(self):
        self.model_dir = 'outputs/trained_model'
        
        print("🔄 Loading emergency classification model...")
        
        # Load label mapping
        with open(f"{self.model_dir}/label_mapping.json", 'r') as f:
            label_config = json.load(f)
        self.id_to_label = {int(k): v for k, v in label_config['id_to_label'].items()}
        self.label_names = list(self.id_to_label.values())
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
        
        # Load model (lazy loading)
        self.model = None
        
        # Resource mapping for each category
        self.resource_mapping = {
            'police': {
                'primary': '🚔 Police Department',
                'phone': '100 / 112',
                'resources': ['Police', 'Law Enforcement', 'Security'],
                'actions': [
                    'Contact local police station immediately',
                    'Provide exact location and situation details',
                    'Stay in safe location if possible',
                    'Do not approach dangerous situations'
                ],
                'description': 'Criminal activities, threats, violence, theft, robbery, assault'
            },
            'fire': {
                'primary': '🔥 Fire Department',
                'phone': '101 / 112',
                'resources': ['Fire Department', 'Fire Brigade', 'Emergency Services'],
                'actions': [
                    'Call fire department immediately',
                    'Evacuate the area if safe to do so',
                    'Do not use elevators during fire',
                    'Alert nearby people about the fire',
                    'Close doors to contain fire if possible'
                ],
                'description': 'Fires, smoke, gas leaks, explosions, burning buildings'
            },
            'ambulance': {
                'primary': '🚑 Ambulance / Medical Emergency',
                'phone': '102 / 112',
                'resources': ['Ambulance', 'Medical Emergency Services', 'Hospital'],
                'actions': [
                    'Call ambulance/medical emergency immediately',
                    'Provide patient condition and location',
                    'Do not move injured person unless in immediate danger',
                    'Apply first aid if trained',
                    'Keep patient calm and comfortable'
                ],
                'description': 'Medical emergencies, injuries, accidents, health crises'
            },
            'women_helpline': {
                'primary': '🛡️ Women Helpline',
                'phone': '1091 / 112',
                'resources': ['Women Helpline', 'Police', 'Support Services'],
                'actions': [
                    'Call women helpline immediately',
                    'Contact police if in immediate danger',
                    'Move to a safe location if possible',
                    'Document incident if safe to do so',
                    'Seek support from trusted individuals'
                ],
                'description': 'Harassment, stalking, domestic violence, safety concerns for women'
            },
            'disaster': {
                'primary': '🌊 Disaster Management',
                'phone': '108 / 112',
                'resources': ['Disaster Management', 'NDRF', 'Emergency Services', 'Police'],
                'actions': [
                    'Contact disaster management helpline',
                    'Follow evacuation procedures if issued',
                    'Move to higher ground if flooding',
                    'Stay away from damaged structures',
                    'Listen to official announcements'
                ],
                'description': 'Natural disasters, floods, earthquakes, landslides, cyclones'
            }
        }
        
        print("✅ Emergency Response System Ready!\n")
    
    def load_model(self):
        """Load model on first use"""
        if self.model is None:
            print("📥 Loading AI model (first time only, please wait)...")
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
    
    def display_response(self, result):
        """Display emergency response with resource recommendations"""
        category = result['predicted_category']
        resources = self.resource_mapping[category]
        
        print("\n" + "="*80)
        print("🚨 EMERGENCY CLASSIFICATION RESULT")
        print("="*80)
        print(f"\n📝 Input Text:")
        print(f"   \"{result['text']}\"")
        
        print(f"\n🎯 Detected Emergency Type: {category.upper().replace('_', ' ')}")
        print(f"   Confidence: {result['confidence']*100:.2f}%")
        
        print("\n" + "="*80)
        print("📞 IMMEDIATE ACTION REQUIRED")
        print("="*80)
        print(f"\n{resources['primary']}")
        print(f"📱 Emergency Number: {resources['phone']}")
        print(f"   (Dial 112 for universal emergency services)")
        
        print(f"\n📋 Description:")
        print(f"   {resources['description']}")
        
        print(f"\n🔧 Resources to Inform:")
        for i, resource in enumerate(resources['resources'], 1):
            print(f"   {i}. {resource}")
        
        print(f"\n⚡ Recommended Actions:")
        for i, action in enumerate(resources['actions'], 1):
            print(f"   {i}. {action}")
        
        print("\n" + "="*80)
        print("📊 ALL CATEGORY PROBABILITIES")
        print("="*80)
        for label in self.label_names:
            prob = result['all_probabilities'][label]
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            marker = "👉" if label == category else "  "
            print(f"{marker} {label:20s} {prob*100:6.2f}% |{bar}|")
        
        print("="*80 + "\n")
    
    def show_quick_reference(self):
        """Show quick reference for all emergency types"""
        print("\n" + "="*80)
        print("📋 QUICK REFERENCE - ALL EMERGENCY TYPES")
        print("="*80)
        for category, resources in self.resource_mapping.items():
            print(f"\n{resources['primary']}")
            print(f"   Phone: {resources['phone']}")
            print(f"   Type: {category.replace('_', ' ').title()}")
        print("="*80 + "\n")


def main():
    """Main interactive demo"""
    print("🚨 Emergency Response System - AI-Powered Classification")
    print("="*80)
    print("This system analyzes emergency text and recommends which resources to contact.")
    print("Categories: Police | Fire | Ambulance | Women Helpline | Disaster")
    print("\nType 'quit' or 'exit' to stop")
    print("Type 'help' to see all emergency numbers")
    print("="*80 + "\n")
    
    demo = EmergencyResponseDemo()
    
    # Example emergency scenarios
    examples = [
        "Help someone is attacking me with knife urgent",
        "Fire in the building smoke everywhere need help",
        "Medical emergency heart attack symptoms chest pain",
        "Man stalking me following me feeling very unsafe",
        "Landslide blocked road multiple people trapped need rescue",
        "Robbery happening near market send police immediately",
        "Gas leak detected strong smell evacuate immediately",
        "Tourist fell from cliff severe injury bleeding",
        "Being harassed by group of men need help",
        "Flood water rising fast emergency evacuation needed"
    ]
    
    print("💡 Example Emergency Scenarios:")
    for i, ex in enumerate(examples, 1):
        print(f"   {i:2d}. {ex}")
    print()
    
    while True:
        try:
            # Get user input
            text = input("\n🔍 Enter emergency description (or 'quit'/'help'): ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Stay safe! Goodbye!")
                break
            
            if text.lower() in ['help', 'h']:
                demo.show_quick_reference()
                continue
            
            if not text:
                print("⚠️  Please enter an emergency description.\n")
                continue
            
            # Predict
            print("\n🔄 Analyzing emergency...")
            result = demo.predict(text)
            
            # Display response
            demo.display_response(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Stay safe! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or check if the model files are available.\n")


if __name__ == "__main__":
    main()

