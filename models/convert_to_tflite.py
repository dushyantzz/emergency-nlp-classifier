import os
import json
import numpy as np
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import shutil


CONFIG = {
    'model_dir': 'outputs/trained_model',
    'output_dir': 'outputs',
    'tflite_filename': 'emergency_classifier.tflite',
    'max_length': 128,
    'quantization': 'INT8',  # Options: INT8, FP16, DYNAMIC, NONE
}


class TFLiteConverter:
    """Converter for DistilBERT to TFLite"""
    
    def __init__(self, config=CONFIG):
        self.config = config
        
        if not os.path.exists(config['model_dir']):
            raise FileNotFoundError(
                f"Model directory not found: {config['model_dir']}. "
                "Run 'python models/train_model.py' first."
            )
        
        print("🔄 TFLite Converter Initialized")
        print(f"Model directory: {config['model_dir']}")
        print(f"Quantization: {config['quantization']}")
    
    def load_tensorflow_model(self):
        """Load trained TensorFlow model"""
        print("\n📂 Loading TensorFlow model...")
        
        # Load label mapping
        with open(f"{self.config['model_dir']}/label_mapping.json", 'r') as f:
            label_config = json.load(f)
        
        num_labels = label_config['num_labels']
        print(f"Number of labels: {num_labels}")
        
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(self.config['model_dir'])
        
        # Load TensorFlow model
        print("\n📥 Loading TensorFlow model...")
        model = TFDistilBertForSequenceClassification.from_pretrained(
            self.config['model_dir'],
            num_labels=num_labels
        )
        
        print("✅ Model loaded successfully")
        
        return model, tokenizer, label_config
    
    def create_concrete_function(self, model):
        """Create concrete function for TFLite conversion"""
        print("\n🔨 Creating concrete function...")
        
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[1, self.config['max_length']], dtype=tf.int32, name='input_ids'),
                tf.TensorSpec(shape=[1, self.config['max_length']], dtype=tf.int32, name='attention_mask')
            ]
        )
        def serving_fn(input_ids, attention_mask):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                training=False
            )
            return {'logits': outputs.logits}
        
        concrete_func = serving_fn.get_concrete_function()
        print("✅ Concrete function created")
        
        return concrete_func
    
    def get_representative_dataset(self, tokenizer):
        """Generate representative dataset for quantization"""
        print("\n📊 Generating representative dataset for quantization...")
        
        sample_texts = [
            "Help someone is attacking me with knife",
            "Fire in the building smoke everywhere",
            "Medical emergency heart attack symptoms",
            "Man following me feeling unsafe",
            "Landslide blocked road people trapped",
            "Robbery happening near hotel help",
            "Gas leak detected evacuate immediately",
            "Tourist fell from cliff severe injury",
            "Harassment at tourist spot need help",
            "Flood water rising fast emergency",
            "Someone stole my wallet and phone",
            "Kitchen fire spreading rapidly",
            "Accident multiple people injured",
            "Being stalked need women helpline",
            "Earthquake felt building shaking",
            "Violence happening send police",
            "Smoke coming from forest fire",
            "Breathing difficulty need ambulance",
            "Inappropriate behavior by group",
            "Road washed away natural disaster"
        ]
        
        def representative_dataset_gen():
            for text in sample_texts:
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config['max_length'],
                    return_tensors='tf'
                )
                
                input_ids = tf.cast(encoding['input_ids'], tf.int32)
                attention_mask = tf.cast(encoding['attention_mask'], tf.int32)
                
                yield [input_ids, attention_mask]
        
        print(f"Generated {len(sample_texts)} representative samples")
        return representative_dataset_gen
    
    def convert_to_tflite(self, concrete_func, tokenizer):
        """Convert to TensorFlow Lite with optimizations"""
        print(f"\n🚀 Converting to TFLite ({self.config['quantization']} quantization)...")
        
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        
        # Apply optimizations based on config
        if self.config['quantization'] == 'INT8':
            print("Applying INT8 quantization (smallest size, best for mobile)...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.get_representative_dataset(tokenizer)
            # Use TFLITE_BUILTINS instead of TFLITE_BUILTINS_INT8 to allow int32 inputs
            # This quantizes weights but keeps I/O types flexible
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            # Don't set inference types - allows int32 inputs (token IDs) and float32 outputs
            
        elif self.config['quantization'] == 'FP16':
            print("Applying FP16 quantization (balanced size/accuracy)...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            
        elif self.config['quantization'] == 'DYNAMIC':
            print("Applying dynamic range quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            
        else:
            print("No quantization applied (largest size, best accuracy)...")
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
        
        # Convert
        tflite_model = converter.convert()
        
        print("✅ Conversion successful!")
        
        return tflite_model
    
    def save_model(self, tflite_model, tokenizer, label_config):
        """Save TFLite model and supporting files"""
        print(f"\n💾 Saving model to {self.config['output_dir']}...")
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Save TFLite model
        tflite_path = os.path.join(
            self.config['output_dir'],
            self.config['tflite_filename']
        )
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✅ TFLite model saved: {tflite_path}")
        print(f"   Size: {model_size_mb:.2f} MB")
        
        # Save vocabulary and all tokenizer files needed for Android
        print("\n📦 Copying tokenizer files for Android deployment...")
        
        # Save vocabulary
        vocab_path = os.path.join(self.config['output_dir'], 'vocab.txt')
        tokenizer.save_vocabulary(self.config['output_dir'])
        if os.path.exists(os.path.join(self.config['output_dir'], 'vocab.txt')):
            print(f"✅ Vocabulary saved: {vocab_path}")
            print("   Note: '[unused]' tokens are normal for DistilBERT - they are placeholder tokens")
        
        # Copy tokenizer configuration files
        tokenizer_files = [
            'tokenizer_config.json',
            'special_tokens_map.json'
        ]
        
        # Check if tokenizer.json exists (HuggingFace format)
        tokenizer_json_path = os.path.join(self.config['model_dir'], 'tokenizer.json')
        if os.path.exists(tokenizer_json_path):
            tokenizer_files.append('tokenizer.json')
        
        for filename in tokenizer_files:
            src_path = os.path.join(self.config['model_dir'], filename)
            dst_path = os.path.join(self.config['output_dir'], filename)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"✅ {filename} copied to outputs/")
            else:
                print(f"⚠️  {filename} not found in model directory")
        
        # Save label mapping
        label_path = os.path.join(self.config['output_dir'], 'label_mapping.json')
        with open(label_path, 'w') as f:
            json.dump(label_config, f, indent=2)
        print(f"✅ Label mapping saved: {label_path}")
        
        # Create deployment info
        deployment_info = {
            'model_name': 'emergency_classifier',
            'version': '1.0.0',
            'framework': 'TensorFlow Lite',
            'quantization': self.config['quantization'],
            'input_shape': [1, self.config['max_length']],
            'output_shape': [1, label_config['num_labels']],
            'max_sequence_length': self.config['max_length'],
            'model_size_mb': round(model_size_mb, 2),
            'categories': list(label_config['label_map'].keys()),
            'usage': {
                'input_format': 'Text string (2-3 sentences)',
                'output_format': 'Logits array (5 classes)',
                'categories': label_config['label_map'],
                'preprocessing': 'Tokenize with vocab.txt, pad to max_length=128',
                'postprocessing': 'Apply softmax to logits, argmax for prediction'
            }
        }
        
        deployment_path = os.path.join(self.config['output_dir'], 'deployment_info.json')
        with open(deployment_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        print(f"✅ Deployment info saved: {deployment_path}")
        
        return tflite_path, model_size_mb
    
    def test_tflite_model(self, tflite_path, tokenizer):
        """Test the converted TFLite model"""
        print("\n🧪 Testing TFLite model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\nInput details: {len(input_details)} inputs")
        for inp in input_details:
            print(f"  - {inp['name']}: shape {inp['shape']}, dtype {inp['dtype']}")
        
        print(f"\nOutput details: {len(output_details)} outputs")
        for out in output_details:
            print(f"  - {out['name']}: shape {out['shape']}, dtype {out['dtype']}")
        
        # Test with sample text
        test_text = "Help someone is attacking me urgent"
        print(f"\nTest input: '{test_text}'")
        
        encoding = tokenizer(
            test_text,
            truncation=True,
            padding='max_length',
            max_length=self.config['max_length'],
            return_tensors='np'
        )
        
        # Set input tensors
        interpreter.set_tensor(input_details[0]['index'], encoding['input_ids'].astype(np.int32))
        interpreter.set_tensor(input_details[1]['index'], encoding['attention_mask'].astype(np.int32))
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        logits = interpreter.get_tensor(output_details[0]['index'])
        
        # Apply softmax
        probs = tf.nn.softmax(logits[0]).numpy()
        predicted_idx = np.argmax(probs)
        
        categories = ['police', 'fire', 'ambulance', 'women_helpline', 'disaster']
        
        print(f"\n🎯 Prediction: {categories[predicted_idx]}")
        print(f"Confidence: {probs[predicted_idx]*100:.2f}%")
        print(f"\nAll probabilities:")
        for i, cat in enumerate(categories):
            print(f"  {cat}: {probs[i]*100:.2f}%")
        
        print("\n✅ TFLite model test successful!")


def main():
    """Main execution"""
    print("🔧 TensorFlow Lite Converter - SIH 2025")
    print("="*80)
    
    # Initialize converter
    converter = TFLiteConverter()
    
    # Load model
    model, tokenizer, label_config = converter.load_tensorflow_model()
    
    # Create concrete function
    concrete_func = converter.create_concrete_function(model)
    
    # Convert to TFLite
    tflite_model = converter.convert_to_tflite(concrete_func, tokenizer)
    
    # Save model
    tflite_path, model_size = converter.save_model(tflite_model, tokenizer, label_config)
    
    # Test model
    converter.test_tflite_model(tflite_path, tokenizer)
    
        # Create Android deployment file list
        android_files_path = os.path.join(self.config['output_dir'], 'ANDROID_FILES_LIST.txt')
        with open(android_files_path, 'w') as f:
            f.write("Files Required for Android Deployment\n")
            f.write("=" * 50 + "\n\n")
            f.write("Copy these files from outputs/ to app/src/main/assets/:\n\n")
            f.write("1. emergency_classifier.tflite (TFLite model)\n")
            f.write("2. vocab.txt (Vocabulary - includes [unused] tokens - this is normal!)\n")
            f.write("3. tokenizer_config.json (Tokenizer configuration)\n")
            f.write("4. special_tokens_map.json (Special tokens mapping)\n")
            f.write("5. label_mapping.json (Label to ID mapping)\n")
            f.write("\nOptional:\n")
            f.write("- deployment_info.json (Model metadata)\n")
            f.write("- ANDROID_DEPLOYMENT_GUIDE.md (Detailed integration guide)\n")
            f.write("\n" + "=" * 50 + "\n")
            f.write("IMPORTANT: The '[unused]' tokens in vocab.txt are NORMAL!\n")
            f.write("They are part of DistilBERT vocabulary - DO NOT remove them.\n")
        
        print("\n" + "="*80)
        print("✨ Conversion pipeline complete!")
        print(f"📦 Model ready for deployment: {tflite_path}")
        print(f"📊 Final model size: {model_size:.2f} MB")
        print("\n📱 Android Deployment Files:")
        print("   ✅ emergency_classifier.tflite")
        print("   ✅ vocab.txt (Note: '[unused]' tokens are normal - part of DistilBERT)")
        print("   ✅ tokenizer_config.json")
        print("   ✅ special_tokens_map.json")
        print("   ✅ label_mapping.json")
        print("\n📖 See ANDROID_DEPLOYMENT_GUIDE.md for detailed integration instructions")
        print("\n👉 Next steps:")
        print("   1. Copy all files from outputs/ to your Android app's assets/ folder")
        print("   2. Read ANDROID_DEPLOYMENT_GUIDE.md for integration code")
        print("   3. Integrate TFLite interpreter in Kotlin code")
        print("   4. Test on mobile device")
        print("="*80)


if __name__ == "__main__":
    main()