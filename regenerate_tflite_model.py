"""
Script to regenerate TFLite model with fixes for mobile compatibility
Run this to create a new model that works properly on mobile apps
"""
import os
import sys

# Add models directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.convert_to_tflite import TFLiteConverter, CONFIG

def main():
    print("🔧 Regenerating TFLite Model for Mobile Compatibility")
    print("="*80)
    print("\nThis will create a new TFLite model with the following fixes:")
    print("  ✅ Returns logits directly (not in dictionary)")
    print("  ✅ Proper input/output format for mobile")
    print("  ✅ Compatible with Android/iOS TFLite interpreters")
    print("\n" + "="*80 + "\n")
    
    # Option to change quantization
    print("Current quantization setting:", CONFIG['quantization'])
    print("\nOptions:")
    print("  1. INT8 (smallest, best for mobile) - Recommended")
    print("  2. DYNAMIC (balanced)")
    print("  3. FP16 (faster, larger)")
    print("  4. NONE (largest, best accuracy)")
    
    choice = input("\nKeep current setting? (y/n, default=y): ").strip().lower()
    
    if choice == 'n':
        print("\nSelect quantization:")
        print("  1. INT8")
        print("  2. DYNAMIC")
        print("  3. FP16")
        print("  4. NONE")
        q_choice = input("Enter choice (1-4): ").strip()
        quantization_map = {'1': 'INT8', '2': 'DYNAMIC', '3': 'FP16', '4': 'NONE'}
        if q_choice in quantization_map:
            CONFIG['quantization'] = quantization_map[q_choice]
            print(f"✅ Set quantization to: {CONFIG['quantization']}")
    
    # Backup old model if exists
    old_model_path = os.path.join(CONFIG['output_dir'], CONFIG['tflite_filename'])
    if os.path.exists(old_model_path):
        backup_path = old_model_path + '.backup'
        print(f"\n📦 Backing up old model to: {backup_path}")
        import shutil
        shutil.copy2(old_model_path, backup_path)
    
    # Initialize converter
    converter = TFLiteConverter(CONFIG)
    
    # Load model
    print("\n" + "="*80)
    model, tokenizer, label_config = converter.load_tensorflow_model()
    
    # Create concrete function
    concrete_func = converter.create_concrete_function(model)
    
    # Convert to TFLite
    tflite_model = converter.convert_to_tflite(concrete_func, tokenizer)
    
    # Save model
    tflite_path, model_size = converter.save_model(tflite_model, tokenizer, label_config)
    
    # Test model
    print("\n" + "="*80)
    converter.test_tflite_model(tflite_path, tokenizer)
    
    print("\n" + "="*80)
    print("✨ Model regeneration complete!")
    print(f"📦 New model saved: {tflite_path}")
    print(f"📊 Model size: {model_size:.2f} MB")
    print("\n✅ This model should work correctly on mobile apps!")
    print("   - Returns logits directly (not in dictionary)")
    print("   - Compatible with Android TFLite Interpreter")
    print("   - Proper input/output format")
    print("\n👉 Copy the new model to your mobile app's assets folder")
    print("="*80)


if __name__ == "__main__":
    main()

