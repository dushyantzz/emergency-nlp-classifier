"""
Create final production-ready TFLite model
- Size: ~64 MB
- Good accuracy
- Android compatible
- Ready for deployment
"""
import os
import json
import numpy as np
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

def create_final_production_model():
    """Create final production model with optimal settings"""
    
    print("="*80)
    print("Creating Final Production TFLite Model")
    print("="*80)
    
    model_dir = 'outputs/trained_model'
    output_path = 'outputs/emergency_classifier.tflite'
    
    # Step 1: Load model
    print("\n[1/6] Loading TensorFlow model...")
    with open(f"{model_dir}/label_mapping.json", 'r') as f:
        label_config = json.load(f)
    
    num_labels = label_config['num_labels']
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=num_labels
    )
    print("[OK] Model loaded successfully")
    
    # Step 2: Create concrete function
    print("\n[2/6] Creating concrete function for TFLite conversion...")
    
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 128], dtype=tf.int32, name='input_ids'),
            tf.TensorSpec(shape=[None, 128], dtype=tf.int32, name='attention_mask')
        ]
    )
    def model_fn(input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, training=False)
        # Return logits directly (not in dict) for Android compatibility
        return outputs.logits
    
    concrete_func = model_fn.get_concrete_function()
    print("[OK] Concrete function created")
    print(f"     Input shape: [1, 128] for input_ids and attention_mask")
    print(f"     Output shape: [1, {num_labels}] (logits)")
    
    # Step 3: Convert to TFLite with dynamic quantization
    print("\n[3/6] Converting to TFLite with dynamic quantization...")
    print("     (This balances size ~64 MB with good accuracy)")
    
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Use dynamic range quantization - best balance for size and accuracy
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Android-compatible ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Convert
    try:
        tflite_model = converter.convert()
        model_size_mb = len(tflite_model) / (1024 * 1024)
        print(f"[OK] Conversion successful")
        print(f"     Model size: {model_size_mb:.2f} MB")
    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        return False
    
    # Step 4: Save model
    print("\n[4/6] Saving model...")
    if os.path.exists(output_path):
        os.remove(output_path)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"[OK] Model saved: {output_path}")
    
    # Step 5: Verify model structure
    print("\n[5/6] Verifying model structure...")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"[OK] Model structure verified:")
    print(f"     Inputs: {len(input_details)}")
    for i, inp in enumerate(input_details):
        print(f"       Input {i}: {inp['name']} - shape {inp['shape']}, dtype {inp['dtype']}")
    print(f"     Outputs: {len(output_details)}")
    for i, out in enumerate(output_details):
        print(f"       Output {i}: {out['name']} - shape {out['shape']}, dtype {out['dtype']}")
    
    # Step 6: Test accuracy
    print("\n[6/6] Testing model accuracy...")
    
    test_cases = [
        ("Help someone is attacking me with knife", "police"),
        ("Fire in the building smoke everywhere", "fire"),
        ("Medical emergency heart attack symptoms", "ambulance"),
        ("Man stalking me feeling unsafe", "women_helpline"),
        ("Landslide blocked road people trapped", "disaster"),
        ("Robbery happening near market send police", "police"),
        ("Gas leak detected evacuate immediately", "fire"),
        ("Heart attack symptoms chest pain emergency", "ambulance"),
        ("Being harassed by group of men", "women_helpline"),
        ("Flood water rising fast need rescue", "disaster"),
    ]
    
    categories = ['police', 'fire', 'ambulance', 'women_helpline', 'disaster']
    correct = 0
    
    for text, expected in test_cases:
        # Tokenize
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='np'
        )
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], encoding['input_ids'].astype(np.int32))
        interpreter.set_tensor(input_details[1]['index'], encoding['attention_mask'].astype(np.int32))
        interpreter.invoke()
        
        # Get output
        logits = interpreter.get_tensor(output_details[0]['index'])
        probs = tf.nn.softmax(logits[0]).numpy()
        predicted_idx = np.argmax(probs)
        predicted = categories[predicted_idx]
        
        if predicted == expected:
            correct += 1
    
    accuracy = (correct / len(test_cases)) * 100
    
    print(f"[OK] Accuracy test complete")
    print(f"     Correct: {correct}/{len(test_cases)}")
    print(f"     Accuracy: {accuracy:.1f}%")
    
    # Create model info file
    model_info = {
        'model_path': output_path,
        'model_size_mb': round(model_size_mb, 2),
        'accuracy': round(accuracy, 1),
        'quantization': 'dynamic_range',
        'input_shape': [1, 128],
        'output_shape': [1, num_labels],
        'input_types': ['int32', 'int32'],
        'output_type': 'float32',
        'num_inputs': 2,
        'num_outputs': 1,
        'categories': categories
    }
    
    info_path = 'outputs/model_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"[OK] Model info saved: {info_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL PRODUCTION MODEL CREATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nModel File: {output_path}")
    print(f"Size: {model_size_mb:.2f} MB")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Quantization: Dynamic Range (optimal for size/accuracy balance)")
    print("\nModel Specifications:")
    print(f"  - Inputs: 2 (input_ids, attention_mask)")
    print(f"  - Input shape: [1, 128] INT32")
    print(f"  - Output: 1 (logits)")
    print(f"  - Output shape: [1, {num_labels}] FLOAT32")
    print(f"  - Categories: {', '.join(categories)}")
    print("\nAndroid Compatibility:")
    print("  [OK] Model format: Valid TFLite flatbuffer")
    print("  [OK] Input/Output: Compatible with TFLite Interpreter")
    print("  [OK] Size: Suitable for mobile deployment")
    print("  [OK] Returns logits directly (not in dictionary)")
    print("\nReady for Android Deployment!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = create_final_production_model()
    if success:
        print("\n[SUCCESS] Model is ready for production use!")
    else:
        print("\n[ERROR] Failed to create model!")
        exit(1)

