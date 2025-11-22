"""
Comprehensive verification of all files needed for Android implementation
"""
import os
import json
import tensorflow as tf
from transformers import DistilBertTokenizer

def verify_all_files():
    """Verify all files needed for Android implementation"""
    
    print("="*80)
    print("Comprehensive File Verification for Android Implementation")
    print("="*80)
    
    all_ok = True
    issues = []
    
    # 1. Verify TFLite model
    print("\n[1/6] Verifying TFLite Model...")
    model_path = 'outputs/emergency_classifier.tflite'
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        all_ok = False
        issues.append("TFLite model file missing")
    else:
        try:
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"[OK] Model file exists: {model_path}")
            print(f"     Size: {file_size:.2f} MB")
            
            # Verify model structure
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"[OK] Model structure valid:")
            print(f"     Inputs: {len(input_details)}")
            for inp in input_details:
                print(f"       - {inp['name']}: shape {inp['shape']}, dtype {inp['dtype']}")
            print(f"     Outputs: {len(output_details)}")
            for out in output_details:
                print(f"       - {out['name']}: shape {out['shape']}, dtype {out['dtype']}")
            
            # Test inference
            test_input = tf.ones((1, 128), dtype=tf.int32)
            interpreter.set_tensor(input_details[0]['index'], test_input.numpy())
            interpreter.set_tensor(input_details[1]['index'], test_input.numpy())
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            if output.shape == (1, 5):
                print(f"[OK] Model inference works correctly")
                print(f"     Output shape: {output.shape}")
            else:
                print(f"[ERROR] Wrong output shape: {output.shape}")
                all_ok = False
                issues.append("Model output shape incorrect")
                
        except Exception as e:
            print(f"[ERROR] Model verification failed: {e}")
            all_ok = False
            issues.append(f"Model verification error: {e}")
    
    # 2. Verify vocab.txt
    print("\n[2/6] Verifying vocab.txt...")
    vocab_path = 'outputs/vocab.txt'
    
    if not os.path.exists(vocab_path):
        print(f"[ERROR] vocab.txt not found: {vocab_path}")
        all_ok = False
        issues.append("vocab.txt missing")
    else:
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_lines = f.readlines()
            
            vocab_size = len(vocab_lines)
            print(f"[OK] vocab.txt exists")
            print(f"     Total tokens: {vocab_size}")
            
            # Check for special tokens
            vocab_list = [line.strip() for line in vocab_lines]
            special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
            found_special = []
            
            for token in special_tokens:
                if token in vocab_list:
                    idx = vocab_list.index(token)
                    found_special.append(f"{token} (index {idx})")
            
            print(f"[OK] Special tokens found: {', '.join(found_special)}")
            
            # Check for [unused] tokens (should be present)
            unused_count = sum(1 for line in vocab_lines if line.strip().startswith('[unused'))
            if unused_count > 0:
                print(f"[OK] [unused] tokens present: {unused_count} (this is normal)")
            
            if vocab_size < 1000:
                print(f"[WARNING] Vocabulary seems too small")
            else:
                print(f"[OK] Vocabulary size is reasonable")
                
        except Exception as e:
            print(f"[ERROR] vocab.txt verification failed: {e}")
            all_ok = False
            issues.append(f"vocab.txt error: {e}")
    
    # 3. Verify label_mapping.json
    print("\n[3/6] Verifying label_mapping.json...")
    label_path = 'outputs/label_mapping.json'
    
    if not os.path.exists(label_path):
        print(f"[ERROR] label_mapping.json not found")
        all_ok = False
        issues.append("label_mapping.json missing")
    else:
        try:
            with open(label_path, 'r') as f:
                label_config = json.load(f)
            
            required_keys = ['label_map', 'id_to_label', 'num_labels']
            missing_keys = [k for k in required_keys if k not in label_config]
            
            if missing_keys:
                print(f"[ERROR] Missing keys: {missing_keys}")
                all_ok = False
                issues.append("label_mapping.json incomplete")
            else:
                print(f"[OK] label_mapping.json valid")
                print(f"     Number of labels: {label_config['num_labels']}")
                print(f"     Labels: {list(label_config['label_map'].keys())}")
                
                # Verify mapping consistency
                label_map = label_config['label_map']
                id_to_label = {int(k): v for k, v in label_config['id_to_label'].items()}
                
                consistent = True
                for label, label_id in label_map.items():
                    if id_to_label.get(label_id) != label:
                        consistent = False
                        break
                
                if consistent:
                    print(f"[OK] Label mappings are consistent")
                else:
                    print(f"[ERROR] Label mappings are inconsistent")
                    all_ok = False
                    issues.append("Label mapping inconsistency")
                    
        except Exception as e:
            print(f"[ERROR] label_mapping.json verification failed: {e}")
            all_ok = False
            issues.append(f"label_mapping.json error: {e}")
    
    # 4. Verify tokenizer_config.json
    print("\n[4/6] Verifying tokenizer_config.json...")
    tokenizer_config_path = 'outputs/tokenizer_config.json'
    
    if not os.path.exists(tokenizer_config_path):
        print(f"[ERROR] tokenizer_config.json not found")
        all_ok = False
        issues.append("tokenizer_config.json missing")
    else:
        try:
            with open(tokenizer_config_path, 'r') as f:
                tokenizer_config = json.load(f)
            
            required_keys = ['tokenizer_class', 'do_lower_case', 'model_max_length', 'cls_token', 'sep_token', 'pad_token', 'unk_token']
            missing_keys = [k for k in required_keys if k not in tokenizer_config]
            
            if missing_keys:
                print(f"[WARNING] Missing keys: {missing_keys}")
            else:
                print(f"[OK] tokenizer_config.json valid")
                print(f"     Tokenizer class: {tokenizer_config['tokenizer_class']}")
                print(f"     Max length: {tokenizer_config['model_max_length']}")
                print(f"     Lowercase: {tokenizer_config['do_lower_case']}")
                
        except Exception as e:
            print(f"[ERROR] tokenizer_config.json verification failed: {e}")
            all_ok = False
            issues.append(f"tokenizer_config.json error: {e}")
    
    # 5. Verify special_tokens_map.json
    print("\n[5/6] Verifying special_tokens_map.json...")
    special_tokens_path = 'outputs/special_tokens_map.json'
    
    if not os.path.exists(special_tokens_path):
        print(f"[ERROR] special_tokens_map.json not found")
        all_ok = False
        issues.append("special_tokens_map.json missing")
    else:
        try:
            with open(special_tokens_path, 'r') as f:
                special_tokens = json.load(f)
            
            required_tokens = ['cls_token', 'sep_token', 'pad_token', 'unk_token']
            missing_tokens = [t for t in required_tokens if t not in special_tokens]
            
            if missing_tokens:
                print(f"[ERROR] Missing special tokens: {missing_tokens}")
                all_ok = False
                issues.append("special_tokens_map.json incomplete")
            else:
                print(f"[OK] special_tokens_map.json valid")
                print(f"     Special tokens: {list(special_tokens.keys())}")
                
        except Exception as e:
            print(f"[ERROR] special_tokens_map.json verification failed: {e}")
            all_ok = False
            issues.append(f"special_tokens_map.json error: {e}")
    
    # 6. Test tokenizer with vocab
    print("\n[6/6] Testing tokenizer compatibility...")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('outputs/trained_model')
        
        # Test tokenization
        test_text = "Help fire in building"
        encoding = tokenizer(
            test_text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='np'
        )
        
        if 'input_ids' in encoding and 'attention_mask' in encoding:
            print(f"[OK] Tokenizer works correctly")
            print(f"     Test input: '{test_text}'")
            print(f"     Tokenized length: {len(encoding['input_ids'][0])}")
            print(f"     Input IDs shape: {encoding['input_ids'].shape}")
            print(f"     Attention mask shape: {encoding['attention_mask'].shape}")
        else:
            print(f"[ERROR] Tokenizer output incomplete")
            all_ok = False
            issues.append("Tokenizer output incomplete")
            
    except Exception as e:
        print(f"[ERROR] Tokenizer test failed: {e}")
        all_ok = False
        issues.append(f"Tokenizer error: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    if all_ok:
        print("\n[SUCCESS] All files are valid and ready for Android implementation!")
        print("\nFiles verified:")
        print("  [OK] emergency_classifier.tflite - Model file")
        print("  [OK] vocab.txt - Vocabulary file")
        print("  [OK] label_mapping.json - Label mappings")
        print("  [OK] tokenizer_config.json - Tokenizer configuration")
        print("  [OK] special_tokens_map.json - Special tokens")
        print("  [OK] Tokenizer compatibility - Working correctly")
        
        print("\n" + "="*80)
        print("READY FOR ANDROID DEPLOYMENT!")
        print("="*80)
        print("\nNext steps for Android developer:")
        print("  1. Copy all files from outputs/ to app/src/main/assets/")
        print("  2. Use MappedByteBuffer to load the TFLite model")
        print("  3. Load vocab.txt for tokenization")
        print("  4. Use label_mapping.json for category mapping")
        print("  5. Follow ANDROID_DEPLOYMENT_GUIDE.md for integration")
        print("="*80)
        
    else:
        print("\n[ERROR] Some issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before deployment.")
        print("="*80)
    
    return all_ok

if __name__ == "__main__":
    success = verify_all_files()
    exit(0 if success else 1)

