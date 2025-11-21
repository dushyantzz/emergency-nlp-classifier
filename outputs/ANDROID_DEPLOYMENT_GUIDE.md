# Android Deployment Guide

## 📦 Required Files for Android App

Copy these files from `outputs/` to your Android app's `assets/` folder:

### Essential Files:
1. **`emergency_classifier.tflite`** - The TensorFlow Lite model
2. **`vocab.txt`** - Vocabulary file for tokenization
3. **`tokenizer_config.json`** - Tokenizer configuration
4. **`special_tokens_map.json`** - Special tokens mapping
5. **`label_mapping.json`** - Label to ID mapping

### Optional but Recommended:
6. **`deployment_info.json`** - Model metadata and usage info

---

## 📝 Understanding vocab.txt

### About "[unused]" Tokens

The `vocab.txt` file contains entries like:
```
[PAD]
[unused0]
[unused1]
[unused2]
...
```

**This is NORMAL and EXPECTED!** ✅

- These `[unused]` tokens are **placeholder tokens** that are part of the DistilBERT vocabulary
- They are reserved for potential future use or fine-tuning
- **You should NOT remove or modify them**
- The tokenizer will work correctly with these tokens present
- Your Android app should read the entire vocab.txt file as-is

### How to Use vocab.txt in Android

1. **Read the file line by line** - each line is a token
2. **Create a mapping**: `token → index` (line number - 1, since indexing starts at 0)
3. **Use this mapping** to convert words to token IDs

Example:
- Line 1: `[PAD]` → token ID = 0
- Line 2: `[unused0]` → token ID = 1
- Line 3: `[unused1]` → token ID = 2
- ... and so on

---

## 🔧 Tokenizer Files Explained

### 1. `vocab.txt`
- Contains all vocabulary tokens (one per line)
- Used to map words to integer IDs
- **Total: 30,523 tokens** (including unused tokens)

### 2. `tokenizer_config.json`
Contains tokenizer settings:
- `do_lower_case: true` - Convert text to lowercase
- `model_max_length: 512` - Maximum sequence length
- `cls_token: "[CLS]"` - Classification token
- `sep_token: "[SEP]"` - Separator token
- `pad_token: "[PAD]"` - Padding token
- `unk_token: "[UNK]"` - Unknown token

### 3. `special_tokens_map.json`
Maps special token names to their string values:
- `cls_token: "[CLS]"`
- `sep_token: "[SEP]"`
- `pad_token: "[PAD]"`
- `unk_token: "[UNK]"`
- `mask_token: "[MASK]"`

### 4. `label_mapping.json`
Maps emergency categories to IDs:
```json
{
  "label_map": {
    "police": 0,
    "fire": 1,
    "ambulance": 2,
    "women_helpline": 3,
    "disaster": 4
  }
}
```

---

## 📱 Android Integration Steps

### Step 1: Add Dependencies

In your `build.gradle` (app level):
```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    // Or latest version
}
```

### Step 2: Copy Files to Assets

1. Create `app/src/main/assets/` folder if it doesn't exist
2. Copy all required files from `outputs/` to `assets/`

### Step 3: Load Tokenizer

```kotlin
// Load vocab.txt
val vocab = mutableMapOf<String, Int>()
assets.open("vocab.txt").bufferedReader().useLines { lines ->
    lines.forEachIndexed { index, token ->
        vocab[token] = index
    }
}

// Load tokenizer config
val tokenizerConfig = JSONObject(
    assets.open("tokenizer_config.json").bufferedReader().use { it.readText() }
)
val maxLength = tokenizerConfig.getInt("model_max_length")
```

### Step 4: Preprocess Text

```kotlin
fun preprocessText(text: String, vocab: Map<String, Int>, maxLength: Int = 128): Pair<IntArray, IntArray> {
    // Convert to lowercase
    val lowerText = text.lowercase()
    
    // Tokenize (simple word-level tokenization)
    val words = lowerText.split(" ")
    
    // Convert words to token IDs
    val tokenIds = mutableListOf<Int>()
    val attentionMask = mutableListOf<Int>()
    
    // Add [CLS] token
    tokenIds.add(vocab["[CLS]"] ?: vocab["[UNK]"]!!)
    attentionMask.add(1)
    
    // Add word tokens
    for (word in words) {
        val tokenId = vocab[word] ?: vocab["[UNK]"]!!
        tokenIds.add(tokenId)
        attentionMask.add(1)
    }
    
    // Add [SEP] token
    tokenIds.add(vocab["[SEP]"] ?: vocab["[UNK]"]!!)
    attentionMask.add(1)
    
    // Pad to maxLength
    while (tokenIds.size < maxLength) {
        tokenIds.add(vocab["[PAD]"] ?: 0)
        attentionMask.add(0)
    }
    
    // Truncate if too long
    val inputIds = tokenIds.take(maxLength).toIntArray()
    val mask = attentionMask.take(maxLength).toIntArray()
    
    return Pair(inputIds, mask)
}
```

### Step 5: Load and Run TFLite Model

```kotlin
// Load model
val interpreter = Interpreter(loadModelFile("emergency_classifier.tflite"))

// Prepare input
val inputIds = preprocessText(userInput, vocab).first
val attentionMask = preprocessText(userInput, vocab).second

// Run inference
val inputBuffer = ByteBuffer.allocateDirect(4 * inputIds.size * 2) // 2 inputs
inputBuffer.order(ByteOrder.nativeOrder())

// Add input_ids
for (id in inputIds) {
    inputBuffer.putInt(id)
}

// Add attention_mask
for (mask in attentionMask) {
    inputBuffer.putInt(mask)
}
inputBuffer.rewind()

// Output buffer
val outputBuffer = ByteBuffer.allocateDirect(4 * 5) // 5 classes
outputBuffer.order(ByteOrder.nativeOrder())

// Run
interpreter.run(inputBuffer, outputBuffer)

// Get predictions
outputBuffer.rewind()
val logits = FloatArray(5)
outputBuffer.asFloatBuffer().get(logits)

// Apply softmax and get prediction
val probs = softmax(logits)
val predictedIndex = probs.indices.maxByOrNull { probs[it] } ?: 0

// Map to label
val labels = arrayOf("police", "fire", "ambulance", "women_helpline", "disaster")
val predictedLabel = labels[predictedIndex]
```

---

## ⚠️ Important Notes

1. **Don't modify vocab.txt** - The "[unused]" tokens are part of the model
2. **Use lowercase** - Tokenizer is configured for lowercase (see `tokenizer_config.json`)
3. **Max length is 128** - Pad or truncate to exactly 128 tokens
4. **Two inputs required** - `input_ids` and `attention_mask` (both Int32 arrays of length 128)
5. **Output is logits** - Apply softmax to get probabilities

---

## 🧪 Testing

Test with these sample inputs:
- "Help someone is attacking me with knife" → Should predict: **police**
- "Fire in the building smoke everywhere" → Should predict: **fire**
- "Medical emergency heart attack symptoms" → Should predict: **ambulance**
- "Man following me feeling unsafe" → Should predict: **women_helpline**
- "Landslide blocked road people trapped" → Should predict: **disaster**

---

## 📞 Support

If you encounter issues:
1. Verify all files are in `assets/` folder
2. Check that vocab.txt is read correctly (should have 30,523 lines)
3. Ensure text is lowercased before tokenization
4. Verify input shape: [1, 128] for both input_ids and attention_mask
5. Check output shape: [1, 5] for logits

