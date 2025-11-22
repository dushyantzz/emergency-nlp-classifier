# 🚨 Emergency Classifier - Interactive Demo

## Quick Start

Run the interactive demo to test your model with custom sentences:

```bash
python models/demo_model.py
```

## What It Does

The demo allows you to:
- ✅ Enter any emergency text and see predictions
- ✅ View confidence scores for all categories
- ✅ See visual probability bars
- ✅ Test multiple sentences interactively

## Example Usage

```
🚨 Emergency NLP Classifier - Interactive Demo
================================================================================
Categories: police, fire, ambulance, women_helpline, disaster
Type 'quit' or 'exit' to stop

💡 Example sentences you can try:
   1. Help someone is attacking me with knife
   2. Fire in the building smoke everywhere
   3. Medical emergency heart attack symptoms
   4. Man following me feeling unsafe
   5. Landslide blocked road people trapped

Enter emergency text (or 'quit' to exit): Help fire in my house
```

## Output Format

The demo shows:
- **Predicted Category**: The emergency type (police, fire, ambulance, women_helpline, disaster)
- **Confidence**: Percentage confidence in the prediction
- **All Categories**: Probability scores for all 5 categories with visual bars

## Categories

1. **police** - Law enforcement emergencies
2. **fire** - Fire-related emergencies
3. **ambulance** - Medical emergencies
4. **women_helpline** - Women's safety issues
5. **disaster** - Natural disasters

## Tips

- Enter natural emergency descriptions (2-3 sentences work best)
- The model works best with clear, descriptive text
- Type 'quit' or 'exit' to stop the demo
- Press Ctrl+C to exit at any time

## Alternative: Batch Testing

If you want to test multiple sentences at once, use the test script:

```bash
python models/test_model.py
```

This runs predefined test cases and shows accuracy metrics.

