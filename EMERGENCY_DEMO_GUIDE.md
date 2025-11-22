# 🚨 Emergency Response Demo - User Guide

## Overview

This demo analyzes emergency text descriptions and automatically recommends:
- **Which resources to contact** (Police, Fire, Ambulance, Women Helpline, Disaster Management)
- **Emergency phone numbers** for each category
- **Immediate actions** to take
- **Resource allocation** recommendations

## Quick Start

```bash
python models/emergency_demo.py
```

## Features

✅ **AI-Powered Classification** - Automatically detects emergency type  
✅ **Resource Recommendations** - Tells you which services to contact  
✅ **Emergency Numbers** - Provides correct helpline numbers  
✅ **Action Steps** - Lists immediate actions to take  
✅ **Confidence Scores** - Shows prediction confidence for all categories  

## Example Usage

### Input:
```
Enter emergency description: Fire in my building smoke everywhere urgent help
```

### Output:
```
🚨 EMERGENCY CLASSIFICATION RESULT
================================================================================

📝 Input Text:
   "Fire in my building smoke everywhere urgent help"

🎯 Detected Emergency Type: FIRE
   Confidence: 94.52%

================================================================================
📞 IMMEDIATE ACTION REQUIRED
================================================================================

🔥 Fire Department
📱 Emergency Number: 101 / 112
   (Dial 112 for universal emergency services)

📋 Description:
   Fires, smoke, gas leaks, explosions, burning buildings

🔧 Resources to Inform:
   1. Fire Department
   2. Fire Brigade
   3. Emergency Services

⚡ Recommended Actions:
   1. Call fire department immediately
   2. Evacuate the area if safe to do so
   3. Do not use elevators during fire
   4. Alert nearby people about the fire
   5. Close doors to contain fire if possible
```

## Emergency Categories & Resources

### 1. 🚔 Police (100 / 112)
- **When to use**: Criminal activities, threats, violence, theft, robbery, assault
- **Resources**: Police, Law Enforcement, Security
- **Actions**: Contact police, provide location, stay safe

### 2. 🔥 Fire (101 / 112)
- **When to use**: Fires, smoke, gas leaks, explosions
- **Resources**: Fire Department, Fire Brigade, Emergency Services
- **Actions**: Call fire department, evacuate, alert others

### 3. 🚑 Ambulance (102 / 112)
- **When to use**: Medical emergencies, injuries, accidents, health crises
- **Resources**: Ambulance, Medical Emergency Services, Hospital
- **Actions**: Call ambulance, provide location, apply first aid if trained

### 4. 🛡️ Women Helpline (1091 / 112)
- **When to use**: Harassment, stalking, domestic violence, safety concerns
- **Resources**: Women Helpline, Police, Support Services
- **Actions**: Call helpline, contact police if immediate danger, seek safe location

### 5. 🌊 Disaster (108 / 112)
- **When to use**: Natural disasters, floods, earthquakes, landslides
- **Resources**: Disaster Management, NDRF, Emergency Services, Police
- **Actions**: Contact disaster management, follow evacuation procedures

## Commands

- **Enter text**: Type your emergency description
- **'help'**: Show all emergency numbers and categories
- **'quit'**: Exit the demo

## Example Scenarios

Try these example sentences:

1. **Police**: "Help someone is attacking me with knife urgent"
2. **Fire**: "Fire in the building smoke everywhere need help"
3. **Ambulance**: "Medical emergency heart attack symptoms chest pain"
4. **Women Helpline**: "Man stalking me following me feeling very unsafe"
5. **Disaster**: "Landslide blocked road multiple people trapped need rescue"

## Tips for Best Results

✅ **Be descriptive** - Include key details (fire, injury, threat, etc.)  
✅ **Use natural language** - Write as you would describe the emergency  
✅ **Include urgency** - Words like "urgent", "immediate", "help" improve detection  
✅ **2-3 sentences** - Optimal length for accurate classification  

## Technical Details

- **Model**: DistilBERT-based classifier
- **Categories**: 5 emergency types
- **Confidence**: Shows probability for all categories
- **Processing**: Real-time inference on CPU/GPU

## Safety Note

⚠️ **This is a demo system for educational purposes.**  
In real emergencies, always:
- Call emergency services directly (112 in India)
- Don't rely solely on AI predictions
- Follow official emergency procedures
- Stay safe and prioritize your safety

## Support

For issues or questions:
- Check that model files exist in `outputs/trained_model/`
- Ensure all dependencies are installed (`pip install -r requirements.txt`)
- Verify the model was trained successfully

