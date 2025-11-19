#!/usr/bin/env python3
"""
Emergency Dataset Generator for SIH 2025 - Team Daredevils
Generates comprehensive emergency classification dataset with 5000+ examples

Categories:
- police: Theft, robbery, assault, threats, kidnapping
- fire: Fire emergencies, gas leaks, explosions
- ambulance: Medical emergencies, accidents, injuries
- women_helpline: Harassment, stalking, women safety
- disaster: Natural disasters, landslides, floods
"""

import pandas as pd
import random
import os
from typing import List, Dict


class EmergencyDatasetGenerator:
    def __init__(self):
        self.categories = [
            'police',
            'fire',
            'ambulance',
            'women_helpline',
            'disaster'
        ]
        
        # Template scenarios for each category
        self.templates = self._create_templates()
        
        # Indian context additions
        self.locations = [
            "near hotel", "at tourist spot", "in market", "on highway",
            "at temple", "near railway station", "in restaurant",
            "at bus stand", "near monument", "in park", "at beach",
            "in shopping mall", "near airport", "on mountain road",
            "in forest area", "at camping site", "near river",
            "in remote village", "on trek route", "at viewpoint"
        ]
        
        self.urgency_words = [
            "urgent", "help", "quickly", "emergency", "please",
            "now", "immediately", "fast", "asap", "desperate"
        ]
        
        self.time_contexts = [
            "at night", "in the evening", "early morning",
            "late night", "afternoon", "during daytime"
        ]
    
    def _create_templates(self) -> Dict[str, List[str]]:
        """Create base templates for each emergency category"""
        return {
            'police': [
                "Someone is following me with {weapon}",
                "I am being robbed {location}",
                "Group of people {action} me",
                "Theft in progress {location}",
                "Someone stole my {item}",
                "Being threatened by {person}",
                "Suspicious activity near {place}",
                "Tourist {victim} being attacked",
                "Kidnapping attempt happening",
                "Violence broke out {location}",
                "Someone snatched my {item}",
                "Assault happening {location}",
                "Pickpocket caught me {location}",
                "Being chased by {group}",
                "Tourist scam happening {location}",
                "Extortion attempt by {person}",
                "Breaking and entering in room",
                "Confrontation with locals",
                "Physical fight at {location}",
                "Vandalism happening {location}"
            ],
            'fire': [
                "Fire broke out in {location}",
                "Smoke coming from {place}",
                "Building caught fire {location}",
                "Vehicle on fire {location}",
                "Forest fire spreading {location}",
                "Gas leak detected {location}",
                "Explosion happened {location}",
                "Hotel room on fire",
                "Kitchen fire at {location}",
                "Electrical fire {location}",
                "Chemical fire hazard {location}",
                "Fire spreading rapidly {location}",
                "People trapped in fire",
                "Smoke inhalation risk {location}",
                "Burning smell {location}",
                "Flames visible from {location}",
                "Fire emergency {location}",
                "Wildfire approaching {location}",
                "Bonfire out of control",
                "Burning vehicle blocking road"
            ],
            'ambulance': [
                "Tourist fell from {location}",
                "Severe injury {location}",
                "Heart attack symptoms",
                "Breathing difficulty {location}",
                "Accident happened {location}",
                "Heavy bleeding after {incident}",
                "Unconscious person {location}",
                "Broken bones suspected",
                "Snake bite incident",
                "Food poisoning case",
                "Allergic reaction severe",
                "Chest pain emergency",
                "Stroke symptoms visible",
                "Heat stroke victim {location}",
                "Drowning incident {location}",
                "Electric shock victim",
                "Multiple injuries {location}",
                "Critical condition patient",
                "Seizure happening {location}",
                "Diabetic emergency {location}"
            ],
            'women_helpline': [
                "Man following me {location}",
                "Being harassed by {person}",
                "Feeling unsafe {location}",
                "Stalker spotted {location}",
                "Inappropriate behavior {location}",
                "Eve teasing incident",
                "Molestation attempt {location}",
                "Being cornered by men",
                "Uncomfortable situation {location}",
                "Need women support {location}",
                "Verbal abuse by {person}",
                "Threatening woman tourist",
                "Unwanted advances {location}",
                "Being followed persistently",
                "Harassment at {location}",
                "Safety concern for woman",
                "Inappropriate touching {location}",
                "Catcalling incident {location}",
                "Woman in distress {location}",
                "Gender based violence risk"
            ],
            'disaster': [
                "Landslide blocked road {location}",
                "Flood water rising {location}",
                "Earthquake felt {location}",
                "Rockfall on highway",
                "Bridge collapsed {location}",
                "Heavy rainfall flooding",
                "Mudslide threatening {location}",
                "Avalanche risk {location}",
                "Road washed away {location}",
                "Cloudburst happening {location}",
                "Glacial lake burst danger",
                "Cyclone approaching {location}",
                "Storm damage severe",
                "People stranded {location}",
                "Natural disaster {location}",
                "Infrastructure collapsed",
                "River overflowing {location}",
                "Mountain road blocked",
                "Emergency evacuation needed",
                "Disaster zone {location}"
            ]
        }
    
    def generate_variations(self, template: str, category: str) -> List[str]:
        """Generate multiple variations of a template"""
        variations = []
        
        # Placeholder replacements
        replacements = {
            '{weapon}': ['knife', 'stick', 'weapon', 'rod', 'sharp object'],
            '{location}': self.locations,
            '{action}': ['harassing', 'threatening', 'surrounding', 'attacking', 'following'],
            '{item}': ['wallet', 'phone', 'bag', 'passport', 'camera', 'money', 'jewelry'],
            '{person}': ['stranger', 'group', 'local', 'individual', 'unknown person'],
            '{place}': ['hotel', 'restaurant', 'shop', 'area', 'building'],
            '{victim}': ['woman', 'couple', 'group', 'elderly person', 'family'],
            '{group}': ['gang', 'group of people', 'mob', 'locals', 'strangers'],
            '{incident}': ['fall', 'accident', 'attack', 'cut', 'crash']
        }
        
        # Generate base variations
        for _ in range(3):
            text = template
            for placeholder, options in replacements.items():
                if placeholder in text:
                    text = text.replace(placeholder, random.choice(options))
            
            # Add urgency
            if random.random() > 0.5:
                text += ' ' + random.choice(self.urgency_words)
            
            # Add location
            if random.random() > 0.6 and '{location}' not in template:
                text += ' ' + random.choice(self.locations)
            
            # Add time context
            if random.random() > 0.7:
                text += ' ' + random.choice(self.time_contexts)
            
            variations.append(text.strip())
        
        return variations
    
    def generate_dataset(self, examples_per_category: int = 1000) -> pd.DataFrame:
        """Generate complete dataset"""
        data = []
        
        print("🔄 Generating emergency classification dataset...")
        print(f"Target: {examples_per_category} examples per category\n")
        
        for category in self.categories:
            print(f"Generating {category} examples...")
            templates = self.templates[category]
            examples_needed = examples_per_category
            
            while len([d for d in data if d['label'] == category]) < examples_needed:
                for template in templates:
                    variations = self.generate_variations(template, category)
                    for variation in variations:
                        if len([d for d in data if d['label'] == category]) < examples_needed:
                            data.append({
                                'text': variation,
                                'label': category
                            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n✅ Dataset generated successfully!")
        print(f"Total examples: {len(df)}")
        print(f"\nDistribution:")
        print(df['label'].value_counts())
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filepath: str = 'data/emergency_dataset.csv'):
        """Save dataset to CSV"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"\n💾 Dataset saved to: {filepath}")


def main():
    """Main execution"""
    generator = EmergencyDatasetGenerator()
    
    # Generate 1000 examples per category = 5000 total
    df = generator.generate_dataset(examples_per_category=1000)
    
    # Save dataset
    generator.save_dataset(df)
    
    # Display sample
    print("\n📊 Sample examples:")
    print("="*80)
    for category in generator.categories:
        samples = df[df['label'] == category].sample(2)
        print(f"\n{category.upper()}:")
        for _, row in samples.iterrows():
            print(f"  - {row['text']}")
    print("="*80)
    
    print("\n✨ Dataset generation complete!")
    print("Next step: Run 'python models/train_model.py' to train the model.")


if __name__ == "__main__":
    main()