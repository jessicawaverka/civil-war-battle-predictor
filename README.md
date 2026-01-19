# Civil War Battle Outcome Predictor - Setup Guide

## What You Need

1. Python installed on your computer
2. The battle data CSV file
3. The app code file (battle_predictor_app.py)

## Step-by-Step Instructions

### Step 1: Install Required Libraries

Open Terminal (Mac) or Command Prompt (Windows) and type:

```bash
pip install streamlit pandas scikit-learn matplotlib
```

Wait for it to finish (takes 1-2 minutes).

### Step 2: Organize Your Files

Put these files in the same folder:
- `battle_predictor_app.py` (the code)
- `civil_war_battles_100plus.csv` (the battle data)

### Step 3: Run the App

In Terminal/Command Prompt, navigate to your folder:

```bash
cd path/to/your/folder
```

Then run:

```bash
streamlit run battle_predictor_app.py
```

### Step 4: View in Browser

A web browser will automatically open showing your app!

If not, go to: http://localhost:8501

## What the App Does

1. **Loads 106 Civil War battles** - Ground and naval engagements from 1861-1865

2. **Trains an AI** - Uses a decision tree to learn patterns from 80% of battles

3. **Tests the AI** - Checks accuracy on the remaining 20% of battles it never saw

4. **Shows predictions** - Displays which battles the AI got right or wrong

5. **Visualizes the decision tree** - See the exact rules the AI learned

6. **Interactive predictor** - Enter your own hypothetical battle and get a prediction!

7. **Shows what matters most** - Which factors (troop numbers, terrain, etc.) were most important

8. **Discusses limitations** - Why AI can't capture everything about war

## For Your Counselor Discussion

Be ready to explain:

**Objectives:**
- Predict Civil War battle outcomes using machine learning
- Understand what AI can and cannot do with historical data

**Data Requirements:**
- 106 battles from 1861-1865
- Features: troop numbers, terrain, weather, defender, battle type
- Target: who won (Union/Confederate/Draw)

**Process:**
- Collected historical battle data
- Encoded categorical variables (turned words into numbers)
- Split data 80/20 for training and testing
- Used decision tree algorithm
- Evaluated accuracy and predictions

**Results:**
- AI achieved approximately 60-70% accuracy on test battles
- Most important factors: troop numbers and terrain
- Shows clear patterns but also limitations

**Ethical Considerations:**
- AI cannot measure leadership quality
- Cannot account for morale, tactics, surprise
- Historical records may be biased
- War involves human decisions beyond just data
- Demonstrates both power and limits of AI

## Troubleshooting

**Error: "ModuleNotFoundError"**
- Make sure you installed all libraries (Step 1)

**Error: "FileNotFoundError"**
- Make sure the CSV file is in the same folder as the Python file

**App won't open**
- Try manually going to http://localhost:8501 in your browser

## Understanding the Code (Optional)

The code does these things:

1. **Imports libraries** - Gets tools we need (pandas, scikit-learn, streamlit)
2. **Loads data** - Reads the CSV file
3. **Encodes data** - Turns words like "Union" into numbers like 1
4. **Splits data** - Separates training and testing battles
5. **Trains model** - DecisionTreeClassifier learns patterns
6. **Makes predictions** - Tests on unseen battles
7. **Shows results** - Creates interactive web interface

You don't need to understand every line, but this shows how AI works!

## Tips for Presenting

- Run the app and show your counselor the interactive predictor
- Try entering a famous battle and see if the AI gets it right
- Explain why the AI sometimes gets it wrong (leadership, morale, etc.)
- Show the decision tree visualization
- Discuss what you learned about AI's capabilities and limitations

Good luck with your merit badge! 🎖️
