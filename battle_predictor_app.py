import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Civil War Battle Predictor", page_icon="⚔️", layout="wide")

# Title
st.title("⚔️ Civil War Battle Outcome Predictor")
st.markdown("### An AI Merit Badge Project by Theo")
st.markdown("---")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('civil_war_battles_100plus.csv')
    return df

df = load_data()

# Show intro
st.markdown("""
**This AI learns patterns from 106 Civil War battles to predict who would win based on:**
- Number of forces (troops or ships)
- Type of battle (ground or naval)
- Terrain
- Weather conditions
- Who was defending
""")

# Sidebar - Show the data
with st.sidebar:
    st.header("📊 Battle Data")
    st.write(f"Total battles: {len(df)}")
    st.write(f"Union victories: {len(df[df['Winner'] == 'Union'])}")
    st.write(f"Confederate victories: {len(df[df['Winner'] == 'Confederate'])}")
    st.write(f"Draws: {len(df[df['Winner'] == 'Draw'])}")
    
    if st.checkbox("Show raw battle data"):
        st.dataframe(df)

# Prepare the data for AI
st.markdown("---")
st.header("🤖 Training the AI")

# Encode categorical variables (turn words into numbers the AI can understand)
le_battle_type = LabelEncoder()
le_terrain = LabelEncoder()
le_weather = LabelEncoder()
le_defender = LabelEncoder()
le_winner = LabelEncoder()

df['Battle_Type_Encoded'] = le_battle_type.fit_transform(df['Battle Type'])
df['Terrain_Encoded'] = le_terrain.fit_transform(df['Terrain'])
df['Weather_Encoded'] = le_weather.fit_transform(df['Weather'])
df['Defender_Encoded'] = le_defender.fit_transform(df['Defender'])
df['Winner_Encoded'] = le_winner.fit_transform(df['Winner'])

# Features (what the AI looks at) and Target (what we want to predict)
features = ['Union Forces', 'Confederate Forces', 'Battle_Type_Encoded', 
            'Terrain_Encoded', 'Weather_Encoded', 'Defender_Encoded']
X = df[features]
y = df['Winner_Encoded']

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write(f"✅ Using {len(X_train)} battles to train the AI")
st.write(f"✅ Using {len(X_test)} battles to test the AI")

# Train the decision tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Calculate accuracy
train_accuracy = model.score(X_train, y_train) * 100
test_accuracy = model.score(X_test, y_test) * 100

col1, col2 = st.columns(2)
with col1:
    st.metric("Training Accuracy", f"{train_accuracy:.1f}%")
with col2:
    st.metric("Testing Accuracy", f"{test_accuracy:.1f}%")

st.success(f"The AI correctly predicted {test_accuracy:.1f}% of battles it had never seen before!")

# Show predictions on test set
st.markdown("---")
st.header("🎯 Predictions vs Reality")

# Get predictions for test set
y_pred = model.predict(X_test)
test_results = X_test.copy()
test_results['Actual Winner'] = le_winner.inverse_transform(y_test)
test_results['AI Predicted'] = le_winner.inverse_transform(y_pred)
test_results['Correct?'] = test_results['Actual Winner'] == test_results['AI Predicted']

# Add battle names back
test_indices = X_test.index
test_results['Battle Name'] = df.loc[test_indices, 'Battle Name'].values

# Show results
st.write("Here's what the AI predicted for battles it had never seen:")
display_cols = ['Battle Name', 'Actual Winner', 'AI Predicted', 'Correct?']
st.dataframe(test_results[display_cols], use_container_width=True)

correct_count = test_results['Correct?'].sum()
st.write(f"**Got {correct_count} out of {len(test_results)} correct!**")

# Visualize the decision tree
st.markdown("---")
st.header("🌳 The AI's Decision Rules")
st.write("This tree shows how the AI makes decisions:")

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(model, 
          feature_names=['Union Forces', 'Confederate Forces', 'Battle Type', 
                        'Terrain', 'Weather', 'Defender'],
          class_names=le_winner.classes_,
          filled=True, 
          rounded=True,
          ax=ax,
          fontsize=10)
st.pyplot(fig)

st.info("💡 Read the tree from top to bottom. Each box shows a decision rule the AI learned!")

# Interactive predictor
st.markdown("---")
st.header("🎮 Try Your Own Battle!")
st.write("Enter details of a hypothetical battle and see what the AI predicts:")

col1, col2, col3 = st.columns(3)

with col1:
    union_forces = st.number_input("Union Forces", min_value=0, max_value=200000, value=50000, step=1000)
    confederate_forces = st.number_input("Confederate Forces", min_value=0, max_value=200000, value=40000, step=1000)

with col2:
    battle_type = st.selectbox("Battle Type", ['Ground', 'Naval'])
    terrain = st.selectbox("Terrain", ['Open', 'Forest', 'Hills', 'Mixed', 'Urban', 'Coastal', 'River', 'Open Sea'])

with col3:
    weather = st.selectbox("Weather", ['Clear', 'Rain', 'Snow', 'Fog'])
    defender = st.selectbox("Who is Defending?", ['Union', 'Confederate', 'Neither'])

if st.button("🔮 Predict Winner!", type="primary"):
    # Encode the inputs
    battle_type_enc = le_battle_type.transform([battle_type])[0]
    terrain_enc = le_terrain.transform([terrain])[0]
    weather_enc = le_weather.transform([weather])[0]
    defender_enc = le_defender.transform([defender])[0]
    
    # Make prediction
    input_data = [[union_forces, confederate_forces, battle_type_enc, 
                   terrain_enc, weather_enc, defender_enc]]
    prediction = model.predict(input_data)[0]
    predicted_winner = le_winner.inverse_transform([prediction])[0]
    
    # Get probability
    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100
    
    # Display result
    st.markdown("### 🏆 Prediction Result:")
    if predicted_winner == "Union":
        st.success(f"**The AI predicts: UNION victory** (Confidence: {confidence:.1f}%)")
    elif predicted_winner == "Confederate":
        st.error(f"**The AI predicts: CONFEDERATE victory** (Confidence: {confidence:.1f}%)")
    else:
        st.warning(f"**The AI predicts: DRAW** (Confidence: {confidence:.1f}%)")

# Feature importance
st.markdown("---")
st.header("📈 What Matters Most?")
st.write("Which factors were most important for the AI's predictions?")

importances = model.feature_importances_
feature_names = ['Union Forces', 'Confederate Forces', 'Battle Type', 'Terrain', 'Weather', 'Defender']

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.barh(importance_df['Feature'], importance_df['Importance'])
ax2.set_xlabel('Importance')
ax2.set_title('Feature Importance in Predicting Battle Outcomes')
st.pyplot(fig2)

# Ethical considerations section
st.markdown("---")
st.header("🤔 Ethical Considerations & Limitations")

st.warning("""
**What the AI Cannot Know:**
- Leadership quality (Grant vs. Bragg, Lee vs. McClellan)
- Soldier morale and fighting spirit
- Element of surprise or brilliant tactics
- Supply line conditions
- Political pressure and strategic objectives
- Weather changes during battle
- Individual soldier bravery and unit cohesion

**Historical Bias:**
- Records may favor the winning side
- Troop numbers may be estimates or exaggerated
- Some perspectives (enslaved people, civilians) are missing from military records

**What This Teaches:**
- AI can find patterns in data
- But war is ultimately about human decisions, not just numbers
- History is complex and can't be fully captured by statistics
""")

st.info("💡 This project shows both the power and limitations of AI!")

# Footer
st.markdown("---")
st.markdown("*Created for AI Merit Badge - Boy Scouts of America*")
