import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib
import pandas as pd
import requests
import os
from scipy.sparse import hstack, vstack, csr_matrix
import csv

# Page config
st.set_page_config(
    page_title="Dota 2 Match Duration Predictor",
    page_icon="⚔️",
    layout="wide"
)

@st.cache_resource
def load_model_and_data():
    """Load model and mapping data"""
    try:
        model_dir = './'
        
        # Load the scripted model
        loaded_model = torch.jit.load(f"{model_dir}/poc_02_scripted_model.pt", map_location='cpu')
        
        # Force model to CPU mode
        loaded_model = loaded_model.cpu()
        loaded_model.eval()
        
        # Load mappings
        with open(f"{model_dir}/idx_to_name_dataset_02.pkl", "rb") as f:
            idx_to_name = pickle.load(f)
        
        with open(f"{model_dir}/name_to_idx_dataset_02.pkl", "rb") as f:
            name_to_idx = pickle.load(f)
        
        return loaded_model, idx_to_name, name_to_idx
    except Exception as e:
        st.error(f"Error loading model or data: {str(e)}")
        return None, None, None

@st.cache_resource
def load_model_v2():
    """Load model v2 (poc_04_scripted_model.pt)"""
    try:
        model_dir = './'
        loaded_model_v2 = torch.jit.load(f"{model_dir}/poc_04_scripted_model.pt", map_location='cpu')
        loaded_model_v2 = loaded_model_v2.cpu()
        loaded_model_v2.eval()
        return loaded_model_v2
    except Exception as e:
        st.error(f"Error loading model v2: {str(e)}")
        return None

@st.cache_resource
def load_role_artifacts():
    """Optionally load encoder and model for hero role prediction if available."""
    try:
        model_dir = './'
        encoder_path = os.path.join(model_dir, 'one_hot_encoder_filtered.pkl')
        model_path = os.path.join(model_dir, 'hero_role_classifier_enhanced.pkl')
        if not (os.path.exists(encoder_path) and os.path.exists(model_path)):
            return None, None
        model_enhanced = joblib.load("hero_role_classifier_enhanced.pkl")
        encoder_filtered = joblib.load("one_hot_encoder_filtered.pkl")
        return encoder_filtered, model_enhanced
    except Exception as e:
        st.warning(f"Could not load role artifacts: {e}")
        return None, None

def get_desc_from_label_id(label_id):
    desc_from_label_id = {
        0: '<=10',
        1: '10-20',
        2: '20-30',
        3: '30-40',
        4: '40-50',
        5: '50-60',
        6: '60-70',
        7: '70-80',
        8: '80-90',
        9: '>90'
    }
    return desc_from_label_id[label_id]

def get_desc_from_label_id_v2(label_id):
    desc_from_label_id = {
        0: '<=10',
        1: '10-15',
        2: '15-20',
        3: '20-25',
        4: '25-30',
        5: '30-35',
        6: '35-40',
        7: '40-45',
        8: '45-50',
        9: '50-55',
        10: '55-60',
        11: '60-65',
        12: '65-70',
        13: '70-75',
        14: '75-80',
        15: '80-85',
        16: '85-90',
        17: '>90'
    }
    return desc_from_label_id[label_id]

def load_id_name_mapping(filename):
    mapping = {}
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            hero_id = int(row['id'])
            name = row['name']
            mapping[hero_id] = name
    return mapping

def get_hero_name_from_id(hero_id):
    """Get hero name from OpenDota hero ID and map to model's naming convention"""
    # OpenDota hero ID to name mapping
    hero_id_to_name = load_id_name_mapping("heroes_data.csv")
    
    # Get the OpenDota name
    opendota_name = hero_id_to_name.get(hero_id)
    if not opendota_name:
        return None
    
    # Map OpenDota names to model names (handle naming differences)
    name_mapping = {
        "Skeleton King": "Wraith King",
        "Outworld Destroyer": "Outworld Devourer"
        # Add more mappings as needed
    }
    
    # Return the mapped name if it exists, otherwise return the original
    return name_mapping.get(opendota_name, opendota_name)

def get_hero_id_from_name(hero_name: str):
    """Map hero display name to OpenDota hero ID using the same mapping list used in get_hero_name_from_id."""
    hero_id_to_name = load_id_name_mapping("heroes_data.csv")
    name_to_id = {v: k for k, v in hero_id_to_name.items()}
    # Handle known renames
    if hero_name == "Skeleton King":
        hero_name = "Wraith King"
    if hero_name == "Outworld Destroyer":
        hero_name = "Outworld Devourer"
    return name_to_id.get(hero_name)

def fetch_match_heroes(match_id, name_to_idx):
    """Fetch hero names from OpenDota match ID (without prediction)"""
    try:
        # Fetch match data from OpenDota API
        url = f"https://api.opendota.com/api/matches/{match_id}"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        match_data = response.json()

        # Extract hero IDs
        radiant_hero_ids_raw = [player['hero_id'] for player in match_data['players'][:5]]
        dire_hero_ids_raw = [player['hero_id'] for player in match_data['players'][5:]]
        heroes_ids_raw = radiant_hero_ids_raw + dire_hero_ids_raw

        hero_names = []
        missing_heroes = []
        
        for hero_id_raw in heroes_ids_raw:
            # Get hero name from ID
            hero_name = get_hero_name_from_id(hero_id_raw)
            
            if hero_name and hero_name in name_to_idx:
                hero_names.append(hero_name)
            else:
                missing_heroes.append(f"ID {hero_id_raw} ({hero_name if hero_name else 'Unknown'})")
        
        # Check if we have all 10 heroes
        if len(hero_names) != 10:
            st.error(f"❌ Missing heroes in mapping: {', '.join(missing_heroes)}")
            st.error(f"Found {len(hero_names)}/10 heroes. Cannot populate dropdowns.")
            return None

        return hero_names

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching match data: {e}")
        return None
    except KeyError:
        st.error("Error: Could not extract hero data from match details. Please check the match ID.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def predict_from_match_id(match_id, model, name_to_idx):
    """Predict match duration from OpenDota match ID"""
    try:
        # Fetch match data from OpenDota API
        url = f"https://api.opendota.com/api/matches/{match_id}"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        match_data = response.json()

        # Extract hero IDs
        radiant_hero_ids_raw = [player['hero_id'] for player in match_data['players'][:5]]
        dire_hero_ids_raw = [player['hero_id'] for player in match_data['players'][5:]]
        heroes_ids_raw = radiant_hero_ids_raw + dire_hero_ids_raw

        heroes_ids = []
        hero_names = []
        missing_heroes = []
        
        for hero_id_raw in heroes_ids_raw:
            # Get hero name from ID
            hero_name = get_hero_name_from_id(hero_id_raw)
            
            if hero_name and hero_name in name_to_idx:
                heroes_ids.append(name_to_idx[hero_name])
                hero_names.append(hero_name)
            else:
                missing_heroes.append(f"ID {hero_id_raw} ({hero_name if hero_name else 'Unknown'})")
        
        # Check if we have all 10 heroes
        if len(heroes_ids) != 10:
            st.error(f"❌ Missing heroes in mapping: {', '.join(missing_heroes)}")
            st.error(f"Found {len(heroes_ids)}/10 heroes. Cannot make prediction.")
            return None, None, None

        # Make prediction using the loaded model
        model.eval()
        with torch.no_grad():
            # Ensure the input tensor is on CPU
            input_tensor = torch.tensor([heroes_ids], dtype=torch.long, device='cpu')
            logits = model(input_tensor)
            outputs = F.softmax(logits, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            predicted_duration = get_desc_from_label_id(predicted.item())

        return outputs, predicted_duration, hero_names

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching match data: {e}")
        return None, None, None
    except KeyError:
        st.error("Error: Could not extract hero data from match details. Please check the match ID.")
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None, None

def predict_hero_roles_enhanced(radiant_hero_ids, dire_hero_ids, encoder_filtered, model_enhanced):
    """Predict 3-role classification for each hero given team composition features."""
    if encoder_filtered is None or model_enhanced is None:
        return {'radiant': [], 'dire': []}

    # Compute team features (sum of one-hots) as sparse
    radiant_team_features = None
    dire_team_features = None
    if radiant_hero_ids:
        radiant_team_features = encoder_filtered.transform(np.array(radiant_hero_ids).reshape(-1, 1)).sum(axis=0)
    if dire_hero_ids:
        dire_team_features = encoder_filtered.transform(np.array(dire_hero_ids).reshape(-1, 1)).sum(axis=0)

    # Determine encoded feature dimension for zero vectors
    if radiant_team_features is not None:
        team_feat_dim = radiant_team_features.shape[1]
    elif dire_team_features is not None:
        team_feat_dim = dire_team_features.shape[1]
    else:
        team_feat_dim = encoder_filtered.transform(np.array([list(encoder_filtered.categories_[0])[0]]).reshape(-1, 1)).shape[1]

    enhanced_features_list = []

    # Radiant players
    for hero_id in radiant_hero_ids:
        indiv = encoder_filtered.transform(np.array([hero_id]).reshape(-1, 1))
        team_vec = radiant_team_features if radiant_team_features is not None else csr_matrix((1, team_feat_dim))
        combined = hstack([indiv, team_vec])
        enhanced_features_list.append(combined)

    # Dire players
    for hero_id in dire_hero_ids:
        indiv = encoder_filtered.transform(np.array([hero_id]).reshape(-1, 1))
        team_vec = dire_team_features if dire_team_features is not None else csr_matrix((1, team_feat_dim))
        combined = hstack([indiv, team_vec])
        enhanced_features_list.append(combined)

    if enhanced_features_list:
        enhanced_features_matrix_predict = vstack(enhanced_features_list)
    else:
        return {'radiant': [], 'dire': []}

    # Predict
    predicted_roles = model_enhanced.predict(enhanced_features_matrix_predict)

    # Build output
    radiant_predictions = []
    current_idx = 0
    for hero_id in radiant_hero_ids:
        radiant_predictions.append({'hero_id': hero_id, 'predicted_role': int(predicted_roles[current_idx])})
        current_idx += 1

    dire_predictions = []
    for hero_id in dire_hero_ids:
        dire_predictions.append({'hero_id': hero_id, 'predicted_role': int(predicted_roles[current_idx])})
        current_idx += 1

    return {'radiant': radiant_predictions, 'dire': dire_predictions}

def draw_distribution(probabilities, correct_label=0, heros=[], idx_to_name=None, label_fn=get_desc_from_label_id):
    """Draw probability distribution for duration labels"""
    # Convert the probabilities tensor to a numpy array after moving it to CPU
    probabilities_np = probabilities.squeeze().cpu().numpy()
    radiants = []
    dires = []
    
    for pos, i in enumerate(heros):
        # Handle both tensor and regular integer inputs
        if hasattr(i, 'cpu'):
            hero_idx = i.cpu().item()
        else:
            hero_idx = i
            
        if pos < 5:
            radiants.append(idx_to_name[hero_idx])
        else:
            dires.append(idx_to_name[hero_idx])

    # Use label_fn for class labels
    class_labels = [label_fn(label_id) for label_id in np.arange(len(probabilities_np))]

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(class_labels, probabilities_np, color='steelblue', alpha=0.7)
    
    # Highlight the highest probability
    max_idx = np.argmax(probabilities_np)
    bars[max_idx].set_color('orange')
    
    ax.set_xlabel('Duration Label Class')
    ax.set_ylabel('Probability')
    ax.set_title(f'Probability Distribution of Duration Labels for a Sample Test Match, correct={correct_label}\nradiants: {radiants}\ndires: {dires}')
    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities_np)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def predict_duration(model, hero_indices):
    """Predict match duration"""
    try:
        # Convert to tensor and ensure it's on CPU
        hero_tensor = torch.tensor(hero_indices, dtype=torch.long, device='cpu').unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            model.eval()
            output = model(hero_tensor)
            probabilities = F.softmax(output, dim=1)
        
        return probabilities
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Main app
def main():
    st.title("⚔️ Dota 2 Match Duration Predictor")
    st.markdown("---")
    
    # Load model and data
    model, idx_to_name, name_to_idx = load_model_and_data()
    model_v2 = load_model_v2()
    
    if model is None or idx_to_name is None or name_to_idx is None:
        st.error("Failed to load model or data files. Please check if the files exist in the current directory.")
        return
    
    # Get hero names for dropdown
    hero_names = list(name_to_idx.keys())
    hero_names.sort()
    
    # Match ID input section
    st.markdown("### 🎮 Predict from Match ID")
    st.markdown("Enter a Dota 2 match ID from OpenDota to automatically populate heroes and predict duration.")
    
    # Show hero mapping info
    with st.expander("ℹ️ Hero Mapping Info"):
        st.info(f"Your model supports {len(name_to_idx)} heroes. Some OpenDota hero names may be mapped to different names in the model.")
        st.markdown("**Common mappings:**")
        st.markdown("- Skeleton King → Wraith King")
        st.markdown("- Outworld Destroyer → Outworld Devourer")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        match_id_input = st.text_input("Match ID:", placeholder="e.g., 1234567890", help="Enter the match ID from OpenDota URL")
    with col2:
        fetch_match_btn = st.button("🔍 Fetch Match", type="secondary")
    
    # Handle match ID fetching (only populate dropdowns, don't predict)
    if fetch_match_btn and match_id_input:
        if match_id_input.isdigit():
            with st.spinner("Fetching match data from OpenDota..."):
                # Only fetch hero names, don't make prediction yet
                hero_names_from_match = fetch_match_heroes(int(match_id_input), name_to_idx)
            
            if hero_names_from_match is not None:
                st.success(f"✅ Match data fetched successfully! Heroes populated in dropdowns below.")
                st.info("💡 You can now review/modify the hero selection and click 'Predict Match Duration' when ready.")
                
                # Store the fetched heroes in session state for later use
                st.session_state.fetched_heroes = hero_names_from_match
                st.session_state.match_id = match_id_input
                
        else:
            st.error("❌ Please enter a valid numeric match ID.")
    
    # Display match info if heroes were fetched
    if 'fetched_heroes' in st.session_state and 'match_id' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Match Information")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"**Match ID:** {st.session_state.match_id}")
            st.markdown(f"**Heroes fetched from:** [OpenDota Match](https://www.opendota.com/matches/{st.session_state.match_id})")
        with col2:
            if st.button("🔄 Clear Match Data", type="secondary"):
                # Clear session state
                del st.session_state.fetched_heroes
                del st.session_state.match_id
                st.rerun()
    
    st.markdown("---")
    st.markdown("### 🎯 Hero Selection")
    st.markdown("Select heroes for each team (heroes will be auto-populated if you fetched a match):")
    
    # Create two columns for Radiant and Dire
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌟 Radiant Team")
        radiant_heroes = []
        for i in range(5):
            # Use fetched heroes if available, otherwise use default index
            default_index = 0
            if 'fetched_heroes' in st.session_state and len(st.session_state.fetched_heroes) == 10:
                # Find the hero in the list and get its index
                fetched_hero = st.session_state.fetched_heroes[i]
                try:
                    default_index = hero_names.index(fetched_hero)
                except ValueError:
                    default_index = 0
            
            hero = st.selectbox(
                f"Radiant Hero {i+1}:",
                options=hero_names,
                key=f"radiant_{i}",
                index=default_index
            )
            radiant_heroes.append(hero)
    
    with col2:
        st.markdown("#### 🌙 Dire Team")
        dire_heroes = []
        for i in range(5):
            # Use fetched heroes if available, otherwise use default index
            default_index = 0
            if 'fetched_heroes' in st.session_state and len(st.session_state.fetched_heroes) == 10:
                # Find the hero in the list and get its index
                fetched_hero = st.session_state.fetched_heroes[i+5]  # Dire heroes start from index 5
                try:
                    default_index = hero_names.index(fetched_hero)
                except ValueError:
                    default_index = 0
            
            hero = st.selectbox(
                f"Dire Hero {i+1}:",
                options=hero_names,
                key=f"dire_{i}",
                index=default_index
            )
            dire_heroes.append(hero)
    
    # Check for duplicate heroes
    all_selected_heroes = radiant_heroes + dire_heroes
    if len(set(all_selected_heroes)) != len(all_selected_heroes):
        st.warning("⚠️ Warning: You have selected duplicate heroes. Each hero should be unique in a match.")
    
    st.markdown("---")
    
    # Predict button V1
    if st.button("🔮 Predict Match Duration (v1)", type="primary", use_container_width=True):
        if len(set(all_selected_heroes)) == 10:
            hero_indices = [name_to_idx[hero] for hero in all_selected_heroes]
            probabilities = predict_duration(model, hero_indices)
            if probabilities is not None:
                predicted_class = torch.argmax(probabilities, dim=1).item()
                predicted_prob = torch.max(probabilities, dim=1)[0].item()
                predicted_duration = get_desc_from_label_id(predicted_class)
                st.markdown("### 📊 Prediction Results (v1)")
                st.metric("Predicted Duration", f"{predicted_duration} minutes")
                st.metric("Confidence", f"{predicted_prob:.1%}")
                fig = draw_distribution(probabilities, correct_label=predicted_class, 
                                      heros=[torch.tensor(idx, device='cpu') for idx in hero_indices], idx_to_name=idx_to_name)
                st.pyplot(fig)
        else:
            st.error("❌ Please select 10 unique heroes (5 for each team) before predicting.")

    # Predict button V2
    if st.button("🔮 Predict Match Duration (v2)", type="primary", use_container_width=True):
        if model_v2 is None:
            st.error("Model v2 not found. Please add poc_04_scripted_model.pt to the project directory.")
            return
        if len(set(all_selected_heroes)) == 10:
            hero_indices = [name_to_idx[hero] for hero in all_selected_heroes]
            probabilities_v2 = predict_duration(model_v2, hero_indices)
            if probabilities_v2 is not None:
                predicted_class_v2 = torch.argmax(probabilities_v2, dim=1).item()
                predicted_prob_v2 = torch.max(probabilities_v2, dim=1)[0].item()
                predicted_duration_v2 = get_desc_from_label_id_v2(predicted_class_v2)
                st.markdown("### 📊 Prediction Results (v2)")
                st.metric("Predicted Duration (v2)", f"{predicted_duration_v2} minutes")
                st.metric("Confidence (v2)", f"{predicted_prob_v2:.1%}")
                fig_v2 = draw_distribution(
                    probabilities_v2,
                    correct_label=predicted_class_v2,
                    heros=[torch.tensor(idx, device='cpu') for idx in hero_indices],
                    idx_to_name=idx_to_name,
                    label_fn=get_desc_from_label_id_v2
                )
                st.pyplot(fig_v2)
        else:
            st.error("❌ Please select 10 unique heroes (5 for each team) before predicting.")

    # Compare button (move below the two predict buttons)
    if st.button("🔬 Compare v1 & v2", type="primary", use_container_width=True):
        if model is None or model_v2 is None:
            st.error("Missing model v1 or v2. Please check your model files.")
            return
        if len(set(all_selected_heroes)) == 10:
            hero_indices = [name_to_idx[hero] for hero in all_selected_heroes]
            probabilities_v1 = predict_duration(model, hero_indices)
            probabilities_v2 = predict_duration(model_v2, hero_indices)
            if probabilities_v1 is not None and probabilities_v2 is not None:
                predicted_class_v1 = torch.argmax(probabilities_v1, dim=1).item()
                predicted_prob_v1 = torch.max(probabilities_v1, dim=1)[0].item()
                predicted_duration_v1 = get_desc_from_label_id(predicted_class_v1)

                predicted_class_v2 = torch.argmax(probabilities_v2, dim=1).item()
                predicted_prob_v2 = torch.max(probabilities_v2, dim=1)[0].item()
                predicted_duration_v2 = get_desc_from_label_id_v2(predicted_class_v2)

                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.markdown("### 📊 Prediction Results (v1)")
                    st.metric("Predicted Duration", f"{predicted_duration_v1} minutes")
                    st.metric("Confidence", f"{predicted_prob_v1:.1%}")
                    fig_v1 = draw_distribution(
                        probabilities_v1,
                        correct_label=predicted_class_v1,
                        heros=[torch.tensor(idx, device='cpu') for idx in hero_indices],
                        idx_to_name=idx_to_name,
                        label_fn=get_desc_from_label_id
                    )
                    st.pyplot(fig_v1)
                with col_v2:
                    st.markdown("### 📊 Prediction Results (v2)")
                    st.metric("Predicted Duration", f"{predicted_duration_v2} minutes")
                    st.metric("Confidence", f"{predicted_prob_v2:.1%}")
                    fig_v2 = draw_distribution(
                        probabilities_v2,
                        correct_label=predicted_class_v2,
                        heros=[torch.tensor(idx, device='cpu') for idx in hero_indices],
                        idx_to_name=idx_to_name,
                        label_fn=get_desc_from_label_id_v2
                    )
                    st.pyplot(fig_v2)
            else:
                st.error("Error occurred during prediction with model v1 or v2.")
        else:
            st.error("❌ Please select 10 unique heroes (5 for each team) before comparing.")

    st.markdown("---")
    
    # Predict hero roles button
    if st.button("🛡️ Predict Hero Roles", type="secondary", use_container_width=True):
        role_to_lane = {
            1: "Safe Lane", 
            2: "Mid Lane", 
            3: "Off Lane",
        }
        encoder_filtered, model_enhanced = load_role_artifacts()
        if encoder_filtered is None or model_enhanced is None:
            st.warning("Role prediction artifacts not found. Place 'role_encoder.pkl' and 'role_model.pkl' in the project root.")
        else:
            # Map selected hero names to OpenDota IDs
            radiant_ids = []
            for name in radiant_heroes:
                hid = get_hero_id_from_name(name)
                if hid is not None:
                    radiant_ids.append(hid)
            dire_ids = []
            for name in dire_heroes:
                hid = get_hero_id_from_name(name)
                if hid is not None:
                    dire_ids.append(hid)

            role_preds = predict_hero_roles_enhanced(radiant_ids, dire_ids, encoder_filtered, model_enhanced)
            if not role_preds['radiant'] and not role_preds['dire']:
                st.error("Could not produce role predictions.")
            else:
                st.markdown("### 🛡️ Predicted Hero Roles")
                col_r, col_d = st.columns(2)
                with col_r:
                    st.markdown("**Radiant**")
                    for item in role_preds['radiant']:
                        st.write(f"{get_hero_name_from_id(item['hero_id'])}: {role_to_lane[item['predicted_role']]}")
                with col_d:
                    st.markdown("**Dire**")
                    for item in role_preds['dire']:
                        st.write(f"{get_hero_name_from_id(item['hero_id'])}: {role_to_lane[item['predicted_role']]}")

    # Add some information about the model
    with st.expander("ℹ️ About This Predictor"):
        st.markdown("""
        This model predicts Dota 2 match duration based on the heroes selected for both teams.
        
        **Duration Categories:**
        - ≤10: Very short matches (10 minutes or less)
        - 10-20: Short matches (10-20 minutes)
        - 20-30: Below average matches (20-30 minutes)
        - 30-40: Average matches (30-40 minutes)
        - 40-50: Above average matches (40-50 minutes)
        - 50-60: Long matches (50-60 minutes)
        - 60-70: Very long matches (60-70 minutes)
        - 70-80: Extremely long matches (70-80 minutes)
        - 80-90: Marathon matches (80-90 minutes)
        - >90: Epic matches (over 90 minutes)
        
        **How to use:**
        1. Select 5 heroes for Radiant team
        2. Select 5 heroes for Dire team
        3. Make sure all heroes are unique
        4. Click "Predict Match Duration" to see the results
        5. Use "Compare v1 & v2" to compare both models
        """)

if __name__ == "__main__":
    main()