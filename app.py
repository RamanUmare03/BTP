import streamlit as st
import torch
import numpy as np
import pandas as pd
import random
import io
import re
from transformers import AutoTokenizer, EsmModel
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================================
# 1. UI Setup & Session State
# ==========================================
st.set_page_config(page_title="Salt Concentration Predictor", layout="wide")
st.title("🧪 Salt Concentration Predictor")
st.markdown(
    "Predict protein crystallization salt molarity using Deep Learning (ESM-2) and Biophysical features.")

# Initialize a history list in the session state to store predictions
if 'history' not in st.session_state:
    st.session_state.history = []

# ==========================================
# 2. Cached Model Training & Setup
# ==========================================


@st.cache_resource
def load_and_train_model():
    with st.spinner("Initializing ESM-2 and training model..."):
        # Synthetic Data Generation
        NUM_SAMPLES = 50
        AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
        SALT_TYPES = ["NaCl", "AmSO4", "LiSO4"]

        # Creating fake training data
        sequences = ["".join(random.choices(
            AMINO_ACIDS, k=random.randint(50, 150))) for _ in range(NUM_SAMPLES)]
        salts = random.choices(SALT_TYPES, k=NUM_SAMPLES)
        target_molarity = np.random.uniform(0.1, 3.0, size=NUM_SAMPLES)

        # Load ESM-2 (Smallest version for CPU efficiency)
        model_name = "facebook/esm2_t12_35M_UR50D"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Batch Extraction of ESM-2 Embeddings
        all_embeddings = []
        for i in range(0, len(sequences), 4):
            batch_seqs = sequences[i:i + 4]
            inputs = tokenizer(batch_seqs, return_tensors="pt",
                               padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            lhs = outputs.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(
                -1).expand(lhs.size()).float()
            mean_pooled = (torch.sum(lhs * mask, dim=1) /
                           torch.clamp(mask.sum(dim=1), min=1e-9)).cpu().numpy()
            all_embeddings.append(mean_pooled)

        embeddings = np.vstack(all_embeddings)
        esm2_cols = [f"esm2_{i}" for i in range(embeddings.shape[1])]
        df_esm2 = pd.DataFrame(embeddings, columns=esm2_cols)

        # Hand-Crafted Feature Extraction
        hand_crafted_data = []
        for seq in sequences:
            analysis = ProteinAnalysis(seq)
            hand_crafted_data.append({
                "pI": analysis.isoelectric_point(),
                "GRAVY": analysis.gravy(),
                "MW": analysis.molecular_weight(),
                "Charge_pH7": analysis.charge_at_pH(7.0)
            })
        df_features = pd.DataFrame(hand_crafted_data)

        # Build Machine Learning Pipeline
        df_main = pd.DataFrame(
            {"Salt_Type": salts, "Target_Molarity": target_molarity})
        df_final = pd.concat([df_features, df_esm2, df_main], axis=1)

        X = df_final.drop(columns=["Target_Molarity"])
        y = df_final["Target_Molarity"]

        numeric_cols = [col for col in X.columns if col != "Salt_Type"]
        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Salt_Type"])
        ])

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", xgb.XGBRegressor(n_estimators=100,
             learning_rate=0.1, max_depth=5, random_state=42))
        ])
        pipeline.fit(X, y)
        return pipeline, tokenizer, model, device


# Load the trained model and resources
pipeline, tokenizer, esm_model, device = load_and_train_model()

# ==========================================
# 3. Input & Prediction Logic
# ==========================================
col_in, col_metrics = st.columns([2, 1])

with col_in:
    st.subheader("Input Sequence")
    raw_input = st.text_area("Paste Amino Acid Sequence here:", height=150)
    test_salt = st.selectbox("Select Salt Type:", ["NaCl", "AmSO4", "LiSO4"])

    if st.button("Predict Molarity", type="primary"):
        # --- STAGE 1: THE CLEANING (REGEX) ---
        # 1. Remove specific Unicode Zero-Width Spaces (\u200b)
        clean_sequence = re.sub(r'\u200b', '', raw_input)
        # 2. Remove all standard whitespace (spaces, tabs, newlines)
        clean_sequence = re.sub(r'\s+', '', clean_sequence)
        # 3. Force to Uppercase and strip edges
        clean_sequence = clean_sequence.strip().upper()

        if not clean_sequence:
            st.error("Error: The sequence input is empty.")
        elif not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in clean_sequence):
            st.warning(
                "Warning: Sequence contains non-standard amino acid characters. Results may be inaccurate.")
            # We proceed anyway, but Biopython might throw its own error if it's too messy

        try:
            with st.spinner("Analyzing sequence..."):
                # --- STAGE 2: BIOPHYSICAL ANALYSIS ---
                analysis = ProteinAnalysis(clean_sequence)
                feat = {
                    "pI": round(analysis.isoelectric_point(), 2),
                    "GRAVY": round(analysis.gravy(), 3),
                    "MW": round(analysis.molecular_weight(), 1),
                    "Charge_pH7": round(analysis.charge_at_pH(7.0), 2),
                    "Salt_Type": test_salt
                }

                # --- STAGE 3: ESM-2 EMBEDDINGS ---
                inputs = tokenizer([clean_sequence], return_tensors="pt",
                                   padding=True, truncation=True, max_length=1024)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = esm_model(**inputs)
                lhs = outputs.last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(
                    -1).expand(lhs.size()).float()
                emb = (torch.sum(lhs * mask, dim=1) /
                       torch.clamp(mask.sum(dim=1), min=1e-9)).cpu().numpy()

                # Merge biophysical features and embeddings into one row
                feature_dict = {**feat}
                for i in range(emb.shape[1]):
                    feature_dict[f"esm2_{i}"] = emb[0, i]

                # --- STAGE 4: PREDICTION ---
                df_new = pd.DataFrame([feature_dict])
                prediction = pipeline.predict(df_new)[0]

                # Update Session History
                result_entry = {
                    "Salt": test_salt,
                    "Prediction (M)": round(float(prediction), 3),
                    "pI": feat["pI"],
                    "MW (Da)": feat["MW"],
                    "GRAVY": feat["GRAVY"],
                    "Charge@pH7": feat["Charge_pH7"]
                }
                st.session_state.history.append(result_entry)

                st.success(f"### Predicted Molarity: {prediction:.3f} M")
        except Exception as e:
            st.error(f"Processing Error: {str(e)}")

with col_metrics:
    st.subheader("Current Results")
    if st.session_state.history:
        latest = st.session_state.history[-1]
        st.metric("Isoelectric Point (pI)", latest["pI"])
        st.metric("MW (Daltons)", f"{latest['MW (Da)']}")
        st.metric("Hydrophobicity (GRAVY)", latest["GRAVY"])
        st.metric("Net Charge @ pH 7", latest["Charge@pH7"])
    else:
        st.info("Results will appear here after prediction.")

# ==========================================
# 4. History Table & Export
# ==========================================
st.divider()
st.subheader("📊 Session History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    # Generate CSV Download
    csv_data = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download All Results as CSV",
        data=csv_data,
        file_name="protein_salt_predictions.csv",
        mime="text/csv",
    )
else:
    st.write("No history to display.")
