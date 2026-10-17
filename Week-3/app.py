"""
Week 3 Mini-Project: Interactive Shakespearean Text Generator (Streamlit)
=======================================================================
This application generates novel Shakespearean-style text using an LSTM neural
network trained character-by-character on a public-domain Shakespeare corpus.
Users can provide a custom prompt, adjust sampling temperature, and control
maximum generation length.
"""

import os
import sys
import time
from pathlib import Path
import streamlit as st

# Add current directory to path so model_utils can be imported
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from model_utils import load_artifacts, generate_text
except ImportError:
    st.error("Failed to import model_utils. Please ensure model_utils.py is in the application directory.")
    st.stop()


# Page Configuration
st.set_page_config(
    page_title="Shakespearean Text Generator — Week 3",
    page_icon="📜",
    layout="wide",
)

MODELS_DIR = APP_DIR / "models"


@st.cache_resource(show_spinner=False)
def get_model_and_vocab():
    """Load and cache the trained LSTM model and vocabulary artifacts."""
    if not (MODELS_DIR / "text_generator.keras").exists():
        return None, None, None, None
    try:
        model, char_to_idx, idx_to_char, config = load_artifacts(str(MODELS_DIR))
        return model, char_to_idx, idx_to_char, config
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None, None


# Quick Sample Prompts
SAMPLE_PROMPTS = [
    "First Citizen:\nBefore we proceed any further, hear me speak.",
    "To be, or not to be, that is the question:",
    "GLOUCESTER:\nNow is the winter of our discontent",
    "ROMEO:\nWhat light through yonder window breaks?",
    "Shall I compare thee to a summer's day?",
]


def main():
    st.title("📜 Shakespearean Text Generation with LSTM")
    st.markdown(
        """
        Welcome to the **Week 3 Deep Learning Interactive Text Generator**.
        This application utilizes a recurrent neural network (**LSTM**) trained character-by-character 
        on William Shakespeare's works. Supply a seed prompt, choose a sampling temperature, 
        and observe how the neural network autoregressively predicts each successive character.
        """
    )

    # Sidebar: Model status, architecture details & documentation
    with st.sidebar:
        st.header("⚙️ Model & Architecture")
        model, char_to_idx, idx_to_char, config = get_model_and_vocab()

        if model is None:
            st.warning("⚠️ Model artifacts not found!")
            st.info(
                """
                To train and export the model:
                1. Run `05_lstm_text_generation_app.ipynb` OR
                2. Run `python Week-3/train_character_model.py`
                
                Artifacts will be saved to `Week-3/models/`.
                """
            )
        else:
            st.success("✅ Model & Vocabulary Loaded")
            st.markdown(f"**Vocabulary Size:** `{config.get('vocab_size', len(char_to_idx))}` unique characters")
            st.markdown(f"**Sequence Length:** `{config.get('seq_length', 40)}` tokens")
            st.markdown(f"**Embedding Dim:** `{config.get('embedding_dim', 64)}`")
            st.markdown(f"**LSTM Units:** `{config.get('lstm_units', 128)}`")
            if "final_accuracy" in config:
                st.markdown(f"**Training Accuracy:** `{config['final_accuracy'] * 100:.1f}%`")
            if "final_loss" in config:
                st.markdown(f"**Final Loss:** `{config['final_loss']:.4f}`")

        st.markdown("---")
        st.subheader("🌡️ Understanding Temperature")
        st.markdown(
            """
            - **Low (< 0.5):** Conservative, repetitive, and adheres strictly to high-probability tokens.
            - **Balanced (0.7 – 0.9):** Natural balance of creativity and grammatical structure.
            - **High (> 1.0):** Highly diverse and creative, but prone to spelling errors and chaos.
            """
        )

    # Main layout: Controls & Generation
    col_input, col_config = st.columns([2, 1])

    with col_config:
        st.subheader("Generation Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.5,
            value=0.8,
            step=0.05,
            help="Higher values produce more diverse output; lower values produce more conservative output.",
        )
        gen_length = st.slider(
            "Characters to Generate",
            min_value=50,
            max_value=500,
            value=200,
            step=25,
            help="Number of new characters to predict autoregressively.",
        )

        st.markdown("**Quick Preset Prompts:**")
        preset_cols = st.columns(2)
        for i, prompt in enumerate(SAMPLE_PROMPTS[:4]):
            col = preset_cols[i % 2]
            label = prompt.split(":")[0] if ":" in prompt else prompt[:18] + "..."
            if col.button(label, key=f"preset_{i}"):
                st.session_state["seed_input"] = prompt

    with col_input:
        st.subheader("Seed Prompt")
        default_seed = "First Citizen:\nBefore we proceed any further, hear me speak."
        current_seed = st.session_state.get("seed_input", default_seed)

        seed_text = st.text_area(
            "Enter initial text to prime the LSTM:",
            value=current_seed,
            height=130,
            key="seed_area",
            help="The network reads this sequence to initialize its hidden state and predict the next character.",
        )

        # Check for characters outside vocabulary
        if char_to_idx is not None and seed_text:
            unknown_chars = sorted(list(set([ch for ch in seed_text if ch not in char_to_idx])))
            if unknown_chars:
                st.info(
                    f"ℹ️ Note: Characters {unknown_chars} are not in the Shakespeare vocabulary "
                    "and will be safely ignored during sequence vectorization."
                )

        btn_col1, btn_col2 = st.columns([1, 1])
        generate_clicked = btn_col1.button("✨ Generate Text", type="primary", use_container_width=True)
        reset_clicked = btn_col2.button("🔄 Reset to Default", use_container_width=True)

        if reset_clicked:
            st.session_state["seed_input"] = default_seed
            st.session_state["generated_result"] = None
            st.rerun()

    # Execution & Results
    st.markdown("---")
    st.subheader("Generated Output")

    if generate_clicked:
        if model is None:
            st.error("Cannot generate text: Model artifacts are missing. Please train the model first.")
        elif not seed_text.strip():
            st.warning("Please provide a non-empty seed prompt to prime the generator.")
        else:
            with st.spinner(f"Generating {gen_length} characters at temperature {temperature}..."):
                start_time = time.time()
                seq_length = config.get("seq_length", 40)
                try:
                    generated_new = generate_text(
                        model=model,
                        char_to_idx=char_to_idx,
                        idx_to_char=idx_to_char,
                        seq_length=seq_length,
                        seed_text=seed_text,
                        gen_length=gen_length,
                        temperature=temperature,
                    )
                    elapsed = time.time() - start_time
                    st.session_state["generated_result"] = {
                        "seed": seed_text,
                        "generated": generated_new,
                        "temp": temperature,
                        "len": gen_length,
                        "time": elapsed,
                    }
                except Exception as ex:
                    st.error(f"Generation error: {ex}")

    # Display result if present in session state
    result = st.session_state.get("generated_result", None)
    if result:
        m1, m2, m3 = st.columns(3)
        m1.metric("Temperature", f"{result['temp']:.2f}")
        m2.metric("Characters Generated", f"{len(result['generated'])}")
        m3.metric("Latency", f"{result['time']:.2f} s")

        full_output = f"{result['seed']}{result['generated']}"
        st.markdown(
            f"""
            <div style="background-color: #0e1117; color: #f0f2f6; padding: 20px; border-radius: 8px; border: 1px solid #30363d; font-family: monospace; white-space: pre-wrap; line-height: 1.5; font-size: 15px;">
<span style="color: #58a6ff; font-weight: bold;">{result['seed']}</span><span style="color: #3fb950;">{result['generated']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Blue: User Seed Prompt | Green: Autoregressively Generated Text")
    else:
        st.info("Click **✨ Generate Text** above to produce new Shakespearean text.")


if __name__ == "__main__":
    main()
