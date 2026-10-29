import io
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="AI Fashion Pattern Studio",
    page_icon="👗",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "fashion_pattern_generator.keras"


@st.cache_resource
def load_generator():
    return tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )


def to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_pattern(generator, seed):
    noise = tf.random.stateless_normal(
        (1, 100), seed=(int(seed), 123)
    )
    result = generator(noise, training=False)[0].numpy()
    result = np.clip((result + 1) * 127.5, 0, 255)
    return Image.fromarray(result.astype(np.uint8))


def apply_pattern(garment, pattern):
    from scipy import ndimage

    original = np.asarray(
        garment.convert("RGB")
    ).astype(np.float32)

    height, width = original.shape[:2]

    texture = np.asarray(
        pattern.convert("RGB").resize(
            (width, height), Image.Resampling.BICUBIC
        )
    ).astype(np.float32)

    brightness = original.mean(axis=2)
    saturation = (
        original.max(axis=2) - original.min(axis=2)
    )

    # Detect the light fabric.
    mask = (brightness > 195) & (saturation < 75)

    # Fill small holes caused by shadows and image noise.
    mask = ndimage.binary_closing(
        mask, structure=np.ones((11, 11))
    )
    mask = ndimage.binary_fill_holes(mask)

    # Smooth the mask edges.
    alpha = ndimage.gaussian_filter(
        mask.astype(np.float32), sigma=2
    )
    alpha = np.clip(alpha, 0, 1)[:, :, None]

    # Preserve the shirt's original lighting and shading.
    shade = np.clip(
        brightness[:, :, None] / 235.0, 0.65, 1.15
    )
    texture = np.clip(texture * shade, 0, 255)

    result = (
        original * (1 - alpha * 0.78)
        + texture * (alpha * 0.78)
    )

    return Image.fromarray(
        np.clip(result, 0, 255).astype(np.uint8)
    )

st.title("AI Fashion Pattern Studio")
st.write(
    "Generate fabric patterns using your trained GAN "
    "and preview them on a garment."
)

try:
    generator = load_generator()
except Exception as error:
    st.error(f"Unable to load the GAN model: {error}")
    st.stop()

st.header("1. Generate a Fabric Pattern")

seed = st.number_input(
    "Pattern number",
    min_value=1,
    max_value=999999,
    value=42,
    step=1
)

if st.button("Generate Pattern", type="primary"):
    try:
        with st.spinner("Generating pattern..."):
            st.session_state["pattern"] = generate_pattern(
                generator, seed
            )
            st.session_state.pop("redesigned", None)
    except Exception as error:
        st.error(f"Pattern generation failed: {error}")

if "pattern" in st.session_state:
    st.image(
        st.session_state["pattern"],
        caption="GAN-generated fabric pattern",
        width=320
    )
    st.download_button(
        "Download Fabric Pattern",
        data=to_png_bytes(st.session_state["pattern"]),
        file_name="generated_bandhani_pattern.png",
        mime="image/png"
    )

st.divider()
st.header("2. Upload Your Garment")

uploaded = st.file_uploader(
    "Choose a plain white or light-coloured shirt/dress",
    type=["png", "jpg", "jpeg"],
    key="garment_upload"
)

if uploaded is not None:
    garment = Image.open(uploaded).convert("RGB")
    st.image(garment, caption="Original garment", width=400)

    if "pattern" not in st.session_state:
        st.warning("Generate a pattern first.")
    else:
        if st.button("Apply Pattern to Garment", type="primary"):
            with st.spinner("Applying pattern..."):
                st.session_state["redesigned"] = apply_pattern(
                    garment, st.session_state["pattern"]
                )

        if "redesigned" in st.session_state:
            st.header("3. Redesigned Garment")
            col1, col2 = st.columns(2)

            with col1:
                st.image(garment, caption="Before", use_container_width=True)

            with col2:
                st.image(
                    st.session_state["redesigned"],
                    caption="After applying pattern",
                    use_container_width=True
                )

            st.download_button(
                "Download Redesigned Garment",
                data=to_png_bytes(st.session_state["redesigned"]),
                file_name="redesigned_garment.png",
                mime="image/png"
            )

st.caption(
    "Prototype: pattern placement uses basic image processing, "
    "so the background may also be affected."
)
