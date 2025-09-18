# app.py
import streamlit as st
import tempfile
import os
from prediction import classify_objects, draw_annotations, estimate_weight

# --- STREAMLIT UI ---
st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="centered")

st.title("🗑️ Waste Detection & Classification")
st.markdown("Upload an image to detect and classify waste objects.")

uploaded_file = st.file_uploader("📁 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_image_path = tmp_file.name

    try:
        st.image(temp_image_path, caption="📸 Original Image", use_container_width=True)

        if st.button("🔍 Analyze Waste"):
            results = classify_objects(temp_image_path)

            if not results:
                st.warning("No waste objects detected.")
            else:
                output_path = "annotated_result.jpg"
                annotated_img_path = draw_annotations(temp_image_path, results, output_path)
                st.image(annotated_img_path, caption=" Annotated Result", use_container_width=True)

                with st.expander("📋 Detection Details"):
                    for i, obj in enumerate(results, 1):
                        bbox = obj['bbox']
                        area_px = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        px_to_cm = 0.1
                        area_cm2 = area_px * (px_to_cm ** 2)
                        weight_kg = estimate_weight(obj['category'], area_cm2)

                        color_hex = {
                            "Dry Waste": "#00FF00",
                            "Wet Waste": "#FFFF00",
                            "Hazardous Waste": "#FF0000",
                            "Electronic Waste": "#FF00FF",
                            "Construction Waste": "#00FFFF",
                            "Biomedical Waste": "#800080"
                        }.get(obj['category'], "#FFFFFF")

                        # Show weight only if not Hazardous or Construction Waste
                        weight_line = ""
                        if obj['category'] not in ["Hazardous Waste", "Construction Waste"]:
                            weight_line = f"\n⚖️ *Tentative Weight*: **{weight_kg:.2f} kg**"

                        st.markdown(f"""
                        **{i}. {obj['object']}**  
                        🗃️ *Category*: <span style="color:{color_hex}; font-weight:bold;">{obj['category']}</span>   
                        📐 *Area*: **{area_cm2:.1f} cm²**{weight_line}  
                        🧹 *How to Dispose*: {obj['disposal']}
                        """, unsafe_allow_html=True)

                with open(annotated_img_path, "rb") as f:
                    st.download_button(
                        label="💾 Download Annotated Image",
                        data=f,
                        file_name="waste_detection_result.jpg",
                        mime="image/jpeg"
                    )

    finally:
        if os.path.exists(temp_image_path):
            try:
                os.unlink(temp_image_path)
            except PermissionError:
                st.warning("⚠️ Could not delete temporary file. Please close any programs using it.")
            except Exception as e:
                st.error(f"Error deleting temp file: {e}")

# Footer
st.markdown("---")
