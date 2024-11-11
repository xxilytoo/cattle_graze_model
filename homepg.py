import streamlit as st

# Streamlit Multi-paging
pg = st.navigation([st.Page("yolov11ft.py", title="Cattle Aerial Identifier Yolov11 Fine Tuned Model"), st.Page("about.py", title="About the Project")])
st.set_page_config(page_title="Cattle Detection Pro", page_icon=":material/edit:")
pg.run()