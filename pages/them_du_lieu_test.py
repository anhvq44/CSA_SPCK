from datetime import datetime

import pandas as pd
import streamlit as st

from app import menu

menu()

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            display: none
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("Thêm dữ liệu mới")

listing_date = st.date_input("Ngày niêm yết", datetime.today())
listing_date = listing_date.strftime("%m/%d/%Y")
area = st.number_input("Diện tích (m2)", min_value=1.0, value=50.0, step=1.0)
num_bedroom = st.number_input("Số phòng ngủ", min_value=0, value=1, step=1)
floors = st.number_input("Số tầng", min_value=1, value=1, step=1)
num_bathroom = st.number_input("Số phòng vệ sinh", min_value=0, value=1, step=1)