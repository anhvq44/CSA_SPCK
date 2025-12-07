import streamlit as st
import os

def menu():
    st.sidebar.page_link("app.py", label="Home")
    st.sidebar.page_link("pages\phan_tich_du_lieu.py", label="Phân Tích Dữ Liệu")
    st.sidebar.page_link("pages/them_du_lieu_test.py", label="Thêm Dữ Liệu Tập Test")
    st.sidebar.page_link("pages\du_doan_model.py", label="Phân Tích Dự Đoán")
    
    
if __name__ == "__main__":
    st.set_page_config(
        page_title="Phân tích giá nhà ở tại Thành Phố Hồ Chí Minh",
        layout="centered",
        page_icon="👋",
    )
    
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

    st.title("Phân tích giá nhà ở Hà Nội")  # tên sản phẩm
    st.header("Chức năng")
    st.markdown(
        """
    1. Xem phân tích tập dữ liệu ...
    2. Thêm dữ liệu mới và cập nhật các biểu đồ
    3. Sử dụng AI để dự đoán ...
    """
    )

    st.subheader("Credits")
    st.markdown(
        """
        Ứng dựng được xây dựng với [streamlit](https://streamlit.io) và [Plotly](https://plotly.com/).
        
        Được phát triển bởi [Quế Anh](https://github.com/anhvq44/CSA_SPCK)
        """
    )

    menu()