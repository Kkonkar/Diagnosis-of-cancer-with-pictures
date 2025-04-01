import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from PIL import Image
import time
import locale

# دعم اللغة العربية
locale.setlocale(locale.LC_ALL, 'ar_EG.utf8')

# تحميل البيانات الأساسية
@st.cache_data
def load_data():
    data = pd.read_csv('Social_Network_Ads.csv')
    return data

# تدريب نموذج التشخيص بالبيانات
@st.cache_resource
def train_model(data):
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    X = data[['Age', 'EstimatedSalary', 'Gender']]
    y = data['Purchased']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

# تحليل الصور (محاكاة للتدريب)
def analyze_medical_image(image):
    time.sleep(2)
    return "سليم", 95.3  # نتيجة تجريبية

# الواجهة الرئيسية
def main():
    st.set_page_config(page_title="النظام الشامل للتشخيص الطبي", layout="wide")
    
    # تحميل البيانات والموديل
    data = load_data()
    model, scaler = train_model(data)
    
    # ------------- الشريط الجانبي -------------
    with st.sidebar:
        st.header("⚕️ بيانات المريض الأساسية")
        
        with st.expander("المعلومات الشخصية"):
            age = st.slider("العمر", 18, 100, 30)
            gender = st.selectbox("الجنس", ["ذكر", "أنثى"])
            salary = st.number_input("الدخل الشهري (USD)", 0, 20000, 3000)
        
        if st.button("🩺 تشخيص بالبيانات"):
            gender_encoded = 0 if gender == "ذكر" else 1
            new_data = scaler.transform([[age, salary, gender_encoded]])
            prediction = model.predict(new_data)
            proba = model.predict_proba(new_data)[0]
            confidence = round(max(proba)*100, 2)
            status = "مصاب" if prediction[0] == 1 else "سليم"
            
            if prediction[0] == 1:
                st.error(f"""
                ### حالة المريض: 🚨 {status}
                #### مستوى الثقة: {confidence}%
                """)
            else:
                st.success(f"""
                ### حالة المريض: ✅ {status}
                #### مستوى الثقة: {confidence}%
                """)

    # ------------- الصفحة الرئيسية -------------
    st.title("📷 التشخيص عبر الصور الطبية")
    
    uploaded_file = st.file_uploader("اسحب وأفلت الصورة هنا أو انقر للاختيار", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("الصورة المرفوعة")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="الصورة المرفوعة")
        
        with col2:
            st.subheader("نتائج التحليل")
            
            if st.button("🔬 بدء الفحص"):
                with st.spinner('جارِ تحليل الصورة... الرجاء الانتظار'):
                    diagnosis, confidence = analyze_medical_image(image)
                    st.balloons()
                    if diagnosis == "سليم":
                        st.success(f"""
                        <div style='border-radius:10px; padding:20px; background-color:#e8f5e9;'>
                        <h3 style='color:#2e7d32;'>الحالة: ✅ {diagnosis}</h3>
                        <p style='font-size:18px;'>مستوى الدقة: {confidence}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"""
                        <div style='border-radius:10px; padding:20px; background-color:#ffebee;'>
                        <h3 style='color:#c62828;'>الحالة: 🚨 {diagnosis}</h3>
                        <p style='font-size:18px;'>مستوى الدقة: {confidence}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.info("""
                    **التوصيات الطبية:**
                    1. مراجعة طبيب متخصص للتأكد
                    2. إجراء الفحوصات المخبرية
                    3. المتابعة الدورية كل 3 أشهر
                    """)
    
    st.markdown("---")
    with st.expander("عرض البيانات المرجعية"):
        st.dataframe(data.head(10).style.highlight_max(color='#fff59d'), height=400)

if __name__ == "__main__":
    main()
