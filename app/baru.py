import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Judul Aplikasi
st.title("Analisis Faktor yang Mempengaruhi Kepuasan Penumpang Maskapai Untuk Meningkatkan Kualitas Maskapai di Indonesia")
st.divider()
# Fitur Upload Dataset
with st.expander("Upload File CSV"):
    uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.balloons()

    data = df.dropna(axis=0)
    new_df = data.drop(['Unnamed: 0', 'id', 'Age', 'Gender', 'Customer Type', 'Type of Travel', 'Class', 'Flight Distance'], axis=1)
    dff = new_df.drop(['Departure/Arrival time convenient', 'Gate location', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'], axis=1)
    # Sidebar Menu
    with st.sidebar:
        st.sidebar.title("Menu Utama")
        main_menu = st.sidebar.selectbox("Pilih Menu", ["⏳ Preprocessing", "🔍 Analisis Data", "📊 Visualisasi Data"])

    st.divider()
    # Menu Preprocessing
    if main_menu == "⏳ Preprocessing":
        st.header("Preprocessing Dataset")
        st.subheader("Informasi Dataset")
        txt = st.text_area(
            "Summary",
            "Dataset ini berisi survei kepuasan penumpang maskapai penerbangan. "
            "Terdapat berbagai faktor yang memengaruhi kepuasan penumpang, "
            "oleh sebab itu melalui aplikasi ini, maskapai penerbangan dapat mengetahui "
            "faktor apa saja yang memiliki korelasi tinggi dengan kepuasan "
            "(atau ketidakpuasan) penumpang\n\n"
            "- Gender: Gender of the passengers (Female, Male)\n"
            "- Customer Type: The customer type (Loyal customer, disloyal customer)\n"
            "- Age: The actual age of the passengers\n"
            "- Type of Travel: Purpose of the flight of the passengers (Personal Travel, Business Travel)\n"
            "- Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus)\n"
            "- Flight distance: The flight distance of this journey\n"
            "- Inflight wifi service: Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)\n"
            "- Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient\n"
            "- Ease of Online booking: Satisfaction level of online booking\n"
            "- Gate location: Satisfaction level of Gate location\n"
            "- Food and drink: Satisfaction level of Food and drink\n"
            "- Online boarding: Satisfaction level of online boarding\n"
            "- Seat comfort: Satisfaction level of Seat comfort\n"
            "- Inflight entertainment: Satisfaction level of inflight entertainment\n"
            "- On-board service: Satisfaction level of On-board service\n"
            "- Leg room service: Satisfaction level of Leg room service\n"
            "- Baggage handling: Satisfaction level of baggage handling\n"
            "- Check-in service: Satisfaction level of Check-in service\n"
            "- Inflight service: Satisfaction level of inflight service\n"
            "- Cleanliness: Satisfaction level of Cleanliness\n"
            "- Departure Delay in Minutes: Minutes delayed when departure\n"
            "- Arrival Delay in Minutes: Minutes delayed when Arrival\n"
            "- Satisfaction: Airline satisfaction level(Satisfaction, neutral or dissatisfaction)",
        )

        with st.expander("Tampilkan Data"):
            st.write(df)
            st.write(f"Dataset ini terdiri dari {df.shape[0]} baris dan {df.shape[1]} kolom.")
            st.write(data.dtypes)

        with st.expander("Cek Nilai Null"):
            st.write(df.isnull().sum())

        data = df.dropna(axis=0)
        with st.expander("Menghapus Nilai Null"):
            st.write(data)
            st.write(f"Berhasil menghapus nilai null!! Dataset ini terdiri dari {data.shape[0]} baris dan {data.shape[1]} kolom.")

        with st.expander("Cek Hasil Drop Nilai Null"):
            st.write(data.isnull().sum())

        new_df = data.drop(['Unnamed: 0', 'id', 'Age', 'Gender', 'Customer Type', 'Type of Travel', 'Class', 'Flight Distance'], axis=1)
        with st.expander("Feature Elimination"):
            st.text_area(
                "",
                "Kolom 'Unnamed: 0', 'id', 'Age', 'Gender', 'Customer Type', 'Type of Travel', 'Class', 'Flight Distance'" 
                "tidak termasuk atribut yang memengaruhi tingkat kepuasan penumpang, maka untuk meningkatkan" 
                "akurasi model, kolom tersebut dapat dihapus.",
                )
            st.write(new_df)
            st.write(f"Berhasil menghapus atribut yang dipilih!!")
            st.write(f"Dataset ini terdiri dari {new_df.shape[0]} baris dan {new_df.shape[1]} kolom.")

        with st.expander("Tipe Data Setiap Atribut"):
            st.write(new_df.dtypes)

        with st.expander("Encoding Atribut Satisfaction"):
            new_df['satisfaction'] = new_df["satisfaction"].replace({"satisfied": 1, "neutral or dissatisfied": 0})
            new_df['satisfaction'] = new_df['satisfaction'].astype('int64')
            st.write(new_df)

        with st.expander("Tipe Data Setiap Atribut setelah Encoding"):
            st.write(new_df.dtypes)

        with st.expander("Matriks Korelasi"):
            fig = plt.figure(figsize=(12, 9))
            sns.heatmap(new_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
            st.pyplot(fig)

        dff = new_df.drop(['Departure/Arrival time convenient', 'Gate location', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'], axis=1)
        with st.expander("Drop Atribut yang Berkorelasi Negatif"):
            st.text_area(
                "", "Drop atribut yang berkorelasi negatif karena dapat menurunkan akurasi model. Drop atribut 'Departure/Arrival time convenient', 'Gate location', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'")
            st.write(dff)
            st.write(f"Berhasil menghapus atribut yang dipilih!")
            st.write(f"Dataset ini terdiri dari {dff.shape[0]} baris dan {dff.shape[1]} kolom.")

    # Menu Analisis Data
    elif main_menu == "🔍 Analisis Data":
        st.header("Analisis Data")
        
        # Split data
        x = dff.drop('satisfaction', axis=1)
        y = dff['satisfaction']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Train model with class weights to handle potential imbalance
        rdf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, 
                                        random_state=42, class_weight="balanced")
        rdf_model.fit(x_train, y_train)

        # Show feature importance to analyze which features impact satisfaction
        st.subheader("Feature Paling Berpengaruh Terhadap Satisfaction")
        feature_importances = pd.Series(rdf_model.feature_importances_, index=x.columns).sort_values(ascending=False)
        st.bar_chart(feature_importances)

        # Show model accuracy
        predict = rdf_model.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, predict)
        st.write("Tingkat Akurasi Model:", round(accuracy * 100, 2), "%")

        y_test_pred = rdf_model.predict(x_test)
        st.subheader('Classification Report Testing model Random Forest Classifier')

        #visualisasi data testing menggunakan confusion matrix
        confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_test_pred), ('0', '1'), ('0', '1'))
        fig = plt.figure()
        heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size':14}, fmt='d', cmap='YlGnBu')
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=18)
        plt.title('CM untuk testing model Random Forest', fontsize=14, color='darkblue')
        plt.xlabel('True label', fontsize=14)
        plt.ylabel('Predicted label', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)

    elif main_menu == "📊 Visualisasi Data":
        st.header("Visualisasi Data")

        # "Persentase Jumlah Penerbangan Berdasarkan Type of Travel"
        type_of_travel_counts = df['Type of Travel'].value_counts()
        fig = px.pie(
            names=type_of_travel_counts.index,
            values=type_of_travel_counts.values,
            title='Percentage of Type of Travel',
            color_discrete_map={'Thur': 'lightcyan', 'Fri': 'cyan'}
        )
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label'
        )

        fig.update_layout(showlegend=True, legend_title_text='Type of Travel')
        st.plotly_chart(fig)
        txt = st.text_area(
            "Penjelasan",
            "Visualisasi tersebut menunjukkan bahwa sebagian besar penumpang menggunakan "
            "pesawat terbang untuk perjalanan bisnis",
        )

        data['Class'] = data['Class'].astype(str)
        data = data.dropna(subset=['Online boarding', 'Class'])

        fig = px.histogram(
            data, x='Class', 
            title='Jumlah Penumpang Berdasarkan Class', 
            labels={'Class': 'Tipe Class', 'count': 'Jumlah'},
            color_discrete_sequence=['#636EFA'] 
        )

        fig.update_layout(
            xaxis_title="Tipe Class",
            yaxis_title="Count",
            title_font_size=16,
            title_x=0  
        )
        st.plotly_chart(fig)
        txt = st.text_area(
            "Penjelasan",
            "Visualisasi tersebut menunjukkan bahwa walaupun sebagian besar perjalanan  "
            "bertujuan untuk kepentingan bisnis, jumlah pemesanan tiket untuk kelas ekonomi "
            "tidak jauh berbeda dari kelas bisnis. Artinya, walaupun dalam rangka perjalanan bisnis, "
            "tiket yang dipesan tidak harus untuk kelas bisnis juga",
            )

        # Perbandingan Jumlah Satisfaction Berdasarkan Class Penerbangan
        fig = px.histogram(
            data, x='Class', color='satisfaction',
            title='Perbandingan Jumlah Satisfaction Berdasarkan Class Penerbangan',
            labels={'Class': 'Class Penerbangan', 'count': 'Jumlah Penumpang'},
            barmode='group',  # Mengelompokkan berdasarkan nilai 'satisfaction'
            color_discrete_map={
                'High': 'mediumslateblue',   
                'Low': 'mediumturquoise'   
            } 
        )
        fig.update_layout(
            xaxis_title="Class Penerbangan",
            yaxis_title="Jumlah Penumpang",
            title_font_size=16,
            title_x=0 
        )
        st.plotly_chart(fig)
        txt = st.text_area(
            "Penjelasan",
            "Visualisasi tersebut menunjukkan bahwa sebagian besar penumpang yang puas berasal dari kelas bisnis, "
            "sedangkan penumpang yang tidak puas kebanyakan berasal dari kelas ekonomi",
            )
        
        # Perbandingan Jumlah Satisfaction Berdasarkan Seat Comfort
        # Perbandingan Jumlah Satisfaction Berdasarkan Seat Comfort
        fig = px.histogram(
            data, 
            x='Class', 
            color='Online boarding', 
            barmode='group',  
            title='Perbandingan Jumlah Penumpang Setiap Class Berdasarkan Seat Comfort',
            labels={'Class': 'Class', 'Online boarding': 'Online Boarding', 'count': 'Jumlah Penumpang'},
            color_discrete_map={
                1: 'indigo',             
                2: 'mediumslateblue',    
                3: 'slateblue',        
                4: 'mediumpurple',       
                5: 'lavender'            
            } 
        )
        fig.update_layout(
            xaxis_title="Class",
            yaxis_title="Jumlah Penumpang",
            title_font_size=16,
            title_x=0
        )
        st.plotly_chart(fig)
        txt = st.text_area(
            "Penjelasan",
            "Online boarding merupakan faktor yang paling berpengaruh dalam menentukan kepuasan penumpang.Oleh karena itu, "
            "dengan menggunakan indikator tersebut, dibuat visualisasi yang menunjukkan tingkat kepuasan penumpang "
            "tiap kelas berdasarkan Online boarding. Berdasarkan visualisasi tersebut, penilaian Online boarding dari kelas"
            " ekonomi rata-rata dari rentang 2-4, sedangkan untuk kelas bisnis sebagian besar penumpang memberi "
            " penilaian > 4 yang mengindikasikan bahwa penumpang puas dengan tingkat kenyamanan seat yang disediakan"
            "berdasarkan hasil tersebut maskapai dapat melakukan evaluasi dengan kondisi dan tingkat kenyaman dari seat yang"
            " disediakan pada kelas ekonomi",
            )
        
        st.divider()
        sentiment_mapping = ["one", "two", "three", "four", "five"]
        selected = st.feedback("stars")
        if selected is not None:
            st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")
        st.write("Give your feedback 😊 ")

else:
    st.warning("Silakan upload file CSV untuk memulai analisis.")
