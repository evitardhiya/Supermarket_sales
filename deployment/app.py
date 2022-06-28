import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import scipy.stats as stats

#set page
st.set_page_config(
    page_title="Milestone 1",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.github.com/evitardhiya',
        'Report a bug': 'https://www.google.com',
        'About': 'Milestone 1, by: Evita Ardhiya Ramadhani Batch 11-Hacktiv8 Full Time Data Science'
    }
)

#@st.cache
# load data
def load_data():
    df = pd.read_csv('E:\Hacktiv8\Assignments\Phase0\week4\p0---ftds011---ml1-evitardhiya\supermarket_sales - Sheet1.csv')
    return df

data = load_data()

st.title('Milestone 1')
st.caption('Hacktiv8 Full Time Data Science')

st.sidebar.subheader('Page Menu')

pages = st.sidebar.selectbox('Select page:', options=['Visualisasi', 'Hypotesis Testing'])


if pages == 'Visualisasi':

    st.header('Data Visualisasi')

    if st.checkbox('Tampilkan dataset Supermarket Sales'):
        st.subheader('Raw data')
        st.write(data)

    # -----------------------visualisasi 1--------------------------------------------
    st.header('Visualisasi 1')
    st.write('Pendapatan supermarket dari bulan januari hingga Maret')

    data['Date'] = data['Date'].astype('datetime64[ns]')
    sorted_data = data.sort_values(by='Date', ascending=True).groupby('Date').sum()

    fig, ax = plt.subplots(figsize=(19,5))
    plt.title('Pendapatan Harian Supermarket')
    sd = sorted_data['income']
    st.area_chart(sd)

    # -----------------------visualisasi 2-------------------------------------------
    st.header('Visualisasi 2')
    st.write('Jumlah Product line pada tiap City')

    selected = st.radio('Select City:', options=['Yangon', 'Mandalay', 'Naypyitaw'])

    fig2, ax2 = plt.subplots(figsize=(5,2))
    plt.title('Jumlah Product line')
    product = data[(data['City'] == selected)]['Product line'].value_counts()
    product.plot(kind='barh', color='red')

    st.pyplot(fig2)

    # -----------------------visualisasi 3-------------------------------------------
    st.header('Visualisasi 3')
    st.write('Jam kunjungan customer terbanyak')

    data['Time'] = pd.to_datetime(data['Time'])
    data['Hour'] = (data['Time']).dt.hour
    tm_sort = data.sort_values(by='Hour', ascending=True).groupby('Hour').sum()

    fig3, ax3 = plt.subplots(figsize=(19,5))
    plt.title('Jam kunjungan customer')
    tm = tm_sort['Quantity']
    st.line_chart(tm)


    # # -----------------------visualisasi 1-------------------------------------------
    st.header('Visualisasi 4')
    st.write('Jenis pembayaran yang digunakan Perempuan dan Laki-laki')

    selected2 = st.radio('Select Gender:', options=['Female', 'Male'])

    fig4, ax4 = plt.subplots(figsize=(7,4))
    plt.title('Payment')
    pay = data[(data['Gender'] == selected2)]['Payment'].value_counts()
    pay.plot(kind='bar', color='pink')
    plt.xticks(rotation=0)

    st.pyplot(fig4)

else:

    st.subheader('Hypotesis Testing')
    st.caption('Two Samples Independent Two Tailed test, Confident interval 95%')

    st.write('Tujuan pengujian hipotesis, menguji apakah rata-rata Pendapatan harian dari suatu Kota berbeda secara signifikan atau tidak. Sampel yang akan diuji meggunakan Yangon dan Mandalay.')

    if st.checkbox('Rata-rata pendapatan'):
        st.write('Rata-rata pendapatan harian Yangon: 1136.0')
        st.write('Rata-rata pendapatan harian Mandalay: 1176.0')

    if st.checkbox('Hipotesis'):
        st.write('H0 : μ_yangon = μ_mandalay')
        st.write('H1 : μ_yangon != μ_mandalay')

    if st.checkbox('Distribusi'):
        # Menghitung rata-rata pendapatan harian dari kota Yangon
        daily_yangon = data[(data['City'] == 'Yangon')][['Date', 'income']].groupby('Date').sum()
        mean_yangon = np.round(daily_yangon['income'].mean())

        # Menghitung rata-rata pendapatan harian dari kota Mandalay
        daily_mandalay = data[(data['City'] == 'Mandalay')][['Date', 'income']].groupby('Date').sum()
        mean_mandalay = np.round(daily_mandalay['income'].mean())

        # menguji hipotesis menggunakan t-test ind
        t_stat, p_val = stats.ttest_ind(daily_yangon, daily_mandalay)

        # Distribusi Yangon dan Mandalay
        yangon = np.random.normal(daily_yangon.income.mean() , daily_yangon.income.std(),10000)
        mandalay = np.random.normal(daily_mandalay.income.mean(),daily_mandalay.income.std(),10000)

        # Membuat confident interval 95%
        ci = stats.norm.interval(0.95, daily_yangon.income.mean(), daily_yangon.income.std())

        # Membuat batasan hipotesis alternatif dari Yangon
        alt_hipo1 = yangon.mean() + t_stat[0]*yangon.std()
        alt_hipo2 = yangon.mean() - t_stat[0]*yangon.std()

        # plot distribusi normal
        fig5, ax = plt.subplots(figsize=(17,7))
        sns.distplot(yangon, label='Rata-rata pendapatan harian Yangon',color='purple')
        sns.distplot(mandalay, label='Rata-rata pendapatan harian Mandalay',color='yellow')

        # mean dari Yangon dan Mandalay
        plt.axvline(mean_yangon, color='purple', linewidth=2, label='Yangon mean')
        plt.axvline(mean_mandalay, color='yellow',  linewidth=2, label='Mandalay mean')

        # garis confident interval 95%
        plt.axvline(ci[1], color='red', linestyle='dashed', label='Confident interval 95%', linewidth=2)
        plt.axvline(ci[0], color='red', linestyle='dashed', linewidth=2)

        # garis hipotesis alternatif
        plt.axvline(alt_hipo1, color='black', linestyle='dashed', linewidth=2, label = 'Hipotesis Alternatif')
        plt.axvline(alt_hipo2, color='black', linestyle='dashed', linewidth=2)

        plt.legend()

        st.pyplot(fig5)

    if st.checkbox('Kesimpulan'):
        st.write('Diperoleh nilai p-value = 0.7377341666768823 yang berarti nilai p-value lebih besar dari nilai alfa.\
        Berdasarkan perhitungan nilai p-value dan visualisasi distribusi dapat diketahui bahwa kita **fail to reject H0**.\
        Artinya, rata-rata pendapatan harian Yangon dan Mandalay **tidak memiliki perbedaan yang signifikan**.')






