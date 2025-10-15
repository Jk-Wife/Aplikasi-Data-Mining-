
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
from collections import Counter
import re
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Market Basket Analysis - Apriori",
    page_icon="🛒",
    layout="wide"
)

def preprocess_product_string(product_string):
    """Bersihkan dan split string produk dengan handling yang lebih robust"""
    if pd.isna(product_string) or product_string == '':
        return []
    
    try:
        # Convert to string dan bersihkan
        products = str(product_string).split(',')
        cleaned_products = []
        
        for product in products:
            # Hilangkan content dalam kurung (kuantitas, dll)
            cleaned = re.sub(r'\([^)]*\)', '', product)
            
            # Hilangkan angka dan karakter khusus
            cleaned = re.sub(r'\d+', '', cleaned)
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
            
            # Standardisasi nama produk
            cleaned = cleaned.strip()
            cleaned = cleaned.lower()
            
            # Fix common typos
            replacements = {
                'milksahke': 'milkshake',
                'blackpaper': 'blackpepper',
                'blackpeper': 'blackpepper',
                'coffe': 'coffee',
                'es teh': 'teh',
                'americano': 'coffee americano',
                'vege': 'vegetable',
                'geprek': 'ayam geprek',
                'kentang': 'potato',
                'nasi': 'rice',
                'mineral': 'water',
                'fanta': 'soft drink fanta',
                'sprite': 'soft drink sprite',
                'coca-cola': 'soft drink coca cola',
                'teh': 'tea',
                'kopi': 'coffee',
                'cheesy': 'cheese',
                'chessy': 'cheese'
            }
            
            for wrong, correct in replacements.items():
                cleaned = cleaned.replace(wrong, correct)
            
            # Hilangkan spasi berlebih
            cleaned = ' '.join(cleaned.split())
            
            if cleaned and cleaned not in ['', 'nan', 'null']:
                # Capitalize first letter of each word
                cleaned = ' '.join(word.capitalize() for word in cleaned.split())
                cleaned_products.append(cleaned)
        
        return cleaned_products
    
    except Exception as e:
        return []

def create_transaction_data(df, product_column):
    """Buat data transaksi dari kolom produk"""
    transactions = []
    skipped_transactions = 0
    
    for index, row in df.iterrows():
        product_string = row[product_column]
        products = preprocess_product_string(product_string)
        
        if len(products) >= 1:
            transactions.append(products)
        else:
            skipped_transactions += 1
    
    if skipped_transactions > 0:
        st.warning(f"⚠️ {skipped_transactions} transaksi dilewati karena tidak ada produk valid")
    
    return transactions

def load_data(uploaded_file):
    """Load data dari file CSV atau Excel"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            file_type = "CSV"
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            file_type = "Excel"
        else:
            st.error("Format file tidak didukung. Harap upload file CSV atau Excel.")
            return None, None
        
        return df, file_type
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None

def safe_apriori_analysis(transactions, min_support=0.01, max_len=3):
    """Implementasi Apriori yang aman dari error encoding"""
    try:
        # Encoding transaksi
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        
        # Pastikan dataframe encoded berisi boolean
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)
        df_encoded = df_encoded.astype(bool)
        
        # Jalankan Apriori
        frequent_itemsets = apriori(
            df_encoded, 
            min_support=min_support, 
            use_colnames=True,
            max_len=max_len
        )
        
        return frequent_itemsets, df_encoded, None
        
    except Exception as e:
        return None, None, str(e)

def main():
    st.title("🛒 Market Basket Analysis dengan Algoritma Apriori")
    st.markdown("Analisis pola pembelian produk menggunakan Association Rules")
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload File Data (CSV atau Excel)", 
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            with st.spinner('Memuat data...'):
                df, file_type = load_data(uploaded_file)
            
            if df is not None:
                st.success(f"✅ File {file_type} berhasil dimuat: {len(df)} records")
                
                # Tampilkan data preview
                st.subheader("📊 Preview Data")
                st.dataframe(df.head(10))
                
                # Pilih kolom produk
                st.subheader("🎯 Pilih Kolom Analisis")
                
                product_column = st.selectbox(
                    "Pilih kolom yang berisi data produk:",
                    options=df.columns.tolist(),
                    index=3 if 'Produk Yang Dibeli/Dijual' in df.columns else 0
                )
                
                # Parameter algoritma
                st.subheader("📈 Parameter Algoritma Apriori")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    min_support = st.slider(
                        "Minimum Support",
                        min_value=0.01,
                        max_value=0.3,
                        value=0.05,
                        step=0.01,
                        help="Frekuensi minimal kemunculan itemset"
                    )
                
                with col2:
                    min_confidence = st.slider(
                        "Minimum Confidence",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.5,
                        step=0.05,
                        help="Kekuatan hubungan antar items"
                    )
                
                with col3:
                    min_lift = st.slider(
                        "Minimum Lift",
                        min_value=1.0,
                        max_value=10.0,
                        value=1.2,
                        step=0.1,
                        help="Tingkat dependensi antar items"
                    )
                
                # Preprocessing data
                st.subheader("🔄 Preprocessing Data")
                
                with st.spinner('Memproses data transaksi...'):
                    transactions = create_transaction_data(df, product_column)
                    
                    # Filter transaksi dengan minimal 2 items
                    filtered_transactions = [t for t in transactions if len(t) >= 2]
                    
                    if len(filtered_transactions) == 0:
                        st.error("❌ Tidak ada transaksi dengan minimal 2 items. Coba gunakan 'Minimal Items per Transaksi = 1'")
                        filtered_transactions = transactions
                    
                    # Analisis data transaksi
                    all_products = [product for transaction in filtered_transactions for product in transaction]
                    product_counts = Counter(all_products)
                
                # Tampilkan statistik
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Transaksi Valid", len(filtered_transactions))
                
                with col2:
                    st.metric("Total Produk Unik", len(product_counts))
                
                with col3:
                    avg_products = np.mean([len(t) for t in filtered_transactions]) if filtered_transactions else 0
                    st.metric("Rata-rata Produk/Transaksi", f"{avg_products:.2f}")
                
                # Tampilkan produk paling populer
                if product_counts:
                    st.write("**📊 Produk Paling Populer:**")
                    top_products = pd.DataFrame(
                        product_counts.most_common(10), 
                        columns=['Produk', 'Frekuensi']
                    )
                    top_products['Support'] = (top_products['Frekuensi'] / len(filtered_transactions)).round(3)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(top_products)
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        top_10 = top_products.head(10)
                        y_pos = np.arange(len(top_10))
                        
                        bars = ax.barh(y_pos, top_10['Frekuensi'], color='lightblue')
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(top_10['Produk'])
                        ax.set_xlabel('Frekuensi')
                        ax.set_title('10 Produk Paling Populer')
                        
                        for bar, freq in zip(bars, top_10['Frekuensi']):
                            width = bar.get_width()
                            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                                   f'{freq}', ha='left', va='center')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # Jalankan algoritma Apriori
                st.subheader("🎯 Menjalankan Algoritma Apriori")
                
                if st.button("🚀 Jalankan Analisis Apriori", type="primary"):
                    if len(filtered_transactions) < 2:
                        st.error("❌ Tidak cukup transaksi untuk analisis. Minimal 2 transaksi.")
                        return
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.text("Encoding data transaksi...")
                        progress_bar.progress(30)
                        
                        # Gunakan safe apriori function
                        frequent_itemsets, df_encoded, error = safe_apriori_analysis(
                            filtered_transactions, 
                            min_support=min_support, 
                            max_len=3
                        )
                        
                        if error:
                            st.error(f"❌ Error dalam encoding data: {error}")
                            return
                        
                        status_text.text("Menghitung association rules...")
                        progress_bar.progress(70)
                        
                        # Association rules
                        rules_found = False
                        rules = pd.DataFrame()
                        
                        if frequent_itemsets is not None and len(frequent_itemsets) > 0:
                            try:
                                rules = association_rules(
                                    frequent_itemsets, 
                                    metric="confidence", 
                                    min_threshold=min_confidence
                                )
                                
                                # Filter berdasarkan lift
                                if len(rules) > 0:
                                    rules = rules[rules['lift'] >= min_lift]
                                    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
                                    rules_found = True
                            
                            except Exception as e:
                                st.warning(f"⚠️ Tidak bisa menghitung association rules: {e}")
                        
                        progress_bar.progress(100)
                        status_text.text("Analisis selesai!")
                        
                        # Tampilkan hasil
                        display_results(frequent_itemsets, rules, rules_found, filtered_transactions)
                        
                    except Exception as e:
                        progress_bar.progress(100)
                        st.error(f"❌ Error selama proses Apriori: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error memproses file: {str(e)}")
    
    else:
        # Tampilan default
        st.info("👈 Silakan upload file CSV atau Excel untuk memulai analisis")

def display_results(frequent_itemsets, rules, rules_found, transactions):
    """Tampilkan hasil analisis dengan formatting yang aman"""
    st.success("✅ Analisis selesai!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        itemset_count = len(frequent_itemsets) if frequent_itemsets is not None else 0
        st.metric("Frequent Itemsets", itemset_count)
    
    with col2:
        rules_count = len(rules) if rules_found else 0
        st.metric("Association Rules", rules_count)
    
    with col3:
        if rules_found and len(rules) > 0:
            max_conf = rules['confidence'].max()
            st.metric("Max Confidence", f"{max_conf:.3f}")
        else:
            st.metric("Max Confidence", "0")
    
    with col4:
        if rules_found and len(rules) > 0:
            max_lift = rules['lift'].max()
            st.metric("Max Lift", f"{max_lift:.3f}")
        else:
            st.metric("Max Lift", "0")
    
    # Frequent Itemsets
    st.write("### 📦 Frequent Itemsets")
    if frequent_itemsets is not None and len(frequent_itemsets) > 0:
        # Buat copy untuk display
        frequent_display = frequent_itemsets.copy()
        frequent_display['itemsets'] = frequent_display['itemsets'].apply(
            lambda x: ', '.join(list(x))
        )
        frequent_display['support_pct'] = (frequent_display['support'] * 100).round(2)
        frequent_display = frequent_display.sort_values('support', ascending=False)
        
        # Tampilkan hanya kolom yang diperlukan
        display_columns = ['itemsets', 'support_pct']
        st.dataframe(
            frequent_display[display_columns].rename(columns={'support_pct': 'Support (%)'}),
            use_container_width=True,
            height=400
        )
    else:
        st.warning("Tidak ada frequent itemsets yang ditemukan. Coba turunkan minimum support.")
    
    # Association Rules
    if rules_found and len(rules) > 0:
        st.write("### 🔗 Association Rules")
        
        # Buat copy untuk display - GUNAKAN KOLOM ASLI
        rules_display = rules.copy()
        rules_display['antecedents'] = rules_display['antecedents'].apply(
            lambda x: ' + '.join(list(x))
        )
        rules_display['consequents'] = rules_display['consequents'].apply(
            lambda x: ' + '.join(list(x))
        )
        
        # Format untuk display - TANPA mengganti nama kolom asli
        rules_display['support'] = (rules_display['support'] * 100).round(2)
        rules_display['confidence'] = (rules_display['confidence'] * 100).round(2)
        rules_display['lift'] = rules_display['lift'].round(3)
        
        # Tampilkan dengan nama kolom yang jelas
        display_df = rules_display[[
            'antecedents', 'consequents', 'support', 'confidence', 'lift'
        ]].copy()
        
        # Rename untuk display saja, bukan mengubah dataframe asli
        display_df = display_df.rename(columns={
            'support': 'Support (%)',
            'confidence': 'Confidence (%)',
            'lift': 'Lift',
            'antecedents': 'Jika Membeli',
            'consequents': 'Maka Juga Membeli'
        })
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=500
        )
        
        # Business Insights
        st.write("### 💡 Business Insights & Rekomendasi")
        
        # Top rules
        top_rules = rules_display.head(5)
        
        for i, (idx, rule) in enumerate(top_rules.iterrows()):
            st.success(
                f"**#{i+1}:** Jika pelanggan membeli **{rule['antecedents']}**, "
                f"maka **{rule['confidence']:.1f}%** juga membeli **{rule['consequents']}** "
                f"(Support: {rule['support']:.1f}%, Lift: {rule['lift']:.2f})"
            )
        
        # Visualisasi
        if len(rules_display) >= 3:
            st.write("### 📊 Visualisasi Hasil")
            create_visualizations(rules_display)
        
        # Download hasil
        st.write("### 💾 Download Hasil Analisis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if frequent_itemsets is not None and len(frequent_itemsets) > 0:
                # Siapkan data untuk download
                download_frequent = frequent_itemsets.copy()
                download_frequent['itemsets'] = download_frequent['itemsets'].apply(
                    lambda x: ', '.join(list(x))
                )
                download_frequent['support'] = (download_frequent['support'] * 100).round(2)
                
                csv_frequent = download_frequent[['itemsets', 'support']].to_csv(index=False)
                st.download_button(
                    label="📥 Download Frequent Itemsets",
                    data=csv_frequent,
                    file_name="frequent_itemsets.csv",
                    mime="text/csv"
                )
        
        with col2:
            if rules_found:
                # Siapkan data untuk download
                download_rules = rules.copy()
                download_rules['antecedents'] = download_rules['antecedents'].apply(
                    lambda x: ', '.join(list(x))
                )
                download_rules['consequents'] = download_rules['consequents'].apply(
                    lambda x: ', '.join(list(x))
                )
                download_rules['support'] = (download_rules['support'] * 100).round(2)
                download_rules['confidence'] = (download_rules['confidence'] * 100).round(2)
                download_rules['lift'] = download_rules['lift'].round(3)
                
                csv_rules = download_rules[[
                    'antecedents', 'consequents', 'support', 'confidence', 'lift'
                ]].to_csv(index=False)
                st.download_button(
                    label="📥 Download Association Rules",
                    data=csv_rules,
                    file_name="association_rules.csv",
                    mime="text/csv"
                )
    
    elif rules_found and len(rules) == 0:
        st.warning(
            "❌ Tidak ada association rules yang ditemukan. Coba:\n"
            "- Turunkan minimum confidence (0.3-0.5)\n" 
            "- Turunkan minimum lift (1.0-1.5)\n"
            "- Turunkan minimum support (0.03-0.05)"
        )

def create_visualizations(rules_display):
    """Buat visualisasi untuk results"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Support vs Confidence
        scatter = ax1.scatter(rules_display['support'], 
                            rules_display['confidence'],
                            c=rules_display['lift'], 
                            cmap='viridis', 
                            s=80, 
                            alpha=0.7)
        ax1.set_xlabel('Support (%)')
        ax1.set_ylabel('Confidence (%)')
        ax1.set_title('Support vs Confidence')
        plt.colorbar(scatter, ax=ax1, label='Lift')
        
        # Plot 2: Top rules by confidence
        top_5 = rules_display.head(5)
        y_pos = np.arange(len(top_5))
        bars = ax2.barh(y_pos, top_5['confidence'], color='lightcoral', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f'Rule {i+1}' for i in range(len(top_5))])
        ax2.set_xlabel('Confidence (%)')
        ax2.set_title('Top 5 Rules by Confidence')
        
        for bar, conf in zip(bars, top_5['confidence']):
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{conf:.1f}%', ha='left', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Tidak dapat membuat visualisasi: {e}")

if __name__ == "__main__":
    main()