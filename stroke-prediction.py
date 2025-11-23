import pandas as pd
import matplotlib as plt
import numpy as np

veriler=pd.read_csv("healthcare-dataset-stroke-data (1).csv")
#bütün verileri gösterme
print(veriler)

# Eksik verileri kontrol etme
print(veriler.isnull().sum())

# BMI Sütununu Temizleme
# 'N/A' değerlerini NaN ile değiştir ve float tipine dönüştür.
veriler['bmi'] = veriler['bmi'].replace('N/A', np.nan)
veriler['bmi'] = pd.to_numeric(veriler['bmi'])

# Korelasyon hesaplamaları için eksik BMI değerlerini içeren satırları geçici olarak düşür.
# (Korelasyon analizi eksik değerlerle sağlıklı yapılamaz.)
df_clean = veriler.dropna(subset=['bmi'])

# BMI ile Sayısal Sütunlar Arasındaki Korelasyonu Hesaplama
# 'age', 'avg_glucose_level' ve ikili (binary) olan 'heart_disease' sütunları kullanılır.
numerical_cols = ['age', 'avg_glucose_level', 'heart_disease']

# df.corr()['bmi'] ifadesi tüm sütunların 'bmi' ile olan korelasyonunu verir.
correlations = df_clean[numerical_cols + ['bmi']].corr()['bmi'].drop('bmi')

print("--- BMI ile Sayısal Sütunlar Arasındaki Korelasyonlar (Pearson) ---")
print(correlations.sort_values(ascending=False))

# BMI'ın Cinsiyete Göre Medyanını Hesaplama
# df.groupby('gender')['bmi'].median() ifadesi her bir cinsiyet grubu için medyan BMI'ı hesaplar.
bmi_by_gender = df_clean.groupby('gender')['bmi'].median()

print("\n BMI'ın Cinsiyete Göre Medyan Değerleri")
print(bmi_by_gender)

# Eksik Değerleri Yaş Ortalamalarına Göre Doldur
# Her 'age' grubu için ortalamayı hesapla ve eksik değerlere atayalım
veriler['bmi'] = veriler['bmi'].fillna(veriler.groupby('age')['bmi'].transform('mean'))

# Eğer bazı yaşlarda hiç veri yoksa ve hala eksik kaldıysa genel ortalama ile doldurma
veriler['bmi'] = veriler['bmi'].fillna(veriler['bmi'].mean())

# Sonucu Kaydet
output_file = "healthcare-dataset-stroke-data_imputed_age_mean.csv"
veriler.to_csv(output_file, index=False)
print(f"BMI eksik değerleri yaş ortalamalarıyla dolduruldu ve '{output_file}' dosyasına kaydedildi.")
print(veriler.head())

# Eksik verileri tekrar kontrol etme
print(veriler.isnull().sum())


#klasifikasyon(sınıflandırma) işlemleri...




