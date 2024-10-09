import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTETomek
import numpy as np

# 讀取CSV文件
df = pd.read_csv('./input/data2.csv')

# 將Actual列的值轉換為1和0
df['HasTransaction'] = df['Actual'].apply(lambda x: 1 if x else 0)
df['Content'] = df.apply(lambda row: f"{row['Instruction']} {row['Chunk']}", axis=1)

# 移除原始列
df = df.drop(columns=['Actual', 'Chunk', 'Contains Transactions'])

# 使用TF-IDF將文本轉換為數值特徵
vectorizer = TfidfVectorizer(max_features=1000)  # 你可以調整max_features
X_tfidf = vectorizer.fit_transform(df['Content'])

# 進行結合採樣
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_tfidf, df['HasTransaction'])

# 將採樣後的結果轉換回DataFrame
feature_names = vectorizer.get_feature_names_out()
X_resampled_df = pd.DataFrame(X_resampled.toarray(), columns=feature_names)

# 將採樣後的文本特徵轉換回最相似的原始文本
def get_closest_text(row):
    similarities = X_tfidf.dot(row.values.reshape(-1, 1)).flatten()
    closest_idx = similarities.argmax()
    return df['Content'].iloc[closest_idx]

resampled_content = X_resampled_df.apply(get_closest_text, axis=1)

# 創建新的平衡數據集
df_resampled = pd.DataFrame({
    'Content': resampled_content,
    'HasTransaction': y_resampled
})

# 將數據分為訓練集和驗證集
train_df, val_df = train_test_split(df_resampled, test_size=0.2, random_state=42)

# 保存訓練集和驗證集到CSV文件
train_df.to_csv('./input/training_balanced.csv', index=False)
val_df.to_csv('./input/validation_balanced.csv', index=False)

# 保存驗證集內容到sample_new.txt
with open('./input/sample_new_balanced.txt', 'w', encoding='utf-8') as file:
    for content in val_df['Content']:
        file.write(content.replace('\n', ' ') + '\n')

print("原始數據集大小:", len(df))
print("平衡後數據集大小:", len(df_resampled))
print("正樣本數量:", sum(df_resampled['HasTransaction']))
print("負樣本數量:", len(df_resampled) - sum(df_resampled['HasTransaction']))