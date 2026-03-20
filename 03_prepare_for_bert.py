import pandas as pd
df = pd.read_csv('data/fake_or_real_news_cleaned.csv')
#label encoding
df['label_num']= df['label'].map({'fake':0,'real':1})
print(df[['label','label_num']].head())
df.to_csv('data/fake_or_real_news_encoded.csv',index=False)
print("Dataset with numeric labels saved as fake_or_real_news_encoded.csv")
#Train_test split
from sklearn.model_selection import train_test_split
x = df['content']
y = df['label_num'] #features and labels
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
print("training samples:",len(x_train))
print("testing samples:",len(x_test))






