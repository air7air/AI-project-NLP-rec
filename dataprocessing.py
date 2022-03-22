
import pandas as pd
import tqdm


df = pd.read_csv("retail.csv")
df.to_csv('data1.csv', encoding = 'UTF-8')

df = pd.read_csv("retail.csv")

df.dropna(inplace=True)
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
df.drop(['Payment Way', 'Date'],axis=1, inplace=True

def check():
    Detail = df['Detail'].unique()
    for i in Detail:
        if i.isupper() == False:
            print(i)

df[df['Amount'] <= 0.]
idx = df[df['Amount'] <= 0.].index
df.drop(idx, inplace=True)
l = ['Manual', 'This is a test product.', 'Discount', 'Adjustment by Peter on 24/05/2010 1',\
     'Bank Charges', 'Adjustment by john on 26/01/2010 17',  'Adjustment by john on 26/01/2010 16', 'Adjust bad debt', 'Adjustment by Peter on Jun 25 2010', 'POSTAGE', 'CARRIAGE']
for v in l:
    idx2 = df[df['Detail'] == v].index
    df.drop(idx2, inplace=True)
idx3 = df[df['Price'] <= 0.].index
df.drop(idx3, inplace=True)

df.drop(['Amount', 'Price', 'ItemCode'],axis=1, inplace=True)

detail = df['Detail'].unique()

def sortdf(data, column): # 정렬 함수
    data.sort_values(column, inplace=True)
    data.reset_index(inplace=True)
    data.pop('index')
    return data

df_invoice = df.copy()
df_invoice['Invoice'] = df_invoice['Invoice'].apply(lambda x: str(x))
sortdf(df_invoice, ['Invoice'])

Invoice = df_invoice['Invoice'].unique()
Invoice = sorted(Invoice)
len(Invoice)

df3 = pd.DataFrame(columns=['Invoice', 'Detail'])
values = df_invoice['Invoice'].value_counts().index.sort_values()

for i, v in enumerate(tqdm.tqdm(values)):
    df3.loc[i] = (v, '&&'.join(df_invoice[df_invoice['Invoice'] == v]['Detail']))

df3.pop('Invoice')

df_series = df3['Detail'].str.split('&&')
df_series[0]

def create_sequences(values, window_size, step_size): #values 들어가는게 리스트로 만들어야 해서 위에서 리스트로 만든것임
    sequences = []
    start_index = 0
    while True:
        print("start:",start_index)
        end_index = start_index + window_size
        print("end:",end_index)
        seq = values[start_index:end_index]
        print(seq)
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
        print(sequences)
    return sequences


sequence_length = 5
step_size = 2

def create_sequences(values, window_size, step_size):
    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences

for i, v in enumerate(df_series):
    df_series[i] = create_sequences(v, sequence_length, step_size)
df_series[0]

df_list = []
for v in df_series:
    for value in v:
        df_list.append(value)
df_list[:2]

label_list = []

for i in range(len(df_list)):
    l = df_list[i][-1]
    df_list[i].pop()
    label_list.append(l)
label_list[:2]

df_new = pd.DataFrame(columns=['Detail', 'label'])
df_new['Detail'] = df_list
df_new['label'] = label_list
df_new['Detail'] = df_new['Detail'].apply(lambda x: '&&'.join(map(str, x)))
df_new.dropna(inplace=True)
print(df_new.duplicated().sum())
df_new.drop_duplicates(inplace=True)
df_new.reset_index(inplace=True)
df_new.pop('index')
df_new