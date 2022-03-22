import pandas as pd
from tqdm import tqdm_notebook
import tensorflow as tf
import re
from main import transformer, tokenizer

NUM_LAYERS = 1
D_MODEL = 128
NUM_HEADS = 4
UNITS = 256
DROPOUT = 0.2
VOCAB_SIZE=2716
MAX_LENGTH = 128

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)


path='my_model2'
model.load_weights(path+f'Instacart_Transformer_{EPOCHS}_weights_layers_{NUM_LAYERS}')


START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
def evaluate(sentence):
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  # 디코더의 예측 시작 output이 벡터로 나오기 때문에 우리가 읽을수 잇는 스트링 형태로 바꿔줌
  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)


def predict(sentence):
  prediction = evaluate(sentence)
  # print("예측:",prediction)
  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

#   print('Input: {}'.format(sentence))
#   print('Output: {}'.format(predicted_sentence))

  return predicted_sentence


def preprocess_sentence(sentence):
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = sentence.strip()
  return sentence


filtering_test_df = pd.read_csv("test_data.csv", index_col=0)
predict_item_test = filtering_test_df['split_train_token'].progress_apply(predict)
filtering_test_df['predict_item'] = predict_item_test

TFoutputs = filtering_test_df
pred = TFoutputs['predict_item']

def find_item(item, n=10):
    return list(dict(model.wv.most_similar(item, topn=n)))

def recommend(n=10, acc=False, pred=pred, TFoutputs=TFoutputs):
    item_list = []
    for v in tqdm_notebook(pred.to_list()):
        if n > 1:
            pp = find_item(v, n - 1)
            pp.insert(0, v)
            item_list.append(pp)
        else:
            item_list.append(v)
    TFoutputs['list'] = item_list
    TFoutputs['accuracy'] = [True if label in item else False for label, item in
                          tqdm_notebook(zip(TFoutputs['label'], TFoutputs['list']), total=(
                              len(TFoutputs)))]
    if acc:
        return TFoutputs[TFoutputs['accuracy'] == True].shape[0] / (TFoutputs.shape[0])

    return TFoutputs