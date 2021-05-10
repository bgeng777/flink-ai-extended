import tensorflow as tf
import census_dataset
import pandas as pd
import numpy as np

# For simplicity, we make evaluation and validation use the same remote dataset
EVAL_FILE = '/tmp/census_data/adult.evaluate'
VALIDATE_FILE = '/tmp/census_data/adult.validate'

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def preprocess(input_dict):

    feature_dict = {
        'age': _float_feature(value=int(input_dict['age'])),
        'workclass': _bytes_feature(value=input_dict['workclass'].encode()),
        'fnlwgt': _float_feature(value=int(input_dict['fnlwgt'])),
        'education': _bytes_feature(value=input_dict['education'].encode()),
        'education_num': _float_feature(value=int(input_dict['education_num'])),
        'marital_status': _bytes_feature(value=input_dict['marital_status'].encode()),
        'occupation': _bytes_feature(value=input_dict['occupation'].encode()),
        'relationship': _bytes_feature(value=input_dict['relationship'].encode()),
        'race': _bytes_feature(value=input_dict['race'].encode()),
        'gender': _bytes_feature(value=input_dict['gender'].encode()),
        'capital_gain': _float_feature(value=int(input_dict['capital_gain'])),
        'capital_loss': _float_feature(value=int(input_dict['capital_loss'])),
        'hours_per_week': _float_feature(value=float(input_dict['hours_per_week'])),
        'native_country': _bytes_feature(value=input_dict['native_country'].encode()),
    }
    model_input = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    model_input = model_input.SerializeToString()
    return model_input

def predict(path):
    inputs = []
    df = pd.read_csv(path, header=None)
    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(path))
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        classes = tf.equal(labels, '>50K')  # binary classification
        return features, classes
    tmp = preprocess()
    output_dict = _predictor({'inputs': [model_input]})

test_data = '/tmp/census_data/adult.evaluate'
dataset = census_dataset.input_fn(test_data, 1, False, 32)
# dataset = tf.data.TextLineDataset(test_data)



path = '/tmp/census_export/1620615875'
path2 = '/tmp/ff.h5'
# model = tf.keras.models.load_model(path)
# print(dir(model))
# print(model.summary())

# Saving the Model in H5 Format
# tf.keras.models.save_model(model, path2)

# Loading the H5 Saved Model
# loaded_model_from_h5 = tf.keras.models.load_model(path2)

# with tf.Session() as session:
#     loaded = tf.saved_model.load(session, [tf.saved_model.tag_constants.SERVING],path)
#     print(dir(loaded))
# loaded = model
#
# for i in dir(loaded):
#     if i[0] != '_':
#         print(i)
# test_df = pd.read_csv('test.csv', ...)

# Loading the estimator
# predict_fn = loaded.signatures['predict']
_predictor = tf.contrib.predictor.from_saved_model(path)
# 40,Private,217120,10th,6,Divorced,Craft-repair,Not-in-family,White,Male,0,0,50,United-States,<=50K
# print(_predictor({
#     'age': 40,
#     'workclass': 'Private', 'fnlwgt':217120, 'education':'10th', 'education_num': 6,
#     'marital_status':'Divorced', 'occupation':'Craft-repair', 'relationship':'Not-in-family',
#     'race':'White', 'gender':'Male',
#     'capital_gain':0, 'capital_loss':0, 'hours_per_week':50, 'native_country':'United-States',
# }))
# output_dict = _predictor({'inputs': [dataset]})
# print(output_dict)
# for i in dir(_predictor):
#     if i[0] != '_':
#         print(i)
# Predict
# predictions = predict_fn(dataset.make_initializable_iterator)
# print(predictions)

# class LayerFromSavedModel(tf.keras.layers.Layer):
#     def __init__(self):
#         super(LayerFromSavedModel, self).__init__()
#         self.vars = loaded.variables
#
#     def call(self, inputs):
#         return loaded.ps(inputs)
# #
# #

# print(loaded.signatures['serving_default'](dataset))
# for element in dataset.output_shapes:
#     print(element)
# input = tf.keras.Input(shape=(13,))
# #
# model = tf.keras.Model(input, LayerFromSavedModel()(input))
# print(dir(model))
# print(model.summary())
# k_model = tf.keras.Model()
# loss, metric = loaded_model_from_h5.evaluate(dataset, verbose=0)

# with tf.Session() as session:
#     tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], path)
#     self._predictor = tf.contrib.predictor.from_saved_model(self._exported_model)
#
#
age=40
workclass='Private'
fnlwgt=217120
education='10th'
education_num=6
marital_status='Divorced'
occupation='Craft-repair'
relationship='Not-in-family'
race='White'
gender='Male'
capital_gain=0
capital_loss=0
hours_per_week=50
native_country='United-States'

feature_dict = {
    'age': _float_feature(value=int(age)),
    'workclass': _bytes_feature(value=workclass.encode()),
    'fnlwgt': _float_feature(value=int(fnlwgt)),
    'education': _bytes_feature(value=education.encode()),
    'education_num': _float_feature(value=int(education_num)),
    'marital_status': _bytes_feature(value=marital_status.encode()),
    'occupation': _bytes_feature(value=occupation.encode()),
    'relationship': _bytes_feature(value=relationship.encode()),
    'race': _bytes_feature(value=race.encode()),
    'gender': _bytes_feature(value=gender.encode()),
    'capital_gain': _float_feature(value=int(capital_gain)),
    'capital_loss': _float_feature(value=int(capital_loss)),
    'hours_per_week': _float_feature(value=float(hours_per_week)),
    'native_country': _bytes_feature(value=native_country.encode()),
}


model_input = tf.train.Example(features=tf.train.Features(feature=feature_dict))
model_input = model_input.SerializeToString()


feature_dict2 = {
    'age': _float_feature(value=int(age+10)),
    'workclass': _bytes_feature(value=workclass.encode()),
    'fnlwgt': _float_feature(value=int(fnlwgt)),
    'education': _bytes_feature(value=education.encode()),
    'education_num': _float_feature(value=int(education_num)),
    'marital_status': _bytes_feature(value=marital_status.encode()),
    'occupation': _bytes_feature(value=occupation.encode()),
    'relationship': _bytes_feature(value=relationship.encode()),
    'race': _bytes_feature(value=race.encode()),
    'gender': _bytes_feature(value=gender.encode()),
    'capital_gain': _float_feature(value=int(capital_gain)),
    'capital_loss': _float_feature(value=int(capital_loss)),
    'hours_per_week': _float_feature(value=float(hours_per_week)),
    'native_country': _bytes_feature(value=native_country.encode()),
}

# \]
model_input2 = tf.train.Example(features=tf.train.Features(feature=feature_dict))
model_input2 = model_input2.SerializeToString()
print(type(model_input2))
# output_dict = _predictor({'inputs': [model_input2]})
# print(output_dict)
# a = np.argmax(output_dict['scores'])
# # print(a)
data_file = path
def parse_csv(value):
    tf.logging.info('Parsing {}'.format(data_file))
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('income_bracket')
    classes = tf.equal(labels, '>50K')  # binary classification
    return features, classes

res, t = parse_csv("40,Private,217120,10th,6,Divorced,Craft-repair,Not-in-family,White,Male,0,0,50,United-States,<=50K")
print(res)
print(t)
# dataset = tf.data.TextLineDataset(data_file)
# dataset = dataset.map(parse_csv, num_parallel_calls=5)
# iter = dataset.make_initializable_iterator()

# model_input2 = tf.train.Example(features=tf.train.Features(feature=res))
# model_input2 = model_input2.SerializeToString()
# output_dict = _predictor({'inputs': [model_input2]})
# print(dataset.take(1))

import pandas as pd
data = pd.read_csv(EVAL_FILE, names=_CSV_COLUMNS)
label = data.pop('income_bracket')
data_dict = {col: list(data[col]) for col in data.columns}

for _, row in data.iterrows():
    tmp = dict(zip(_CSV_COLUMNS[:-1], row))
    print(tmp)
    break

test_data = '/tmp/census_data/adult.evaluate'

predictor = tf.contrib.predictor.from_saved_model(path)
data = pd.read_csv(test_data, names=census_dataset._CSV_COLUMNS)
label = data.pop('income_bracket')
label = label.map({'<=50K': 0, '>50K': 1})
res = []
from sklearn.metrics import f1_score, accuracy_score

inputs = []
for _, row in data.iterrows():
    tmp = dict(zip(census_dataset._CSV_COLUMNS[:-1], row))
    tmp =preprocess(tmp)
    inputs.append(tmp)

output_dict = predictor({'inputs': inputs})
print(output_dict)
res = [np.argmax(output_dict['scores'][i]) for i in range(0, len(output_dict['scores']))]
print(accuracy_score(label, res))