# Natural Language Processing

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer


# List of sample sentences that we want to tokenize
sentences = ['I love my dog',
             'I love my cat',
             'you love my dog!',
             'Do you think my dog is amazing?',
             ]

mig_sms = ['Ginga, from today on, I’ll start to focus on IT and computer science, python, and as much as I can so that I can help in the development. Even if we find people who know, it’ll be better if we do it together so that we can save money on the early stage. If I want to stay in Australia I’ll have to study in an university here, so I’m doing my research to see which is the best option for me.']

sex = ['I love sex, I love fucking girls, invite a girl to the home or a hotel and go to bed, セックス']
sex_words = ['pakopako, deepkiss, shikoshiko, peropero, fera, kunni, shiofuki, teman, kijyoui, seijyoui, sounyuu, yarimoku, sefure, ecchi, chinko, manko, nakadashi, oppai, yaru, sukebe, hentai, manjiru, seishi, fuck, creampie, blowjob, pussy, hookup, boobs']
# initializing a tokenizer that can index
# num_words is the maximum number words that can be kept
# tokenizer will automatically help in choosing most frequent words
tokenizer = Tokenizer(num_words=500, oov_token="<oov>")
tokenizer.fit_on_texts(mig_sms)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts=mig_sms)

print(word_index)
print(sequences)
sexinizer = Tokenizer(num_words=10, oov_token="<oov>")
sexinizer.fit_on_texts(sex_words)
sex_index = sexinizer.word_index
sex_sequence = sexinizer.texts_to_sequences(texts=sex_words)
sex_test = ['creampie, nakadashi, deepkiss, teman, shiofuki']
sex_test_seq = sexinizer.texts_to_sequences(sex_test)
print(sex_index)
print(sex_sequence)
print(sex_test_seq)

import nltk
# import re
# import tensorflow as tf
#
# # Importing the libraries
# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics import confusion_matrix
# from sklearn.naive_bayes import GaussianNB
#
# import sys
# import os
#
# from tensorflow.python.ops import lookup_ops
#
# tf.compat.v1.app.flags.DEFINE_integer('training_iteration', 1000,
#                                       'number of training iterations.')
# tf.compat.v1.app.flags.DEFINE_integer('model_version', 1,
#                                       'version number of the model.')
# tf.compat.v1.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
# FLAGS = tf.compat.v1.app.flags.FLAGS
#
# # Importing the dataset
# dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
#
# # Cleaning the texts
# #nltk.download('stopwords')
# corpus = []
# for i in range(0, 1000):
#     review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
#     review = review.lower()
#     review = review.split()
#     ps = PorterStemmer()
#     review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#     review = ' '.join(review)
#     corpus.append(review)
#
# # Creating the Bag of Words model
# cv = CountVectorizer(max_features=1500)
# X = cv.fit_transform(corpus).toarray()
# y = dataset.iloc[:, 1].values
#
# # Splitting the dataset into the Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
#
# # Fitting Naive Bayes to the Training set
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
#
# # Making the Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
#
# # Export model
#   # WARNING(break-tutorial-inline-code): The following code snippet is
#   # in-lined in tutorials, please update tutorial documents accordingly
#   # whenever code changes.
# export_path_base = sys.argv[-1]
# export_path = os.path.join(
#     tf.compat.as_bytes(export_path_base),
#     tf.compat.as_bytes(str(FLAGS.model_version)))
# print('Exporting trained model to', export_path)
# builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)
#
# # Build the signature_def_map.
# classification_inputs = tf.compat.v1.saved_model.utils.build_tensor_info(
#     serialized_tf_example)
# classification_outputs_classes = tf.compat.v1.saved_model.utils.build_tensor_info(
#     prediction_classes)
# classification_outputs_scores = tf.compat.v1.saved_model.utils.build_tensor_info(
#     values)
#
# classification_signature = (
#     tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
#         inputs={
#             tf.compat.v1.saved_model.signature_constants.CLASSIFY_INPUTS:
#                 classification_inputs
#         },
#         outputs={
#             tf.compat.v1.saved_model.signature_constants
#             .CLASSIFY_OUTPUT_CLASSES:
#                 classification_outputs_classes,
#             tf.compat.v1.saved_model.signature_constants
#             .CLASSIFY_OUTPUT_SCORES:
#                 classification_outputs_scores
#         },
#         method_name=tf.compat.v1.saved_model.signature_constants
#         .CLASSIFY_METHOD_NAME))
#
# tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
# tensor_info_y = tf.compat.v1.saved_model.utils.build_tensor_info(y)
#
# prediction_signature = (
#     tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
#         inputs={'images': tensor_info_x},
#         outputs={'scores': tensor_info_y},
#         method_name=tf.compat.v1.saved_model.signature_constants
#         .PREDICT_METHOD_NAME))
#
# builder.add_meta_graph_and_variables(
#     sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
#     signature_def_map={
#         'predict_images':
#             prediction_signature,
#         tf.compat.v1.saved_model.signature_constants
#         .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#             classification_signature,
#     },
#     main_op=tf.compat.v1.tables_initializer(),
#     strip_default_attrs=True)
#
# builder.save()