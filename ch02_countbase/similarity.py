import sys
sys.path.append('/Users/ats432/projects/Matsuzaki_lab/DeepLearning_Scratch_NLP')
import numpy as np 
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar

text = 'You say goodbye and I say hello.'
#コーパスの前処理
corpus, word_to_id, id_to_word = preprocess(text)
print("corpus:\n{0}".format(corpus),"辞書:\n{0}".format(id_to_word),sep="\n")

#共起行列の生成
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
print("共起行列:\n{0}".format(C))

c0 = C[word_to_id['you']] #"you"の単語ベクトル
c1 = C[word_to_id['i']]   #"i"の単語ベクトル

#コサイン類似度
print("類似度(コサイン類似度):\n{0}".format(cos_similarity(c0,c1)))

#類似度ランキング
most_similar('you', word_to_id, id_to_word, C, top=5)
