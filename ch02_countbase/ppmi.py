import sys
sys.path.append('/Users/ats432/projects/Matsuzaki_lab/DeepLearning_Scratch_NLP')
import numpy as np 
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi

text = 'You say goodbye and I say hello.'
#コーパスの前処理
corpus, word_to_id, id_to_word = preprocess(text)
print("corpus:\n{0}".format(corpus), "辞書:\n{0}".format(id_to_word), '-'*50, sep="\n")

#共起行列の生成
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
print("共起行列:\n{0}".format(C), '-'*50, sep='\n')

#PPMI(相互情報量)
W = ppmi(C)
print('PPMI:\n{0}'.format(W), '-'*50, sep='\n')