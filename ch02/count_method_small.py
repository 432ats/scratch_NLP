import sys
sys.path.append('..')
import numpy as np 
import matplotlib.pyplot as plt
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

#SVD(特異値分解)
U, S, V = np.linalg.svd(W)

print('共起行列:\n{0}'.format(C[0]), '-'*50, sep='\n')
print('PPMI行列:\n{0}'.format(W[0]), '-'*50, sep='\n')
print('SVD:\n{0}'.format(U[0]), '-'*50, sep='\n')
print('SVDの先頭の2要素:\n{0}'.format(U[0, :2]), '-'*50, sep='\n')

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()

