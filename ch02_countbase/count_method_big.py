import sys
sys.path.append('/Users/ats432/projects/Matsuzaki_lab/DeepLearning_Scratch_NLP')
import numpy as np 
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, most_similar, ppmi
from dataset import ptb

window_size = 2
wordvec_size = 100

#コーパスの前処理
corpus, word_to_id, id_to_word = ptb.load_data('train')


#共起行列の生成
vocab_size = len(word_to_id)
print('counting 共起行列 ...')
C = create_co_matrix(corpus, vocab_size, window_size)


#PPMI(相互情報量)
print('calculating PPMI ...')
W = ppmi(C)


print('calculating SVD ...')
#SVD(特異値分解)
try:
    # truncated SVD (fast!)
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5,
                             random_state=None)
except ImportError:
    # SVD (slow)
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]


#類似度ランキング
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

