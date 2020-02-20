import sys
sys.path.append('..')
import numpy as np 

#コーパスの前処理
def preprocess(text):
    #テキストを全て小文字化する"You"→"you"
    text = text.lower()
    #"."→" ."と置き換える
    text = text.replace('.', ' .')

    #スペース区切りで単語をリスト化する
    words = text.split(' ')
   
    #辞書化
    #キー：単語,　値：ID
    word_to_id = {}
    #キー：ID,　値：単語
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    #辞書→numpy配列
    corpus = [word_to_id[w] for w in words]
    corpus = np.array(corpus)

    return corpus, word_to_id, id_to_word

#共起行列の生成
def create_co_matrix(corpus, vocab_size, window_size=1):
    '''共起行列の作成
    :param corpus: コーパス（単語IDのリスト）
    :param vocab_size:語彙数
    :param window_size:ウィンドウサイズ（ウィンドウサイズが1のときは、単語の左右1単語がコンテキスト）
    :return: 共起行列
    '''
    #コーパスのサイズ
    corpus_size = len(corpus)
    #共起行列の初期化
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    #enumerate:インデックスと要素を同時に生成
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx-i
            right_idx = idx+i

            if left_idx>=0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            
            if right_idx<corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
            


    return co_matrix

#コサイン類似度:x*y/||x||*||y||
def cos_similarity(x, y, eps=1e-8):
    '''コサイン類似度の算出
    :param x: ベクトル
    :param y: ベクトル
    :param eps: ”0割り”防止のための微小値
    :return:
    '''
    nx = x/np.sqrt(np.sum(x**2)+eps)
    ny = y/np.sqrt(np.sum(y**2)+eps)

    return np.dot(nx, ny)

#類似単語ランキング
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''類似単語の検索
    :param query: クエリ（テキスト）
    :param word_to_id: 単語から単語IDへのディクショナリ
    :param id_to_word: 単語IDから単語へのディクショナリ
    :param word_matrix: 単語ベクトルをまとめた行列。各行に対応する単語のベクトルが格納されていることを想定する
    :param top: 上位何位まで表示するか
    '''
    #❶クエリ（単語）を取り出す
    if query not in word_to_id:
        print('%s is not found'%query)
        return
    
    print('\n[query] '+query)
    query_id =word_to_id[query]
    query_vec = word_matrix[query_id]

    #❷コサイン類似度の算出
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    
    #❸コサイン類似度の結果から、その値を高い順に出力
    count = 0
    for i in (-1*similarity).argsort():
        if id_to_word[i]==query:
            continue
        print(' %s: %s'%(id_to_word[i],similarity[i]))

        count += 1
        if count >= top:
            return
            
#ppmi相互情報量
def ppmi(C, verbose=False, eps=1e-8):
    '''PPMI（正の相互情報量）の作成
    :param C: 共起行列
    :param verbose: 進行状況を出力するかどうか
    :return:
    '''
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0]*C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j]*N/(S[j]*S[i])+eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print('%.1f%% done'%(100*cnt/total))
    return M

#コンテキスト(入力)とターゲット(出力)の生成
def create_contexts_target(corpus, window_size):
    '''コンテキストとターゲットの作成
    :param corpus: コーパス（単語IDのリスト）
    :param window_size: ウィンドウサイズ（ウィンドウサイズが1のときは、単語の左右1単語がコンテキスト）
    :return:
    '''
    target = corpus[window_size: -window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size+1):
            if t == 0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    
    return np.array(contexts), np.array(target)

#one-hot表現への変換
def convert_one_hot(corpus, vocab_size):
    '''one-hot表現への変換
    :param corpus: 単語IDのリスト（1次元もしくは2次元のNumPy配列）
    :param vocab_size: 語彙数
    :return: one-hot表現（2次元もしくは3次元のNumPy配列）
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate