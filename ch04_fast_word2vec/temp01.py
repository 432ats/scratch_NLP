import numpy as np 
from negative_sample_layer import UnigramSampler
# 0から9の数字の中から一つの数字をランダムにサンプリング
print(np.random.choice(10))
print(np.random.choice(10))

# wordsから一つだけサンプリング
words = ['you', 'say', 'goodbye', 'and', 'i', 'hello', '.']
print(np.random.choice(words))

# 5つだけランダムサンプリング（重複あり）
print(np.random.choice(words, size=5))

# 5つだけランダムサンプリング（重複なし）
print(np.random.choice(words, size=5, replace=False))

#確率分布に従ってサンプリング
p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1, 0.0]
print(np.random.choice(words, size=3, replace=False, p=p))

p = [0.7, 0.29, 0.01]
new_p = np.power(p, 0.75)
new_p /= np.sum(new_p)

print(new_p)

corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
power = 0.75
sample_size = 2

sampler = UnigramSampler(corpus, power, sample_size)
target = np.array([1, 3, 0])
negative_sample = sampler.get_negative_sample(target)
print(negative_sample)