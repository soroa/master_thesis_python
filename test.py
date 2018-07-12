import operator

import matplotlib.pyplot as plt
import numpy as np
a = [0.196917808219,
0.33918128655,
0.232014388489,
0.20145631068,
0.1875,
0.421487603306,
0.238938053097,
0.227810650888,
0.197879858657,
0.217002237136,
0.218705035971,
0.60824742268,
0.244755244755,
0.224256292906,
0.527093596059,
0.19776119403,
0.214285714286,
0.242206235012,
0.215841584158,
0.190972222222,
0.240143369176,
0.411764705882,
0.243944636678,
0.218348623853,
0.263793103448,
0.145161290323,
0.282937365011,
0.244036697248,
0.231448763251,
0.251497005988,
0.243956043956,
0.253913043478,
0.234782608696,
0.249329758713,
0.238532110092,
0.227099236641,
0.223123732252,
0.200381679389,
0.274725274725,
0.126984126984,
0.251893939394,
0.195833333333,
0.271983640082]


# sorted_x = sorted(a.items(), key=operator.itemgetter(1))

# for i in sorted_x:
#     print(i)


a = np.asarray(a)
max = np.max(a)
min = np.min(a)
std = np.std(a)
avg = np.mean(a)

hist, bins = np.histogram(a, bins=8)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.ylabel('#Samples')
plt.xlabel('Test accuracy - L1O over N')
plt.savefig('bins_svc.png')
print("max " + str(max))
print("min " + str(min))
print("std " + str(std))
print("avg " + str(avg))
