import pickle
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


pol_list = ['min-spe', 'max-spe', 'min-rpe', 'max-rpe']
NN_list = []
for pol in range(4):
    NN_df = []
    for sbj in range(82):
        with open('history_results/Analysis-Object-' + pol_list[pol] + '-{0:02d}_NN.pkl'.format(sbj),'rb') as f:
            data = pickle.load(f)
        fc_df=[data.fc1.weight.data.tolist(), data.fc2.weight.data.tolist(), data.fc3.weight.data.tolist()]
        NN_df.append(fc_df)
    NN_list.append(NN_df)

NNs=[]
for pol in range(4):
    NN_pol = []
    for sbj in range(82):
        NNs_df=[]
        for layer in [2]:#range(3):
            for index in range(len(NN_list[0][sbj][layer])):
                for index2 in range(len(NN_list[0][sbj][layer][index])):
                    NNs_df.append(NN_list[0][sbj][layer][index][index2])
        NN_pol.append(NNs_df)
    NNs.append(NN_pol)


for pol in range(4):
    print(pol)
    sample_tsne = TSNE(n_components=2, learning_rate=200)
    tsne_results = sample_tsne.fit_transform(NNs[pol])
    tsne_x = []
    tsne_y = []
    for ii in range(82):
      tsne_x.append(tsne_results[ii][0])
      tsne_y.append(tsne_results[ii][1])
    scatter = plt.scatter(tsne_x, tsne_y)
    print(len(tsne_x))
    plt.title(pol_list[pol])
    plt.savefig('NN tSNE result in the '+ pol_list[pol] +'.png')
    plt.clf()


