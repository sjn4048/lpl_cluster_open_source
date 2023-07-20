from matplotlib import pyplot as plt
import os
from result_plotter import ResultPlotter
from trainer import ClusterTrainer
from dim_reducer import TSNEDimReducer
from data_collector import get_tournament_data

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ['OMP_NUM_THREADS'] = '1'

# hyperparams
algorithm = 'kmeans'
tsne_perplexity = 40
tsne_iter = 300
default_n_clusters = 5
train_min_occur = 10
valid_min_occur = 10
ranked = True

# data preprocess
train_data, valid_data, valid_names, valid_labels_gt, feature_names = get_tournament_data(['S13-summer-LPL'], 'all')

# algorithm
use_k = int(os.getenv('n_clusters', default_n_clusters))
print('training clustering')
trainer = ClusterTrainer(algorithm)
trainer.feed(train_data, valid_data)

if best_k := os.getenv('find_best_k', ''):
    print('finding best k')
    best_k = trainer.find_best_k(kmax=int(best_k))

valid_labels_pred, centers = trainer.train_and_cluster(n_clusters_=use_k)

print('reducing dim')
# reducer
reducer = TSNEDimReducer(perplexity=tsne_perplexity, n_iter=tsne_iter)
_, valid_pca_points, valid_center_points = reducer.fit_transform([train_data, valid_data, centers])

# plotting the results
print('plotting results')
plotter = ResultPlotter('result')
plotter.plot(valid_pca_points, valid_center_points, valid_labels_pred,
             valid_labels_gt, valid_names, feature_names, valid_data, lite=True)
