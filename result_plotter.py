import numpy as np
from matplotlib import pyplot as plt
from collections import Counter, defaultdict
from adjustText import adjust_text
import os
import scipy
import numpy.ma as ma
import pandas as pd
import sys

if sys.platform == 'win32':
    # windows下adjustText库有点bug的感觉
    def adjust_text(a, *args, **kw):
        pass


class ResultPlotter:
    def __init__(self, path_dir: str):
        # init matplotlib
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        os.environ['OMP_NUM_THREADS'] = '1'
        self.figsize = (8, 4.5)
        self.text_size = 6
        self.dpi = 240
        self.path_dir = path_dir
        os.makedirs(path_dir, exist_ok=True)

    @staticmethod
    def ranked_(arr: np.ndarray) -> np.ndarray:
        sorted_index = np.argsort(np.argsort(-arr, axis=0), axis=0) + 1
        return sorted_index

    def plot(self, points, centers, labels_pred, labels_gt, names, feature_names, valid_data, lite: bool = False):
        self._plot_cluster_result(points, labels_pred, names, centers)
        self._write_inconsistent_points(names, labels_pred, labels_gt)
        # self._player_stats(names, feature_names, valid_data, labels_gt)
        self._cluster_stats(feature_names, labels_pred, valid_data)
        self._plot_each_cluster(labels_pred, points, names, centers)
        self._plot_each_feature(feature_names, valid_data, points, names, lite)
        self._plot_cluster_feature(labels_pred, feature_names, valid_data, points, names)

    def savefig(self, name: str):
        path = os.path.join(self.path_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=self.dpi, facecolor='white')

    def write(self, content: str, name: str):
        path = os.path.join(self.path_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    def save_df(self, df: pd.DataFrame, name: str):
        path = os.path.join(self.path_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)

    def _cluster_stats(self, feature_names, labels_pred, valid_data):
        label_res = []
        for i in np.unique(labels_pred):
            label_dict = {}
            tmp_res = []
            for fidx, fn in enumerate(feature_names):
                fdata = valid_data[labels_pred == i, fidx].mean()
                label_dict[fn] = fdata
                tmp_res.append(fdata)
            label_res.append(tmp_res)
        label_ranks = self.ranked_(np.array(label_res))
        df = pd.DataFrame(label_ranks, columns=feature_names, index=np.unique(labels_pred))
        self.save_df(df, 'cluster_rank.csv')
        df = pd.DataFrame(label_res, columns=feature_names, index=np.unique(labels_pred))
        self.save_df(df, 'cluster_norm.csv')
        # for i in np.unique(labels_pred):
        #     for rk, value, fn in zip(label_ranks[i], label_res[i], feature_names, strict=True):
        #         print(f'{fn}: {value}(#{rk})')

    def _plot_cluster_result(self, points, labels, names, centers_):
        plt.figure(figsize=self.figsize)
        plt.axis('off')
        plt.tight_layout(pad=0)

        xs = points[:, 0]
        ys = points[:, 1]
        margin = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.05

        for i in np.unique(labels):
            plt.scatter(points[labels == i, 0], points[labels == i, 1], label=i, s=25)

        # text labels
        if len(names) > 0:
            texts = []
            for p, name in zip(points, names, strict=True):
                texts.append(plt.text(p[0] + margin * 0.1, p[1] + margin * 0.1, name, size=self.text_size))
            adjust_text(texts, only_move={'points': 'y', 'texts': 'y'})

        # centroids
        if len(centers_) == len(labels):
            centroids_x = centers_[:, 0]
            centroids_y = centers_[:, 1]
            plt.scatter(centroids_x, centroids_y, marker="x", s=15, linewidths=1, zorder=10)

        # save figure
        self.savefig('pca.png')

    def _plot_each_cluster(self, labels_, points_, names_, centers_):
        for i in np.unique(labels_):
            plt.figure(figsize=self.figsize)
            ax = plt.gca()
            xs = points_[labels_ == i, 0]
            ys = points_[labels_ == i, 1]
            margin = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.05
            ax.set_xlim([min(xs) - margin, max(xs) + margin])
            ax.set_ylim([min(ys) - margin, max(ys) + margin])
            plt.axis('off')

            plt.scatter(xs, ys, label=i, s=15)

            # centers
            if len(centers_) == 0:
                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)
            else:
                center_x = centers_[i][0]
                center_y = centers_[i][1]
            plt.scatter(center_x, center_y, marker="x", s=40, linewidths=2, zorder=10)

            # annot text
            if len(names_) > 0:
                texts = []
                for p, name, label in zip(points_, names_, labels_):
                    if label == i:
                        texts.append(plt.text(p[0] + margin * 0.1, p[1] + margin * 0.1, name, size=self.text_size))
                adjust_text(texts)

            # save
            plt.tight_layout(pad=margin)
            self.savefig(f'clusters/pca-{i}.png')
            plt.close()

    def _write_inconsistent_points(self, names_, labels_pred_, labels_gt):
        # 每个类别的结果
        pos_label = defaultdict(lambda: defaultdict(int))
        name_label_pos = {}
        label_pos = {}
        with open(os.path.join(self.path_dir, 'result.txt'), 'w') as f:
            for n, p, l in zip(names_, labels_gt, labels_pred_, strict=True):
                name_label_pos[n] = (l, p)
                pos_label[l][p] += 1
            for l, res in pos_label.items():
                f.write(f'类别#{l}: {dict(res)}\n')
                label_pos[l] = Counter(res).most_common(1)[0][0]
            if len(names_) == 0:
                return

            for n in names_:
                label, real_pos = name_label_pos[n]
                pred_pos = label_pos[label]
                if real_pos != pred_pos:
                    f.write(f'{n}: {real_pos} -> {pred_pos}\n')

    def _plot_each_feature(self, features_, data_, points_, names_, lite: bool = False):
        x = points_[:, 0]
        y = points_[:, 1]
        margin = max(max(x) - min(x), max(y) - min(y)) * 0.05
        for f_idx, f_name in enumerate(features_):
            # plot for each feature
            feat_data = data_[:, f_idx]

            plt.figure(figsize=self.figsize)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.scatter(points_[:, 0], points_[:, 1], c=feat_data, s=25, cmap='bwr')
            # text labels
            if len(names_) > 0:
                texts = []
                for p, name in zip(points_, names_, strict=True):
                    texts.append(plt.text(p[0] + margin * 0.1, p[1] + margin * 0.1, name, size=self.text_size))
                adjust_text(texts, only_move={'points': 'y', 'texts': 'y'})

            # save figure
            self.savefig(f'feature_scatter/pca-{f_name}.png')
            plt.close()

            if not lite:
                # Tricontour
                xy = np.column_stack([x.flat, y.flat])
                z = data_[:, f_idx]

                plt.figure(figsize=self.figsize)
                plt.axis('off')
                plt.tricontourf(x, y, z, cmap='RdBu_r')
                # text labels
                if len(names_) > 0:
                    texts = []
                    for p, name in zip(points_, names_):
                        texts.append(plt.text(p[0] + margin * 0.1, p[1] + margin * 0.1, name, size=self.text_size))
                    adjust_text(texts, only_move={'points': 'y', 'texts': 'y'})

                # save figure
                plt.title(f'{f_name}')
                self.savefig(f'feature_tricontour/{f_name}.png')
                plt.close()

                # Interpolate and generate heatmap:
                grid_x, grid_y = np.mgrid[x.min():x.max():1000j, y.min():y.max():1000j]
                for method in ['nearest', 'linear', 'cubic']:
                    plt.figure(figsize=self.figsize)
                    grid_z = scipy.interpolate.griddata(xy, z, (grid_x, grid_y), method=method)

                    plt.pcolormesh(grid_x, grid_y, ma.masked_invalid(grid_z), cmap='RdBu_r', vmin=np.nanmin(grid_z),
                                   vmax=np.nanmax(grid_z))
                    # text labels
                    if len(names_) > 0:
                        texts = []
                        for p, name in zip(points_, names_, strict=True):
                            texts.append(plt.text(p[0] + margin * 0.1, p[1] + margin * 0.1, name, size=self.text_size))
                    plt.title(f'{f_name}')
                    plt.colorbar()
                    self.savefig(f'feature_interpolation/{f_name}-{method}.png')
                    plt.clf()
                    plt.close()

    def _plot_cluster_feature(self, labels_, features_, data_, points_, names_):
        for i in np.unique(labels_):
            xs = points_[labels_ == i, 0]
            ys = points_[labels_ == i, 1]
            l_points = np.array(points_)[labels_ == i]
            l_names = np.array(names_)[labels_ == i]
            margin = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.05
            for f_idx, f_name in enumerate(features_):
                # plot for each feature
                plt.figure(figsize=self.figsize)
                ax = plt.gca()
                ax.set_xlim([min(xs) - margin, max(xs) + margin])
                ax.set_ylim([min(ys) - margin, max(ys) + margin])
                plt.axis('off')
                # plot for each feature
                feat_data = data_[labels_ == i, f_idx]

                plt.scatter(xs, ys, c=feat_data, s=60, cmap='bwr')
                # text labels
                if len(names_) > 0:
                    texts = []
                    for p, name in zip(l_points, l_names, strict=True):
                        texts.append(plt.text(p[0] + margin * 0.1, p[1] + margin * 0.1, name, size=self.text_size))
                    adjust_text(texts, only_move={'points': 'y', 'texts': 'y'})
                plt.title(f'{f_name}')
                self.savefig(f'clusters_feature/{i}/{f_name}.png')
                plt.close()

    def _player_stats(self, names_, feature_names_, data_, labels_gt_):
        labels_gt_ = np.array(labels_gt_)
        names_ = np.array(names_)
        data_ = np.array(data_)

        with open(os.path.join(self.path_dir, 'player_stats.txt'), 'w') as f:
            for label in np.unique(labels_gt_):
                label_ranks = self.ranked_(data_[labels_gt_ == label, :])
                label_names = names_[labels_gt_ == label]

                for name, rank in zip(label_names, label_ranks, strict=True):
                    f.write(f'{name} ({label})\n')
                    for feat, r in zip(feature_names_, rank, strict=True):
                        if feat == '平均死亡':
                            r = len(label_names) - r + 1
                        f.write(f'{feat}: #{r}\n')
                    f.write(f'平均排名: #{rank.mean():.1f}')
                    f.write('\n\n')

    def _player_stats_single(self, names_, feature_names_, data_, labels_gt_, name_, target_label_):
        labels_gt_ = np.array(labels_gt_)
        player_data = np.array([data_[names_.index(name_)]])
        names_ = np.array(names_)
        data_ = np.array(data_)

        def ranked_(arr: np.ndarray) -> np.ndarray:
            sorted_index = np.argsort(np.argsort(-arr, axis=0), axis=0) + 1
            return sorted_index

        with open('player_stats_pred.txt', 'w+') as f:
            label_ranks = ranked_(np.concatenate([data_[labels_gt_ == target_label_, :], player_data]))
            label_names = np.concatenate([names_[labels_gt_ == target_label_], np.array([name_])])

            f.write(f'{name_} ({target_label_})\n')

            for feat, r in zip(feature_names_, label_ranks[-1], strict=True):
                if feat == '平均死亡':
                    r = len(label_names) - r + 1
                f.write(f'{feat}: #{r}\n')
            f.write(f'平均排名: #{label_ranks[-1].mean():.1f}')
            f.write('\n\n')
