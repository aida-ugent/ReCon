import numpy as np
from ot_local.ot_exact import fpbm_new, emd
import torch
from matplotlib import pyplot as plt



class OTEvaluation:
    def __init__(self, user_num, item_num, do_print=True):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.do_print = do_print

    def compute_metrics(self, recom_model, predictions, top_k, compute_imbalance=True, lorenz_file_path_prefix=None, do_print=True):
        result_dict = dict()
        if predictions is not None:
            predictions_matrix = predictions.reshape((self.user_num, self.item_num))
            if compute_imbalance:
                self.compute_exact_imbalance(predictions_matrix, do_print=do_print)
            congestion, coverage, gini_index = self.compute_congestion_coverage_items(predictions_matrix, top_k, lorenz_file_path_prefix, do_print=do_print)
            result_dict['congestion_%d_items' % top_k] = congestion
            result_dict['coverage_%d_items' % top_k] = coverage
            result_dict['gini_%d_items' % top_k] = gini_index

            congestion, coverage, gini_index = self.compute_congestion_coverage_users(predictions_matrix, top_k, lorenz_file_path_prefix, do_print=do_print)
            result_dict['congestion_%d_users' % top_k] = congestion
            result_dict['coverage_%d_users' % top_k] = coverage
            result_dict['gini_%d_users' % top_k] = gini_index
        elif type(recom_model) is dict:
            congestion, coverage, gini_index = self.compute_congestion_coverage_items_from_dict(recom_model, top_k, lorenz_file_path_prefix, do_print=do_print)
            result_dict['congestion_%d_items' % top_k] = congestion
            result_dict['coverage_%d_items' % top_k] = coverage
            result_dict['gini_%d_items' % top_k] = gini_index

        else:
            congestion, coverage, gini_index = self.compute_congestion_coverage_items_from_model(recom_model, top_k, lorenz_file_path_prefix, do_print=do_print)
            result_dict['congestion_%d_items' % top_k] = congestion
            result_dict['coverage_%d_items' % top_k] = coverage
            result_dict['gini_%d_items' % top_k] = gini_index

            congestion, coverage, gini_index = self.compute_congestion_coverage_users_from_model(recom_model, top_k, lorenz_file_path_prefix, do_print=do_print)
            result_dict['congestion_%d_users' % top_k] = congestion
            result_dict['coverage_%d_users' % top_k] = coverage
            result_dict['gini_%d_users' % top_k] = gini_index
        return result_dict


    def compute_exact_imbalance(self, predictions_matrix, do_print=True):
        M = -torch.log(predictions_matrix)
        D = M.cpu().detach().numpy().squeeze()
        imb = emd(D)
        if do_print:
            print("imbalance: %.3f" % imb)
        return imb

    def compute_congestion_coverage_items_from_dict(self, recom_model, top_k, lorenz_file_path_prefix, do_print=True):
        market_shares = self._compute_items_market_share_from_dict(recom_model, top_k)
        return self.return_measures_items(lorenz_file_path_prefix, market_shares, do_print=do_print)

    def compute_congestion_coverage_items_from_model(self, recom_model, top_k, lorenz_file_path_prefix, do_print=True):
        market_shares = self._compute_items_market_share_from_model(recom_model, top_k)
        return self.return_measures_items(lorenz_file_path_prefix, market_shares, do_print=do_print)

    def compute_congestion_coverage_users_from_model(self, recom_model, top_k, lorenz_file_path_prefix, do_print=True):
        market_shares = self._compute_users_market_share_from_model(recom_model, top_k)
        return self.return_measures_items(lorenz_file_path_prefix, market_shares, do_print=do_print)

    def compute_congestion_coverage_items(self, predictions_matrix, top_k, lorenz_file_path_prefix, do_print=True):
        market_shares = self._compute_items_market_share(predictions_matrix, top_k)
        return self.return_measures_items(lorenz_file_path_prefix, market_shares, do_print=do_print)

    def return_measures_items(self, lorenz_file_path_prefix, market_shares, do_print=True):
        congestion = self._compute_congestion_from_market_shares(market_shares, self.item_num)
        coverage = self._compute_coverage_from_market_shares(market_shares, self.item_num)
        if torch. is_tensor(market_shares):
            market_shares = market_shares.cpu().detach().numpy()
        gini_index = self.gini(market_shares, self.item_num)
        if lorenz_file_path_prefix is not None:
            self.lorenz_curve(market_shares, lorenz_file_path_prefix+'_items.png')
        if do_print:
            print("congestion items: %.3f" % congestion)
            print("coverage items: %.3f" % coverage)
            print("gini items: %.3f" % gini_index)
        return congestion, coverage, gini_index

    def compute_congestion_coverage_users(self, predictions_matrix, top_k, lorenz_file_path_prefix, do_print=True):
        market_shares = self._compute_users_market_share(predictions_matrix, top_k)
        return self.return_measures_user(lorenz_file_path_prefix, market_shares, do_print=do_print)

    def return_measures_user(self, lorenz_file_path_prefix, market_shares, do_print=True):
        congestion = self._compute_congestion_from_market_shares(market_shares, self.user_num)
        coverage = self._compute_coverage_from_market_shares(market_shares, self.user_num)
        if torch. is_tensor(market_shares):
            market_shares = market_shares.cpu().detach().numpy()
        gini_index = self.gini(market_shares, self.user_num)
        if lorenz_file_path_prefix is not None:
            self.lorenz_curve(market_shares, lorenz_file_path_prefix+'_users.png')
        if do_print:
            print("congestion users: %.3f" % congestion)
            print("coverage users: %.3f" % coverage)
            print("gini users: %.3f" % gini_index)
        return congestion, coverage, gini_index

    def user_item_pairs(self, user_based=True, count=1):
        if user_based:
            items = torch.LongTensor([int(i) for i in range(0, self.item_num)])
            if torch.cuda.is_available():
                items = items.cuda()
            for start_user_id in range(0, self.user_num, count):
                user_ids = [user_id for user_id in range(start_user_id, min(self.user_num, start_user_id+count))]
                user = torch.LongTensor([int(user_id) for user_id in user_ids])
                if torch.cuda.is_available():
                    user = user.cuda()
                yield user.repeat_interleave(self.item_num), items.repeat(min(self.user_num, start_user_id+count)-start_user_id)
        else:
            users = torch.LongTensor([int(u) for u in range(0, self.user_num)])
            if torch.cuda.is_available():
                users = users.cuda()
            for start_item_id in range(0, self.item_num, count):
                item_ids = [item_id for item_id in range(start_item_id, min(self.item_num, start_item_id+count))]
                item = torch.LongTensor([int(item_id) for item_id in item_ids])
                if torch.cuda.is_available():
                    item = item.cuda()
                yield users.repeat_interleave(min(self.item_num, start_item_id+count)-start_item_id), item.repeat(self.user_num)

    def _compute_items_market_share_from_dict(self, result_dict, top_k):
        recommendation_list = list()
        for user in range(self.user_num):
            recommendations = result_dict[user][:top_k]
            recommendation_list += recommendations

        counter = {}
        for word in recommendation_list:
            if word not in counter:
                counter[word] = 1
            else:
                counter[word] += 1


        market_shares = np.array(list(counter.values()))/(self.user_num*top_k)
        return market_shares


    def _compute_items_market_share_from_model(self, recom_model, top_k):
        tensor_list = []
        for user, items in self.user_item_pairs(user_based=True, count=100):
            prediction = recom_model(user, items)
            predictions_matrix = prediction.reshape((-1, self.item_num))
            top_k_indices = torch.topk(predictions_matrix, top_k).indices
            tensor_list.append(top_k_indices)
        top_k_indices = torch.cat(tensor_list, dim=0)
        counts = top_k_indices.unique(return_counts=True)[1]
        market_shares = counts/(self.user_num*top_k)
        return market_shares

    def _compute_users_market_share_from_model(self, recom_model, top_k):
        tensor_list = []
        for user, items in self.user_item_pairs(user_based=False, count=100):
            prediction = recom_model(user, items)
            predictions_matrix = prediction.reshape((self.user_num, -1))
            top_k_indices = torch.topk(predictions_matrix, top_k, dim=0).indices
            tensor_list.append(top_k_indices)
        top_k_indices = torch.cat(tensor_list, dim=1)
        counts = top_k_indices.unique(return_counts=True)[1]
        market_shares = counts/(self.item_num*top_k)
        return market_shares

    def _compute_items_market_share(self, predictions_matrix, top_k):
        top_k_indices = torch.topk(predictions_matrix, top_k).indices
        counts = top_k_indices.unique(return_counts=True)[1]
        market_shares = counts/(self.user_num*top_k)
        return market_shares

    def _compute_users_market_share(self, predictions_matrix, top_k):
        top_k_indices = torch.topk(predictions_matrix, top_k, dim=0).indices
        counts = top_k_indices.unique(return_counts=True)[1]
        market_shares = counts/(self.item_num*top_k)
        return market_shares

    def _compute_congestion_from_market_shares(self, market_shares, number_of_objects_recommending):
        if torch. is_tensor(market_shares):
            return torch.sum(torch.log(market_shares) * market_shares)/np.log(number_of_objects_recommending)
        else:
            return np.sum(np.log(market_shares) * market_shares)/np.log(number_of_objects_recommending)
        
    def _compute_coverage_from_market_shares(self, market_shares, number_of_items):
        return market_shares.shape[0]/(number_of_items)

    @staticmethod
    def gini(arr, total_count):
        ## first sort
        sorted_arr = arr.copy()
        sorted_arr.sort()
        n = total_count
        offset = total_count - sorted_arr.size
        coef_ = 2. / n
        const_ = (n + 1.) / n
        weighted_sum = sum([(i+1+offset)*yi for i, yi in enumerate(sorted_arr)])
        return coef_*weighted_sum/(sorted_arr.sum()) - const_

    @staticmethod
    def lorenz_curve(X, filepath):
        X_lorenz = X.cumsum() / X.sum()
        X_lorenz = np.insert(X_lorenz, 0, 0) 
        X_lorenz[0], X_lorenz[-1]
        fig, ax = plt.subplots(figsize=[6,6])
        ## scatter plot of Lorenz curve
        ax.scatter(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz, 
                marker='x', color='darkgreen', s=100)
        ## line plot of equality
        ax.plot([0,1], [0,1], color='k')
        plt.savefig(filepath, dpi=300)
