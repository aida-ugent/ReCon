from data_handler import data_utils
import torch.utils.data as data


def get_data(args, config, ot_method=None):
		if ot_method is None:
			ot_method = args.ot_method
		train_data, test_data_classification, test_data_ranking, user_num ,item_num, train_mat, val_data = data_utils.load_all(config=config)
		top_k_list_for_rec_performance = [1, 10, 100]
		top_k_list = [1, 10, 100]


		train_dataset = data_utils.CustomDatasetTrain(
		train_data, user_num, item_num, train_mat, args.num_ng, True)
		train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=data_utils.custom_collate)

		val_dataset = data_utils.CustomDatasetTrain(
		val_data, user_num, item_num, train_mat, args.num_ng, True)
		val_loader = data.DataLoader(val_dataset,
		batch_size=500, shuffle=False, num_workers=4, collate_fn=data_utils.custom_collate)

		test_dataset_classification = data_utils.CustomDataset(
		test_data_classification, user_num, item_num, train_mat, 0, False)
		test_dataset_ranking = data_utils.CustomDataset(
		test_data_ranking, user_num, item_num, train_mat, 0, False)
		
		test_loader_classification = data.DataLoader(test_dataset_classification,
		batch_size=args.batch_size, shuffle=False, num_workers=0)
		test_loader_ranking = data.DataLoader(test_dataset_ranking,
		batch_size=args.batch_size, shuffle=False, num_workers=0)
			
		return user_num,item_num,top_k_list_for_rec_performance,top_k_list,train_loader,test_loader_classification, test_loader_ranking, val_loader
