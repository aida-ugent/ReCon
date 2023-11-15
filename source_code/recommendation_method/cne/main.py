import torch
from omegaconf import OmegaConf
import os
import time
import argparse
from ot_local.ot_plugin import OTPlugin
from data_handler.config import DatasetConfig
from data_handler.data_common import get_data
from recommendation_method.common.common import train
from recommendation_method.cne.model import IdentityModel


def get_ot_param_dict(args, train_loader, user_num, item_num):
	ot_plugin = OTPlugin(user_num, item_num, args.sinkhorn_gamma, args.sinkhorn_maxiter)
	if args.use_ot == 1:
		ot_param_dict = {"ot_plugin": ot_plugin, "lambda_p": args.lambda_p, 'train_dataset': train_loader.dataset, 'ot_method': args.ot_method}
	else:
		ot_param_dict = {"ot_plugin": ot_plugin, "lambda_p": args.lambda_p, 'train_dataset': train_loader.dataset}
	return ot_param_dict, ot_plugin


parser = argparse.ArgumentParser()
parser.add_argument("--path_to_config", 
	type=str,
	default='recommendation_method/cne/config.yml', 
	help="config path, parameters learning_rate, weight_decay, and dim are reinitialized using arguments in this file")
parser.add_argument("--base_path_to_config", 
	type=str,
	default='recommendation_method/cne/config.yml', 
	help="base config path, not used")

parser.add_argument("--batch_size", 
	type=int, 
	default=4096, 
	help="batch size for training")
parser.add_argument("--base_batch_size", 
	type=int, 
	default=4096, 
	help="base batch size for training")

parser.add_argument("--min_epochs", 
	type=int,
	default=0,  
	help="training epoches")
parser.add_argument("--epochs", 
	type=int,
	default=40000,  
	help="training epoches")
parser.add_argument("--base_epochs", 
	type=int,
	default=40000,  
	help="base training epoches")
parser.add_argument("--base_training_repeat", 
	type=int,
	default=0, 
	help="not used")
parser.add_argument("--base_monitor", 
	type=str,
	default='valid_loss', 
	help="valid_loss, valid_auc")

parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--out", 
	type=int,
	default=1,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
parser.add_argument("--device", 
	type=str,
	default="gpu",  
	help="cpu/gpu")
parser.add_argument("--use_ot", 
	type=int,
	default=0, 
	help="0 or 1")
parser.add_argument("--lambda_p", 
	type=float,
	default=0.005, 
	help="regularization weight")
parser.add_argument("--sinkhorn_gamma", 
	type=float,
	default=10.0, 
	help="sinkhorn gamma")
parser.add_argument("--sinkhorn_maxiter", 
	type=int,
	default=5, 
	help="sinkhorn max iterations")
parser.add_argument("--dataset", 
	type=str,
	default='career_builder_small', 
	help="")
parser.add_argument("--main_path", 
	type=str,
	default='Data/', 
	help="main path for data")
parser.add_argument("--model_path", 
	type=str,
	default='models/', 
	help="model path to save/load")
parser.add_argument("--figures_path", 
	type=str,
	default='figures/', 
	help="not used")
parser.add_argument("--use_pretrained", 
	type=int,
	default=0, 
	help="not used")
parser.add_argument("--early_stop", 
	type=int,
	default=1, 
	help="do early stopping?")

parser.add_argument("--patience", 
	type=int,
	default=5, 
	help="patience in early stopping")
parser.add_argument("--base_patience", 
	type=int,
	default=5, 
	help="base patience in early stopping")

parser.add_argument("--monitor", 
	type=str,
	default='valid_loss', 
	help="valid_loss, valid_auc")

parser.add_argument("--mode", 
	type=str,
	default='min', 
	help="min or max")
parser.add_argument("--ot_method", 
	type=str,
	default='all', 
	help="")

parser.add_argument("--popularity", 
	type=int,
	default=1, 
	help="whether to add popularity model or not")
parser.add_argument("--gradient_clipping", 
	type=float,
	default=0, 
	help="gradient clipping")
parser.add_argument("--precision", 
	type=int,
	default=32, 
	help="mixed precision")


	# cne configs
parser.add_argument("--dim", 
	type=int,
	default=8, 
	help="embedding_dimension")
parser.add_argument("--learning_rate", 
	type=float,
	default=0.01, 
	help="learning_rate")
parser.add_argument("--weight_decay", 
	type=float,
	default=0.01, 
	help="weight_decay")


parser.add_argument("--max_time", 
	type=float,
	default=-1, 
	help="max training time in hours. If < 0, does not set a limit")


if __name__ == "__main__":
	args = parser.parse_args()
	print(args)
	config = DatasetConfig('cne', args.model_path, args.main_path, args.dataset)

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	user_num,item_num,top_k_list_for_rec_performance,top_k_list,train_loader,test_loader_classification, test_loader_ranking, val_loader = get_data(args, config)
	print(user_num)
	print(item_num)

	print("load config")
	cne_config = OmegaConf.load(args.path_to_config)
	base_cne_config = OmegaConf.load(args.base_path_to_config)

	cne_config.interaction.parameters.dim = args.dim
	cne_config.interaction.parameters.learning_rate = args.learning_rate
	cne_config.interaction.parameters.weight_decay = args.weight_decay

	popularity_epochs = min(50, args.epochs)
	popularity_patience = 5
	popularity_model_file_path = '{}popularity_{}.pth'.format(args.model_path, args.dataset)
	base_model_file_path = '{}cne_{}_{}.pth'.format(args.model_path, 0, args.dataset)
	model_file_path = '{}cne_{}_{}.pth'.format(args.model_path, args.use_ot, args.dataset)

	if args.popularity:
		popularity_model = IdentityModel(user_num ,item_num, None, cne_config.popularity.parameters)
		if torch.cuda.is_available():
			popularity_model = popularity_model.cuda()

		########################### CREATE MODEL POPULARITY #################################
		if not os.path.exists(popularity_model_file_path):
			print("popularity model not found")
			popularity_model = train(args, config, popularity_model, IdentityModel, model_name='popularity', model_file_path=model_file_path, epochs=popularity_epochs, patience=popularity_patience, monitor=args.base_monitor, ot_method='all')
			if args.out:
				if not os.path.exists(args.model_path):
					os.mkdir(args.model_path)
				popularity_model.zero_grad()
				torch.save(popularity_model.state_dict(), 
					popularity_model_file_path)

		else:
			print("popularity model found")
			if torch.cuda.is_available():
				state = torch.load(popularity_model_file_path)
			else:
				state = torch.load(popularity_model_file_path, map_location=torch.device('cpu'))
			popularity_model.load_state_dict(state)


	class_ = IdentityModel
	if args.popularity:
		if torch.cuda.is_available():
			popularity_model.cuda()
		d = {'embeddings': {'emb_u': popularity_model.emb_u.weight.detach(), 'emb_i': popularity_model.emb_i.weight.detach()}, 
		'emb_operator': popularity_model._emb_operator}
		dependent_models = {'popularity': d}
	else:
		dependent_models = dict()
	cne_model = class_(user_num ,item_num, dependent_models, cne_config.interaction.parameters, ot_param_dict=dict(), use_ot=args.use_ot)
	if torch.cuda.is_available():
		cne_model.cuda()

	ot_param_dict, ot_plugin = get_ot_param_dict(args, train_loader, user_num, item_num)

	if not os.path.exists(model_file_path):
		print("model not found")

		########################### CREATE MODEL CNE #################################
		start_time = time.time()
		cne_model = class_(user_num ,item_num, dependent_models, cne_config.interaction.parameters, use_ot=args.use_ot, ot_param_dict=ot_param_dict)
		if torch.cuda.is_available():
			cne_model = cne_model.cuda()


		########################### TRAINING CNE #####################################
		cne_model = train(args, config, cne_model, class_, model_name='cne', ot_param_dict=ot_param_dict, dependent_models=dependent_models, model_file_path=model_file_path, epochs=args.epochs, patience=args.patience, monitor=args.monitor)
		if torch.cuda.is_available():
			cne_model = cne_model.cuda()
		end_time = time.time()
		total_time = end_time - start_time
		print("The total time is: " + 
			time.strftime("%H: %M: %S", time.gmtime(total_time)))

		if args.out:
			if not os.path.exists(args.model_path):
				os.mkdir(args.model_path)
			cne_model.zero_grad()
			torch.save(cne_model.state_dict(), 
				model_file_path)

	else:
		print("model found")
		if torch.cuda.is_available():
			state = torch.load(model_file_path)
		else:
			state = torch.load(model_file_path, map_location=torch.device('cpu'))
		cne_model.load_state_dict(state)
		cne_model.init_part2(args.use_ot, ot_param_dict, dependent_models, user_num, item_num)
		if torch.cuda.is_available():
			cne_model.cuda()
