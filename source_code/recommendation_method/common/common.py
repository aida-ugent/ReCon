import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from data_handler.data_common import get_data
from recommendation_method.common.custom_early_stopping import LogicalOREarlyStopping
import torch


def train(args, config, r_model, ModelClass, model_name, model_file_path, epochs, patience, monitor=None, ot_param_dict=None, dependent_models=None, ot_method=None):
	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	torch.set_float32_matmul_precision('medium')

	min_delta = 0.0001
	if monitor is None:
		monitor = args.monitor
	monitor_metrics = monitor.split(',')
	
	user_num,item_num,top_k_list_for_rec_performance,top_k_list,train_loader,test_loader_classification, test_loader_ranking, val_loader = get_data(args, config, ot_method=ot_method)
	# initialize trainer  
	if args.early_stop:
		if ',' in monitor:
			callbacks = [LogicalOREarlyStopping(monitor=monitor_metrics, min_delta=min_delta, 
	patience=patience, verbose=True, mode=args.mode)]
		else:
			print("EarlyStopping initialized with patience %d" % patience)
			callbacks = [EarlyStopping(monitor=monitor, min_delta=min_delta, 
	patience=patience, verbose=True, mode=args.mode)]
	else:
		callbacks = []      
	checkpoint_callbacks = list()
	for m in monitor_metrics:
		checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=m, mode=args.mode)
		checkpoint_callbacks.append(checkpoint_callback)
		callbacks.append(checkpoint_callback)

	if torch.cuda.is_available():
		r_model = r_model.cuda()
	if args.gradient_clipping > 0:
		gradient_clip_val = args.gradient_clipping
	else:
		gradient_clip_val = None

	if args.max_time <= 0:
		max_time_dict = None
	else:
		max_time_dict = {'hours': float(args.max_time)}
	trainer = pl.Trainer(callbacks=callbacks, num_sanity_val_steps=0, enable_checkpointing=True, gradient_clip_val=gradient_clip_val, precision=args.precision,
		    default_root_dir=os.path.join(args.model_path, 'model_checkpoints', model_file_path.split('/')[-1].replace('.pth', '')),
			logger=False, min_epochs=args.min_epochs, max_epochs=epochs, 
			accelerator=args.device, check_val_every_n_epoch=1, log_every_n_steps=1, max_time=max_time_dict)

	r_model.train()
	print("trainer.fit")
	trainer.fit(r_model, train_loader, val_loader)
	print("trainer.fit finished")

	best_checkpoint_epoch = -1
	best_checkpoint_path = ''
	for checkpoint_callback in checkpoint_callbacks:
		checkpoint_epoch = int(checkpoint_callback.best_model_path.split('epoch=')[1].split('-step')[0])
		if checkpoint_epoch > best_checkpoint_epoch:
			print(checkpoint_callback.monitor, checkpoint_callback.best_model_score)
			best_checkpoint_epoch = checkpoint_epoch
			best_checkpoint_path = checkpoint_callback.best_model_path

	best_model = ModelClass.load_from_checkpoint(best_checkpoint_path)
	if model_name == 'cne':
		best_model.init_part2(args.use_ot, ot_param_dict, dependent_models, user_num, item_num)

	return best_model
