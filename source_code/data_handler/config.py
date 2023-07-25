class DatasetConfig:
    def __init__(self, model, model_path, main_path, dataset):
        super().__init__()
        self.model = model
        self.model_path = model_path
        self.main_path = main_path
        self.dataset = dataset
        self.train_rating = main_path + '{}/{}.train.rating'.format(dataset, dataset)
        self.val_rating = main_path + '{}/{}.val.rating'.format(dataset, dataset)
        self.test_rating_classification = main_path + '{}/{}.test_classification.rating'.format(dataset, dataset)
        self.test_rating_ranking = main_path + '{}/{}.test_ranking.rating'.format(dataset, dataset)
        self.test_negative = main_path + '{}/{}.test.negative'.format(dataset, dataset)
