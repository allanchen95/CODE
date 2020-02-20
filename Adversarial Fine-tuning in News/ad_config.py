configs = {
    "train_neg_sample":5,
    # "test_neg"
    # "seed": 2,
    # #paper train
    # "train_max_papers_each_author": 12,
    # "train_max_news_each_para_length": 64,
    # "train_max_paper_length": 192,

    # #paper_test
    # "test_author_neg_sample": 9,
    # "min_papers_each_author": 10,


    # #news
    # "test_max_news_each_para_length": 64,
    # "test_news_max_papers_each_author": 100,
    # "test_max_paper_length": 192,
    # "test_paper_max_papers_each_author": 100,

    "seed": 2,
    #paper train
    "train_max_papers_each_author": 9,
    "train_max_news_each_para_length": 64,
    "train_max_paper_length": 192,
    # "train_max_paper_name_length": 42,
    # "train_max_paper_attr_length": 180,
    #paper_test
    "test_author_neg_sample": 17,
    "min_papers_each_author": 9,


    #news
    "test_max_news_each_para_length": 64,
    "test_news_max_papers_each_author": 200,
    "test_max_paper_length": 192,
    "test_paper_max_papers_each_author": 200,

    
    "domain_paper_batch_size": 1,
    "domain_news_batch_size": 64,

    "n_epoch": 20,
    "hidden_size": 768,

    "train_bert_learning_rate" : 2e-5,
    "train_knrm_learning_rate": 2e-3,

    "bert_learning_rate" : 1e-5,
    "knrm_learning_rate": 1e-3,
    # "rl_learning_rate" : 1e-3,
    "adversarial_learning_rate" : 1e-3,
    
    "sample_num" : 4,

    "domain_accum_step": 2,
    "domain_scheduler_step": 500,

    "local_accum_step": 16,
    "scheduler_step" : 32
}
