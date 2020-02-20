configs = {
    "train_neg_sample":5,
    # "test_neg"
    "seed": 2,
    #paper train
    "train_max_papers_each_author": 16,
    "train_max_news_each_para_length": 64,
    "train_max_paper_length": 192,

    #paper_test
    "test_author_neg_sample": 9,
    "min_papers_each_author": 10,


    #news
    "test_max_news_each_para_length": 64,
    "test_news_max_papers_each_author": 100,
    "test_max_paper_length": 164,
    "test_paper_max_papers_each_author": 100,


    
    "domain_paper_batch_size": 4,
    "domain_news_batch_size": 32,

    "n_epoch": 20,
    "bert_size": 768,
    "hidden_size": 300,

    "train_bert_learning_rate" : 2e-5,
    "train_knrm_learning_rate": 2e-3,

    "bert_learning_rate" : 2e-5,
    "knrm_learning_rate": 2e-3,
    # "rl_learning_rate" : 1e-3,
    "adversarial_learning_rate" : 1e-3,
    
    "sample_num" : 4,

    "domain_accum_step": 4,
    "domain_scheduler_step": 500,

    "local_accum_step": 16,
    "scheduler_step" : 32
}
