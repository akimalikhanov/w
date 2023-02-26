PATH_DICT={
    'application_train': '../data/application_train.csv',
    'application_test': '../data/application_test.csv',
    'bur': '../data/bureau.csv',
    'bb': '../data/bureau_balance.csv',
    'previous': '../data/previous_application.csv',
    'cash': '../data/POS_CASH_balance.csv',
    'installments': '../data/installments_payments.csv',
    'card_balance': '../data/credit_card_balance.csv',
    'ohe_dict': '../models/pipeline/ohe_dict.pkl',
    'corr_matrix': '../data/corr_matrix.csv',
    'missing_columns_drop': '../models/pipeline/missing_columns_drop.pkl',
    'zero_variance_drop': '../models/pipeline/zero_var_columns_drop.pkl',
    'zero_imp_drop': '../models/pipeline/zero_imp.pkl',
    'model_file': '../models/model.txt',
    'lgb_ohe': '../models/pipeline/lgb_ohe.pkl',
    'train_ready_file': '../data/train_ready.csv',
    'test_ready_file': '../data/test_ready.csv',
    'dtypes': '../models/pipeline/dtypes.pkl',
    'submit': '../data/submit.csv',
    }


PARAMS={
    'num_boost_round': 10000,
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'subsample': 0.8,
    'n_jobs': -1,
    'random_state': 5,
    'verbose': -1
    }