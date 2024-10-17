import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.svm import SVC, SVR
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from empath import Empath
from nltk.corpus import stopwords
import os
from tqdm import tqdm
import pdb
from joblib import dump
from utils import noise_labels
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=os.path.join(os.getenv('WORK_DIR'), 'data/causal_text/hk/'))
    parser.add_argument('--csv-path', type=str, default='hk_rct/HKRepData_textfull.csv')
    parser.add_argument('--test-csv-path', type=str, default='hk_speeches/target_corpus.csv')
    parser.add_argument('--g', type=str, default='lr')
    parser.add_argument('--noise-seed', type=int, default=240621)

    args = parser.parse_args()

    return args

def encode_empath(df, text_col, lexicon):
    rows = []
    for i in tqdm(range(df.shape[0])):
        cat_counts = lexicon.analyze(df[text_col][i], categories=hk_categories)
        # sentences = nltk.sent_tokenize(df[args.text][i])
        # cat_counts['numtexts'] = len(sentences)
        rows.append(cat_counts)

    df_empath = pd.DataFrame(rows)
    df_empath[df_empath > 1] = 1
    df_empath.columns = ['treatycommit', 'brave', 'evil', 'flag', 'threat', 'economy', 'treatyviolation']

    return df_empath

args = get_args()

df = pd.read_csv(os.path.join(args.data_dir, args.csv_path))
df_test = pd.read_csv(os.path.join(args.data_dir, args.test_csv_path))

X = df[['treatycommit', 'brave', 'evil', 'flag', 'threat', 'economy', 'treatyviolation']]
y = df['resp']

if args.g == 'lr':
    g = RidgeCV(cv=5)
elif args.g == 'svm':
    g = SVR()
elif args.g == 'gbm':
    g = GradientBoostingRegressor()

g.fit(X, y)
dump(g, './models/hk/outcome_model_{}.joblib'.format(args.g))

# Evaluate the regression model

lexicon = Empath()
hk_categories = ['commitment', 'bravery', 'mistreatment', 'flags', 'threat', 'economy', 'violation']
csvs = ['HKarmstreatyobligation.csv', 'HKarmsbrave.csv', 'HKarmsevil.csv', 'HKarmsflag.csv', 'HKarmsthreat.csv', 'HKarmseconomy.csv', 'HKarmstreatyviolation.csv']
stop_words = set(stopwords.words('english'))
for i in range(len(hk_categories)):
    csv = pd.read_csv(os.path.join(args.data_dir, 'hk_rct/{}'.format(csvs[i])), header=0, names=['text'])
    words = ' '.join(csv.text.values.tolist()).split()
    words = [word for word in words if word.lower() not in stop_words]
    # words = list(set(words))
    lexicon.create_category(hk_categories[i], [hk_categories[i]] + words, model='nytimes')

df_empath = encode_empath(df_test, 'text_full', lexicon)
X_test = df_empath[['treatycommit', 'brave', 'evil', 'flag', 'threat', 'economy', 'treatyviolation']]

y_pred = g.predict(X_test)
np.random.seed(args.noise_seed)
mu = 0 
sigma = np.sqrt(np.var(df['resp'].values))
df_empath['resp'] = y_pred + np.random.normal(mu, sigma, len(y_pred))
df_empath['text_full'] = df_test['text_full']

df_empath.to_csv(
    os.path.join(args.data_dir, 'hk_speeches/speeches_w_outcome_{}.csv'.format(args.g)), index=False)

print('CV R2: {:.3f}'.format(g.score(X, y)))

# pdb.set_trace()