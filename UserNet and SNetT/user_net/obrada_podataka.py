import pandas as pd
from glob import glob
import pickle

# Učitavanje objava
dataPath = 'data/reddit2008/submissions_2008_asm/'
files = glob(dataPath + '*.csv')

data = []

for file in files:
    data.append(pd.read_csv(file, low_memory=False))

submissions = pd.concat(data, axis=0, ignore_index=True)

# Učitavanje komentara
dataPath = 'data/reddit2008/comments_2008_asm/'
files = glob(dataPath + '*.csv')

data = []

for file in files:
    data.append(pd.read_csv(file, low_memory=False))

comments = pd.concat(data, axis=0, ignore_index=True)

print("Završeno učitavanje")

# Provera komentara
print(comments['parent_id'].isnull().values.any())
print(comments['author'].isnull().values.any())
print(comments)

# Provera objava
print(submissions['id'].isnull().values.any())
print(submissions['id'].is_unique)
print(submissions['author'].isnull().values.any())
print(submissions)

# Brisanje [deleted] autora
submissions = submissions[submissions['author'] != '[deleted]']
comments = comments[comments['author'] != '[deleted]']

# Obrada podataka
comment_submission_author = {}
for submission in submissions.itertuples():
    comment_submission_author[submission.id] = submission.author

for comment in comments.itertuples():
    comment_submission_author[comment.id] = comment.author

comments['parent_author'] = 'None'

parent_author_column = comments.shape[1] - 1

noneAuthors = 0

for comment in comments.itertuples():
    # print(comment.Index)
    author = comment_submission_author.get(comment.parent_id.split('_')[1])
    if author is None:
        noneAuthors += 1
        comments.at[comment.Index, 'parent_author'] = '-1'
    else:
        comments.at[comment.Index, 'parent_author'] = author

print('Ukupno komentara %d' % len(comments))

comments = comments[comments['parent_author'] != '-1']

print(comments)

print("None authors %d" % noneAuthors)

with open('data/data_cleaned/comments', 'wb') as file:
    pickle.dump(comments, file)

print('Gotovo')
