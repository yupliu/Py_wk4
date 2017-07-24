#Modeling weeks
#Written by Lan

import graphlab
people = graphlab.SFrame('D:\\ML_Learning\\people_wiki.csv')
eltonjohn = people[people['name']=='Elton John']
eltonjohn['word_count'] = graphlab.text_analytics.count_words(eltonjohn['text'])
eltonjohn_tbl = eltonjohn[['word_count']].stack('word_count',new_column_name=['word','count'])
eltonjohn_tbl.sort('count',ascending=False)
people['word_count'] = graphlab.text_analytics.count_words(people['text'])
people['tfidf'] = graphlab.text_analytics.tf_idf(people['word_count'])

#have to rerun this, otherwise it cannot find column tfidf
eltonjohn = people[people['name']=='Elton John']
eltonjohn[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)

VB = people[people['name']=='Victoria Beckham']
PM = people[people['name']=='Paul McCartney']

graphlab.distances.cosine(eltonjohn['tfidf'][0],VB['tfidf'][0])
graphlab.distances.cosine(eltonjohn['tfidf'][0],PM['tfidf'][0])

knn_count = graphlab.nearest_neighbors.create(people,features=['word_count'],label='name',distance='cosine')
knn_tfidf = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name',distance = 'cosine')



knn_count.query(eltonjohn)
knn_tfidf.query(eltonjohn)

knn_count.query(VB)
knn_tfidf.query(VB)