import pandas as pd
from sklearn import tree


train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

print('訓練データ：',train.shape, ',', 'テストデータ：', test.shape)
# print('訓練データ：{}' .format(train.shape), ', テストデータ：{}' .format(test.shape))
train.describe()
test.describe()
def null_table(df):
    null_value = df.isnull().sum()
    null_percent = null_value/len(df)
    null_sheet = pd.concat([null_value, null_percent], axis=1)
    null_sheet_rename = null_sheet.rename(columns={0:'欠損値', 1:'%'})
    return null_sheet_rename

null_table(test)

train.Age = train.Age.fillna(train.Age.median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

train['Sex'] = train.Sex.replace({'male':0, 'female':1})
# train.loc[train.Sex=='male', 'Sex'] = 0
# train.loc[train.Sex=='female', 'Sex'] = 1
# train.Sex[train.Sex=='male'] = 0
# train.Sex[train.Sex=='female'] = 1
train['Embarked'] = train['Embarked'].replace({'S':0, 'C':1, 'Q':2})

test['Age'] = test['Age'].fillna(test['Age'].median())
test[test['Fare'].isnull()]
test['Fare'][152] = test['Fare'].median()

test['Sex'] = test['Sex'].replace({'male':0, 'female':1})
test['Embarked'] = test['Embarked'].replace({'S':0, 'C':1, 'Q':2})

target = train['Survived']
feature = train[['Pclass','Sex','Age','Embarked']]

model = tree.DecisionTreeClassifier()
lng = model.fit(feature, target)

feature_2 = train[['Pclass','Sex','Age','Fare','SibSp','Parch','Embarked']]
model_2 = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=5)
lng_2 = model_2.fit(feature_2, target)


test_feature = test[['Pclass','Sex','Age','Embarked']]
pred = lng.predict(test_feature)
PassengerId = test['PassengerId'].values
solution = pd.DataFrame(pred, index=PassengerId,columns=['Survived'])
solution.to_csv('my_tree.csv', index_label='PassengerId')

test_feature_2 = test[['Pclass','Sex','Age','Fare','SibSp','Parch','Embarked']]
pred_2 = lng_2.predict(test_feature_2)
PassengerId = test['PassengerId'].values
solution_2 = pd.DataFrame(pred_2, index=PassengerId,columns=['Survived'])
solution_2.to_csv('my_tree_2.csv', index_label='PassengerId')
