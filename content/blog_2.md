Title: Don't be a Dummy, (For)get_dummies!
Date: 2019-04-10 10:20
Tags: python
Slug: blog_2


pd.get_dummies is a pandas function that transforms a column containing categorical variables into a series of new columns composed of 1s and 0s (true and false). The significance in this process is that in order for any model or algorithm to process categorical features, they must first be transformed into numerical representation. Why not assign categorical features a number between 1 and 10 and keep them in one column? This process would result in assigning irrelevant values to items in a categorical column that would be misinterpreted by an algorithm. More columns composed of 1s and 0s is preferred over a single column, because it has the benefit of categories not weighting a value improperly.

To show how Get_Dummies works, I will create a dictionary of hockey players and transform it into a DataFrame:


```python
from IPython.display import HTML
import pandas as pd
hockey_players = ({
    
    'name': 'Sidney Crosby',
    'age': '31',
    'team': 'Pittsburgh Penguins',
    'position': 'C',
    'nationality': 'Canadian',
    'hart': 'yes',
    'stanley_cup': 'yes'
},
{
    'name': 'Nikita Kucherov',
    'age': '25',
    'team': 'Tampa Bay Lightning',
    'position': 'RW',
    'nationality': 'Russian',
    'hart': 'no',
    'stanley_cup': 'no'
},
{
    'name': 'Connor McDavid',
    'age': '22',
    'team': 'Edmonton Oilers',
    'position': "C",
    'nationality': 'Canadian',
    'hart': 'yes',
    'stanley_cup': 'no'
},
{
    'name': 'Mikko Rantanen',
    'age': '22',
    'team': "Colorado Avalanche", 
    'position': 'RW',
    'nationality': 'Finnish',
    'hart': 'no',
    'stanley_cup': 'no'
},
{
    'name': 'Alex Ovechkin',
    'age': '33',
    'team': 'Washington Capitals',
    'position': 'LW',
    'nationality': 'Russian',
    'hart': 'yes',
    'stanley_cup': 'yes'
},
{
    'name': 'Braden Holtby',
    'age': '29',
    'team': 'Washington Capitals',
    'position': 'G',
    'nationality': 'Canadian',
    'hart': 'no',
    'stanley_cup': 'yes'
},
{
    'name': 'Brent Burns',
    'age': '34',
    'team': 'San Jose Sharks',
    'position': 'D',
    'nationality': 'Canadian',
    'hart': 'no',
    'stanley_cup': 'no'
},
{
    'name': 'John Tavares',
    'age': '28',
    'team': 'Toronto Maple Leafs',
    'position': 'C',
    'nationality': 'Canadian',
    'hart': 'no',
    'stanley_cup': 'no'
}, 
{
    'name': 'Mark Scheiele',
    'age': '26',
    'team': 'Winnipeg Jets',
    'position': 'C',
    'nationality': 'Canadian',
    'hart': 'no',
    'stanley_cup': 'no'
},
{
    'name': 'Carey Price',
    'age': '31',
    'team': 'Montreal Canadians',
    'position': 'G',
    'nationality': 'Canadian',
    'hart': 'yes',
    'stanley_cup': 'no'
},
{
    'name': 'Morgan Rielly',
    'age': '25',
    'team': 'Toronto Maple Leafs',
    'position': 'D',
    'nationality': 'American',
    'hart': 'no',
    'stanley_cup': 'no'

},
{
    'name': 'Nathan MacKinnon',
    'age': '23',
    'team': 'Colorado Avalanche',
    'position': 'C',
    'nationality': 'Canadian',
    'hart': 'no',
    'stanley_cup': 'no'
},
{
    'name': 'Evgeni Malkin',
    'age': '32',
    'team': 'Pittsburgh Penguins',
    'position': 'C',
    'nationality': 'Russian',
    'hart': 'yes',
    'stanley_cup': 'yes'
})
```


```python
hockey_players = pd.DataFrame(hockey_players)
hockey_players = hockey_players[['name', 'age', 'nationality', 'team', 'position', 'stanley_cup', 'hart']]
hockey_players.head()

HTML(hockey_players.head(5).to_html(classes="table table-stripped table-hover"))
```




<table border="1" class="dataframe table table-stripped table-hover">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>nationality</th>
      <th>team</th>
      <th>position</th>
      <th>stanley_cup</th>
      <th>hart</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sidney Crosby</td>
      <td>31</td>
      <td>Canadian</td>
      <td>Pittsburgh Penguins</td>
      <td>C</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nikita Kucherov</td>
      <td>25</td>
      <td>Russian</td>
      <td>Tampa Bay Lightning</td>
      <td>RW</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Connor McDavid</td>
      <td>22</td>
      <td>Canadian</td>
      <td>Edmonton Oilers</td>
      <td>C</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mikko Rantanen</td>
      <td>22</td>
      <td>Finnish</td>
      <td>Colorado Avalanche</td>
      <td>RW</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alex Ovechkin</td>
      <td>33</td>
      <td>Russian</td>
      <td>Washington Capitals</td>
      <td>LW</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>



Above is a simple dataset consisiting of 13 National Hockey League players. For each player, we are provided with

        -name 
        -age 
        -nationality 
        -team 
        -position 
        -whether he has won a Stanley Cup 
        -whether he has won the Hart trophy.

For the purpose of this blog, let's imagine that this is a much larger dataset consisting of all NHL players. As I mentioned above, in order to compare categorical variables, we need to transform them into 1s and 0s. 

If a column is categorical and based on two inputs, forexample 'yes' and 'no', this can be as simple as: 


```python
hockey_players['stanley_cup'] = hockey_players['stanley_cup'].map({'yes': 1, 'no': 0})
```

                                OR


```python
hockey_players['hart'] = hockey_players['hart'].replace(['yes', 'no'], [1, 0])
```


```python
hockey_players.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>nationality</th>
      <th>team</th>
      <th>position</th>
      <th>stanley_cup</th>
      <th>hart</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sidney Crosby</td>
      <td>31</td>
      <td>Canadian</td>
      <td>Pittsburgh Penguins</td>
      <td>C</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nikita Kucherov</td>
      <td>25</td>
      <td>Russian</td>
      <td>Tampa Bay Lightning</td>
      <td>RW</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Connor McDavid</td>
      <td>22</td>
      <td>Canadian</td>
      <td>Edmonton Oilers</td>
      <td>C</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mikko Rantanen</td>
      <td>22</td>
      <td>Finnish</td>
      <td>Colorado Avalanche</td>
      <td>RW</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alex Ovechkin</td>
      <td>33</td>
      <td>Russian</td>
      <td>Washington Capitals</td>
      <td>LW</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Whenever the column is more complex, this is when someone would reach for pd.get_dummies.

Say we thought that the nationality of a player was indicative of performance. In order to use this categorical value in a model, we would need to transform these values into 1s and 0s. 

Now I am going to show how to use pd.get_dummies to transform the nationality column into multiple columns that correspond with the nationality of each player. 


```python
dummy = pd.get_dummies(hockey_players['nationality'])
dummy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>American</th>
      <th>Canadian</th>
      <th>Finnish</th>
      <th>Russian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The function provided us with 4 columns each representing 1 of 4 nationalities in the dataset. In order to compare this information to the original dataset, we will concatenate onto the original DataFrame. While we are doing that, we will drop the original column for nationality, as well as the name category, as each player is already identfied by an id number.


```python
hockey_players = pd.concat([hockey_players, dummy], axis=1)
hockey_players = hockey_players.drop(columns=['nationality'])
hockey_players = hockey_players.drop(columns=['name'])
hockey_players.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>team</th>
      <th>position</th>
      <th>stanley_cup</th>
      <th>hart</th>
      <th>American</th>
      <th>Canadian</th>
      <th>Finnish</th>
      <th>Russian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31</td>
      <td>Pittsburgh Penguins</td>
      <td>C</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25</td>
      <td>Tampa Bay Lightning</td>
      <td>RW</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>Edmonton Oilers</td>
      <td>C</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22</td>
      <td>Colorado Avalanche</td>
      <td>RW</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>Washington Capitals</td>
      <td>LW</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



At first sight, this seems like a very useful tool provided by pandas. We can now compare the nationality of each player to the values in the other columns.

However, there is one very large downside to using pd.get_dummies, which I will demonstrate below.

Imagine that we were attempting to see whether all these factors could be used to make a model that predicted whether a player would win the Hart trophy. First we will identify our features and our target.





```python
features = [cols for cols in hockey_players if cols != 'hart']

X = hockey_players[features]
y = hockey_players['hart']
```

Since the dataset is clean (no missing values), the next thing we will do is split the data into training and testing sets.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
```

Now we have two sets of data:
    
    -One to train our model on (X_train)
    -One to test our final model on (X_test)
    
Next we look at our X_train and we see that we will need to use pd.get_dummies on the categorical columns for position and team. 


```python
X_train_with_dummy = pd.get_dummies(X_train, columns=['team', 'position'], drop_first=True)
```


```python
X_train_with_dummy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>stanley_cup</th>
      <th>American</th>
      <th>Canadian</th>
      <th>Finnish</th>
      <th>Russian</th>
      <th>team_Montreal Canadians</th>
      <th>team_Pittsburgh Penguins</th>
      <th>team_Tampa Bay Lightning</th>
      <th>team_Toronto Maple Leafs</th>
      <th>team_Washington Capitals</th>
      <th>position_D</th>
      <th>position_G</th>
      <th>position_LW</th>
      <th>position_RW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>25</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>29</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Next, we will do the same thing for the testing data.


```python
X_test_with_dummy = pd.get_dummies(X_test, columns=['team', 'position'], drop_first=True)
```



Now with our training and testing data hot-encoded, we are ready to model! Let's instantiate the Logistic Regression and fit and score our model.





```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_with_dummy, y_train)
model.score(X_test_with_dummy, y_test)
```

    /anaconda3/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-13-7057dac4b7c1> in <module>
          4 
          5 model.fit(X_train_with_dummy, y_train)
    ----> 6 model.score(X_test_with_dummy, y_test)
    

    /anaconda3/lib/python3.6/site-packages/sklearn/base.py in score(self, X, y, sample_weight)
        286         """
        287         from .metrics import accuracy_score
    --> 288         return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        289 
        290 


    /anaconda3/lib/python3.6/site-packages/sklearn/linear_model/base.py in predict(self, X)
        279             Predicted class label per sample.
        280         """
    --> 281         scores = self.decision_function(X)
        282         if len(scores.shape) == 1:
        283             indices = (scores > 0).astype(np.int)


    /anaconda3/lib/python3.6/site-packages/sklearn/linear_model/base.py in decision_function(self, X)
        260         if X.shape[1] != n_features:
        261             raise ValueError("X has %d features per sample; expecting %d"
    --> 262                              % (X.shape[1], n_features))
        263 
        264         scores = safe_sparse_dot(X, self.coef_.T,


    ValueError: X has 9 features per sample; expecting 15


VALUE ERROR! 


When we used pd.get_dummies to make dummy columns that were hot-encoded, we over-looked the fact that the testing data was much smaller and it did not have examples for each categorical variable. Although we performed pd.get_dummies on the testing data, as we did on the training data, there was no way for the transformed testing data to know that it was missing certain columns that were present in the original data set and still existed in the training data.

Observe the shapes of our two data sets:


```python
X_train_with_dummy.shape
```




    (10, 15)




```python
X_test_with_dummy.shape
```




    (3, 9)





Some may argue that this could easily be avoided by using pd.get_dummies before splitting the data into training and testing sets. However, every good Data Scientist knows the importance of using train_test_split first before tampering with your dataset.

These two ideas seem at odds, because they are! What to do?

Luckily SkLearn provides us with a better and more advanced function which considers this possible dilemna. Because of this, SkLearn's Label Binarizer should be used instead of pd.get_dummies.


```python
from sklearn.preprocessing import LabelBinarizer

features = [cols for cols in hockey_players if cols != 'hart']

X_train = hockey_players[features]
y = hockey_players['hart']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
```


```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>team</th>
      <th>position</th>
      <th>stanley_cup</th>
      <th>American</th>
      <th>Canadian</th>
      <th>Finnish</th>
      <th>Russian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>32</td>
      <td>Pittsburgh Penguins</td>
      <td>C</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>25</td>
      <td>Toronto Maple Leafs</td>
      <td>D</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22</td>
      <td>Colorado Avalanche</td>
      <td>RW</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>23</td>
      <td>Colorado Avalanche</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>Edmonton Oilers</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
lb = LabelBinarizer()

lb.fit(X_train['position'])
lb.classes_      #This is where the magic happens!
lb.transform(X_train['position'])

X_train = X_train.join(
    pd.DataFrame(lb.fit_transform(X_train['position']),
    columns=lb.classes_,
    index=X_train.index))

X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>team</th>
      <th>position</th>
      <th>stanley_cup</th>
      <th>American</th>
      <th>Canadian</th>
      <th>Finnish</th>
      <th>Russian</th>
      <th>C</th>
      <th>D</th>
      <th>G</th>
      <th>LW</th>
      <th>RW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>32</td>
      <td>Pittsburgh Penguins</td>
      <td>C</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>25</td>
      <td>Toronto Maple Leafs</td>
      <td>D</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22</td>
      <td>Colorado Avalanche</td>
      <td>RW</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>23</td>
      <td>Colorado Avalanche</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>Edmonton Oilers</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>29</td>
      <td>Washington Capitals</td>
      <td>G</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>34</td>
      <td>San Jose Sharks</td>
      <td>D</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>31</td>
      <td>Pittsburgh Penguins</td>
      <td>C</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25</td>
      <td>Tampa Bay Lightning</td>
      <td>RW</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>Washington Capitals</td>
      <td>LW</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The built in method call 'classes_' is the saviour in this function. By fitting the training data and than labeling the columns based on the method call 'classes_', the existence of all 5 positions is remembered. Simply put, the attribute 'classes_' holds the label for each class. Now we will transform our test data.


```python
X_test = X_test.join(
    pd.DataFrame(lb.transform(X_test['position']),
    columns=lb.classes_,
    index=X_test.index))
```


```python
X_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>team</th>
      <th>position</th>
      <th>stanley_cup</th>
      <th>American</th>
      <th>Canadian</th>
      <th>Finnish</th>
      <th>Russian</th>
      <th>C</th>
      <th>D</th>
      <th>G</th>
      <th>LW</th>
      <th>RW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>28</td>
      <td>Toronto Maple Leafs</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>26</td>
      <td>Winnipeg Jets</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>31</td>
      <td>Montreal Canadians</td>
      <td>G</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, even though none of the players in the testing data played wing, instantiating Label Binarizer and utilizing the method 'classes_' ensures that these columns were preserved in the testing data.

Now let's perform LabelBinarizer on the team column as well!


```python
lb = LabelBinarizer()

lb.fit(X_train['team'])
lb.classes_     
lb.transform(X_train['team'])

X_train = X_train.join(
    pd.DataFrame(lb.fit_transform(X_train['team']),
    columns=lb.classes_,
    index=X_train.index))

X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>team</th>
      <th>position</th>
      <th>stanley_cup</th>
      <th>American</th>
      <th>Canadian</th>
      <th>Finnish</th>
      <th>Russian</th>
      <th>C</th>
      <th>D</th>
      <th>G</th>
      <th>LW</th>
      <th>RW</th>
      <th>Colorado Avalanche</th>
      <th>Edmonton Oilers</th>
      <th>Pittsburgh Penguins</th>
      <th>San Jose Sharks</th>
      <th>Tampa Bay Lightning</th>
      <th>Toronto Maple Leafs</th>
      <th>Washington Capitals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>32</td>
      <td>Pittsburgh Penguins</td>
      <td>C</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>25</td>
      <td>Toronto Maple Leafs</td>
      <td>D</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22</td>
      <td>Colorado Avalanche</td>
      <td>RW</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>23</td>
      <td>Colorado Avalanche</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>Edmonton Oilers</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test = X_test.join(
    pd.DataFrame(lb.transform(X_test['team']),
    columns=lb.classes_,    #Remembering the classes created from the original categorical column.
    index=X_test.index))

X_test = X_test.drop(columns=['team', 'position'])
X_train = X_train.drop(columns=['team', 'position'])
```

Now let's see if we can model our data...


```python
model = LogisticRegression()

model.fit(X_train, y_train)
model.score(X_test, y_test)
```

    /anaconda3/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)





    0.6666666666666666



SUCCESS THEY'RE THE SAME SIZE!!!

LabelBinarizer is better than pd.get_dummies because when you fit your training data it gives you back a class_ method which remembers the different categories that were created when you hot encoded your column into multiple columns. When you transform your testing data and call the class attributes that were stored when you fit your training data, even though your test data may not have samples from that specific column, it remembers that it needs to represent the column even though there are no 1s present.


```python

```