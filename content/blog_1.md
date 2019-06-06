Title: The Functionality of Functions
Date: 2019-03-28 10:20
Tags: python
Slug: blog_1

*The Functionality of Functions*

Coming from an academic background based in History, Philosophy, and Law, learning Data Science can be  overwhelming; especially when you haven't taken a Computer or Math course in TWELVE years. The biggest issues I have had in the first three weeks deal with syntax. Learning Python is the equivalent of learning a new language. The one area I have continued to struggle with is how to develop new functions. While this is by far not the hardest task thrown at us, it is essential to most tasks a Data Scientist faces daily.

To start, writing a basic function is very simple.


```python
def print_my_name(first_name, last_name):
    return (last_name, first_name)
```


```python
print_my_name('kristi', 'gourlay')
```




    ('gourlay', 'kristi')



OR


```python
def adder(number1, number2):
    return number1 + number2
```


```python
adder(2, 3)
```




    5



They get slightly more difficult when you're asked to count something or create a list. 

Consider the following function that counts the vowels in a provided word.


```python
def vowel_counter(word):
    vowels = 'aeiou'
    counter = 0
    for char in word:
            if char in vowels:
                counter += 1
                return counter
            
```


```python
vowel_counter('elephant')
```




    1



WAIT! That's not right!

Lesson One: Make sure the return is outside the loop!
This placement of the return call had me pulling my hair out for the entire first week. I could not understand why my functions, that looked just like my classmates' functions, were not working. 


```python
def vowel_counter(word):
    vowels = 'aeiou'
    counter = 0
    for char in word:
            if char in vowels:
                counter += 1
    
    return counter
            
```


```python
vowel_counter('elephant')
```




    3



While writing a simple function is easy, the next step is learning to write a function that can be used on data or a dictionary. I struggled with this in our first lab.

Below I have created a dictionary of tv shows:


```python

tvshows = [
{
    'name': 'curb your enthusiasm',
    'category': 'comedy',
    'network': 'hbo'
},
{
    'name': 'the wire',
    'category': 'drama',
    'network': 'hbo'
},
{
    'name': 'shameless',
    'category': 'dramedy',
    'network': 'showtime'
},
{
    'name': 'the sopranos',
    'category': 'drama',
    'network': 'hbo'
},
{
    'name': 'game of thrones',
    'category': 'drama',
    'network': 'hbo'
},
{
    'name': 'house of cards',
    'category': 'drama',
    'network': 'netflix'
},
{
    'name': 'kimmy schmidt',
    'category': 'comedy',
    'network': 'netflix'
}]
    
print(tvshows)
```

    [{'name': 'curb your enthusiasm', 'category': 'comedy', 'network': 'hbo'}, {'name': 'the wire', 'category': 'drama', 'network': 'hbo'}, {'name': 'shameless', 'category': 'dramedy', 'network': 'showtime'}, {'name': 'the sopranos', 'category': 'drama', 'network': 'hbo'}, {'name': 'game of thrones', 'category': 'drama', 'network': 'hbo'}, {'name': 'house of cards', 'category': 'drama', 'network': 'netflix'}, {'name': 'kimmy schmidt', 'category': 'comedy', 'network': 'netflix'}]


For example, I want to make a function to iterate through the dictionary and return all the shows that are from a specific network.

The first step is to create a 'for loop' OR use a 'list comprehension' to write what you want to get back. 

In this instance, I would like to see all the tv show names that are on the HBO network.

Name a function and pass it two arguments. 
1)what you are going to iterate through.
2)what you are looking for.


```python
def tv_net(dictionary='tvshows', network='hbo'):
    
    return [tv['name'] for tv in dictionary if tv['network'] == network]

tv_net(tvshows)
```




    ['curb your enthusiasm', 'the wire', 'the sopranos', 'game of thrones']



Since I set the second argument default to hbo, the function automatically looks for shows from the HBO network. Setting a default argument makes life easier, and helps you remember what type of argument you will be looking for. If I wanted to change the network argument, it's as simple as passing a different network through.


```python
tv_net(tvshows, network='netflix')
```




    ['house of cards', 'kimmy schmidt']




```python
tv_net(tvshows, network='showtime')
```




    ['shameless']



The important thing I learned from this process was the fact that you can then take the function you made and use it on similar data later in your code. For example, say I was then provided with a dictionary of prime time tv shows.


```python
primetime = [
{
    'name': 'this is us',
    'category': 'drama',
    'network': 'nbc'
},
{
    'name': 'family guy',
    'category': 'comedy',
    'network': 'fox'
},
{
    'name': 'the good place',
    'category': 'comedy',
    'network': 'nbc'
},
{
    'name': 'the office',
    'category': 'comedy',
    'network': 'nbc'
}
]
```


```python
tv_net(primetime, network='nbc')
```




    ['this is us', 'the good place', 'the office']




```python
tv_net(primetime, 'fox')
```




    ['family guy']



This importance of using functions to iterate through data and dictionaries resurfaces in EDA. If we are looking at a set of data and need to change something about it, and we may need to make similar changes later in our code, it's practical to create a function.
To show this, I will use the tvshow sets from above. (First I will need to place them in a Data Frame)


```python
import pandas as pd

tvshows = pd.DataFrame(tvshows)
primetime = pd.DataFrame(primetime)
```


```python
tvshows
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
      <th>category</th>
      <th>name</th>
      <th>network</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>comedy</td>
      <td>curb your enthusiasm</td>
      <td>hbo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>drama</td>
      <td>the wire</td>
      <td>hbo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dramedy</td>
      <td>shameless</td>
      <td>showtime</td>
    </tr>
    <tr>
      <th>3</th>
      <td>drama</td>
      <td>the sopranos</td>
      <td>hbo</td>
    </tr>
    <tr>
      <th>4</th>
      <td>drama</td>
      <td>game of thrones</td>
      <td>hbo</td>
    </tr>
    <tr>
      <th>5</th>
      <td>drama</td>
      <td>house of cards</td>
      <td>netflix</td>
    </tr>
    <tr>
      <th>6</th>
      <td>comedy</td>
      <td>kimmy schmidt</td>
      <td>netflix</td>
    </tr>
  </tbody>
</table>
</div>



When we look at the dataframe for tvshows, we notice that the category and the name columns need to be swapped. We could run a simple list command, but since we know we have other similar data, it may be easier to create a function so that we don't have to repeat these verbose commands later.


```python
def swap_col(dataframe):
    cols = list(dataframe.columns)
    a, b = cols.index('category'), cols.index('name')
    cols[b], cols[a] = cols[a], cols[b]
    dataframe = dataframe[cols]
    return dataframe
```


```python
swap_col(tvshows)
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
      <th>category</th>
      <th>network</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>curb your enthusiasm</td>
      <td>comedy</td>
      <td>hbo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>the wire</td>
      <td>drama</td>
      <td>hbo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>shameless</td>
      <td>dramedy</td>
      <td>showtime</td>
    </tr>
    <tr>
      <th>3</th>
      <td>the sopranos</td>
      <td>drama</td>
      <td>hbo</td>
    </tr>
    <tr>
      <th>4</th>
      <td>game of thrones</td>
      <td>drama</td>
      <td>hbo</td>
    </tr>
    <tr>
      <th>5</th>
      <td>house of cards</td>
      <td>drama</td>
      <td>netflix</td>
    </tr>
    <tr>
      <th>6</th>
      <td>kimmy schmidt</td>
      <td>comedy</td>
      <td>netflix</td>
    </tr>
  </tbody>
</table>
</div>





Now we import our other dataframe and we see the same item that needs to be fixed. 





```python
primetime
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
      <th>category</th>
      <th>name</th>
      <th>network</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>drama</td>
      <td>this is us</td>
      <td>nbc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>comedy</td>
      <td>family guy</td>
      <td>fox</td>
    </tr>
    <tr>
      <th>2</th>
      <td>comedy</td>
      <td>the good place</td>
      <td>nbc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>comedy</td>
      <td>the office</td>
      <td>nbc</td>
    </tr>
  </tbody>
</table>
</div>




```python
swap_col(primetime)
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
      <th>category</th>
      <th>network</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>this is us</td>
      <td>drama</td>
      <td>nbc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>family guy</td>
      <td>comedy</td>
      <td>fox</td>
    </tr>
    <tr>
      <th>2</th>
      <td>the good place</td>
      <td>comedy</td>
      <td>nbc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>the office</td>
      <td>comedy</td>
      <td>nbc</td>
    </tr>
  </tbody>
</table>
</div>



The ability to make a function to fix mistakes in your data becomes a useful skill to have. It will be a time saver, and that's one of the reasons why learning functions is quite important for Data Science. 

One last point.

I've shown how functions become important when both analyzing and fixing your data. Sometimes you might need your function to include multiple steps and the sheer volume may seem daunting. The number one tip, I have taken away in my first three weeks learning Python is to break things down. If you take everything step by step and work your way up to the return you need, the task becomes much easier.

It's easier to do many smaller pieces of the whole, than to tackle the whole.

STEP ONE: DONT GET OVERWHELMED!

STEP TWO: SIMPLY WRITE DOWN WHAT THE FUNCTION NEEDS TO DO/RETURN

STEP THREE: WRITE DOWN IN ORDER WHAT YOU NEED TO DO TO GET THE DESIRED RETURN


The perfect example of this breaking down a function was presented to me in our first project. We were asked to analyze SAT and ACT data. At one point in the assignment, we were asked to design a function that returns the standard deviation for a column. This question took me 10x longer than any other question, because I kept thinking about the final product. Eventually, I realised I needed to follow the three steps laid out above.

After calming myself down, I answered STEP TWO. The function needs to return the standard deviation of a column in the SAT/ACT dataset. 

STEP THREE: How do you calculate standard deviation?


1. Calculate the mean

    mean = sum(column) / len(column)
    
    
2. For each number, subtract the mean and square the results

        ((n - (sum(column) / len(column)))**2)
        

3. Find the mean of those squared differences( START A LIST before the math. new_list = [] )

    new_list.append(((n - (sum(column) / len(column)))**2))
        

4. Take the square root of that list

    math.sqrt(((sum(new_list))) / (len(column) - 1))


Start with the first thing you need to do and slowly add what you need next. The result is:   


```python
import math

def calc_std(column):
    new_list = []
    for n in column:
        new_list.append((n - (sum(column) / len(column)))**2)
    return math.sqrt(((sum(new_list))) / (len(column) - 1))
```


```python
column = [4, 5, 8, 9, 3, 4, 7, 9]
```


```python
calc_std(column)
```




    2.416461403433896



Writing functions was something that I alone seemed to struggle with in my cohort. However, my focus on improving my ability to write functions, allowed me to highlight other areas in my work that I should keep an eye on. If you're code is not working, read the return for details, if you still cannot figure out what's wrong, double check these few things:
* Placement
* Accuracy
* Brackets!

Placement: Make sure your return is located outside the loop. But also make sure that lists and counters are located in the right place as well. If needed for the function, place the lists right below the function call.

Accuracy: The amount of times my code was not working because I mis-spelt a word in one place. Spelling will always be important, not just in academia!

Brackets: If in doubt look if you're missing a bracket somewhere. I spent an hour with a function not working, to realize I just needed to square bracket the argument I was submitting.

Attention to detail!


```python

```