import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
match=pd.read_csv('matches.csv')
delivery=pd.read_csv('deliveries.csv')
match.head()#check what datas does this dataframe have
#it is giving the output that it has 756 rows and 18 columns
delivery.head()
#but we have many columns that are not needed for this project

'''Columns required
the batting_team and bowling_team in second innings
city
runs_left
balls_left
wickets_left
total_runs_x (runs required)
crr (current run rate)
rrr (required run rate)
result 0 or 1 (the team will win or not)
'''
#Note:this is a classification problem but we need to show the percentage chance of winning for both the teams
#so we use only those type of classification algorithms that uses the feauture of probability along with the classification like for example logistic regression

'''Dataset cleaning'''
#calculating the total runs of both the teams using group by
total_score_df = (
    delivery
    .groupby(['match_id','inning'])['total_runs']
    .sum()
    .reset_index()
)
#gives the total runs for each match (for each match_id) and each of its innings
#each match will have 2 innings
#now we need to separate the first innings alone
#we calculate the percentage based on the current runs in the second innings by comparing it with the first innings
total_score_df=total_score_df[total_score_df['inning']==1] #the number of rows becomes half : To filter rows in pandas, the condition must be placed inside the DataFrame brackets.
#total_score_df = total_score_df['inning'] == 1 : This does NOT return a DataFrame.It returns a boolean Series like: True False True False True...
#merging match and total_score_df into a new data frame match_df
#total_score_df[['match_id','total_runs']] : This selects only two columns from total_score_df
#merge() is used to join two dataframes similar to sql join
#here id column comes from match dataframe, and match_id comes from total_score_df, these are the keys for joining
#we can also add another attribute how : match.merge( total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id',how='inner')
#how='inner' means keep only the rows where the join key exists in BOTH DataFrames
match_df=match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')
#now match will have two more columns "match_id" and "total_runs"
#now we have the current runs of the second innings of each match
#so many teams are not playing currently so we can remove their names from the dataframe
#also some teams have two names so we can replace that with same name
#'Delhi Daredevils' and 'delhi Capitals' are same, also 'Deccan Chargers' and 'Sunrises Hyderabad' are same
#create a list of teams that are currently playing
teams=['Sunrises Hyderabad',
       'Mumbai Indians',
       'Royal Challengers Bangalore',
       'Kolkata Knight Riders',
       'Kings XI Punjab',
       'Chennai Super Kings',
       'Rajasthan Royals',
       'Delhi Capitals']
#we have team1 and team2 names in match_df
#replace the same team with new name
match_df['team1']=match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2']=match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team1']=match_df['team1'].str.replace('Deccan Chargers','Sunrises Hyderabad')
match_df['team2']=match_df['team2'].str.replace('Deccan Chargers','Sunrises Hyderabad')
#keep only the team names which are there in the list given above : 'teams'
match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]
#we have a column named 'dl_applied' this means if Duckworth-Lewis method was applied in the match
#it is a method used when match is interrupted usually due to rain or bad weather
#this method recalculates the target score, adjusts overs and wickets, ensures a fair result despite interruptions
#DLS(Duckworth–Lewis–Stern) matches behave very differently, targets are revised so models can get confused so mostly we remove it
#lets check how many dls are applied just to know
match_df['dl_applied'].value_counts()#gives where it is applied and where it is not: count of 0s and 1s
match_df=match_df[match_df['dl_applied']==0] #take only those columns where dl is not applied
#now we need to merge match_df with delivery
#we take only those columns which we need from match_df and merge it with delivery
match_df=match_df[['match_id','city','winner','total_runs']] #Note:here total runs is the runs of the first innings
delivery_df=match_df.merge(delivery,on='match_id') #merges based on the common column match_id
#now we calculated to total runs of the first innings and separated it from the dataframe in total_score_df
#so we have the value of total_runs in first innings now
#so now we can filter out the second innings score in delivery_df
#delivery_df contains number of runs for each ball in each inning and total runs mentioned again and again
#now remove first innings from it
delivery_df=delivery_df[delivery_df['inning']==2]
#now we need to calculate "runs_left","balls_left","wickets_left","rrr","crr","result"
#after each ball we need to calculate the runs_left to win and balls_left
#we have runs after each ball calculate the total runs after each ball...like after each ball what is the total runs now
#'total_runs_y' this is the column that specifies the number of runs taken for each ball...not the sum but the number...so to find the sum we use groupby('match_id') this separates the data match-wise.
delivery_df['current_score'] = (
    delivery_df.groupby('match_id')['total_runs_y'].cumsum()
)
#now in each ball subtract the current_score from the total required score this will give us the runs left to win after each ball
#so whichever ball the user gives as input we can find the total number of runs needed to win
delivery_df['runs_left']=delivery_df['total_runs_x']-delivery_df['current_score'] #new columns runs_left
#note: What happens when you merge two DataFrames with same column names?Pandas keeps both columns and automatically renames them by adding suffixes like _x for left,_y for right thats why it is total_runs_x
#the dataset contains over number and ball of each over
#  (over - 1) converts 1-based over numbering to completed overs,
# multiplying by 6 converts overs to balls,
# adding ball gives total balls bowled so far,
# subtracting from 120 gives balls left in the innings : 120-(delivery_df['over'-1]*6+delivery_df['ball'])
#but here we are not taking over-1 so we subtract from 126
#store the result in a new column balls_left
delivery_df['balls_left']=126-(delivery_df['over']*6+delivery_df['ball'])
#now we need to get wickets_left column but we do not have any data for that
#the only data we have the names of the player who out if a person is not out then we have NaN value in a columns player_dismissed
#if in any over's ball a player is out then we have the name of the player in that row
delivery_df['player_dismissed']=delivery_df['player_dismissed'].fillna("0") #wherever there is NaN value replace it with string 0
delivery_df['player_dismissed']=delivery_df['player_dismissed'].apply(lambda x: x if x=="0" else "1") #gives value of string 1 for the player names which means that 1 wicket is gone for that particular ball
#the apply() runs a function one element at a time and returns a transformed Series or DataFrame.
delivery_df['player_dismissed']=delivery_df['player_dismissed'].astype(int) #astype(int) converts values like True/False, 0/1, or numeric strings into integers (0 or 1, or whole numbers).
wickets = (
    delivery_df
    .groupby('match_id')['player_dismissed']
    .cumsum()
    .values
)
#It calculates the cumulative number of wickets fallen ball-by-ball for each match.
delivery_df['wickets']=10-wickets #It calculates wickets remaining at each ball.
# deliver_df.tail() It returns the last n rows (default = 5).
#crr=runs/overs
delivery_df['crr']=(delivery_df['current_score']*6)/(120-delivery_df['balls_left'])
'''Why this formula works
current_score → total runs scored so far
120 - balls_left → balls already bowled
* 6 → convert from runs-per-ball to runs-per-over'''
delivery_df['rrr']=(delivery_df['runs_left']*6)/delivery_df['balls_left']
#now result 0 or 1 we have batting team of second innings and winner
#if winner and battingg team are same then 1 else 0
#lets create a func
def result(row):
    return 1 if row['batting_team']==row['winner'] else 0
delivery_df['result']=delivery_df.apply(result,axis=1) #axis=1 means in each row
#take only the required columns remove others
final_df=delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]
final_df=final_df.sample(final_df.shape[0])
#final_df.shape[0] : Returns the number of rows in the DataFrame.
#final_df.sample(n=...) Randomly selects n rows from the DataFrame. By default, it selects without replacement (each row appears only once).
#together it randomly reorders all rows in the dataframe this is done to avoid confusion because same name teams are together 
#this line effectively shuffles the rows of the dataframe
#the finals_df contains some NaN values we got to know that after fitting the data to pipe
#final_df.isnull.sum() : gives the total number of NaN values in each column
final_df.dropna(inplace=True) #dropna removes rows that contain missing values NaN
#if an column has NaN then the entire row is deleted
#final df rrr contains some +inf and -inf values
#this happens when balls_left=0 and we divide by 0 
#so we remove rows with balls_left=0
final_df=final_df[final_df['balls_left']!=0]


'''Train and Test split'''
#iloc stands for Integer Location, it uses 0-based indexing
#it is used to select rows and columns by their numerical index positions, not by names
#synatx: df.iloc[row_selection, column_selection]
x=final_df.iloc[:,:-1] # : all rows, :-1 from first column to second last column
y=final_df.iloc[:,-1] # : all rows, -1 only the last column
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#random_state=1 ensures same split every time
#the batting team, bowling team and city are strings
#so we need to use one hote encoder and column tranformer
trf=ColumnTransformer([('trf',OneHotEncoder(sparse_output=False,drop='first'),['batting_team','bowling_team','city'])],remainder='passthrough') 
#This line creates a preprocessing transformer in scikit-learn to handle categorical + numerical features together.
#It one-hot encodes the categorical columns (batting_team, bowling_team, city) and keeps all other columns unchanged.
#ColumnTransformer:used when some columns need preprocessing(categorical) and others should stay as they are(numerical)
#'trf': Name of the transformer (any name is fine)
#remainder='passthrough': All columns not listed in the transformer are kept as-is.
#drop='first' avoids the dummy variable trap, and sparse=False returns a dense NumPy array instead of a sparse matrix.
#creating pipeline
pipe=Pipeline(steps=[('step1',trf),('step2',LogisticRegression(solver='liblinear'))])
pipe.fit(x_train,y_train)
#A Pipeline is a way to chain steps together so that data flows step-by-step automatically.
#like an assembly line Raw Data → Encoding → Model → Prediction
#A Pipeline chains preprocessing and model steps so that data is transformed and trained consistently without data leakage.
#('step1', trf): This step prepares the data
#step 2 is the ml model the liblinaer Works well for small & medium datasets and is Good for binary classification
#liblinear is the algorithm (solver) used to train Logistic Regression.
#A solver is just the method used to find the best weights (how important a feature is) for the model.
'''pipe.fit(x_train, y_train)
Internally, scikit-learn does this:

step1.fit_transform(x_train)
Learns categories (teams, cities)
Converts categorical → numeric

2️step2.fit(transformed_x, y_train)
Learns weights
Builds the Logistic Regression model
'''
y_pred=pipe.predict(x_test) #It uses the trained pipeline to predict the output (win/lose) for the test data.
accuracy_score(y_test,y_pred)
#we can also use other like random forest regression it more accuracy also but to get better probabilities logistic regression is better


'''Calculating probability'''
def match_progression(x_df,match_id,pipe):  
    match=x_df[x_df['match_id'] == match_id] #Select ONE match
    match=match[(match['ball'] == 6)] #Take end of each over only
    temp_df=match[['batting_team', 'bowling_tean', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x','crr','rrr','result']] #Pick required columns
    temp_df=temp_df[temp_df['balls_left'] != 0] #Remove last over (balls_left = 0)
    result=pipe.predict_proba(temp_df) #Predict win/lose probability
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1) #Store probabilities in %
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1) #Create over numbers
    target=temp_df['total_runs_x'].values[0]
    runs=list(temp_df['runs_left'].values)
    new_runs=runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs) [:-1]-np.array(new_runs) #Calculate runs scored per over
    wickets=list(temp_df['wickets'].values)
    new_wickets= wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w=np.array(wickets)
    nw= np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw-w) [0:temp_df.shape[0]] #Calculate wickets fallen per over
    print("Target-", target)
    temp_df= temp_df[['end_of_over', 'runs_after_over', 'wickets_in_over', 'lose', 'win']] #Final clean table
    return temp_df, target

import pickle
pickle.dump(pipe, open('pipe.pkl', 'wb'))
