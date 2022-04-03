"""
@author: Erik Andres Escayola based on code from Christian Brownlees (http://www.econ.upf.edu/~cbrownlees/)

@title: Partial Correlation Network

@description:
  1. Import time-series data
  2. Preliminary analysis through plots and desriptive stats
  3. Fit partial correlation network
  4. Compute statistics for network
"""


## Step 1 - Data loading and prep
## --------------------------------------------------------------------------- 

# install packages 
# pip install python-igraph 
# pip install cairocffi 
# pip install cffi
# pip install pycairo

# import modules 
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import igraph as ig
from sklearn.linear_model import LinearRegression as linreg
from sklearn.covariance import GraphicalLasso as glasso
from sklearn.covariance import GraphicalLassoCV as glasso_cv

# load index for local press
xlspath = "C:/Users/eandr/Documents/1_PROJECTS/A_PAPERS/ST_LATAM/Indices_tensions.xlsx"
epu_data = pd.read_excel(xlspath, sheet_name="Import_epu", index_col=1, header=1) 
epu_data = epu_data.iloc[: , 1:]
epu_data_agg = epu_data.iloc[: , -1]
epu_data = epu_data.iloc[: , :-1]

# number of countries
N = len(epu_data.columns)
T = len(epu_data.index)

# color dictionary
country_Long = ["Argentina", "Brazil", "Chile", "Colombia", "Mexico", "Peru"]
colours = ["k", "g", "r", "y", "b", "m"]
palettes = ["RdGy", "YlGn",  "OrRd", "YlOrBr", "PuBu", "PiYG"]
color_dict = {"AG": "k", 
              "BR": "g", 
              "CL": "r", 
              "CO": "y", 
              "MX": "b",
              "PE": "m"}

# log difference data
epu_log_data = np.log(epu_data)
epu_diff_data = epu_log_data.diff()
epu_diff_data = epu_diff_data.iloc[1:,:]
epu_log_data_agg = np.log(epu_data_agg)
epu_diff_data_agg = epu_log_data_agg.diff()
epu_diff_data_agg = epu_diff_data_agg.iloc[1:]

# plot all 
epu_data.plot(figsize=(18, 6),legend=False)
plt.show() # data looks stationary ?
epu_diff_data.plot(figsize=(18, 6),legend=False)
plt.show() 

# plot specific countries
epu_diff_data[ 'epuCO' ].plot(figsize=(18, 6))
plt.show()


## Step 2 - Descriptive stats
## ---------------------------------------------------------------------------

# descriptive statistics
epu_data.describe(include='all') 

# plot the heatmap of the correlation matrix 
plt.figure(figsize=(15,12))
sns.heatmap(epu_data.corr(),xticklabels=epu_data.columns,yticklabels=epu_data.columns,cmap='terrain',linecolor='white', linewidths=1, annot=True) # the partial correlation Network is characterized by the inverse of the covariance matrix, so interesting to look at correlation matrix
plt.show() 
plt.figure(figsize=(15,12))
sns.heatmap(epu_diff_data.corr(),xticklabels=epu_diff_data.columns,yticklabels=epu_diff_data.columns,cmap='terrain',linecolor='white', linewidths=1, annot=True) 
plt.show() 


# ## Step 3 - Filter out factors from the panel 
# ## ---------------------------------------------------------------------------

# create the panel of idiosyncratic indices
epu_data_idio = pd.DataFrame(columns=epu_data.columns,index=epu_data.index) # empty dataframe <- regressing each of the tickers to the factor and save the residuals here
epu_diff_data_idio = pd.DataFrame(columns=epu_diff_data.columns,index=epu_diff_data.index)

common_factor_epu = epu_data_agg #epu_data.mean(axis=1) # market factor as mean of all countries
common_factor_epu_diff = epu_diff_data_agg 

# change level vs differenced version
data = epu_diff_data
factor = common_factor_epu_diff
idio = epu_diff_data_idio

betas = np.zeros( (N) ) 
x = factor.values
x = x.reshape(len(x),1)

for i in range(N):
    y = data[ data.columns[i] ].values
    y = y.reshape(len(y), 1)
    reg = linreg().fit( x , y  )
    idio[ idio.columns[i] ] =  y - reg.predict(x)
    betas[i] = abs(reg.coef_[0][0])

data.head(10) # show how matrix changes after filter out influence of common factors
idio.head(10) 

# plot the factor loadings
plt.figure(figsize=(15,12))
barlist=plt.bar( range(N) , betas )
for i in range(N):
    barlist[i].set_color( colours[ i ] ) 
plt.show()

# heatmap of the idiosyncratic correlation 
plt.figure(figsize=(15,12))
sns.heatmap(idio.corr(), xticklabels=idio.columns,yticklabels=idio.columns,cmap='terrain',linecolor='white', linewidths=1, annot=True)
plt.show() # less dependence? influence of common factor?


## Step 4 - Estimate the network 
## ---------------------------------------------------------------------------

# use the igraph package to plot the network
# g = ig.Graph.Barabasi(100,1)
# ig.plot(g) # power law distribution based

# use the GraphicalLasso routine from the sklearn package
# the function requires as imputs alpha=the lasso tuning parameter and the data
net = glasso(alpha=0.3, max_iter=1000).fit( idio ) # first fixed tuning parameter

# also use the following line to tune automatically the shrinkage parameters
net_opt = glasso_cv(max_iter=1000).fit( idio ) # choose optimal shrinkage


# compute a number of outputs associatd with the Glasso 
# get the precision matrix
K = net.precision_

# construct the adjencency matrix
A = np.zeros( (N, N) )
for i in range(N):
    for j in range(i-1):
        if abs(K[i,j] > 0 ):
            A[i,j] = 1
            A[j,i] = 1

# construct partial correlations            
PC = np.zeros( (N, N) )
for i in range(N):
    for j in range(i-1):
        if abs(K[i,j]) > 0 :
            PC[i,j] = -K[i,j]/sqrt( K[i,i] * K[j,j] )

# compute the degree
degree = A.sum(axis = 0) 
print('Number of connections %d '% (sum(degree)/2) )
print("Number of connections %2.2f%%" % ( 100*(sum(degree)/2) / (N*(N-1)/2) ) ) # sparse Network  based on low % of number of connections (out of all possible ones)

# create a graph object
g = ig.Graph.Adjacency( (A>0).tolist() , mode=ig.ADJ_UNDIRECTED )
ig.plot(g) # need to work on how the graph is plotted to make it to have sense


# igraph options appropriately to get a cute plot
# set label, size and color of the vertices
g.vs['label'] = idio.columns
g.vs['label'] = idio.columns
g.vs['size']  = degree*10
#g.vs["color"] = [ color_dict[ sp100_info['GICS.Sector'][ sp100_info.index[2+i] ] ] for i in range(2,N+1) ]

# delete vertices with no link
vertices_to_be_deleted = [v.index for v in g.vs if v['size']==0]
g.delete_vertices(vertices_to_be_deleted)
ig.plot(g) # much more insightfull plot (notice the clustering between colors): partial correlation in graph vs correlation in heatmap

# degree histogram
degree_df = pd.DataFrame({ 'degree': degree } )
degree_df.hist(bins=10,figsize=(15,12)) # degree distribution shows that there indeed this power distribution

# partial correlation histogram
pc_vec = PC.flatten()
pc_vec = pc_vec[ abs(pc_vec) > 0 ]
parcorr_df = pd.DataFrame({ 'parcorr': pc_vec } )
parcorr_df.hist(bins=20,figsize=(15,12)) # vast majority of the linkages should be interpreted as positive/negative partial correlation?

# plot the sparsity pattern recovered by lasso
plt.figure(figsize=(12,12))
sns.heatmap(A,xticklabels=idio.columns,yticklabels=idio.columns,cmap='binary')
plt.title('sparsity pattern recovered by GLASSO')
plt.show() # clear sign of sparsity (white areas) and indeed estimated as exactly zero


## Step 5 - Check statistics in network
## ---------------------------------------------------------------------------

# compute ranking according to the eigenvector centrality (page rank algorithm)
ev_cent = ig.Graph.evcent(g)
ticker_ranking = pd.DataFrame( columns = ['Name' , 'Ranking'] , index=range(len(g.vs['label'])) )
ticker_ranking['Name'] = g.vs['label']
ticker_ranking['Ranking'] = ev_cent
ticker_ranking.sort_values("Ranking", inplace = True, ascending=False) 
ticker_ranking.head(N) # who is most central in the Network (similar results if just looking at the degree of each unit and compare it)


# sparsity as a function of the tuning parameter
lambdas  = np.arange(0.01,1,0.05)
sparsity = np.zeros( (len(lambdas)) )

for l in range( len(lambdas) ):
    print('.')    
    K = glasso(alpha=lambdas[l]).fit( idio ).precision_
    A = np.zeros( (N, N) )
    for i in range(N):
        for j in range(i-1):
            if abs(K[i,j] > 0 ):
                A[i,j] = 1
                A[j,i] = 1
                
    degree = A.sum(axis = 0) 
    sparsity[l] = ( 100*(sum(degree)/2) / (N*(N-1)/2) )
    
print('done!')

# default tuning parameter vs optimal 
K = glasso(alpha=0.3).fit( idio ).precision_
A = np.zeros( (N, N) )
for i in range(N):
    for j in range(i-1):
        if abs(K[i,j] > 0 ):
            A[i,j] = 1
            A[j,i] = 1
                
degree = A.sum(axis = 0) 
sparsity_example = ( 100*(sum(degree)/2) / (N*(N-1)/2) )

K = glasso(alpha=net_opt.alpha_).fit( idio ).precision_
A = np.zeros( (N, N) )
for i in range(N):
    for j in range(i-1):
        if abs(K[i,j] > 0 ):
            A[i,j] = 1
            A[j,i] = 1
                
degree = A.sum(axis = 0) 
sparsity_opt = ( 100*(sum(degree)/2) / (N*(N-1)/2) )

# plots
fig = plt.figure(figsize=(12,12))
ax = plt.axes()
ax.plot(lambdas,sparsity);
plt.scatter( (0.3,net_opt.alpha_), (sparsity_example,sparsity_opt),s=500)
plt.title('sparsity pattern as a function of shrinkage parameter of GLASSO')
plt.show() # y-axis number of edges as % of total number of edges as function of tuning parameter (x-axis)