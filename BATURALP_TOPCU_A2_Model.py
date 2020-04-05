#timeit

# Student Name : BATURALP TOPCU
# Cohort       :MSBA-2

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

# importing libraries
import random as rand # random number generation
import pandas as pd # data science essentials
import numpy as np # numpy array
import seaborn as sns # enhanced data visualization
import matplotlib.pyplot as plt # essential graphical ou
from sklearn.model_selection import train_test_split # train/test split
import statsmodels.formula.api as smf # linear regression (statsmodels)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler     # standard scaler

################################################################################
# Load Data
################################################################################

# use this space to load the original dataset
# MAKE SURE TO SAVE THE ORIGINAL FILE AS original_df
# Example: original_df = pd.read_excel('Apprentice Chef Dataset.xlsx')

original_df ='Apprentice_Chef_Dataset.xlsx'


# reading the file into Python
chef = pd.read_excel(original_df)

# Copy the chef dataset
chef_c = pd.DataFrame.copy(chef)

################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

#Missing Value Flagging
for col in chef_c:

    if chef_c[col].isnull().astype(int).sum() > 0:
         chef_c['m_'+col] = chef_c[col].isnull().astype(int)
            
# string_split
#########################
def string_split(col, df, sep=' ', new_col_name='NUMBER_OF_NAMES'):
    
    df[new_col_name] = 0  
    for index, val in df.iterrows():
        df.loc[index, new_col_name] = len(df.loc[index, col].split(sep =' '))

# calling text_split_feature
string_split(col='NAME',df=chef_c)

#Splitting personal emails

# placeholder list
placeholder_lst = []

# looping over each email address
for index, col in chef_c.iterrows():
    
    # splitting email domain at '@'
    split_email = chef_c.loc[index, 'EMAIL'].split(sep = '@')
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)
    

# converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)


# displaying the results
email_df

# concatenating with original DataFrame

# renaming column to concatenate
email_df.columns = ['0' , 'email_domain']


# concatenating personal_email_domain with friends DataFrame
chef_c = pd.concat([chef_c, email_df['email_domain']],
                     axis = 1)




# email domain types
professional_d = ['@mmm.com','@amex.com','@apple.com','@boeing.com','@caterpillar.com','@chevron.com','@cisco.com','@cocacola.com','@disney.com','@dupont.com','@exxon.com','@ge.org.com','@goldmansacs.com','@homedepot.com','@ibm.com','@intel.com','@jnj.com','@jpmorgan.com','@mcdonalds.com','@merck.com','@microsoft.com','@nike.com','@pfizer.com','@pg.com','@travelers.com','@unitedtech.com','@unitedhealth.com','@verizon.com','@visa.com','@walmart.com']
personal_d  = ['@gmail.com','@yahoo.com','@protonmail.com']
junk_d = ['@me.com','@aol.com','@hotmail.com','@live.com','@msn.com','@passport.com']


# placeholder list
placeholder_lst = []


# looping to group observations by domain type
for domain in chef_c['email_domain']:
    
    if '@' + domain in professional_d:
        placeholder_lst.append('PROFESSIONAL_EMAIL')
        

    elif '@' + domain in personal_d:
        placeholder_lst.append('PERSONAL_EMAIL')

    elif '@' + domain in junk_d:
        placeholder_lst.append('JUNK_EMAIL')
        
    else:
        placeholder_lst.append('UNKNOWN_EMAIL')
        
# concatenating with original DataFrame
chef_c['domain_group'] = pd.Series(placeholder_lst)

# checking results
chef_c['domain_group'].value_counts()

# one hot encoding variables
one_hot_email       = pd.get_dummies(chef_c['domain_group'])

# joining codings together
chef_c = chef_c.join([one_hot_email])

# saving results
chef_c.to_excel('chef_feature_rich.xlsx',
                 index = False)
# loading saved file
chef_c = pd.read_excel('chef_feature_rich.xlsx')

# dropping categorical variables after they've been encoded
chef_c=chef_c.drop(labels=[ 'NAME', 'EMAIL', 'FIRST_NAME',
               'FAMILY_NAME','email_domain','domain_group'],
                     axis=1)


# Outlier Analysis
lst=['REVENUE', 'TOTAL_MEALS_ORDERED',
       'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE',
       'PRODUCT_CATEGORIES_VIEWED', 'AVG_TIME_PER_SITE_VISIT', 'MOBILE_NUMBER',
       'CANCELLATIONS_BEFORE_NOON', 'CANCELLATIONS_AFTER_NOON',
       'TASTES_AND_PREFERENCES', 'MOBILE_LOGINS', 'PC_LOGINS', 'WEEKLY_PLAN',
       'EARLY_DELIVERIES', 'LATE_DELIVERIES', 'PACKAGE_LOCKER',
       'REFRIGERATED_LOCKER', 'FOLLOWED_RECOMMENDATIONS_PCT',
       'AVG_PREP_VID_TIME', 'LARGEST_ORDER_SIZE', 'MASTER_CLASSES_ATTENDED',
       'MEDIAN_MEAL_RATING', 'AVG_CLICKS_PER_VISIT', 'TOTAL_PHOTOS_VIEWED',
       'm_FAMILY_NAME', 'NUMBER_OF_NAMES', 'JUNK_EMAIL', 'PERSONAL_EMAIL',
       'PROFESSIONAL_EMAIL', 'UNKNOWN_EMAIL']


'''#Ploting displots to analyze outliers
for c in lst:
            fig, ax = plt.subplots(figsize = (6, 4))
            sns.distplot(chef_c[c],
                         bins  = 'fd',
                         kde  = True,
                         rug  = True,
                         color = 'r')
            plt.xlabel(c)
#BOX PLOT LOOP
#for c in lst:
#            fig, ax = plt.subplots(figsize = (6, 4))
#            sns.boxplot(chef_c[c])
#            plt.xlabel(c) '''       



# Setting outlier thresholds according to box plots and displots

total_meal_hi         = 180
unique_meals_hi       = 7
contacts_w_hi         = 10
product_cat_lo        = 1
product_cat_hi        = 10
avg_time_per_site_hi  = 190
cancel_before_hi      = 5
cancel_after_hi       = 1
mobile_log_lo         = 5
mobile_log_hi         = 6
pc_log_hi             = 3
weekly_plan_hi        = 13
early_deli_hi         = 7
late_deli_hi          = 7
followed_pct_lo       = 0
followed_pct_hi       = 90
avg_prep_vid_lo       = 100
avg_prep_vid_hi       = 250
largest_order_lo      = 3
largest_order_hi      = 5
master_class_lo       = 0
master_class_hi       = 2
median_meal_lo        = 1
median_meal_hi        = 4
avg_clicks_lo         = 8
avg_clicks_hi         = 19
total_view_hi         = 400
number_of_names_hi    = 3

revenue_lo            = 130
revenue_hi            = 2400

## Feature Engineering (outlier thresholds)                                 ##
##############################################################################

# developing features (columns) for outliers

# TOTAL MEALS ORDERED
chef_c['out_total_meal'] = 0
condition_hi = chef_c.loc[0:,'out_total_meal'][chef_c['TOTAL_MEALS_ORDERED'] > total_meal_hi]

chef_c['out_total_meal'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)


# UNIQUE MEALS PURCHASED
chef_c['out_unique_meals'] = 0
condition_hi = chef_c.loc[0:,'out_unique_meals'][chef_c['UNIQUE_MEALS_PURCH'] > unique_meals_hi]

chef_c['out_unique_meals'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

# CONTACTS WITH CUSTOMER SERVICE 
chef_c['out_contacts_w'] = 0
condition_hi = chef_c.loc[0:,'out_contacts_w'][chef_c['CONTACTS_W_CUSTOMER_SERVICE'] > contacts_w_hi]

chef_c['out_contacts_w'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)



# PRODUCT CATEGORIES VIEWED
chef_c['out_product_cat'] = 0
condition_hi = chef_c.loc[0:,'out_product_cat'][chef_c['PRODUCT_CATEGORIES_VIEWED'] > product_cat_hi]
condition_lo = chef_c.loc[0:,'out_product_cat'][chef_c['PRODUCT_CATEGORIES_VIEWED'] < product_cat_lo]

chef_c['out_product_cat'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chef_c['out_product_cat'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)


# AVERAGE TIME PER SITE VISIT
chef_c['out_avg_time_per_site'] = 0
condition_hi = chef_c.loc[0:,'out_avg_time_per_site'][chef_c['AVG_TIME_PER_SITE_VISIT'] > avg_time_per_site_hi]

chef_c['out_avg_time_per_site'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

#CANCELLATIONS BEFORE NOON
chef_c['out_cancel_before'] = 0
condition_hi = chef_c.loc[0:,'out_cancel_before'][chef_c['CANCELLATIONS_BEFORE_NOON'] > cancel_before_hi]

chef_c['out_cancel_before'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

#CANCELLATIONS AFTER NOON
chef_c['out_cancel_after'] = 0
condition_hi = chef_c.loc[0:,'out_cancel_after'][chef_c['CANCELLATIONS_AFTER_NOON'] > cancel_after_hi]

chef_c['out_cancel_after'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)


#MOBILE LOGINS
chef_c['out_mobile_log'] = 0
condition_hi = chef_c.loc[0:,'out_mobile_log'][chef_c['MOBILE_LOGINS'] > mobile_log_hi]
condition_lo = chef_c.loc[0:,'out_mobile_log'][chef_c['MOBILE_LOGINS'] < mobile_log_lo]

chef_c['out_mobile_log'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chef_c['out_mobile_log'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)




#PC LOGINS
chef_c['out_pc_log'] = 0
condition_hi = chef_c.loc[0:,'out_pc_log'][chef_c['PC_LOGINS'] > pc_log_hi]

chef_c['out_pc_log'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

#WEEKLY PLAN  
chef_c['out_weekly_plan'] = 0
condition_hi = chef_c.loc[0:,'out_weekly_plan'][chef_c['WEEKLY_PLAN'] > weekly_plan_hi]

chef_c['out_weekly_plan'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

#EARLY DELIVERIES
chef_c['out_early_deli'] = 0
condition_hi = chef_c.loc[0:,'out_early_deli'][chef_c['EARLY_DELIVERIES'] > early_deli_hi]

chef_c['out_early_deli'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

#LATE DELIVERIES
chef_c['out_late_deli'] = 0
condition_hi = chef_c.loc[0:,'out_late_deli'][chef_c['LATE_DELIVERIES'] > late_deli_hi]

chef_c['out_late_deli'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)



#FOLLOWED RECOMMENDATIONS PERCENTAGE
chef_c['out_followed_pct'] = 0
condition_hi = chef_c.loc[0:,'out_followed_pct'][chef_c['FOLLOWED_RECOMMENDATIONS_PCT'] > followed_pct_hi]
condition_lo = chef_c.loc[0:,'out_followed_pct'][chef_c['FOLLOWED_RECOMMENDATIONS_PCT'] < followed_pct_lo]

chef_c['out_followed_pct'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chef_c['out_followed_pct'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)


#AVGERAGE PREPERATION VIDEO TIME
chef_c['out_avg_prep_vid'] = 0
condition_hi = chef_c.loc[0:,'out_avg_prep_vid'][chef_c['AVG_PREP_VID_TIME'] > avg_prep_vid_hi]
condition_lo = chef_c.loc[0:,'out_avg_prep_vid'][chef_c['AVG_PREP_VID_TIME'] < avg_prep_vid_lo]

chef_c['out_avg_prep_vid'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chef_c['out_avg_prep_vid'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

#LARGEST ORDER SIZE
chef_c['out_largest_order'] = 0
condition_hi = chef_c.loc[0:,'out_largest_order'][chef_c['LARGEST_ORDER_SIZE'] > largest_order_hi]
condition_lo = chef_c.loc[0:,'out_largest_order'][chef_c['LARGEST_ORDER_SIZE'] < largest_order_lo]

chef_c['out_largest_order'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chef_c['out_largest_order'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)




#MASTER CLASSES ATTENDED
chef_c['out_master_class'] = 0
condition_hi = chef_c.loc[0:,'out_master_class'][chef_c['MASTER_CLASSES_ATTENDED'] > master_class_hi]
condition_lo = chef_c.loc[0:,'out_master_class'][chef_c['MASTER_CLASSES_ATTENDED'] < master_class_lo]

chef_c['out_master_class'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chef_c['out_master_class'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)


#MEDIAN MEAL RATING
chef_c['out_median_meal'] = 0
condition_hi = chef_c.loc[0:,'out_median_meal'][chef_c['MEDIAN_MEAL_RATING'] > median_meal_hi]
condition_lo = chef_c.loc[0:,'out_median_meal'][chef_c['MEDIAN_MEAL_RATING'] < median_meal_lo]

chef_c['out_median_meal'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chef_c['out_median_meal'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)


#AVG CLICKS PER VISIT
chef_c['out_avg_clicks'] = 0
condition_hi = chef_c.loc[0:,'out_avg_clicks'][chef_c['AVG_CLICKS_PER_VISIT'] > avg_clicks_hi]
condition_lo = chef_c.loc[0:,'out_avg_clicks'][chef_c['AVG_CLICKS_PER_VISIT'] < avg_clicks_lo]

chef_c['out_avg_clicks'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chef_c['out_avg_clicks'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)


#TOTAL PHOTOS VIEWED
chef_c['out_total_view'] = 0
condition_hi = chef_c.loc[0:,'out_total_view'][chef_c['TOTAL_PHOTOS_VIEWED'] > total_view_hi]

chef_c['out_total_view'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)


#NUMBER OF NAMES
chef_c['out_number_of_names'] = 0
condition_hi = chef_c.loc[0:,'out_number_of_names'][chef_c['NUMBER_OF_NAMES'] > number_of_names_hi]

chef_c['out_number_of_names'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

#REVENUE

chef_c['out_revenue'] = 0
condition_hi = chef_c.loc[0:,'out_revenue'][chef_c['REVENUE'] > revenue_hi]
condition_lo = chef_c.loc[0:,'out_revenue'][chef_c['REVENUE'] < revenue_lo]

chef_c['out_revenue'].replace(to_replace = condition_hi,
                                   value      = 1,
                                  inplace    = True)

chef_c['out_revenue'].replace(to_replace = condition_lo,
                                   value      = 1,
                                  inplace    = True)

######################################################################
#Created a dictionary wit significant values from regression
chef_dict={ 'logit_significant'    : ['MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON', 'TASTES_AND_PREFERENCES','FOLLOWED_RECOMMENDATIONS_PCT','NUMBER_OF_NAMES','JUNK_EMAIL','out_number_of_names'],
           
           'logit_full' : ['REVENUE', 'TOTAL_MEALS_ORDERED',
       'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE',
       'PRODUCT_CATEGORIES_VIEWED', 'AVG_TIME_PER_SITE_VISIT', 'MOBILE_NUMBER',
       'CANCELLATIONS_BEFORE_NOON', 'CANCELLATIONS_AFTER_NOON',
       'TASTES_AND_PREFERENCES', 'MOBILE_LOGINS', 'PC_LOGINS', 'WEEKLY_PLAN',
       'EARLY_DELIVERIES', 'LATE_DELIVERIES', 'PACKAGE_LOCKER',
       'REFRIGERATED_LOCKER', 'FOLLOWED_RECOMMENDATIONS_PCT',
       'AVG_PREP_VID_TIME', 'LARGEST_ORDER_SIZE', 'MASTER_CLASSES_ATTENDED',
       'MEDIAN_MEAL_RATING', 'AVG_CLICKS_PER_VISIT', 'TOTAL_PHOTOS_VIEWED',
       'm_FAMILY_NAME', 'NUMBER_OF_NAMES', 'JUNK_EMAIL', 'PERSONAL_EMAIL',
       'PROFESSIONAL_EMAIL', 'UNKNOWN_EMAIL', 'out_total_meal',
       'out_unique_meals', 'out_contacts_w', 'out_product_cat',
       'out_avg_time_per_site', 'out_cancel_before', 'out_cancel_after',
       'out_mobile_log', 'out_pc_log', 'out_weekly_plan', 'out_early_deli',
       'out_late_deli', 'out_followed_pct', 'out_avg_prep_vid',
       'out_largest_order', 'out_master_class', 'out_median_meal',
       'out_avg_clicks', 'out_total_view', 'out_number_of_names']
          }

# Data and Target significant features 
chef_data   =  chef_c.loc[ : , chef_dict['logit_significant']]
chef_target =  chef_c.loc[ : , 'CROSS_SELL_SUCCESS']

# INSTANTIATING StandardScaler()
scaler = StandardScaler()


# FITTING the data
scaler.fit(chef_data)


# TRANSFORMING the data
X_scaled     = scaler.transform(chef_data)


# converting to a DataFrame
X_scaled_df  = pd.DataFrame(X_scaled) 





################################################################################
# Train/Test Split
################################################################################

# use this space to set up testing and validation sets using train/test split

# train-test split with the scaled data
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
            X_scaled_df,
            chef_target,
            random_state = 222,
            test_size = 0.25,
            stratify = chef_target)

################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# use this space to instantiate, fit, and predict on your final model

model_name=[]
train_score=[]
test_score=[]
roc_auc=[]

model = GaussianNB()
mod_fit = model.fit(X_train_scaled, y_train_scaled)
y_pred = model.predict(X_test_scaled)
        
model_name.append('GaussianNB')
train_score.append(model.score(X_train_scaled, y_train_scaled).round(3))
test_score.append(model.score(X_test_scaled, y_test_scaled).round(3))
roc_auc.append(roc_auc_score(y_true = y_test_scaled,
                                     y_score = y_pred).round(3))






################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test)

test_score = roc_auc_score(y_true = y_test_scaled,
                                     y_score = y_pred).round(3)

print('AUC Score:',test_score)
