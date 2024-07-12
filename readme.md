## **Project Overview**



Sales forecasting enables businesses to allocate resources for future growth while managing cash flow properly. Sales forecasting also assists firms in precisely estimating their expenditures and revenue, allowing them to predict their short- and long-term success. 

Retail Sales Forecasting also assists retailers in meeting customer expectations by better understanding consumer purchasing trends. This results in more efficient use of shelf and display space within the retail establishment and optimal use of inventory space.
The Bigmart sales forecast project can help you comprehend project creation in a professional atmosphere. This project entails extracting and processing data in the Amazon Redshift database before further processing and building various machine-learning models for sales prediction. 


We will study several data processing techniques, exploratory data analysis, and categorical correlation with Chi-squared, Cramer’s v tests, and ANOVA. In addition to basic statistical models like Linear Regression, we will learn how to design cutting-edge machine-learning models like Gradient Boosting and Generalized Additive Models. We will investigate splines and multivariate adaptive regression splines (MARS), as well as ensemble techniques like model stacking and model blending, and evaluate these models for the best results.



## **Execution Instructions**

* Create a python environment using the command 'python3 -m venv myenv'.

* Activate the environment by running the command 'myenv\Scripts\activate.bat'.

* Install the requirements using the command 'pip install -r requirements.txt'

* Run engine.py with the command 'python3 engine.py'.



## **Commands to create AWS Redshift Cluster**


create table public.data(
Item_Identifier nvarchar(30),
Item_Weight numeric(10,2),
Item_Fat_Content nvarchar(40),
Item_Visibility numeric(10,6),
Item_Type nvarchar(40),
Item_MRP numeric(10,4),
Outlet_Identifier nvarchar(40),
Outlet_Establishment_Year int,
Outlet_Size nvarchar(40),
Outlet_Location_Type nvarchar(40),
Outlet_Type nvarchar(50),
Item_Outlet_Sales numeric(10,4)
)

copy public.data
(item_identifier,item_weight,item_fat_content,item_visibility,item_type,item_mrp,outlet_identifier,outlet_establishment_year
,outlet_size,outlet_location_type,outlet_type,item_outlet_sales)
from 's3://bigmart-data'
iam_role 'arn:aws:iam::143176219551:role/service-role/AmazonRedshift-CommandsAccessRole-20240206T122742'
Csv