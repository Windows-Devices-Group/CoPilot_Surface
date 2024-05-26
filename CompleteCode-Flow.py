import streamlit as st
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import plotly.express as px
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import openai
import pyodbc
import urllib
from sqlalchemy import create_engine
import pandas as pd
from azure.identity import InteractiveBrowserCredential
from pandasai import SmartDataframe
import pandas as pd
from pandasai.llm import AzureOpenAI
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import base64
import pandasql as ps
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import seaborn as sns

#Initializing API Keys to use LLM
os.environ["AZURE_OPENAI_API_KEY"] = "a22e367d483f4718b9e96b1f52ce6d53"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://hulk-openai.openai.azure.com/"

#Reading the dataset
Sentiment_Data  = pd.read_csv("New_CoPilot_Data.csv")


def Sentiment_Score_Derivation(value):
    try:
        if value == "positive":
            return 1
        elif value == "negative":
            return -1
        else:
            return 0
    except Exception as e:
        err = f"An error occurred while deriving Sentiment Score: {e}"
        return err    

#Deriving Sentiment Score and Review Count columns into the dataset
Sentiment_Data["Sentiment_Score"] = Sentiment_Data["Sentiment"].apply(Sentiment_Score_Derivation)
Sentiment_Data["Review_Count"] = 1.0


# In[4]:


def convert_top_to_limit(sql):
    try:
        tokens = sql.upper().split()
        is_top_used = False

        for i, token in enumerate(tokens):
            if token == 'TOP':
                is_top_used = True
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    limit_value = tokens[i + 1]
                    # Remove TOP and insert LIMIT and value at the end
                    del tokens[i:i + 2]
                    tokens.insert(len(tokens), 'LIMIT')
                    tokens.insert(len(tokens), limit_value)
                    break  # Exit loop after successful conversion
                else:
                    raise ValueError("TOP operator should be followed by a number")

        return ' '.join(tokens) if is_top_used else sql
    except Exception as e:
        err = f"An error occurred while converting Top to Limit in SQL Query: {e}"
        return err


# In[5]:


def process_tablename(sql, table_name):
    try:
        x = sql.upper()
        query = x.replace(table_name.upper(), table_name)
        return query
    except Exception as e:
        err = f"An error occurred while processing table name in SQL query: {e}"
        return err


# In[6]:


def get_conversational_chain_quant(history):
    try:
        hist = """"""
        for i in history:
            hist = hist+"\nUser: "+i[0]
            if isinstance(i[1],pd.DataFrame):
                x = i[1].to_string()
            else:
                x = i[1]
            hist = hist+"\nResponse: "+x
        prompt_template = """
        
        If an user is asking for Summarize reviews of any product. Note that user is not seeking for reviews, user is seeking for all the Quantitative things of the product(Net Sentiment & Review Count) and also (Aspect wise sentiment and Aspect wise review count)
        So choose to Provide Net Sentiment and Review Count and Aspect wise sentiment and their respective review count and Union them in single table
        
        Example : If the user Quesiton is "Summarize reviews of CoPilot Produt"
        
        User seeks for net sentiment and aspect wise net sentiment of "CoPilot" Product and their respective review count in a single table

        Product - "CoPilot"
        Different Product_Family = Microsoft Copilot, Windows CoPilot, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile
        These are the different aspects : 'Microsoft Product', 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization','Accessibility'.

        CoPilot is Overall Product and Product_Family are different versions of CoPilot.
        
        IMPORTANT : IMPLEMENT 'LIKE' OPERATOR where every it is possible.
        
        Your response should be : Overall Sentiment is nothing but the net sentiment and overall review count of the product
        
                        Aspect Aspect_SENTIMENT REVIEW_COUNT
                    0 TOTAL 40 15000.0
                    1 Generic 31.8 2302.0
                    2 Microsoft Product 20.2 570.0
                    3 Productivity 58.9 397.0
                    4 Code Generation -1.2 345.0
                    5 Ease of Use 20.1 288.0
                    6 Interface -22.9 271.0
                    7 Connectivity -43.7 247.0
                    8 Compatibility -28.6 185.0
                    9 Innovation 52.9 170.0
                    10 Text Summarization/Generation 19.1 157.0
                    11 Reliability -44.7 152.0
                    12 Price 29.5 95.0
                    13 Customization/Personalization 18.9 90.0
                    14 Security/Privacy -41.3 75.0
                    15 Accessibility 16.7 6.0
                    
                    The Query has to be like this 
                    
                SELECT 'TOTAL' AS Aspect, 
                ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                SUM(Review_Count) AS Review_Count
                FROM Sentiment_Data
                WHERE Product_Family LIKE '%CoPilot for Mobile%'

                UNION

                SELECT Aspect, 
                ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                SUM(Review_Count) AS Review_Count
                FROM Sentiment_Data
                WHERE Product_Family LIKE '%CoPilot for Mobile%'
                GROUP BY Aspect

                ORDER BY Review_Count DESC

                    
                    
                IMPORTANT : if any particular Aspect "Code Generation" in user prompt:
                    

                        SELECT 'TOTAL' AS Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Sentiment_Data
                        WHERE Product_Family LIKE '%CoPilot for Mobile%'

                        UNION

                        SELECT Aspect, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Aspect_Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Sentiment_Data
                        WHERE Product_Family LIKE '%CoPilot for Mobile%'
                        GROUP BY Aspect
                        HAVING Aspect LIKE %'Code Generation'%

                        ORDER BY Review_Count DESC
                
                
        This is aspect wise summary. If a user wants in Geography level 
        
        SELECT 'TOTAL' AS Geography, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Net Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Sentiment_Data
                        WHERE Product_Family LIKE '%CoPilot for Mobile%'
                        
            UNION
        
        
            SELECT Geography, 
                        ROUND((SUM(Sentiment_Score) / SUM(Review_Count)) * 100, 1) AS Net Sentiment, 
                        SUM(Review_Count) AS Review_Count
                        FROM Sentiment_Data
                        WHERE Product_Family LIKE '%CoPilot for Mobile%'
                        GROUP BY Geography

                        ORDER BY Review_Count DESC
                        
            You shold respond like this. Same Goes for all the segregation
            
        It the user wants to compare features of 2 different ProductFamily, let's say "Github CoPilot" and "CoPilot for Microsoft 365". I want the aspect wise sentiment of both the devices in one table.
        
        IMPORTANT : Example USE THIS Query for COMPARISION -  "Compare different features of CoPilot for Mobile and GitHub CoPilot" 
        
        Query: 
        
                    SELECT 'GITHUB COPILOT' AS ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                    FROM Sentiment_Data
                    WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%'

                    UNION All

                    SELECT ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                    FROM Sentiment_Data
                    WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%'
                    GROUP BY ASPECT

                    UNION All

                    SELECT 'COPILOT FOR MICROSOFT 365' AS ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                    FROM Sentiment_Data
                    WHERE PRODUCT_FAMILY LIKE '%COPILOT FOR MICROSOFT 365%'

                    UNION All

                    SELECT ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                    FROM Sentiment_Data
                    WHERE PRODUCT_FAMILY LIKE '%COPILOT FOR MICROSOFT 365%'
                    GROUP BY ASPECT
                    
        IMPORTANT : Example USE THIS Query for COMPARISION Query - :  if only one aspect (Use always 'LIKE' OPERATOR) for ASPECT, GEOGRAPHY, PRODUCT_FAMILY, PRODUCT and so on while performing where condition. 
        
                    SELECT 'GITHUB COPILOT' AS ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                    FROM Sentiment_Data
                    WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%'

                    UNION All

                    SELECT ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                    FROM Sentiment_Data
                    WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%'
                    GROUP BY ASPECT
                    HAVING ASPECT LIKE '%CODE GENERATION%'

                    UNION All

                    SELECT 'COPILOT FOR MICROSOFT 365' AS ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                    FROM Sentiment_Data
                    WHERE PRODUCT_FAMILY LIKE '%COPILOT FOR MICROSOFT 365%'

                    UNION All

                    SELECT ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                    FROM Sentiment_Data
                    WHERE PRODUCT_FAMILY LIKE '%COPILOT FOR MICROSOFT 365%'
                    GROUP BY ASPECT
                    HAVING ASPECT LIKE '%CODE GENERATION%'
                    
                    
                    IMPORTANT : Do not use Order By here.
                    
            IMPORTANT : USE UNION ALL Everytime instead of UNION
                    
           If the user question is : Compare "Microsoft Product" feature of CoPilot for Mobile and GitHub CoPilot or "Compare the reviews for Github Copilot and Copilot for Microsoft 365 for Microsoft Product"
           
           DO NOT respond like :
           
           
            SELECT 'COPILOT FOR MOBILE' AS PRODUCT_FAMILY, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT 
            FROM Sentiment_Data 
            WHERE PRODUCT_FAMILY LIKE '%COPILOT FOR MOBILE%' 
            AND ASPECT = 'Microsoft Product'

            UNION ALL

            SELECT 'GITHUB COPILOT' AS PRODUCT_FAMILY, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT 
            FROM Sentiment_Data WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%' 
            AND ASPECT = 'Microsoft Product'
            
            
            Instead respond like : 
            
            
            SELECT 'COPILOT FOR MOBILE' AS ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT 
            FROM Sentiment_Data 
            WHERE PRODUCT_FAMILY LIKE '%COPILOT FOR MOBILE%' 
            AND ASPECT = '%Microsoft Product%'

            UNION ALL

            SELECT 'GITHUB COPILOT' AS ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT 
            FROM Sentiment_Data WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%' 
            AND ASPECT LIKE '%Microsoft Product%'
            
            IMPORTANT : Do not use Order By here.
            
            CHANGES MADE : USE OF LIKE OPERATOR, ASPECT as alias instead of Product_Family


        IMPORTANT : IT has to be Net sentiment and Aspect Sentiment. Create 2 SQL Query and UNION ALL them
        
        1. Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.You have to only return query as a result.
            2. There is only one table with table name Sentiment_Data where each row is a user review. The table has 10 columns, they are:
                Review: Review of the Copilot Product
                Data_Source: From where is the review taken. It contains different retailers
                Geography: From which Country or Region the review was given. It contains different Geography.
                Title: What is the title of the review
                Review_Date: The date on which the review was posted
                Product: Corresponding product for the review. It contains following values: "COPILOT"
                Product_Family: Which version or type of the corresponding Product was the review posted for. Different Device Names
                Sentiment: What is the sentiment of the review. It contains following values: 'Positive', 'Neutral', 'Negative'.
                Aspect: The review is talking about which aspect or feature of the product. It contains following values: "Audio-Microphone","Software","Performance","Storage/Memory","Keyboard","Browser","Connectivity","Hardware","Display","Graphics","Battery","Gaming","Design","Ports","Price","Camera","Customer-Service","Touchpad","Account","Generic"
                Keyword: What are the keywords mentioned in the product
                Review_Count - It will be 1 for each review or each row
                Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.
                
        ONLY FOLLOW these column names
                
            3. Sentiment mark is calculated by sum of Sentiment_Score.
            4. Net sentiment is calculcated by sum of Sentiment_Score divided by sum of Review_Count. It should be in percentage. Example:
                    SELECT ((SUM(Sentiment_Score)*1.0)/(SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment 
                    FROM Sentiment_Data
                    ORDER BY Net_Sentiment DESC
            5. Net sentiment across country or across region is sentiment mark of a country divided by total reviews of that country. It should be in percentage.
                Example to calculate net sentiment across country:
                    SELECT Geography, ((SUM(Sentiment_Score)*1.0) / (SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment
                    FROM Sentiment_Data
                    GROUP BY Geography
                    ORDER BY Net_Sentiment DESC
            6. Net Sentiment across a column "X" is calculcated by Sentiment Mark for each "X" divided by Total Reviews for each "X".
                Example to calculate net sentiment across a column "X":
                    SELECT X, ((SUM(Sentiment_Score)*1.0) / (SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment
                    FROM Sentiment_Data
                    GROUP BY X
                    ORDER BY Net_Sentiment DESC
            7. Distribution of sentiment is calculated by sum of Review_Count for each Sentiment divided by overall sum of Review_Count
                Example: 
                    SELECT Sentiment, SUM(ReviewCount)*100/(SELECT SUM(Review_Count) AS Reviews FROM Sentiment_Data) AS Total_Reviews 
                    FROM Sentiment_Data 
                    GROUP BY Sentiment
                    ORDER BY Total_Reviews DESC
            
            REMEBER TO USE LIKE OPERATOR whenever you use 'where' clause
                     
                    
            8. Convert numerical outputs to float upto 1 decimal point.
            9. Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
            10. Top Country is based on Sentiment_Score i.e., the Country which have highest sum(Sentiment_Score)
            11. Always use 'LIKE' operator whenever they mention about any Country. Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
            12. If you are using any field in the aggregate function in select statement, make sure you add them in GROUP BY Clause.
            13. Make sure to Give the result as the query so that it can be used on Microsoft SQL server SSMS.
            14. Important: Always show Net_Sentiment in Percentage upto 1 decimal point. Hence always make use of ROUND function while giving out Net Sentiment and Add % Symbol after it.
            15. Important: User can ask question about any categories including Aspects, Geograpgy, Sentiment etc etc. Hence, include the in SQL Query if someone ask it.
            16. Important: You Response should directly starts from SQL query nothing else.
            17. Important: Always use LIKE keyword instead of = symbol while generating SQL query.
            18. Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
            19. Sort all Quantifiable outcomes based on review count
        \n Following is the previous conversation from User and Response, use it to get context only:""" + hist + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Verbatim-Synthesis",
            api_version='2023-12-01-preview',
            temperature = 0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for quantifiable review summarization: {e}"
        return err

#Function to convert user prompt to quantitative outputs for Copilot Review Summarization
def query_quant(user_question, history, vector_store_path="faiss_index_CopilotSample"):
    try:
        # Initialize the embeddings model
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        
        # Load the vector store with the embeddings model
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Rest of the function remains unchanged
        chain = get_conversational_chain_quant(history)
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        SQL_Query = convert_top_to_limit(SQL_Query)
        SQL_Query = process_tablename(SQL_Query,"Sentiment_Data")
        # st.write(SQL_Query)
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        html_table = data.to_html(index=False)
    #     return html_table
        return data_1
    except Exception as e:
        err = f"An error occurred while generating response for quantitative review summarization: {e}"
        return err


def get_conversational_chain_aspect_wise_detailed_summary(history):
    try:
        hist = """"""
        for i in history:
            hist = hist+"\nUser: "+i[0]
            if isinstance(i[1],pd.DataFrame):
                x = i[1].to_string()
            else:
                x = i[1]
            hist = hist+"\nResponse: "+ x
        prompt_template = """
        
       
        1. Your Job is to analyse the Net Sentiment, Aspect wise sentiment and Key word regarding the different aspect and summarize the reviews that user asks for utilizing the reviews and numbers you get. Use maximum use of the numbers and Justify the numbers using the reviews.
        
        
        Your will receive Aspect wise net sentiment of the Product. you have to concentrate on top 4 Aspects based on ASPECT_RANKING.
        For that top 4 Aspect you will get top 2 keywords for each aspect. You will receive each keywords' contribution and +ve mention % and negative mention %
        You will receive reviews of that devices focused on these aspects and keywords.

        For Each Aspect

        Condition 1 : If the net sentiment is less than aspect sentiment, which means that particular aspect is driving the net sentiment Higher for that Product. In this case provide why the aspect sentiment is lower than net sentiment.
        Condition 2 : If the net sentiment is high than aspect sentiment, which means that particular aspect is driving the net sentiment Lower for that Product. In this case provide why the aspect sentiment is higher than net sentiment. 

        IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.

            Your summary should justify the above conditions and tie in with the net sentiment and aspect sentiment and keywords. Mention the difference between Net Sentiment and Aspect Sentiment (e.g., -2% or +2% higher than net sentiment) in your summary and provide justification.

            Your response should be : 

            For Each Aspect 
                    Net Sentiment of the Product and aspect sentiment of that aspect of the Product (Mention Code Generation, Aspect Sentiment) . 
                    Top Keyword contribution and their positive and negative percentages and summarize Reviews what user have spoken regarding this keywords in 2 to 3 lines detailed
                    Top 2nd Keyword contribution and their positive and negative percentages and summarize Reviews what user have spoken regarding this keywords in 2 to 3 lines detailed
                       Limit yourself to top 3 keywords and don't mention as top 1, top 2, top 3 and all. Mention them as pointers
                    Overall Summary

            IMPORTANT : Example Template :

            ALWAYS FOLLOW THIS TEMPLATE : Don't miss any of the below:

            Response : "BOLD ALL THE NUMBERS"

            IMPOPRTANT : Start with : "These are the 4 major aspects users commented about" and mention their review count contributions. These top 4 shold be based on ASPECT_RANKING Column

                           These are the 4 top ranked aspects users commented about - IMPORTANT : These top 4 should be from Aspect Ranking:
                           
                           IMPORTANT : DO NOT CONSIDER GENERIC AS ONE OF THE ASPECTS

                        - Total Review for CoPilot for Mobile Product is 1200
                        - Code Generarion: 4.82% of the reviews mentioned this aspect
                        - Ease of Use: 6% of the reviews mentioned this aspect
                        - Compatibility: 9.71% of the reviews mentioned this aspect
                        - Interface: 7.37% of the reviews mentioned this aspect

                        Code Generation:
                        - The aspect sentiment for price is 52.8%, which is higher than the net sentiment of 38.5%. This indicates that the aspect of price is driving the net sentiment higher for the Vivobook.
                        -  The top keyword for price is "buy" with a contribution of 28.07%. It has a positive percentage of 13.44% and a negative percentage of 4.48%.
                              - Users mentioned that the Vivobook offers good value for the price and is inexpensive.
                        - Another top keyword for price is "price" with a contribution of 26.89%. It has a positive percentage of 23.35% and a negative percentage of 0.24%.
                            - Users praised the affordable price of the Vivobook and mentioned that it is worth the money.

                        Ease of use:
                        - The aspect sentiment for performance is 36.5%, which is lower than the net sentiment of 38.5%. This indicates that the aspect of performance is driving the net sentiment lower for the Vivobook.
                        - The top keyword for performance is "fast" with a contribution of 18.24%. It has a positive percentage of 16.76% and a neutral percentage of 1.47%.
                            - Users mentioned that the Vivobook is fast and offers good speed.
                        - Another top keyword for performance is "speed" with a contribution of 12.06%. It has a positive percentage of 9.12% and a negative percentage of 2.06%.
                            - Users praised the speed of the Vivobook and mentioned that it is efficient.


                        lIKE THE ABOVE ONE EXPLAIN OTHER 2 ASPECTS

                        Overall Summary:
                        The net sentiment for the Vivobook is 38.5%, while the aspect sentiment for price is 52.8%, performance is 36.5%, software is 32.2%, and design is 61.9%. This indicates that the aspects of price and design are driving the net sentiment higher, while the aspects of performance and software are driving the net sentiment lower for the Vivobook. Users mentioned that the Vivobook offers good value for the price, is fast and efficient in performance, easy to set up and use in terms of software, and has a sleek and high-quality design.

                        Some Pros and Cons of the device, 


           IMPORTANT : Do not ever change the above template of Response. Give Spaces accordingly in the response to make it more readable.

           A Good Response should contains all the above mentioned poniters in the example. 
               1. Net Sentiment and The Aspect Sentiment
               2. Total % of mentions regarding the Aspect
               3. A Quick Summary of whether the aspect is driving the sentiment high or low
               4. Top Keyword: "Usable" (Contribution: 33.22%, Positive: 68.42%, Negative: 6.32%)
                    - Users have praised the usable experience on the Cobilot for Mobile, with many mentioning the smooth usage and easy to use
                    - Some users have reported experiencing lag while not very great to use, but overall, the gaming Ease of use is highly rated.

                Top 3 Keywords : Their Contribution, Postitive mention % and Negative mention % and one ot two positive mentions regarding this keywords in each pointer

                5. IMPORTANT : Pros and Cons in pointers (overall, not related to any aspect)
                6. Overall Summary
                    
          Enhance the model’s comprehension to accurately interpret user queries by:
          Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
          Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
          Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
          Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
          Generate acurate response only, do not provide extra information.
            
            Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.\n Following is the previous conversation from User and Response, use it to get context only:""" + hist + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Verbatim-Synthesis",
            api_version='2023-12-01-preview',
            temperature = 0.0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def query_aspect_wise_detailed_summary(user_question, history, vector_store_path="faiss_index_CopilotSample"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_aspect_wise_detailed_summary(history)
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err


# In[8]:


def get_conversational_chain_detailed_compare():
    try:
        prompt_template = """
        
            IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.
            
        Product = Microsoft Copilot, Copilot in Windows 11, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile

        
        1. Your Job is to Summarize the user reviews and sentiment data you get as an input for 2 different Product that user mentioned.
        
        IMPORTANT : Mention their Positive and Negative of each Product for each aspects (What consumer feels) for each aspect.
        
        Example :
        
        Summary of CoPilot for Mobile and Github Copilot:
        

        Positive:

        CoPilot for Mobile has a high sentiment score for aspects such as Productivity, Ease of Use, and Accessibility.
        Users find it helpful for various tasks and appreciate its quick and informative responses.
        The app is praised for its usefulness in both work and everyday life.
        Negative:

        Some users have reported issues with Connectivity and Reliability, mentioning network problems and automatic closing of the app.
        There are concerns about Security/Privacy, with users mentioning the potential for data misuse.
        Compatibility with certain devices and interfaces is also mentioned as an area for improvement.
        Summary of GitHub CoPilot:

        Positive:

        GitHub CoPilot receives positive sentiment for aspects such as Microsoft Product and Innovation.
        Users appreciate its code generation capabilities and find it helpful for their programming tasks.
        The app is praised for its accuracy and ability to provide quick and relevant responses.
        Negative:

        Some users have reported issues with Reliability and Compatibility, mentioning problems with generating images and recognizing certain commands.
        There are concerns about Security/Privacy, with users mentioning the potential for data misuse.
        Users also mention the need for improvements in the app's interface and connectivity.
        Overall, both CoPilot for Mobile and GitHub CoPilot have received positive feedback for their productivity and code generation capabilities. However, there are areas for improvement such as connectivity, reliability, compatibility, and security/privacy. Users appreciate the ease of use and quick responses provided by both apps.
     
        
        IMPORTANT : If user asks to compare any specific aspects of two device, Give detailed summary like how much reviews is being spoken that aspect in each device, net sentiment and theire Pros and cons on that device (Very Detailed).
        
            Summary of Code Generation feature for CoPilot for Mobile:

                    Positive:

                    Users have praised the Code Generation feature of CoPilot for Mobile, with a high sentiment score of 8.5.
                    The feature is described as helpful and efficient in generating code that aligns with project standards and practices.
                    Users appreciate the convenience and time-saving aspect of the Code Generation feature.
                    Negative:

                    No negative reviews or concerns were mentioned specifically for the Code Generation feature of CoPilot for Mobile.
                    Summary of Code Generation feature for GitHub CoPilot:

                    Positive:

                    Users have a positive sentiment towards the Code Generation feature of GitHub CoPilot, with a sentiment score of 5.4.
                    The feature is described as a game-changer for developer productivity.
                    Users appreciate the ability of GitHub CoPilot to generate code that aligns with project standards and practices.
                    Negative:

                    No negative reviews or concerns were mentioned specifically for the Code Generation feature of GitHub CoPilot.
                    Overall, both CoPilot for Mobile and GitHub CoPilot have received positive feedback for their Code Generation capabilities. Users find the feature helpful, efficient, and a game-changer for developer productivity. No negative reviews or concerns were mentioned for the Code Generation feature of either product.
        
        Give a detailed summary for each aspects using the reviews. Use maximum use of the reviews. Do not use your pretrained data. Use the data provided to you. For each aspects. Summary should be 3 ro 4 lines

                    
          Enhance the model’s comprehension to accurately interpret user queries by:
          Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
          Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
          Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
          Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
          Generate acurate response only, do not provide extra information.
            
            Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Verbatim-Synthesis",
            api_version='2023-12-01-preview',
            temperature = 0.2)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def query_detailed_compare(user_question, vector_store_path="faiss_index_CopilotSample"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_detailed_compare()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err
        
        
        
def get_conversational_chain_generic():
    try:
        prompt_template = """
        
            IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.
            
            Given a dataset with these columns: Review, Data_Source, Geography, Product_Family, Sentiment and Aspect (also called Features)
                      
                      Review: This column contains the opinions and experiences of users regarding different product families across geographies, providing insights into customer satisfaction or complaints and areas for improvement.
                      Data_Source: This column indicates the platform from which the user reviews were collected, such as Reddit, Play Store, App Store, Tech Websites, or YouTube videos.
                      Geography: This column lists the countries of the users who provided the reviews, allowing for an analysis of regional preferences and perceptions of the products.
                      Product_Family: This column identifies the broader category of products to which the review pertains, enabling comparisons and trend analysis across different product families.
                      Sentiment: This column reflects the overall tone of the review, whether positive, negative, or neutral, and is crucial for gauging customer sentiment.
                      Aspect: This column highlights the particular features or attributes of the product that the review discusses, pinpointing areas of strength or concern.
                      
                      Perform the required task from the list below, as per user's query: 
                      1. Review Summarization - Summarize the reviews by filtering the relevant Aspect, Geography, Product_Family, Sentiment or Data_Source, only based on available reviews and their sentiments in the dataset.
                      2. Aspect Comparison - Provide a summarized comparison for each overlapping feature/aspect between the product families or geographies ,  only based on available user reviews and their sentiments in the dataset. Include pointers for each aspect highlighting the key differences between the product families or geographies, along with the positive and negative sentiments as per customer perception.
                      3. New Feature Suggestion/Recommendation - Generate feature suggestions or improvements or recommendations based on the frequency and sentiment of reviews and mentioned aspects and keywords. Show detailed responses to user queries by analyzing review sentiment, specific aspects, and keywords.
                      4. Hypothetical Reviews - Based on varying customer sentiments for the reviews in the existing dataset, generate hypothetical reviews for any existing feature updation or new feature addition in any device family across any geography, by simulating user reactions. Ensure to synthesize realistic reviews that capture all types of sentiments and opinions of users, by considering their hypothetical prior experience working with the new feature and generate output based on data present in dataset only. After these, provide solutions/remedies for negative hypothetical reviews. 
                      
                      Enhance the model’s comprehension to accurately interpret user queries by:
                      Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
                      Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
                      Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
                      Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
                      Generate acurate response only, do not provide extra information.
            
            Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
            
            If the user question is not in the data provided. Just mention - "Not in the context". 
            But do not restrict yourself in responding to the user questions like 'hello', 'Hi' and basic chat question
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Verbatim-Synthesis",
            api_version='2023-12-01-preview',
            temperature = 0.2)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def query_detailed_generic(user_question, vector_store_path="faiss_index_CopilotSample"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_generic()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err


# In[9]:


def get_conversational_chain_detailed_summary(history):
    try:
        hist = """"""
        for i in history:
            hist = hist+"\nUser: "+i[0]
            if isinstance(i[1],pd.DataFrame):
                x = i[1].to_string()
            else:
                x = i[1]
            hist = hist+"\nResponse: "+ x
        prompt_template = """
        
            IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.
            
            
            'None' is Unknown Geography
            
            "Total" is Overall Net Sentiment

        
        1. Your Job is to analyse the Net Sentiment, Geo-Wise wise sentiment of particular product or Product-wise sentiment and summarize the reviews that user asks for utilizing the reviews and numbers you get. Use maximum use of the numbers and Justify the numbers using the reviews.
        
        Total Sentiment is the Overall Sentiment. 
        
        You have to decide how your response is going to look like based on the data you received based on whether is it Product-wise net sentiment or Geo-wise net net sentiment or something else.
        
        If the data you receive is Geography wise net sentiment data for a particular product
        
            Overall summary of the data that you receive like, from which Geography most of the reviews are from which have the most and least net sentiment.
            
            And then Go into Geography wise.
            
            With the help of the reviews, summarize reviews from each geography and Provide Pros and Cons from each Geography about that Product
            
            Like wise Provide for all the Geography
    
        2. Same Goes for all the segregation. (It might be Geo-wise sentiment, Product-wise sentiment)
        
        
        Your Response Should Follow the below Template:
        
        1. Based on the provided sentiment data for Github CoPilot reviews from different geographies, here is a summary:

            2. Total Sentiment: The overall net sentiment for Github CoPilot is 6.9, based on a total of 3,735 reviews.

            3. 1st Geography Geography: The net sentiment for reviews with unknown geography is 5.2, based on 2,212 reviews. (It is driving the sentiment high/Low)
            
                Overall summary of the Product from that Geography in 5 to 6 lines
                Give Some Pros and Cons of the Product from the reviews from this Geography

            4. 2nd Geography: The net sentiment for reviews from the United States is 8.1, based on 1,358 reviews.  (It is driving the sentiment high/Low)
            
                Overall summary of the Product from that Geography in 5 to 6 lines
                Give Some Pros and Cons of the Product from the reviews from this Geography

            5. 3rd Geography: The net sentiment for reviews from Japan is 20.0, based on 165 reviews.  (It is driving the sentiment high/Low)
            
                Overall summary of the Product from that Geography in 5 to 6 lines
                Give Some Pros and Cons of the Product from the reviews from this Geography
            
            
            ProductFamily wise summary Template:Donot change the TEMPLATES. Stick with the same template
            
            
                Based on the provided sentiment data for different Product Families, here is a summary:

                1. Total Sentiment: The overall net sentiment for all the reviews is 12.3, based on a total of 50,928 reviews.

                2. Copilot for Mobile: The net sentiment for reviews of Copilot for Mobile is 29.5, based on 18,559 reviews. (It is driving the sentiment high)

                   Overall summary of Copilot for Mobile: Users have highly positive reviews for Copilot for Mobile, praising its functionality and ease of use. They find it extremely helpful in their mobile development tasks and appreciate the regular updates and additions to the toolkit.

                3. Copilot: The net sentiment for reviews of Copilot is -8.0, based on 10,747 reviews. (It is driving the sentiment low)

                   Overall summary of Copilot: Reviews for Copilot are mostly negative, with users expressing dissatisfaction with its performance and suggesting improvements. They mention issues with suggestions and accuracy, leading to frustration and disappointment.

                4. Copilot in Windows 11: The net sentiment for reviews of Copilot in Windows 11 is 8.3, based on 6,107 reviews. (It is driving the sentiment high)

                   Overall summary of Copilot in Windows 11: Users have positive reviews for Copilot in Windows 11, highlighting its compatibility and ease of use. They find it helpful in their development tasks and appreciate the integration with the Windows 11 operating system.

                5. Copilot Pro: The net sentiment for reviews of Copilot Pro is 12.7, based on 5,075 reviews. (It is driving the sentiment high)

                   Overall summary of Copilot Pro: Users have highly positive reviews for Copilot Pro, praising its advanced features and capabilities. They find it valuable for their professional development tasks and appreciate the additional functionalities offered in the Pro version.

                6. Github Copilot: The net sentiment for reviews of Github Copilot is 6.9, based on 3,735 reviews. (It is driving the sentiment high)

                   Overall summary of Github Copilot: Users have generally positive reviews for Github Copilot, mentioning its usefulness in their coding tasks. They appreciate the AI-powered suggestions and find it helpful in improving their productivity.

                7. Microsoft Copilot: The net sentiment for reviews of Microsoft Copilot is -2.4, based on 2,636 reviews. (It is driving the sentiment low)

                   Overall summary of Microsoft Copilot: Reviews for Microsoft Copilot are mostly negative, with users expressing dissatisfaction with its performance and suggesting improvements. They mention issues with accuracy and compatibility, leading to frustration and disappointment.

                8. Copilot for Security: The net sentiment for reviews of Copilot for Security is 9.4, based on 2,038 reviews. (It is driving the sentiment high)

                   Overall summary of Copilot for Security: Users have positive reviews for Copilot for Security, mentioning its effectiveness in enhancing security measures. They find it valuable for protecting sensitive information and appreciate the various customization options offered.

                9. Copilot for Microsoft 365: The net sentiment for reviews of Copilot for Microsoft 365 is 4.0, based on 2,031 reviews. (It is driving the sentiment low)

                   Overall summary of Copilot for Microsoft 365: Reviews for Copilot for Microsoft 365 are mostly neutral, with users expressing mixed opinions about its functionality. Some find it helpful in their Microsoft 365 tasks, while others mention limitations and suggest improvements.

                Based on the sentiment data, it can be observed that Copilot for Mobile, Copilot in Windows 11, Copilot Pro, and Copilot for Security have higher net sentiments, indicating positive user experiences. On the other hand, Copilot, Microsoft Copilot, and Copilot for Microsoft 365 have lower net sentiments, indicating negative or mixed user experiences.
        
        
            For Each Geography

        Condition 1 : If the Overall net sentiment is less than Geography sentiment, which means that particular Geography is driving the net sentiment Higher for that Product. In this case provide why the Geography sentiment is lower than net sentiment.
        Condition 2 : If the Overall net sentiment is high than Geography sentiment, which means that particular Geography is driving the net sentiment Lower for that Product. In this case provide why the Geography sentiment is higher than net sentiment. 

                    
          Enhance the model’s comprehension to accurately interpret user queries by:
          Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
          Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
          Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
          Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
          Generate acurate response only, do not provide extra information.
            
            Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.\n Following is the previous conversation from User and Response, use it to get context only:""" + hist + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Verbatim-Synthesis",
            api_version='2023-12-01-preview',
            temperature = 0.0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def query_detailed_summary(user_question, history, vector_store_path="faiss_index_CopilotSample"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_detailed_summary(history)
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err
        
        

def make_desired_df(data):
    try:
        # Create DataFrame from the dictionary
        df = pd.DataFrame(data)
        
        # Ensure the necessary columns are present
        if 'ASPECT_SENTIMENT' not in df.columns or 'REVIEW_COUNT' not in df.columns or 'ASPECT' not in df.columns:
            raise ValueError("Input data must contain 'ASPECT', 'ASPECT_SENTIMENT' and 'REVIEW_COUNT' columns")
        
        df = df[df['ASPECT_SENTIMENT'] != 0]
        df = df[df['ASPECT'] != 'Generic']

#         df = df[(df['ASPECT_SENTIMENT'] != 0) & (df['ASPECT'] != 'TOTAL') & (df['ASPECT'] != 'Generic')]

        # Compute min and max values for normalization
        min_sentiment = df['ASPECT_SENTIMENT'].min(skipna=True)
        max_sentiment = df['ASPECT_SENTIMENT'].max(skipna=True)
        min_review_count = df['REVIEW_COUNT'].min(skipna=True)
        max_review_count = df['REVIEW_COUNT'].max(skipna=True)

        # Apply min-max normalization for ASPECT_SENTIMENT
        df['NORMALIZED_SENTIMENT'] = df.apply(
            lambda row: (row['ASPECT_SENTIMENT'] - min_sentiment) / (max_sentiment - min_sentiment)
            if pd.notnull(row['ASPECT_SENTIMENT'])
            else None,
            axis=1
        )

        # Apply min-max normalization for REVIEW_COUNT
        df['NORMALIZED_REVIEW_COUNT'] = df.apply(
            lambda row: (row['REVIEW_COUNT'] - min_review_count) / (max_review_count - min_review_count)
            if pd.notnull(row['REVIEW_COUNT'])
            else None,
            axis=1
        )

        # Calculate the aspect ranking based on normalized values
        weight_for_sentiment = 1
        weight_for_review_count = 3
        
#         df['ASPECT_RANKING'] = df.apply(
#             lambda row: (weight_for_review_count * row['NORMALIZED_REVIEW_COUNT'] * (1 - weight_for_review_count*row['NORMALIZED_SENTIMENT'])
#             if pd.notnull(row['NORMALIZED_SENTIMENT']) and pd.notnull(row['NORMALIZED_REVIEW_COUNT'])
#             else None),
#             axis=1
        df['ASPECT_RANKING'] = df.apply(
            lambda row: (weight_for_sentiment * (1 - row['NORMALIZED_SENTIMENT']) + weight_for_review_count * row['NORMALIZED_REVIEW_COUNT'])
            if pd.notnull(row['NORMALIZED_SENTIMENT']) and pd.notnull(row['NORMALIZED_REVIEW_COUNT'])
            else None,
            axis=1
        )
        
        # fg
        # Assign integer rankings based on the 'Aspect_Ranking' score
        df['ASPECT_RANKING'] = df['ASPECT_RANKING'].rank(method='max', ascending=False, na_option='bottom').astype('Int64')

        # Sort the DataFrame based on 'Aspect_Ranking' to get the final ranking
        df_sorted = df.sort_values(by='ASPECT_RANKING')
        df_sorted = df_sorted.drop(columns=['NORMALIZED_SENTIMENT', 'NORMALIZED_REVIEW_COUNT'])
        
        # Extract and display the net sentiment and overall review count
        # try:
            # total_row = df[df['ASPECT'] == 'TOTAL'].iloc[0]
            # net_sentiment = str(int(total_row["ASPECT_SENTIMENT"])) + '%'
            # overall_review_count = int(total_row["REVIEW_COUNT"])
        # except (ValueError, TypeError, IndexError):
            # try:
                # total_row = df[df['ASPECT'] == device_a].iloc[0]
                # net_sentiment = str(int(total_row["ASPECT_SENTIMENT"])) + '%'
                # overall_review_count = int(total_row["REVIEW_COUNT"])
            # except:
                # try:
                    # total_row = df[df['ASPECT'] == device_a].iloc[0]
                    # net_sentiment = str(int(total_row["ASPECT_SENTIMENT"])) + '%'
                    # overall_review_count = int(total_row["REVIEW_COUNT"])
                # except:
                    # st.write("Failed")
                    # net_sentiment = total_row["ASPECT_SENTIMENT"]
                    # overall_review_count = total_row["REVIEW_COUNT"]

        # st.write(f"Net Sentiment: {net_sentiment}")
        # st.write(f"Overall Review Count: {overall_review_count}")

        return df_sorted
    except Exception as e:
        st.error(f"Error in make_desired_df: {str(e)}")
        return pd.DataFrame()


import numpy as np

def custom_color_gradient(val, vmin=-100, vmax=100):
    green_hex = '#347c47'
    middle_hex = '#dcdcdc'
    lower_hex = '#b0343c'
    
    # Adjust the normalization to set the middle value as 0
    try:
        # Normalize the value to be between -1 and 1 with 0 as the midpoint
        normalized_val = (val - vmin) / (vmax - vmin) * 2 - 1
    except ZeroDivisionError:
        normalized_val = 0
    
    if normalized_val <= 0:
        # Interpolate between lower_hex and middle_hex for values <= 0
        r = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[1:3], 16), int(middle_hex[1:3], 16)]))
        g = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[3:5], 16), int(middle_hex[3:5], 16)]))
        b = int(np.interp(normalized_val, [-1, 0], [int(lower_hex[5:7], 16), int(middle_hex[5:7], 16)]))
    else:
        # Interpolate between middle_hex and green_hex for values > 0
        r = int(np.interp(normalized_val, [0, 1], [int(middle_hex[1:3], 16), int(green_hex[1:3], 16)]))
        g = int(np.interp(normalized_val, [0, 1], [int(middle_hex[3:5], 16), int(green_hex[3:5], 16)]))
        b = int(np.interp(normalized_val, [0, 1], [int(middle_hex[5:7], 16), int(green_hex[5:7], 16)]))
    
    # Convert interpolated RGB values to hex format for CSS color styling
    hex_color = f'#{r:02x}{g:02x}{b:02x}'
    
    return f'background-color: {hex_color}; color: black;'


# In[11]:


def get_final_df(aspects_list,device):
    final_df = pd.DataFrame()
    device = device
    aspects_list = aspects_list

    # Iterate over each aspect and execute the query
    for aspect in aspects_list:
        # Construct the SQL query for the current aspect
        query = f"""
        SELECT Keywords,
               COUNT(CASE WHEN Sentiment = 'positive' THEN 1 END) AS Positive_Count,
               COUNT(CASE WHEN Sentiment = 'negative' THEN 1 END) AS Negative_Count,
               COUNT(CASE WHEN Sentiment = 'neutral' THEN 1 END) AS Neutral_Count,
               COUNT(*) as Total_Count
        FROM Sentiment_Data
        WHERE Aspect = '{aspect}' AND Product_Family LIKE '%{device}%'
        GROUP BY Keywords
        ORDER BY Total_Count DESC;
        """

        # Execute the query and get the result in 'key_df'
        key_df = ps.sqldf(query, globals())

        # Calculate percentages and keyword contribution
        total_aspect_count = key_df['Total_Count'].sum()
        key_df['Positive_Percentage'] = (key_df['Positive_Count'] / total_aspect_count) * 100
        key_df['Negative_Percentage'] = (key_df['Negative_Count'] / total_aspect_count) * 100
        key_df['Neutral_Percentage'] = (key_df['Neutral_Count'] / total_aspect_count) * 100
        key_df['Keyword_Contribution'] = (key_df['Total_Count'] / total_aspect_count) * 100

        # Drop the count columns
        key_df = key_df.drop(['Positive_Count', 'Negative_Count', 'Neutral_Count', 'Total_Count'], axis=1)

        # Add the current aspect to the DataFrame
        key_df['Aspect'] = aspect

        # Sort by 'Keyword_Contribution' and select the top 2 for the current aspect
        key_df = key_df.sort_values(by='Keyword_Contribution', ascending=False).head(2)

        # Append the results to the final DataFrame
        final_df = pd.concat([final_df, key_df], ignore_index=True)
        
    return final_df


def classify(user_question):
    try:
        prompt_template = """
            Given an input, classify it into one of two categories:
            
            Product = Microsoft Copilot, Copilot in Windows 11, Copilot, Github Copilot , Copilot for Security, Copilot Pro, Copilot for Microsoft 365, Copilot for Mobile
            
            1stFlow: The user_question should focus more on one Product (How does that Product Perform or Summarize that Product reviews ) Then choose the 1st flow.
            2ndFlow: User is seeking any other information like geography wise performance or any quantitative numbers like what is net sentiment for different product families then categorize as 2ndFlow. It should even choose 2nd flow, if it asks for Aspect wise sentiment of one Product.
            
            Example - Geography wise how products are performing or seeking for information across different product families/products.
            What is net sentiment for any particular product/geography
            
        IMPORTANT : Only share the classified category name, no other extra words.
        IMPORTANT : Don't categorize into 1stFlow or 2ndFlow based on number of products, categorize based on the type of question the user is asking
        Input: User Question
        Output: Category (1stFlow or 2ndFlow)
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Verbatim-Synthesis",
            api_version='2023-12-01-preview',
            temperature = 0.0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)
        if "1stflow" in response["output_text"].lower():
            return "1"
        elif "2ndflow" in response["output_text"].lower():
            return "2"
        else:
            return "Others"+"\nPrompt Identified as:"+response["output_text"]+"\n"
    except Exception as e:
        err = f"An error occurred while generating conversation chain for identifying nature of prompt: {e}"
        return err

# In[13]:


#Function to generate chart based on output dataframe 

def generate_chart(df):
    # Determine the data types of the columns
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime']).columns
    
    if len(df.columns)>3:
        df=df.iloc[:,0:len(df.columns)-1]
    
    # Simple heuristic to determine the most suitable chart
    if len(df.columns)==2:
        
        if len(num_cols) == 1 and len(cat_cols) == 0:

            plt.figure(figsize=(10, 6))
            sns.histplot(df[num_cols[0]], kde=True)
            plt.title(f"Frequency Distribution of '{num_cols[0]}'")
            st.pyplot(plt)


        elif len(num_cols) == 2:
   
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[num_cols[0]], y=df[num_cols[1]])
            plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
            st.pyplot(plt)


        elif len(cat_cols) == 1 and len(num_cols) == 1:
            if df[cat_cols[0]].nunique() <= 5 and df[num_cols[0]].sum()>=99 and df[num_cols[0]].sum()<=101:
                fig = px.pie(df, names=cat_cols[0], values=num_cols[0], title=f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
                st.plotly_chart(fig)

            else:
                num_categories=df[cat_cols[0]].nunique()
                width = 800
                height = max(600,num_categories*50)
                
                bar=px.bar(df,x=num_cols[0],y=cat_cols[0],title=f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'",text=num_cols[0])
                bar.update_traces(textposition='outside', textfont_size=12)
                bar.update_layout(width=width, height=height)
                st.plotly_chart(bar)


        elif len(cat_cols) == 2:

            plt.figure(figsize=(10, 6))
            sns.countplot(x=df[cat_cols[0]], hue=df[cat_cols[1]], data=df)
            plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
            st.pyplot(plt)


        elif len(date_cols) == 1 and len(num_cols) == 1:
   
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=df[date_cols[0]], y=df[num_cols[0]], data=df)
            plt.title(f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
            st.pyplot(plt)


        else:
            sns.pairplot(df)
            st.pyplot(plt)
            
    
            
    elif len(df.columns)==3 and len(cat_cols)>=1:
        
        col_types = df.dtypes

        cat_col = None
        num_cols = []

        for col in df.columns:
            if col_types[col] == 'object' and df[col].nunique() == len(df):
                categorical_col = col
            elif col_types[col] in ['int64', 'float64']:
                num_cols.append(col)

        # Check if we have one categorical and two numerical columns
        if len(cat_cols)==1 and len(num_cols) == 2:
            df[cat_cols[0]]=df[cat_cols[0]].astype(str)
            df[cat_cols[0]]=df[cat_cols[0]].fillna('NA')
            
            if df[cat_cols[0]].nunique() <= 5 and df[num_cols[0]].sum()>=99 and df[num_cols[0]].sum()<=101:
                fig = px.pie(df, names=cat_cols[0], values=num_cols[0], title=f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'")
                fig2 = px.pie(df, names=cat_cols[0], values=num_cols[1], title=f"Distribution of '{num_cols[1]}' across '{cat_cols[0]}'")
                st.plotly_chart(fig)
                st.plotly_chart(fig2)
                

            else:
                num_categories=df[cat_cols[0]].nunique()
                width = 800
                height = max(600,num_categories*50)
                
                bar=px.bar(df,x=num_cols[0],y=cat_cols[0],title=f"Distribution of '{num_cols[0]}' across '{cat_cols[0]}'",text=num_cols[0])
                bar.update_traces(textposition='outside', textfont_size=12)
                bar.update_layout(width=width, height=height)
                st.plotly_chart(bar)
                
                bar2=px.bar(df,x=num_cols[1],y=cat_cols[0],title=f"Distribution of '{num_cols[1]}' across '{cat_cols[0]}'",text=num_cols[1])
                bar2.update_traces(textposition='outside', textfont_size=12)
                bar2.update_layout(width=width, height=height)
                st.plotly_chart(bar2)
                
        elif len(cat_cols)==2 and len(num_cols) == 1:
            df[cat_cols[0]]=df[cat_cols[0]].astype(str)
            df[cat_cols[1]]=df[cat_cols[1]].astype(str)
            df[cat_cols[0]]=df[cat_cols[0]].fillna('NA')
            df[cat_cols[1]]=df[cat_cols[1]].fillna('NA')
            
            
            num_categories=df[cat_cols[0]].nunique()
            num_categories2=df[cat_cols[1]].nunique()
            height = max(800,num_categories2*100)
            width = max(600,num_categories*100)
            
            bar=px.bar(df,y=num_cols[0],x=cat_cols[0],title=f"Distribution of '{num_cols[0]}' across '{cat_cols[1]}' within '{cat_cols[0]}'",text=num_cols[0],color=cat_cols[1])
            bar.update_traces(textposition='outside', textfont_size=12)
            bar.update_layout(width=width, height=height)
            st.plotly_chart(bar)


def get_conversational_chain_detailed_deepdive(history):
    try:
        hist = """"""
        for i in history:
            hist = hist+"\nUser: "+i[0]
            if isinstance(i[1],pd.DataFrame):
                x = i[1].to_string()
            else:
                x = i[1]
            hist = hist+"\nResponse: "+ x
        prompt_template = """
        
        1. Your Job is to analyse the Net Sentiment Aspect wise sentiment and Key word regarding the aspect and summarize the reviews that user asks for utilizing the reviews and numbers you get. Use maximum use of the numbers and Justify the numbers using the reviews.
        
        Overall Sentiment is the Net Sentiment.
        
        Condition 1 : If the net sentiment is less than aspect sentiment, which means that particular aspect is driving the net sentiment Lower for that Product. In this case provide why the aspect sentiment is lower than net sentiment by using reviews as justificaton point.
        Condition 2 : If the net sentiment is high than aspect sentiment, which means that particular aspect is driving the net sentiment Higher for that Product. In this case provide why the aspect sentiment is higher than net sentiment by using reviews as justificaton point.
            
            You must be receiving keywords information. If there are any keywords which have more keyword_contribution mention that keyword with its contribution percentage and Positive, negative percentages. 
            Give the reviews summarized for this aspect 
            
            Give at least top 2 keyword information - (Contribution , Positive and Negative Percentage) and when summarizing reviews focus on those particular keywords.
            

            IMPORTANT: Use only the data provided to you and do not rely on pre-trained documents.

            Your summary should justify the above conditions and tie in with the net sentiment and aspect sentiment and keywords. Mention the difference between Net Sentiment and Aspect Sentiment (e.g., -2% or +2% higher than net sentiment) in your summary and provide justification.
            
            
            Example Template :
            
            IMPORTANT : ALWAYS FOLLOW THIS TEMPLATE : Don't miss any of the below: 1st Template

                    Net Sentiment: 41.9%
                    Aspect Sentiment (Interface): 53.1%

                    75% of the users commented about Interface of this Product. Interface drives the sentiment high for CoPilot for Mobile Product

                    Top Keyword: User-Friendly (Contribution: 33.22%, Positive: 68.42%, Negative: 6.32%)
                    - Users have praised the User-Friendly experience on the CoPilot for Mobile, with many mentioning the good layout and interfacce
                    - Some users have reported experiencing lag while gaming, but overall, the gaming performance is highly rated.

                    Top 2nd Keyword: Graphical (Contribution: 33.22%, Positive: 60%, Negative: 8.42%)
                    - Users appreciate the ability to play various games on the Lenovo Legion, mentioning the enjoyable gaming experience.
                    - A few users have mentioned encountering some issues with certain games, but the majority have had a positive experience.

                    Top 3rd Keyword: Play (Contribution: 16.08%, Positive: 56.52%, Negative: 13.04%)
                    - Users mention the ease of playing games on the Lenovo Legion, highlighting the smooth gameplay and enjoyable experience.
                    - Some users have reported difficulties with certain games, experiencing lag or other performance issues.

                    Pros:
                    1. Smooth gameplay experience
                    2. High FPS and enjoyable gaming performance
                    3. Wide range of games available
                    4. Positive feedback on gaming experience
                    5. Ease of playing games

                    Cons:
                    1. Some users have reported lag or performance issues while gaming
                    2. Occasional difficulties with certain games

                    Overall Summary:
                    The net sentiment for the CoPilot for Mobile is 41.9%, while the aspect sentiment for Inteface is 53.1%. This indicates that the Interface aspect is driving the net sentiment higher for the product. Users have praised the smooth gameplay, high FPS, and enjoyable gaming experience on the Lenovo Legion. The top keywords related to gaming contribute significantly to the aspect sentiment, with positive percentages ranging from 56.52% to 68.42%. However, there are some reports of lag and performance issues with certain games. Overall, the Lenovo Legion is highly regarded for its gaming capabilities, but there is room for improvement in addressing performance issues for a seamless gaming experience.
               
           IMPORTANT : Do not ever change the above template of Response. Give Spaces accordingly in the response to make it more readable.
           
           A Good Response should contains all the above mentioned poniters in the example. 
               1. Net Sentiment and The Aspect Sentiment
               2. Total % of mentions regarding the Aspect
               3. A Quick Summary of whether the aspect is driving the sentiment high or low
               4. Top Keyword: Gaming (Contribution: 33.22%, Positive: 68.42%, Negative: 6.32%)
                    - Users have praised the gaming experience on the Lenovo Legion, with many mentioning the smooth gameplay and high FPS.
                    - Some users have reported experiencing lag while gaming, but overall, the gaming performance is highly rated.
                    
                Top 3 Keywords : Their Contribution, Postitive mention % and Negative mention % and one ot two positive mentions regarding this keywords in each pointer
                
                5. Pros and Cons in pointers
                6. Overall Summary. 
                
        IMPORTANT : Only follow this template. Donot miss out any poniters from the above template

                    
          Enhance the model’s comprehension to accurately interpret user queries by:
          Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
          Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘copilot in windows 11’ as ‘copilot windows’ and ‘copilot for security’ as ‘copilot security’ etc.).
          Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references
          Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families and actionable insights that can help in product development and marketing strategies.
          Generate acurate response only, do not provide extra information.
            
            Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.\n Following is the previous conversation from User and Response, use it to get context only:""" + hist + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Verbatim-Synthesis",
            api_version='2023-12-01-preview',
            temperature = 0.0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def query_detailed_deepdive(user_question, history, vector_store_path="faiss_index_CopilotSample"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_detailed_deepdive(history)
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err
        
        
def get_conversational_chain_quant_classify2():
    try:
#         hist = """"""
#         for i in history:
#             hist = hist+"\nUser: "+i[0]
#             if isinstance(i[1],pd.DataFrame):
#                 x = i[1].to_string()
#             else:
#                 x = i[1]
#             hist = hist+"\nResponse: "+x

#################################################################################################################################################################################################################################################
        prompt_template = """
        
        Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.You have to only return query as a result.
        There is only one table with table name Sentiment_Data where each row is a user review. The table has 10 columns, they are:
                Review: Review of the Copilot Product
                Data_Source: From where is the review taken. It contains different retailers
                Geography: From which Country or Region the review was given. It contains different Geography. The user might mention Geography as Geography/Geographies/Regions
                Title: What is the title of the review
                Review_Date: The date on which the review was posted
                Product: Corresponding product for the review. It contains the value "COPILOT"
                Product_Family: Which version or type of the corresponding Product was the review posted for. Different Product Families are "Windows Copilot" , "Microsoft Copilot" , "Github Copilot" , "Copilot Pro" , "Copilot for Security" , "Copilot for Mobile", "Copilot for Microsoft 365"
                Sentiment: What is the sentiment of the review. It contains following values: 'Positive', 'Neutral', 'Negative'.
                Aspect: The review is talking about which aspect or feature of the product. It contains following values: ['Microsoft Product', 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization']
                Keyword: What are the keywords mentioned in the product
                Review_Count - It will be 1 for each review or each row
                Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.
                
            1. If the user asks for count of column 'X', the query should be like this:
                    SELECT COUNT(DISTINCT ('X')) 
                    FROM Sentiment_Data
            2. If the user asks for count of column 'X' for different values of column 'Y', the query should be like this:
                    SELECT 'Y', COUNT(DISTINCT('X')) AS Total_Count
                    FROM Sentiment_Data 
                    GROUP BY 'Y'
                    ORDER BY TOTAL_COUNT DESC
            3. If the user asks for Net overall sentiment the query should be like this:
                    SELECT ((SUM(Sentiment_Score))/(SUM(Review_Count))) * 100 AS Net_Sentiment 
                    FROM Sentiment_Data
                    ORDER BY Net_Sentiment DESC
                    
            4. If the user asks for Net Sentiment across a column "X", the query should be like this and do not include WHERE condition: 
                    Here X can be Geography/Product Family or even a particular value
                    SELECT X, ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment, SUM(Review_Count) AS Review_Count
                    FROM Sentiment_Data
                    GROUP BY X
                    ORDER BY Net_Sentiment DESC
                    
                 
                    
            5. If the user asks for Net Sentiment across a column "X" for a particular column Y, the query should be like this:
                    SELECT X, ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment, SUM(Review_Count) AS Review_Count
                    FROM Sentiment_Data
                    WHERE Y LIKE '%Y%'
                    GROUP BY X
                    ORDER BY Net_Sentiment DESC
                    
                    
            7. If the user asks for Net Sentiment across column "X" across different values of column "Y" for a particular value "Z", the query should be like this:
                    Example - Give me aspect wise net sentiment for different regions/geographies for Windows Copilot
                    
                    SELECT X, Y, ((SUM(Sentiment_Score)) / (SUM(Review_Count))) * 100 AS Net_Sentiment
                    FROM Sentiment_Data
                    WHERE Y LIKE '%Z%'
                    GROUP BY X,Y
                    ORDER BY Net_Sentiment DESC
                    
                    
            8. If the user asks for overall review count, the query should be like this:
                    SELECT SUM(Review_Count) 
                    FROM Sentiment_Data
            9. If the user asks for review distribution across column 'X', the query should be like this:
                    SELECT 'X', SUM(Review_Count) * 100 / (SELECT SUM(Review_Count) FROM Sentiment_Data) AS Review_Distribution
                    FROM Sentiment_Data 
                    GROUP BY 'X'
                    ORDER BY Review_Distribution DESC
            10. If the user asks for column 'X' Distribution across column 'Y', the query should be like this: 
                    SELECT 'Y', SUM('X') * 100 / (SELECT SUM('X') AS Reviews FROM Sentiment_Data) AS Distribution_PCT
                    FROM Sentiment_Data 
                    GROUP BY 'Y'
                    ORDER BY Distribution_PCT DESC
                    
            11. SELECT ASPECT, ROUND((SUM(SENTIMENT_SCORE) / SUM(REVIEW_COUNT)) * 100, 1) AS ASPECT_SENTIMENT, SUM(REVIEW_COUNT) AS REVIEW_COUNT
                    FROM Sentiment_Data
                    WHERE PRODUCT_FAMILY LIKE '%GITHUB COPILOT%'
                    GROUP BY ASPECT
                    HAVING ASPECT LIKE '%CODE GENERATION%'
                    
                    USE LIKE OPERATOR while filtering (USING WHERE CLAUSE) for PRODUCT_FAMILY, ASPECT, and wherever necessary.
                    
            Important: Always replace '=' operator with LIKE keyword and add '%' before and after filter value for single or multiple WHERE conditions in the generated SQL query 
            Important : Replace '=' operator with LIKE keyword for WHERE conditions for single or multiple columns, for example - product family  or aspect or geography
            If there are multiple columns mentioned in WHERE condition, then also replace "=" operator with LIKE keyword
            Include WHERE condition in the query only if user is asking for a particular column/value.
            Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
            Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
            Important: You Response should directly start from SQL query nothing else.
            Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
        
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
########################################################################################################################################
#########################################################################################
        

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Verbatim-Synthesis",
            api_version='2024-03-01-preview',
            temperature = 0.3)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for quantifiable review summarization: {e}"
        return err

#Function to convert user prompt to quantitative outputs for Copilot Review Summarization
def query_quant_classify2(user_question, vector_store_path="faiss_index_CopilotSample"):
    try:
        # Initialize the embeddings model
        embeddings = AzureOpenAIEmbeddings(azure_deployment="Embedding-Model")
        
        # Load the vector store with the embeddings model
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Rest of the function remains unchanged
        chain = get_conversational_chain_quant_classify2()
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        # st.write(SQL_Query)
        SQL_Query = convert_top_to_limit(SQL_Query)
        SQL_Query = process_tablename(SQL_Query,"Sentiment_Data")
        # st.write(SQL_Query)
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        html_table = data.to_html(index=False)
    #     return html_table
        return data_1
    except Exception as e:
        err = f"An error occurred while generating response for quantitative review summarization: {e}"
        return err



def quantifiable_data(user_question):
    try:
        response = query_quant_classify2(user_question)
        
        return response
    except Exception as e:
        err = f"An error occurred while generating quantitative review summarization: {e}"
        return err


def split_table(data,device_a,device_b):
    # Initialize empty lists for each product
    copilot_index = data[data["ASPECT"] == str(device_b).upper()].index[0]
    if copilot_index != 0:
        device_a_table = data.iloc[:copilot_index]
        device_b_table = data.iloc[copilot_index:]
    else:
        copilot_index = data[data["ASPECT"] == str(device_a).upper()].index[0]
        device_a_table = data.iloc[:copilot_index]
        device_b_table = data.iloc[copilot_index:]

    return device_a_table, device_b_table
    
    
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("672370cd6ca440f2a0327351d4f4d2bf"),  
    api_version="2024-02-01",
    azure_endpoint = os.getenv("https://hulk-openai.openai.azure.com/")
    )
    
deployment_name='SurfaceGenAI'

context_Prompt = """

As a data scientist analyzing the sentiment data of the Copilot product, we have developed several features to facilitate the synthesis of consumer review sentiment data. 

[Questions regarding Pros and Cons, Top verbatims, and similar inquiries are not applicable here.] [Here, ‘Device’ is synonymous with ‘Product_Family’.] We have created the following list of features:

note : Verbatims means raw reviews

List of aspects : ['Microsoft Product', 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization']
List of Product_Families : ["Windows Copilot" , "Microsoft Copilot" , "Github Copilot" , "Copilot Pro" , "Copilot for Security" , "Copilot for Mobile", "Copilot for Microsoft 365"]

Quantifiable and visualization - This feature enables the retrieval and visualization of data for any requested product/feature. It can answer queries like “Which is the best device?” (Based on Net Sentiment) or “Which device is most commonly commented on?” (Based on Review Count), among others.
Comparison - This feature allows users to compare two different Products based on user reviews. Remember that this function only does comparision for 2 different Products.
    IMPORTANT : If the user Question mentions 3 or more different Product Families. Then don't give it as Comparision . Make it as Generic. Example : Compare Github Copilot, Wnidows CopIlot and CoPilot Pro. In this case it should choose "Generic", as 3 Product were mentioned in this user query.
Generic - This category allows users to ask general questions about any Product, such as the Pros and Cons, common complaints associated with a device, and the top verbatims (Reviews) mentioned in product reviews, etc.
Summarization of reviews for a specific Product - This feature provides a summary of the most frequently mentioned aspects of a device, offering both quantifiable and detailed sentiment analysis. (Don't choose this functionc, if the user asks for basic pros and cons, top verbatims and all)

If user question just mentioned verbatims for devices. Provide Generic
Your task is to categorize incoming user queries into one of these four features.
Your response should be one of the following:

“Summarization”
“Quantifiable and visualization”
“Comparison”
“Generic”
"""

def finetuned_prompt(user_question):
    global context_Prompt
    # Append the new question to the context
    full_prompt = context_Prompt + "\nQuestion:\n" + user_question + "\nAnswer:"
    # Send the query to Azure OpenAI
    response = client.completions.create(
        model=deployment_name,
        prompt=full_prompt,
        max_tokens=500,
        temperature=0
    )
    
    # Extract the generated SQL query
    user_query = response.choices[0].text.strip()
    
    # Update context with the latest interaction
    context_Prompt += "\nQuestion:\n" + user_question + "\nAnswer:\n" + user_query
    
    return user_query   
    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["AZURE_OPENAI_API_KEY"] = "672370cd6ca440f2a0327351d4f4d2bf"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://hulk-openai.openai.azure.com/"


client = AzureOpenAI(
    api_key=os.getenv("672370cd6ca440f2a0327351d4f4d2bf"),  
    api_version="2024-02-01",
    azure_endpoint = os.getenv("https://hulk-openai.openai.azure.com/")
    )
    
deployment_name='SurfaceGenAI'

context = """
You are given a list of product names and a mapping file that maps these names to their corresponding product families. Your task is two-fold:

1. Rephrase any input sentence by replacing the product name with its correct product family according to the mapping file.
2. Modify the input sentence into one of the specified Features.

Features and sample prompts:
    1. Comparison - "Compare different features for [Product 1] and [Product 2]"
    2. Specific Feature comparison - "Compare [Feature] feature of [Product 1] and [Product 2]
        Features are : ['Microsoft Product', 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization']
    3. Summarization of reviews - "Summarize the reviews for [Product] / Analyze consumer reviews for [Product]"
    4. Asking net sentiment or review count  - "What is the net sentiment and review count for [product 1]"
        4.1. It can be across any categories such as Product Family, Geography, Data Source etc. Hence repharse the input sentence accordingly.
    5. Asking net sentiment or review count for multiple Products - "What is the net sentiment and review count of different Product families?". Do not enter multiple product names in this case.

IMPORTAT : Net Sentiment is different and aspect sentiment is different. Don't rephrase those questions. If the user askes aspect wise sentiment, let it be there as aspect wise sentiment.

Mapping file:

Copilot in Windows 11 -> Windows Copilot
Copilot for Security -> Copilot for Security
Copilot Pro -> Copilot Pro
Microsoft Copilot -> Microsoft Copilot
Copilot for Microsoft 365 -> Copilot for Microsoft 365
Github Copilot -> Github Copilot
Copilot for Mobile -> Copilot for Mobile
Windows Copilot -> Windows Copilot
Copilot for Windows -> Windows Copilot
Copilot Windows -> Windows Copilot
Win Copilot -> Windows Copilot
Security Copilot -> Copilot for Security
Privacy Copilot -> Copilot for Security
M365 -> Copilot for Microsoft 365
Microsoft 365 -> Copilot for Microsoft 365
Office copilot -> Copilot for Microsoft 365
Github -> Github Copilot
MS Office -> Copilot for Microsoft 365
MSOffice -> Copilot for Microsoft 365
Microsoft Office -> Copilot for Microsoft 365
Office Product -> Copilot for Microsoft 365
Mobile -> Copilot for Mobile
App -> Copilot for Mobile
ios -> Copilot for Mobile
apk -> Copilot for Mobile
Copilot -> Microsoft Copilot

IMPORTANT: If the input sentence mentions a device(Laptop or Desktop) instead of Copilot, keep the device name as it is.

Rephrase the following input with the correct product family and modify it to fit one of the specified functionalities.


Please rephrase and modify the following input sentences with the correct product family names and into one of the specified formats:

[List of input sentences]
"""

def rephrased_prompt(user_question):
    global context
    # Append the new question to the context
    full_prompt = context + "\nQuestion:\n" + user_question + "\nAnswer:"
    
    # Send the query to Azure OpenAI
    response = client.completions.create(
        model=deployment_name,
        prompt=full_prompt,
        max_tokens=500,
        temperature=0
    )
    
    # Extract the generated SQL query
    user_query = response.choices[0].text.strip()
    # st.write(user_query)
    
    # Update context with the latest interaction
    context += "\nQuestion:\n" + user_question + "\nAnswer:\n" + user_query
    
    return user_query
    
def user_ques(user_question):
    if user_question:
        device_list = Sentiment_Data['Product_Family'].to_list()
        sorted_device_list_desc = sorted(device_list, key=lambda x: len(x), reverse=True)

    # Convert user question and product family names to lowercase for case-insensitive comparison
        user_question_lower = user_question.lower()

        # Initialize variables for device names
        device_a = None
        device_b = None

        # Search for product family names in the user question
        for device in sorted_device_list_desc:
            if device.lower() in user_question_lower:
                if device_a is None:
                    device_a = device
                else:
                    if device_a != device and device != 'Copilot':
                        device_b = device
                        break# Found both devices, exit the loop

        st.write(device_a)
        st.write(device_b)

        if device_a != None and device_b != None:
            col1,col2 = st.columns(2) 
            data = query_quant(user_question,[])
            # st.write(data)
            device_a_table,device_b_table = split_table(data,device_a,device_b)   
            with col1:
                device_a_table = device_a_table.dropna(subset=['ASPECT_SENTIMENT'])
                device_a_table = device_a_table[~device_a_table["ASPECT"].isin(["Generic", "Account", "Customer-Service", "Browser"])]
                device_a_table = device_a_table[device_a_table['ASPECT_SENTIMENT'] != 0]
                device_a_table = device_a_table[device_a_table['ASPECT'] != 'Generic']
                device_a_table = device_a_table.sort_values(by='REVIEW_COUNT', ascending=False)
                styled_df_a = device_a_table.style.applymap(lambda x: custom_color_gradient(x, int(-100), int(100)), subset=['ASPECT_SENTIMENT'])
                data_filtered = device_a_table[(device_a_table["ASPECT"] != device_a) | (device_a_table["ASPECT"] != device_b) & (device_a_table["ASPECT"] != 'Generic')]
                top_four_aspects = data_filtered.head(4)
                c = device_a_table.to_dict(orient='records')
                st.dataframe(styled_df_a)

            with col2:

                device_b_table = device_b_table.dropna(subset=['ASPECT_SENTIMENT'])
                device_b_table = device_b_table[~device_b_table["ASPECT"].isin(["Generic", "Account", "Customer-Service", "Browser"])]
                device_b_table = device_b_table[device_b_table['ASPECT_SENTIMENT'] != 0]
                device_b_table = device_b_table[device_b_table['ASPECT'] != 'Generic']
                device_b_table = device_b_table.sort_values(by='REVIEW_COUNT', ascending=False)
                styled_df_b = device_b_table.style.applymap(lambda x: custom_color_gradient(x, int(-100), int(100)), subset=['ASPECT_SENTIMENT'])
                data_filtered = device_b_table[(device_b_table["ASPECT"] != device_b) | (device_b_table["ASPECT"] != device_a) & (device_b_table["ASPECT"] != 'Generic')]
                top_four_aspects = data_filtered.head(4)
                d = device_b_table.to_dict(orient='records')
                st.dataframe(styled_df_b)
            try:
                user_question = user_question.replace("Compare", "Summarize reviews of")
            except:
                pass
            st.write(query_detailed_compare(user_question + "Which have the following sentiment data" + str(c)+str(d)))


        elif (device_a != None and device_b == None) | (device_a == None and device_b == None)  :

            data = query_quant(user_question,[]) 
            # st.write(data)
            try:
                total_reviews = data.loc[data.iloc[:, 0] == 'TOTAL', 'REVIEW_COUNT'].iloc[0]
            except:
                pass
            # total_reviews = data.loc[data['ASPECT'] == 'TOTAL', 'REVIEW_COUNT'].iloc[0]
            try:
                data['REVIEW_PERCENTAGE'] = data['REVIEW_COUNT'] / total_reviews * 100
            except:
                pass
            dataframe_as_dict = data.to_dict(orient='records')

            classify_function = classify(user_question+str(dataframe_as_dict))


            if classify_function == "1":
                data_new = data
                data_new = data_new.dropna(subset=['ASPECT_SENTIMENT'])
                data_new = data_new[~data_new["ASPECT"].isin(["Generic", "Account", "Customer-Service", "Browser"])]
                data_new = make_desired_df(data_new)
                styled_df = data_new.style.applymap(lambda x: custom_color_gradient(x, int(-100), int(100)), subset=['ASPECT_SENTIMENT'])
                data_filtered = data_new[(data_new['ASPECT'] != 'TOTAL') & (data_new['ASPECT'] != 'Generic')]
                top_four_aspects = data_filtered.head(4)
                dataframe_as_dict = data_new.to_dict(orient='records')
                aspects_list = top_four_aspects['ASPECT'].to_list()
        #         formatted_aspects = ', '.join(f"'{aspect}'" for aspect in aspects_list)
                key_df = get_final_df(aspects_list, device)
                b =  key_df.to_dict(orient='records')
                st.write(query_aspect_wise_detailed_summary(user_question+"which have the following sentiment :" + str(dataframe_as_dict) + "these are the imporatnt aspect based on aspect ranking : " + str(aspects_list) + "and their respective keywords" + str(b),[]))
                heat_map = st.checkbox("Would you like to see the Aspect wise sentiment of this Product?")
                if heat_map:
                    st.dataframe(styled_df)
                    aspect_names = ['Microsoft Product', 'Interface', 'Connectivity', 'Privacy','Compatibility', 'Generic', 'Innovation', 'Reliability','Productivity', 'Price', 'Text Summarization/Generation','Code Generation', 'Ease of Use', 'Performance','Personalization/Customization']
                    with st.form(key='my_form'):
                        aspect_wise_sentiment = st.markdown("Verbatims")
                        selected_aspect = st.selectbox('Select an aspect to see consumer reviews:', aspect_names)
                        submitted = st.form_submit_button('Submit')
                        if submitted:
                            query = f"""
                            SELECT Keywords,
                                   COUNT(CASE WHEN Sentiment = 'positive' THEN 1 END) AS Positive_Count,
                                   COUNT(CASE WHEN Sentiment = 'negative' THEN 1 END) AS Negative_Count,
                                   COUNT(CASE WHEN Sentiment = 'neutral' THEN 1 END) AS Neutral_Count,
                                   COUNT(*) as Total_Count
                            FROM Sentiment_Data
                            WHERE Aspect LIKE '%{selected_aspect}%' AND Product_Family LIKE '%{device}%'
                            GROUP BY Keywords
                            ORDER BY Total_Count DESC;
                            """
                            key_df = ps.sqldf(query, globals())
                            total_aspect_count = key_df['Total_Count'].sum()
                            key_df['Positive_Percentage'] = (key_df['Positive_Count'] / key_df['Total_Count']) * 100
                            key_df['Negative_Percentage'] = (key_df['Negative_Count'] / key_df['Total_Count']) * 100
                            key_df['Neutral_Percentage'] = (key_df['Neutral_Count'] / key_df['Total_Count']) * 100
                            key_df['Keyword_Contribution'] = (key_df['Total_Count'] / total_aspect_count) * 100
                            key_df = key_df.drop(['Positive_Count', 'Negative_Count', 'Neutral_Count', 'Total_Count'], axis=1)
                            key_df = key_df.head(10)
                            b =  key_df.to_dict(orient='records')
                            st.write(query_detailed_deepdive("Summarize reviews of" + device + "for " +  selected_aspect +  "Aspect which have following "+str(dataframe_as_dict)+ str(b) + "Reviews: ",[]))

            elif classify_function == "2":
                data= quantifiable_data(user_question)
                dataframe_as_dict = data.to_dict(orient='records')
                st.dataframe(data)
                try:
                    data = data.dropna()
                except:
                    pass
                generate_chart(data)
                try:
                    user_question = user_question.replace("What is the", "Summarize reviews of")
                except:
                    pass
                st.write(query_detailed_summary(user_question + "Which have the following sentiment data : " + str(dataframe_as_dict),[]))

        else:
            print('No Flow')
            
if __name__ == "__main__":
    st.header("Copilot Consumer Review Synthesis Tool")
    user_question = st.text_input("Enter the Prompt: ")
    if user_question:
        classification = 'Generic'
        classification = finetuned_prompt(user_question)
        print(classification)
        if classification != 'Generic':
            user_question_1 = rephrased_prompt(user_question)
            # st.write(user_question_1)
            user_ques(user_question_1)
        else:
            user_question_1 = user_question
            st.write(query_detailed_generic(user_question_1))
