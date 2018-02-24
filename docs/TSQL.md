# Querying Data with Transact-SQL
***
Notes taken during/inspired by the edX course 'Querying Data with Transact-SQL - Microsoft: DAT201x' by Graeme Malcolm and Geoff Allix.

**_Course Handouts_**

* [Course Syllabus](../files/TSQL/Syllabus.PDF)
* [Getting Started Guide Including Install](../files/TSQL/DAT201x.PDF)
* [Adventure Works Entity Relationship Diagram](../files/TSQL/AW_ER.pdf)
* [Adeventure Works db install script](../files/TSQL/AW-oltp-install-script)  
NOTE: Remember to ensure read only access for everyone to the folder containing the .SQL and other files
* [GitHib Repo for course including course materials, slids, labs etc](https://microsoftlearning.github.io/QueryingT-SQL/)
* [A copy of the above materials should they be changed or removed](../files/TSQL/TSQLCourseFiles.zip)

**_Other useful links_**
* [Transact-SQL Refrence](https://docs.microsoft.com/en-gb/sql/t-sql/language-reference)


```r
library(DBI)
```

```
## Loading required package: methods
```

```r
# creates a connection to the SQL database
# note that "con" will be used later in each connection to the database
con <- DBI::dbConnect(odbc::odbc(), 
                      Driver = "SQL Server", 
                      Server = "localhost\\SQLEXPRESS", 
                      Database = "AdventureWorksLT", 
                      Trusted_Connection = "True")

# Sets knitr to use this connection as the default so we don't need to specify it for every chunk
knitr::opts_chunk$set(connection = "con")
```

## Introduction to Transact-SQL

SQL or Structured Query Language was first developed in the 1970s by IBM as a way of interacting with databases. Other vendors have specfic versions of SQL for instance Oracle is PL/SQL, Microsoft's implentation is TSQL or Transact SQL.  Both SQL Server (on prem) and Azure SQL Databases (cloud) use the same query language, however Azure is a subset of full TSQL since it some commands relate to local files and data functions within .NET that relate only to SQL Server.  However, as new features are added to Azure, some new commands are being added to Azure. 

SQL is a declarative language - you express what it is that you want, the results - rather than specifying the steps taken to acheive that - it is not prodecural like other programming languages, it is set theory based.  It is possible to write proceedural elements or steps within TSQL, however if this is ocurring a lot, it is perhaps better done in another language, which may also perform or run better.  

In databases, we typically talk about entities - one type of thing - which is contained in each table.  
* Entities are represented as relations (tables) 
* And entity attributes as domains (columns)

Most relationships are normalized, with relationships between primary and foreign keys.  This helps to reduce duplication, however there are instances where de-normalised data is desired.  

Schemas are namespaces for database objects - is shows a logical layout for all or part of a relational database, _"As part of a data dictionary, a database schema indicates how the entities that make up the database relate to one another, including tables, views, stored procedures, and more."_(see [Lucid Chart on Database Schemas](https://www.lucidchart.com/pages/database-diagram/database-schema)).  The process of creating a database schema is called data modelling.  

When referring to objects in a database, we could use a fully qualified name, such as:

* [server_name.][database_name.][schema_name.]object_name

This is only really relevant for SQL Server, since Azure will only work with one database at a time.  Most of the time we typically just use

* schema_name.object_name

The schema name somwtimes be discarded, but it is considered best practice to include this, since there is sometimes some ambiguity about tables e.g. if we have two tables - Product.Order and Customer.Order  - which order table is being referred to, that in the customer or product schema?

SQL has a number of SQL Statement Types:

* DML or Data Manipulation Language - SELECT, INSERT, UPDATE, DELETE
* DDL or Data Definition Language - CREATE, ALTER, DROP
* DCL or Data Control Language - GRANT, REVOKE, DENY

The course focuses on DML which is typically for working with data.  

**SELECT** statement has a number of possible sub-components:

* FROM [table]
* WHERE [condition for filtering rows]
* GROUP BY [arranges rows by groups]
* HAVING [condition for filtering groups]
* ORDER BY [sorts the output]

Whilst a SQL statement can look like English, it doesn't neccessarily run from top to bottom in terms of the sequence of elements that are run in a query.  For instance, the FROM is the first thing that will be run, then the WHERE filter will be run, then we GROUP BY, then  SELECT the columns we are interested in and finally ORDER the results.  This can be important when running some queries, which will be explored later in the course.  When we run a query, it is not an actual table in a database that is return but a set of rows or record set or subset.

### Data Types

There are a number of different data types in T-SQL as shown below, which are grouped in to a number of different types.  

<div class="figure">
<img src="images/TSQL/TSQLDataTypes.png" alt="Transact-SQL Data Types" width="742" />
<p class="caption">(\#fig:Data Types)Transact-SQL Data Types</p>
</div>

This is more relevant when designing a database, however it is useful to know when querying what data type you have in a broad sense - numeric, data, string and so on - as the types will determine what type of combinations can be combined together in expressions  e.g. you can concatenate strings or add numbers together, but you can't concatenate a string and a number together. 

Sometimes it is neccessary to convert data from one type to another, there are two ways this could happen

* Implicit conversion - compatible data types are automatically converted 
* Explicit conversion - requires an explicit function e.g. CAST / TRY_CAST, STR, PARSE ? TRY_PARSE, CONVERT / TRY_CONVERT

The TRY options will attempt a conversion and if it does not work, a NULL will be returned rather than an error in the non-TRY version.   

### Working with NULLs

There are recognised standards for treating NULL values - ANSI - which says that anythign involving a NULL should return a NULL.  There are functions that help us handle NULL values:

* ISNULL(column/variable, value) - Returns _value_ (which you can specify) if the column or variable is NULL
* NULLIF(column/variable, value) - Returns NULL if the column or variable is a value - we are almost recoding a non-null to a null
* COALESCE (column/variable1, column/variable2, ...) - Returns the value of the first non-NULL column or variable in the list - for instance if contact details, someone might not have an email, so we might want a telephone number, if they don't have that, return an address etc

NULL is used to indicate an unknown or missing value. NULL is **not** equivalent to zero or an empty string.

ISNULL can be used like an IF function in excel, for instance:

> SELECT name, ISNULL(TRY_CAST(size AS Integer), 0) AS NumericSize  
  FROM SalesLT.Product;

In this instance, if there is a value that will be returned, if not, the NULL value will be returned as a 0.  

We can also use a CASE statement to return a value whilst integrating NULL in to our query, e.g.


```sql
SELECT name,  
    CASE size  
      WHEN 'S' THEN 'SMALL'  
      WHEN 'M' THEN 'MEDIUM'  
      WHEN 'L' THEN 'LARGE'  
      WHEN 'XL' THEN 'EXTRA LARGE'  
      ELSE ISNULL(Size, 'N/A')  
    END AS PRODUCT  
  FROM SalesLT.Product; 
```


<div class="knitsql-table">


Table: (\#tab:unnamed-chunk-1)Displaying records 1 - 10

name                         PRODUCT 
---------------------------  --------
HL Road Frame - Black, 58    58      
HL Road Frame - Red, 58      58      
Sport-100 Helmet, Red        N/A     
Sport-100 Helmet, Black      N/A     
Mountain Bike Socks, M       MEDIUM  
Mountain Bike Socks, L       LARGE   
Sport-100 Helmet, Blue       N/A     
AWC Logo Cap                 N/A     
Long-Sleeve Logo Jersey, S   SMALL   
Long-Sleeve Logo Jersey, M   MEDIUM  

</div>

### Lab Exercises

AdventureWorks Cycles is a company that sells directly to retailers, who then sell products to consumers. Each retailer that is an AdventureWorks customer has provided a named contact for all communication from AdventureWorks.

The sales manager at AdventureWorks has asked you to generate some reports containing details of the company's customers to support a direct sales campaign. Let's start with some basic exploration.

First we display the sales person, the customer's title, surname and telephone number


```sql
SELECT SalesPerson, Title + ' ' + LastName AS CustomerName, Phone
FROM SalesLT.Customer;
```


<div class="knitsql-table">


Table: (\#tab:unnamed-chunk-2)Displaying records 1 - 10

SalesPerson                CustomerName     Phone        
-------------------------  ---------------  -------------
adventure-works\pamela0    Mr. Gee          245-555-0173 
adventure-works\david8     Mr. Harris       170-555-0127 
adventure-works\jillian0   Ms. Carreras     279-555-0130 
adventure-works\jillian0   Ms. Gates        710-555-0173 
adventure-works\shu0       Mr. Harrington   828-555-0186 
adventure-works\linda3     Ms. Carroll      244-555-0112 
adventure-works\shu0       Mr. Gash         192-555-0173 
adventure-works\josé1      Ms. Garza        150-555-0127 
adventure-works\josé1      Ms. Harding      926-555-0159 
adventure-works\garrett1   Mr. Caprio       112-555-0191 

</div>

Next we cast the CustomerID column to a VARCHAR and concatenate with the CompanyName column


```sql
SELECT CAST(CustomerID AS VARCHAR) + ': ' + CompanyName AS CustomerCompany
FROM SalesLT.Customer;
```


<div class="knitsql-table">


|CustomerCompany               |
|:-----------------------------|
|1: A Bike Store               |
|2: Progressive Sports         |
|3: Advanced Bike Components   |
|4: Modular Cycle Systems      |
|5: Metropolitan Sports Supply |
|6: Aerobic Exercise Company   |
|7: Associated Bikes           |
|10: Rural Cycle Emporium      |
|11: Sharp Bikes               |
|12: Bikes and Motorbikes      |

</div>

