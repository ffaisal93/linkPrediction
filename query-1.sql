query

select keytable.art_id, keytable.keyword, article.art_year,
      article.citation, article.title, authortable.author_name,
affiliationtable.affiliation_1, affiliationtable.affiliation_2, affiliationtable.country
    from article
        INNER JOIN
(SELECT  art_id, keyword =
    STUFF((SELECT DISTINCT '; ' + keyword
           FROM apnoea_modefied_keywords b
           WHERE b.art_id = a.art_id
          FOR XML PATH('')), 1, 2, '')
FROM apnoea_modefied_keywords a
GROUP BY art_id) AS keytable
on article.id=keytable.art_id
INNER JOIN
        (SELECT  art_id,author_name =
    STUFF((SELECT '; ' + c.author_surname + ' '+ c.author_firstname
           FROM article_author c
           WHERE c.art_id = d.art_id
          FOR XML PATH('')), 1, 2, N'')
FROM article_author d
GROUP BY art_id) as authortable
on article.id=authortable.art_id
INNER JOIN
(SELECT  art_id,affiliation_1 =
    STUFF((SELECT DISTINCT '; ' + e.affiliation_1
           FROM article_affiliation e
           WHERE e.art_id = f.art_id
          FOR XML PATH('')), 1, 2, N''),
          affiliation_2 =
    STUFF((SELECT DISTINCT '; ' + e.affiliation_2
           FROM article_affiliation e
           WHERE e.art_id = f.art_id
          FOR XML PATH('')), 1, 2, N''),
          country =
     STUFF((SELECT DISTINCT '; ' + e.country
           FROM article_affiliation e
           WHERE e.art_id = f.art_id
          FOR XML PATH('')), 1, 2, N'')
FROM article_affiliation f
GROUP BY art_id) as affiliationtable
ON article.id=affiliationtable.art_id;















-----------------------------------------------------------------------------------------------------------





select top 10 percent *
from
(select A.S, A.D, B.art_year, A.art_id
    from
(select a.keyword as S, b.keyword as D, a.art_id
from apnoea_modefied_keywords a
inner join apnoea_modefied_keywords b
    on a.art_id = b.art_id
and a.keyword < b.keyword) as A
inner join article as B
    on  A.art_id=B.id) as mytable order by  mytable.art_year asc , newid();

--------
select id, art_year from article order by art_year;
-------------
select A.keyword, B.title, B.citation, A.article_id
from article_keywords as A
inner join article as B
    on A.article_id = B.id
where A.article_id =159890
-----------------------
select keyword, article_id
from article_keywords
where keyword like 'wc';
----------
select keyword from article_keywords where article_id=121637;
--------------------
select distinct keyword from article_keywords order by keyword asc;
-------------------------

select akk.article_id , keytable.newkey keyword, keytable.id key_id,
       article.art_year year, article.title title
from
(select  max(ck.new_keyword) as newkey,max(ck.old_keyword) as oldkey, max(ck.id) id,count(*) as total
from article_keywords as ak
inner join cleaned_keyword as ck
    on ak.keyword = ck.old_keyword
group by ck.new_keyword
having count(*)>2) as keytable
inner join article_keywords as akk
on akk.keyword = keytable.oldkey
inner join article
    on article.id = akk.article_id
order by article.art_year asc
----------------------------------------------------------
select ak.keyword old_key, ck.new_keyword new_key, ak.article_id art_id, ck.id key_id
from article_keywords ak
inner join cleaned_keyword ck
    on ak.keyword = ck.old_keyword;
----------------------------------------------------------
select  max(ck.new_keyword) as newkey,max(ck.old_keyword) as oldkey, count(*) as total
from article_keywords as ak
inner join cleaned_keyword as ck
    on ak.keyword = ck.old_keyword
group by ck.new_keyword
having count(*)>3






---------------------
select max(keyword), count(keyword) from article_keywords
group by keyword
having count(keyword)>1

---------------------
select keytable.art_id, keytable.keyword, article.art_year,
      article.citation, article.title, authortable.author_name,
affiliationtable.affiliation_1, affiliationtable.affiliation_2, affiliationtable.country
    from article
        INNER JOIN
(SELECT  art_id, keyword =
    STUFF((SELECT DISTINCT '; ' + keyword
           FROM three_degree_keyword_list b
           WHERE b.art_id = a.art_id
          FOR XML PATH('')), 1, 2, '')
FROM three_degree_keyword_list a
GROUP BY art_id) AS keytable
on article.id=keytable.art_id
INNER JOIN
        (SELECT  art_id,author_name =
    STUFF((SELECT '; ' + c.author_surname + ' '+ c.author_firstname
           FROM article_author c
           WHERE c.art_id = d.art_id
          FOR XML PATH('')), 1, 2, N'')
FROM article_author d
GROUP BY art_id) as authortable
on article.id=authortable.art_id
INNER JOIN
(SELECT  art_id,affiliation_1 =
    STUFF((SELECT DISTINCT '; ' + e.affiliation_1
           FROM article_affiliation e
           WHERE e.art_id = f.art_id
          FOR XML PATH('')), 1, 2, N''),
          affiliation_2 =
    STUFF((SELECT DISTINCT '; ' + e.affiliation_2
           FROM article_affiliation e
           WHERE e.art_id = f.art_id
          FOR XML PATH('')), 1, 2, N''),
          country =
     STUFF((SELECT DISTINCT '; ' + e.country
           FROM article_affiliation e
           WHERE e.art_id = f.art_id
          FOR XML PATH('')), 1, 2, N'')
FROM article_affiliation f
GROUP BY art_id) as affiliationtable
ON article.id=affiliationtable.art_id;
---------------------------------------------
select distinct_keyword.id,
       distinct_keyword.keyword new_keyword, cleaned_keyword.old_keyword old_keyword
from cleaned_keyword
inner join distinct_keyword
    on distinct_keyword.keyword=cleaned_keyword.new_keyword
--------------------------------------------
select max(key_id) from three_degree_keyword_list
---------------------------------------------
select * from article
where id=101772


use ResearchDatabaseObesity



select akk.article_id , keytable.newkey keyword, keytable.id key_id,
       article.art_year year, article.title title
from
(select  max(ck.new_keyword) as newkey,max(ck.old_keyword) as oldkey, max(ck.id) id,count(*) as total
from article_keywords as ak
inner join cleaned_keyword as ck
    on ak.keyword = ck.old_keyword
group by ck.new_keyword
having count(*)>0) as keytable
inner join article_keywords as akk
on akk.keyword = keytable.oldkey
inner join article
    on article.id = akk.article_id
order by article.art_year asc

---------------------------------------------------------------------------------------------------------obesity09-11
select keytable.art_id, keytable.keyword, article.art_year,
      article.citation, article.title, authortable.author_name,
affiliationtable.affiliation_1, affiliationtable.affiliation_2, affiliationtable.country
    from article_obesity2_Arif09_13 as article
        INNER JOIN
(SELECT  art_id, keyword =
    STUFF((SELECT DISTINCT '; ' + keyword
           FROM three_degree_keyword_list b
           WHERE b.art_id = a.art_id
          FOR XML PATH('')), 1, 2, '')
FROM three_degree_keyword_list a
GROUP BY art_id) AS keytable
on article.id=keytable.art_id
INNER JOIN
        (SELECT  art_id,author_name =
    STUFF((SELECT '; ' + c.author
           FROM art_author_obesity2_Arif09_13 c
           WHERE c.art_id = d.art_id
          FOR XML PATH('')), 1, 2, N'')
FROM art_author_obesity2_Arif09_13 d
GROUP BY art_id) as authortable
on article.id=authortable.art_id
INNER JOIN
(SELECT  art_id,affiliation_1 =
    STUFF((SELECT DISTINCT '; ' + e.art_aff_1
           FROM art_affiliation_obesity2_Arif09_13 e
           WHERE e.art_id = f.art_id
          FOR XML PATH('')), 1, 2, N''),
          affiliation_2 =
    STUFF((SELECT DISTINCT '; ' + e.art_aff_2
           FROM art_affiliation_obesity2_Arif09_13 e
           WHERE e.art_id = f.art_id
          FOR XML PATH('')), 1, 2, N''),
          country =
     STUFF((SELECT DISTINCT '; ' + e.country
           FROM art_affiliation_obesity2_Arif09_13 e
           WHERE e.art_id = f.art_id
          FOR XML PATH('')), 1, 2, N'')
FROM art_affiliation_obesity2_Arif09_13 f
GROUP BY art_id) as affiliationtable
ON article.id=affiliationtable.art_id;



[0.7175125019217716, 0.7538886106870882, 0.797863500860664, 0.8002849363198428, 0.7982936501691694, 0.7952978004128609, 0.7676728607279769, 0.7932854501319488]
['close', 'cm', 'typeaut', 'typeart', 'typenode', 'y_weight1', 'res_aloc', 'pref']


closeness = 0.72
common neighbour = 0.75
my feature = 0.80
resource allocation = 0.77
preferential attachment = 0.79