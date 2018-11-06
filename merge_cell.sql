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