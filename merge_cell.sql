SELECT art_id, keyword =
    STUFF((SELECT DISTINCT ', ' + keyword
           FROM apnoea_modefied_keywords b
           WHERE b.art_id = a.art_id
          FOR XML PATH('')), 1, 2, '')
FROM apnoea_modefied_keywords a
GROUP BY art_id;