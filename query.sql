
---source and destination joining
select a.keyword as S, b.keyword as D, a.art_id
from apnoea_modefied_keywords a
inner join apnoea_modefied_keywords b
    on a.art_id = b.art_id
and a.keyword < b.keyword
order by a.art_id;

---source and destination joining order by article year
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

