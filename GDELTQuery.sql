SELECT
  g.MonthYear AS MonthYear,
  g.Actor1Name AS Actor1Name,
  g.Actor2Name AS Actor2Name,
  AVG(g.GoldsteinScale) AS GoldsteinScale,
  AVG(g.NumMentions) AS NumMentions,
  AVG(g.NumSources) AS NumSources,
  AVG(g.AvgTone) AS AvgTone,
  ActionGeo_CountryCode AS CountryCode
FROM
  [gdelt-bq:full.events] g
GROUP BY
  MonthYear,
  Actor1Name,
  Actor2Name,
  CountryCode
